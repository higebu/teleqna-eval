package llm

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// Prompt caching is only available on the native Messages API: the
// OpenAI-compatible endpoint documents "Prompt caching is not supported", and
// its usage.prompt_tokens_details is always empty, so a chat-completions run
// pays full price for every re-sent tool result.
const anthropicVersion = "2023-06-01"

type AnthropicClient struct {
	BaseURL, APIKey, Model string
	MaxTokens              int
	Extra                  map[string]any
	HTTP                   *http.Client
}

func (c *AnthropicClient) headers() map[string]string {
	return map[string]string{"x-api-key": c.APIKey, "anthropic-version": anthropicVersion}
}

// MaxOutputTokens asks the Models API for the model's output ceiling. The
// Messages API requires max_tokens, so "-max-tokens 0" (no cap) means "as many
// as this model allows".
func (c *AnthropicClient) MaxOutputTokens() (int, error) {
	req, _ := http.NewRequest("GET", c.BaseURL+"/models/"+c.Model, nil)
	for k, v := range c.headers() {
		req.Header.Set(k, v)
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, err
	}
	var out struct {
		MaxTokens int `json:"max_tokens"`
	}
	if err := json.Unmarshal(data, &out); err != nil || out.MaxTokens == 0 {
		return 0, fmt.Errorf("models/%s: %.200s", c.Model, data)
	}
	return out.MaxTokens, nil
}

type AnthropicBackend struct {
	client *AnthropicClient
	tools  []map[string]any
}

func NewAnthropicBackend(c *AnthropicClient, tools []ToolDef) *AnthropicBackend {
	return &AnthropicBackend{client: c, tools: anthropicTools(tools)}
}

func anthropicTools(tools []ToolDef) []map[string]any {
	var out []map[string]any
	for _, t := range tools {
		out = append(out, map[string]any{
			"name": t.Name, "description": t.Description, "input_schema": t.InputSchema,
		})
	}
	return out
}

func (b *AnthropicBackend) NewConvo(system, user string) Convo {
	// Tools render before system, so a single breakpoint on the system block
	// caches the tool definitions and the system prompt together — the prefix
	// every question shares. (Below the model's minimum cacheable prefix it is
	// silently a no-op, which is the case for the no-tools baseline.)
	return &anthConvo{b: b,
		system: []any{map[string]any{
			"type": "text", "text": system,
			"cache_control": map[string]any{"type": "ephemeral"},
		}},
		messages: []map[string]any{{
			"role":    "user",
			"content": []any{map[string]any{"type": "text", "text": user}},
		}},
	}
}

// anthConvo keeps the history in Anthropic block form. Assistant content blocks
// are echoed back verbatim so thinking blocks survive the tool round-trip.
type anthConvo struct {
	b        *AnthropicBackend
	system   []any
	messages []map[string]any
	marks    []map[string]any // blocks currently carrying cache_control
}

// appendUserBlock merges into the trailing user message when there is one:
// all tool results of a round belong to a single user turn.
func (c *anthConvo) appendUserBlock(block map[string]any) {
	if len(c.messages) > 0 {
		last := c.messages[len(c.messages)-1]
		if blocks, ok := last["content"].([]any); ok && last["role"] == "user" {
			last["content"] = append(blocks, block)
			return
		}
	}
	c.messages = append(c.messages, map[string]any{"role": "user", "content": []any{block}})
}

func (c *anthConvo) AddUser(text string) {
	c.appendUserBlock(map[string]any{"type": "text", "text": text})
}

func (c *anthConvo) AddToolResult(tc ToolCall, content string) {
	c.appendUserBlock(map[string]any{
		"type": "tool_result", "tool_use_id": tc.ID, "content": content,
	})
}

// markLatest keeps a rolling cache breakpoint on the newest content block, so
// each round reads the prefix the previous round wrote. Two markers are kept:
// a breakpoint only walks back 20 blocks looking for a cache entry, and one
// round of parallel tool calls can add more blocks than that.
func (c *anthConvo) markLatest() {
	if len(c.messages) == 0 {
		return
	}
	blocks, ok := c.messages[len(c.messages)-1]["content"].([]any)
	if !ok || len(blocks) == 0 {
		return
	}
	blk, ok := blocks[len(blocks)-1].(map[string]any)
	if !ok {
		return
	}
	if _, marked := blk["cache_control"]; marked {
		return
	}
	blk["cache_control"] = map[string]any{"type": "ephemeral"}
	c.marks = append(c.marks, blk)
	for len(c.marks) > 2 {
		delete(c.marks[0], "cache_control")
		c.marks = c.marks[1:]
	}
}

func (c *anthConvo) Step(withTools bool) (*Turn, error) {
	cl := c.b.client
	c.markLatest()
	reqBody := map[string]any{
		"model":      cl.Model,
		"max_tokens": cl.MaxTokens,
		"system":     c.system,
		"messages":   c.messages,
	}
	for k, v := range cl.Extra {
		reqBody[k] = v
	}
	if withTools && len(c.b.tools) > 0 {
		reqBody["tools"] = c.b.tools
		reqBody["tool_choice"] = map[string]any{"type": "auto"}
	}
	data, err := postJSON(cl.HTTP, cl.BaseURL+"/messages", cl.headers(), reqBody)
	if err != nil {
		return nil, fmt.Errorf("messages: %w", err)
	}
	var out struct {
		Content []json.RawMessage `json:"content"`
		Usage   struct {
			InputTokens              int `json:"input_tokens"`
			OutputTokens             int `json:"output_tokens"`
			CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
			CacheReadInputTokens     int `json:"cache_read_input_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, fmt.Errorf("messages: bad response: %.300s", data)
	}
	// input_tokens is the uncached remainder only; the prompt total is the sum.
	t := &Turn{Usage: Usage{
		Prompt:     out.Usage.InputTokens + out.Usage.CacheCreationInputTokens + out.Usage.CacheReadInputTokens,
		Completion: out.Usage.OutputTokens,
		CacheRead:  out.Usage.CacheReadInputTokens,
		CacheWrite: out.Usage.CacheCreationInputTokens,
	}}
	for _, raw := range out.Content {
		var blk struct {
			Type  string          `json:"type"`
			Text  string          `json:"text"`
			ID    string          `json:"id"`
			Name  string          `json:"name"`
			Input json.RawMessage `json:"input"`
		}
		if err := json.Unmarshal(raw, &blk); err != nil {
			return nil, fmt.Errorf("messages: bad content block: %.300s", raw)
		}
		switch blk.Type {
		case "text":
			t.Content += blk.Text
		case "tool_use":
			var tc ToolCall
			tc.ID = blk.ID
			tc.Type = "function"
			tc.Function.Name = blk.Name
			tc.Function.Arguments = string(blk.Input)
			t.ToolCalls = append(t.ToolCalls, tc)
		}
	}
	// A turn that spent its whole budget thinking comes back with no content;
	// echoing an empty assistant message is rejected, so drop it and let the
	// answer retry proceed.
	if len(out.Content) > 0 {
		c.messages = append(c.messages, map[string]any{"role": "assistant", "content": out.Content})
	}
	return t, nil
}
