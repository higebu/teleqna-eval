package llm

import (
	"encoding/json"
	"fmt"
	"net/http"
)

type ResponsesClient struct {
	BaseURL, APIKey, Model string
	MaxTokens              int
	Temperature            *float64 // nil sends no temperature field at all
	Extra                  map[string]any
	HTTP                   *http.Client
}

type ResponsesBackend struct {
	client *ResponsesClient
	tools  []map[string]any
}

func NewResponsesBackend(c *ResponsesClient, tools []ToolDef) *ResponsesBackend {
	return &ResponsesBackend{client: c, tools: responsesTools(tools)}
}

// responsesTools renders the flat tool shape of the Responses API. Strict mode
// is attempted by default there, so it is disabled for MCP schemas.
func responsesTools(tools []ToolDef) []map[string]any {
	var out []map[string]any
	for _, t := range tools {
		out = append(out, map[string]any{
			"type": "function", "name": t.Name, "description": t.Description,
			"parameters": t.InputSchema, "strict": false,
		})
	}
	return out
}

func (b *ResponsesBackend) NewConvo(system, user string) Convo {
	return &respConvo{b: b, instructions: system,
		input: []any{map[string]any{"role": "user", "content": user}}}
}

// respConvo replays the full item history on every request (store:false).
// Output items — including reasoning items with encrypted_content — are
// echoed back verbatim, as the migration guide requires for stateless use.
type respConvo struct {
	b            *ResponsesBackend
	instructions string
	input        []any
}

func (c *respConvo) AddUser(text string) {
	c.input = append(c.input, map[string]any{"role": "user", "content": text})
}

func (c *respConvo) AddToolResult(tc ToolCall, content string) {
	c.input = append(c.input, map[string]any{
		"type": "function_call_output", "call_id": tc.ID, "output": content,
	})
}

func (c *respConvo) Step(withTools bool) (*Turn, error) {
	cl := c.b.client
	reqBody := map[string]any{
		"model":        cl.Model,
		"instructions": c.instructions,
		"input":        c.input,
		"store":        false,
		"include":      []string{"reasoning.encrypted_content"},
	}
	if cl.Temperature != nil {
		reqBody["temperature"] = *cl.Temperature
	}
	if cl.MaxTokens > 0 {
		reqBody["max_output_tokens"] = cl.MaxTokens
	}
	for k, v := range cl.Extra {
		reqBody[k] = v
	}
	if withTools && len(c.b.tools) > 0 {
		reqBody["tools"] = c.b.tools
		reqBody["tool_choice"] = "auto"
	}
	data, err := postJSON(cl.HTTP, cl.BaseURL+"/responses", bearer(cl.APIKey), reqBody)
	if err != nil {
		return nil, fmt.Errorf("responses: %w", err)
	}
	var out struct {
		Status string            `json:"status"`
		Output []json.RawMessage `json:"output"`
		Usage  struct {
			InputTokens       int `json:"input_tokens"`
			OutputTokens      int `json:"output_tokens"`
			InputTokenDetails struct {
				CachedTokens     int `json:"cached_tokens"`
				CacheWriteTokens int `json:"cache_write_tokens"`
			} `json:"input_tokens_details"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, fmt.Errorf("responses: bad response: %.300s", data)
	}
	// OpenAI caches automatically; input_tokens already includes the cached
	// prefix, so the details are recorded but not added to the total.
	t := &Turn{Usage: Usage{
		Prompt: out.Usage.InputTokens, Completion: out.Usage.OutputTokens,
		CacheRead:  out.Usage.InputTokenDetails.CachedTokens,
		CacheWrite: out.Usage.InputTokenDetails.CacheWriteTokens,
	}}
	for _, raw := range out.Output {
		var item map[string]any
		if err := json.Unmarshal(raw, &item); err != nil {
			return nil, fmt.Errorf("responses: bad output item: %.300s", raw)
		}
		c.input = append(c.input, item) // echo every item back next round
		switch item["type"] {
		case "message":
			var m struct {
				Content []struct {
					Type string `json:"type"`
					Text string `json:"text"`
				} `json:"content"`
			}
			_ = json.Unmarshal(raw, &m)
			for _, blk := range m.Content {
				if blk.Type == "output_text" {
					t.Content += blk.Text
				}
			}
		case "function_call":
			var f struct {
				CallID    string `json:"call_id"`
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			}
			_ = json.Unmarshal(raw, &f)
			var tc ToolCall
			tc.ID = f.CallID
			tc.Type = "function"
			tc.Function.Name = f.Name
			tc.Function.Arguments = f.Arguments
			t.ToolCalls = append(t.ToolCalls, tc)
		}
	}
	if out.Status == "incomplete" && t.Content == "" && len(t.ToolCalls) == 0 {
		return nil, fmt.Errorf("responses: incomplete with no usable output: %.300s", data)
	}
	return t, nil
}
