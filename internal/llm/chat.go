package llm

import (
	"encoding/json"
	"fmt"
	"net/http"
)

type ChatClient struct {
	BaseURL, APIKey, Model string
	MaxTokens              int
	MaxTokensField         string         // "max_tokens", or "max_completion_tokens" for newer OpenAI models
	Extra                  map[string]any // extra request fields, e.g. {"reasoning_effort": "none"}
	HTTP                   *http.Client
}

type ChatBackend struct {
	client *ChatClient
	tools  []map[string]any
}

func NewChatBackend(c *ChatClient, tools []ToolDef) *ChatBackend {
	return &ChatBackend{client: c, tools: chatTools(tools)}
}

func chatTools(tools []ToolDef) []map[string]any {
	var out []map[string]any
	for _, t := range tools {
		out = append(out, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name": t.Name, "description": t.Description, "parameters": t.InputSchema,
			},
		})
	}
	return out
}

type chatMessage struct {
	Role string `json:"role"`
	// ReasoningContent is only ever received (DeepSeek-style reasoning models)
	// and is cleared before echoing.
	Content          string     `json:"content,omitempty"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string     `json:"tool_call_id,omitempty"`
}

type chatUsage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	PromptTokensDetails struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

func (c *ChatClient) complete(messages []chatMessage, tools []map[string]any) (*chatMessage, *chatUsage, error) {
	reqBody := map[string]any{"model": c.Model, "messages": messages}
	if c.MaxTokens > 0 {
		reqBody[c.MaxTokensField] = c.MaxTokens
	}
	for k, v := range c.Extra {
		reqBody[k] = v
	}
	if len(tools) > 0 {
		reqBody["tools"] = tools
		reqBody["tool_choice"] = "auto"
	}
	data, err := postJSON(c.HTTP, c.BaseURL+"/chat/completions", bearer(c.APIKey), reqBody)
	if err != nil {
		return nil, nil, fmt.Errorf("chat: %w", err)
	}
	var out struct {
		Choices []struct {
			Message chatMessage `json:"message"`
		} `json:"choices"`
		Usage chatUsage `json:"usage"`
	}
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, nil, fmt.Errorf("chat: bad response: %.300s", data)
	}
	if len(out.Choices) == 0 {
		return nil, nil, fmt.Errorf("chat: no choices: %.300s", data)
	}
	return &out.Choices[0].Message, &out.Usage, nil
}

func (b *ChatBackend) NewConvo(system, user string) Convo {
	return &chatConvo{b: b, messages: []chatMessage{
		{Role: "system", Content: system},
		{Role: "user", Content: user},
	}}
}

type chatConvo struct {
	b        *ChatBackend
	messages []chatMessage
}

func (c *chatConvo) Step(withTools bool) (*Turn, error) {
	var tools []map[string]any
	if withTools {
		tools = c.b.tools
	}
	msg, u, err := c.b.client.complete(c.messages, tools)
	if err != nil {
		return nil, err
	}
	t := &Turn{Content: msg.Content, Reasoning: msg.ReasoningContent, ToolCalls: msg.ToolCalls}
	if u != nil {
		t.Usage = Usage{Prompt: u.PromptTokens, Completion: u.CompletionTokens,
			CacheRead: u.PromptTokensDetails.CachedTokens}
	}
	msg.ReasoningContent = "" // never echo reasoning back
	// A reasoning model that spends its whole budget thinking returns empty
	// content and no tool calls; echoing that back is rejected ("content or
	// tool_calls must be set"), so drop it and let the answer retry proceed.
	if msg.Content != "" || len(msg.ToolCalls) > 0 {
		c.messages = append(c.messages, *msg)
	}
	return t, nil
}

func (c *chatConvo) AddUser(text string) {
	c.messages = append(c.messages, chatMessage{Role: "user", Content: text})
}

func (c *chatConvo) AddToolResult(tc ToolCall, content string) {
	c.messages = append(c.messages, chatMessage{Role: "tool", ToolCallID: tc.ID, Content: content})
}
