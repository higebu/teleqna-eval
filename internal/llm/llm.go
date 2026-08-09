// Package llm bridges the three model APIs the harness supports (OpenAI
// chat/completions, OpenAI Responses, Anthropic Messages) behind one
// tool-calling conversation interface.
package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// ToolDef is a tool as the MCP server describes it, before it is rendered into
// the wire shape a given API expects.
type ToolDef struct {
	Name        string
	Description string
	InputSchema json.RawMessage
}

type ToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// Usage is one step's token accounting, backend-independent. Prompt is always
// the full prompt size: OpenAI reports cached tokens inside its input count,
// Anthropic reports them alongside it, so the Anthropic backend adds them up.
type Usage struct {
	Prompt     int
	Completion int
	CacheRead  int // prompt tokens served from cache
	CacheWrite int // prompt tokens written to cache (Anthropic/OpenAI charge a premium)
}

// Turn is one assistant step, backend-independent.
type Turn struct {
	Content   string
	Reasoning string
	ToolCalls []ToolCall
	Usage     Usage
}

// Convo is one question's conversation with a model; each backend keeps the
// history in its own wire format.
type Convo interface {
	Step(withTools bool) (*Turn, error)
	AddUser(text string)
	AddToolResult(tc ToolCall, content string)
}

type Backend interface {
	NewConvo(system, user string) Convo
}

func bearer(apiKey string) map[string]string {
	return map[string]string{"Authorization": "Bearer " + apiKey}
}

// postJSON POSTs a JSON body with the given auth headers, retrying 429/5xx and
// network errors with exponential backoff, and returns the response body.
func postJSON(httpc *http.Client, url string, headers map[string]string, reqBody map[string]any) ([]byte, error) {
	body, _ := json.Marshal(reqBody)
	var lastErr error
	for attempt := 0; attempt < 5; attempt++ {
		if attempt > 0 {
			time.Sleep(time.Duration(1<<attempt) * 2 * time.Second)
		}
		req, _ := http.NewRequest("POST", url, bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		for k, v := range headers {
			req.Header.Set(k, v)
		}
		resp, err := httpc.Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		data, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			lastErr = err
			continue
		}
		if resp.StatusCode == 429 || resp.StatusCode >= 500 {
			lastErr = fmt.Errorf("HTTP %d: %.300s", resp.StatusCode, data)
			continue
		}
		if resp.StatusCode != 200 {
			return nil, fmt.Errorf("HTTP %d: %.500s", resp.StatusCode, data)
		}
		return data, nil
	}
	return nil, lastErr
}
