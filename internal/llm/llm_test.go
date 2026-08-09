package llm

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// serve replies with replies[i] to the i-th request (the last one repeats) and
// records every request body.
func serve(t *testing.T, replies ...string) (*http.Client, string, *[]map[string]any) {
	t.Helper()
	var got []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		got = append(got, body)
		i := len(got) - 1
		if i >= len(replies) {
			i = len(replies) - 1
		}
		_, _ = io.WriteString(w, replies[i])
	}))
	t.Cleanup(srv.Close)
	return srv.Client(), srv.URL, &got
}

var testTools = []ToolDef{{Name: "search", Description: "d", InputSchema: json.RawMessage(`{"type":"object"}`)}}

func TestChatStep(t *testing.T) {
	httpc, url, reqs := serve(t, `{"choices":[{"message":{"role":"assistant","content":"thinking",
		"reasoning_content":"hidden","tool_calls":[{"id":"c1","type":"function",
		"function":{"name":"search","arguments":"{\"q\":\"5G\"}"}}]}}],
		"usage":{"prompt_tokens":100,"completion_tokens":20,"prompt_tokens_details":{"cached_tokens":80}}}`,
		`{"choices":[{"message":{"role":"assistant","content":"ANSWER: 1"}}],"usage":{}}`)
	be := NewChatBackend(&ChatClient{BaseURL: url, Model: "m", MaxTokens: 99, MaxTokensField: "max_completion_tokens", HTTP: httpc}, testTools)

	c := be.NewConvo("sys", "user")
	turn, err := c.Step(true)
	if err != nil {
		t.Fatal(err)
	}
	if turn.Content != "thinking" || turn.Reasoning != "hidden" {
		t.Errorf("turn = %+v", turn)
	}
	if len(turn.ToolCalls) != 1 || turn.ToolCalls[0].Function.Name != "search" {
		t.Fatalf("tool calls = %+v", turn.ToolCalls)
	}
	if (turn.Usage != Usage{Prompt: 100, Completion: 20, CacheRead: 80}) {
		t.Errorf("usage = %+v", turn.Usage)
	}
	if (*reqs)[0]["max_completion_tokens"] != float64(99) || (*reqs)[0]["tool_choice"] != "auto" {
		t.Errorf("request = %v", (*reqs)[0])
	}

	c.AddToolResult(turn.ToolCalls[0], "result text")
	if _, err := c.Step(false); err != nil {
		t.Fatal(err)
	}
	req := (*reqs)[1]
	if _, ok := req["tools"]; ok {
		t.Error("tools sent even though withTools=false")
	}
	msgs, _ := json.Marshal(req["messages"])
	if strings.Contains(string(msgs), "hidden") {
		t.Errorf("reasoning echoed back: %s", msgs)
	}
	if !strings.Contains(string(msgs), "result text") {
		t.Errorf("tool result missing: %s", msgs)
	}
}

// A reasoning model can return neither content nor tool calls; that message
// must not be echoed back or the next request is rejected.
func TestChatDropsEmptyAssistant(t *testing.T) {
	httpc, url, reqs := serve(t, `{"choices":[{"message":{"role":"assistant","content":""}}],"usage":{}}`)
	be := NewChatBackend(&ChatClient{BaseURL: url, Model: "m", HTTP: httpc}, nil)
	c := be.NewConvo("sys", "user")
	if _, err := c.Step(false); err != nil {
		t.Fatal(err)
	}
	c.AddUser("answer now")
	if _, err := c.Step(false); err != nil {
		t.Fatal(err)
	}
	msgs, _ := (*reqs)[1]["messages"].([]any)
	if len(msgs) != 3 { // system, user, "answer now"
		t.Errorf("sent %d messages, want 3: %v", len(msgs), msgs)
	}
}

func TestResponsesStep(t *testing.T) {
	httpc, url, reqs := serve(t, `{"status":"completed","output":[
		{"type":"reasoning","id":"r1","encrypted_content":"enc"},
		{"type":"message","content":[{"type":"output_text","text":"ANSWER: 2"}]},
		{"type":"function_call","call_id":"c1","name":"search","arguments":"{}"}],
		"usage":{"input_tokens":50,"output_tokens":5,"input_tokens_details":{"cached_tokens":40,"cache_write_tokens":10}}}`)
	be := NewResponsesBackend(&ResponsesClient{BaseURL: url, Model: "m", HTTP: httpc}, testTools)

	c := be.NewConvo("sys", "user")
	turn, err := c.Step(true)
	if err != nil {
		t.Fatal(err)
	}
	if turn.Content != "ANSWER: 2" || len(turn.ToolCalls) != 1 || turn.ToolCalls[0].ID != "c1" {
		t.Errorf("turn = %+v", turn)
	}
	// input_tokens already includes the cached prefix, so Prompt is not a sum.
	if (turn.Usage != Usage{Prompt: 50, Completion: 5, CacheRead: 40, CacheWrite: 10}) {
		t.Errorf("usage = %+v", turn.Usage)
	}
	if (*reqs)[0]["store"] != false {
		t.Errorf("store = %v, want false", (*reqs)[0]["store"])
	}

	if _, err := c.Step(true); err != nil {
		t.Fatal(err)
	}
	input, _ := json.Marshal((*reqs)[1]["input"])
	if !strings.Contains(string(input), "enc") {
		t.Errorf("reasoning item not echoed back: %s", input)
	}
}

func TestResponsesIncomplete(t *testing.T) {
	httpc, url, _ := serve(t, `{"status":"incomplete","output":[],"usage":{}}`)
	be := NewResponsesBackend(&ResponsesClient{BaseURL: url, Model: "m", HTTP: httpc}, nil)
	if _, err := be.NewConvo("sys", "user").Step(false); err == nil || !strings.Contains(err.Error(), "incomplete") {
		t.Errorf("err = %v", err)
	}
}

func TestAnthropicStep(t *testing.T) {
	httpc, url, reqs := serve(t, `{"content":[{"type":"text","text":"looking"},
		{"type":"tool_use","id":"t1","name":"search","input":{"q":"5G"}}],
		"usage":{"input_tokens":10,"output_tokens":7,"cache_creation_input_tokens":100,"cache_read_input_tokens":900}}`)
	be := NewAnthropicBackend(&AnthropicClient{BaseURL: url, Model: "m", MaxTokens: 64, HTTP: httpc}, testTools)

	c := be.NewConvo("sys", "user")
	turn, err := c.Step(true)
	if err != nil {
		t.Fatal(err)
	}
	if turn.Content != "looking" || len(turn.ToolCalls) != 1 || turn.ToolCalls[0].Function.Arguments != `{"q":"5G"}` {
		t.Errorf("turn = %+v", turn)
	}
	// input_tokens is the uncached remainder only: the prompt total is the sum.
	if (turn.Usage != Usage{Prompt: 1010, Completion: 7, CacheRead: 900, CacheWrite: 100}) {
		t.Errorf("usage = %+v", turn.Usage)
	}
	sys, _ := json.Marshal((*reqs)[0]["system"])
	if !strings.Contains(string(sys), "cache_control") {
		t.Errorf("system block not cached: %s", sys)
	}

	// Two tool results in one round belong to a single user message.
	c.AddToolResult(turn.ToolCalls[0], "a")
	c.AddToolResult(turn.ToolCalls[0], "b")
	if _, err := c.Step(true); err != nil {
		t.Fatal(err)
	}
	msgs, _ := (*reqs)[1]["messages"].([]any)
	if len(msgs) != 3 { // user, assistant, tool results
		t.Fatalf("sent %d messages, want 3: %v", len(msgs), msgs)
	}
	last, _ := msgs[2].(map[string]any)
	blocks, _ := last["content"].([]any)
	if last["role"] != "user" || len(blocks) != 2 {
		t.Errorf("last message = %v", last)
	}
}

func TestAnthropicMaxOutputTokens(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-api-key") != "k" || r.Header.Get("anthropic-version") == "" {
			t.Errorf("headers = %v", r.Header)
		}
		_, _ = io.WriteString(w, `{"max_tokens":64000}`)
	}))
	defer srv.Close()
	cl := &AnthropicClient{BaseURL: srv.URL, APIKey: "k", Model: "claude", HTTP: srv.Client()}
	got, err := cl.MaxOutputTokens()
	if err != nil || got != 64000 {
		t.Fatalf("got %d, err %v", got, err)
	}
}

func TestNew(t *testing.T) {
	httpc, url, _ := serve(t, `{}`)
	for _, api := range []string{"chat", "responses", "anthropic"} {
		be, max, err := New(Config{API: api, BaseURL: url, Model: "m", MaxTokens: 8192, HTTP: httpc, Tools: testTools})
		if err != nil || be == nil || max != 8192 {
			t.Errorf("%s: be=%v max=%d err=%v", api, be, max, err)
		}
	}
	if _, _, err := New(Config{API: "bogus"}); err == nil {
		t.Error("unknown -api accepted")
	}
}

// -api anthropic -max-tokens 0 resolves the model's ceiling instead.
func TestNewAnthropicResolvesMaxTokens(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"max_tokens":32000}`)
	}))
	defer srv.Close()
	httpc := srv.Client()
	httpc.Timeout = 5 * time.Second
	_, max, err := New(Config{API: "anthropic", BaseURL: srv.URL, Model: "m", MaxTokens: 0, HTTP: httpc})
	if err != nil || max != 32000 {
		t.Fatalf("max=%d err=%v", max, err)
	}
}

func TestPostJSONHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "bad request", http.StatusBadRequest)
	}))
	defer srv.Close()
	// 4xx other than 429 fails immediately; 429/5xx would sleep between retries.
	if _, err := postJSON(srv.Client(), srv.URL, nil, map[string]any{}); err == nil || !strings.Contains(err.Error(), "HTTP 400") {
		t.Errorf("err = %v", err)
	}
}
