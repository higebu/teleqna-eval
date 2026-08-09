package mcpclient

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// serve replies to every JSON-RPC request with body, recording the last request.
func serve(t *testing.T, body string, last *map[string]any) *Client {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if last != nil {
			raw, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(raw, last)
		}
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	return New(srv.URL, 5*time.Second)
}

func TestListTools(t *testing.T) {
	var req map[string]any
	c := serve(t, `{"jsonrpc":"2.0","id":1,"result":{"tools":[
		{"name":"search","description":"full-text search","inputSchema":{"type":"object"}}]}}`, &req)
	tools, err := c.ListTools()
	if err != nil {
		t.Fatal(err)
	}
	if len(tools) != 1 || tools[0].Name != "search" || tools[0].Description != "full-text search" {
		t.Fatalf("tools = %+v", tools)
	}
	if string(tools[0].InputSchema) != `{"type":"object"}` {
		t.Errorf("schema = %s", tools[0].InputSchema)
	}
	if req["method"] != "tools/list" {
		t.Errorf("method = %v", req["method"])
	}
}

func TestListToolsSSE(t *testing.T) {
	c := serve(t, "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"tools\":[{\"name\":\"get_section\"}]}}\n\n", nil)
	tools, err := c.ListTools()
	if err != nil {
		t.Fatal(err)
	}
	if len(tools) != 1 || tools[0].Name != "get_section" {
		t.Fatalf("tools = %+v", tools)
	}
}

func TestCallTool(t *testing.T) {
	var req map[string]any
	c := serve(t, `{"jsonrpc":"2.0","id":1,"result":{"content":[
		{"type":"text","text":"5.2.1 Overview"},{"type":"image","data":"..."}]}}`, &req)
	text, isErr, err := c.CallTool("get_section", json.RawMessage(`{"spec":"23.501"}`))
	if err != nil || isErr {
		t.Fatalf("err=%v isErr=%v", err, isErr)
	}
	if text != "5.2.1 Overview\n[image content omitted]\n" {
		t.Errorf("text = %q", text)
	}
	params, _ := req["params"].(map[string]any)
	if params["name"] != "get_section" {
		t.Errorf("params = %v", params)
	}
}

func TestCallToolIsError(t *testing.T) {
	c := serve(t, `{"jsonrpc":"2.0","id":1,"result":{"isError":true,"content":[{"type":"text","text":"no such spec"}]}}`, nil)
	text, isErr, err := c.CallTool("get_section", nil)
	if err != nil {
		t.Fatal(err)
	}
	if !isErr || !strings.Contains(text, "no such spec") {
		t.Errorf("isErr=%v text=%q", isErr, text)
	}
}

func TestRPCError(t *testing.T) {
	c := serve(t, `{"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"method not found"}}`, nil)
	if _, _, err := c.CallTool("nope", nil); err == nil || !strings.Contains(err.Error(), "method not found") {
		t.Errorf("err = %v", err)
	}
}

// One Client is shared by every worker, so the request id must be race-free.
func TestConcurrentCalls(t *testing.T) {
	c := serve(t, `{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"ok"}]}}`, nil)
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, _, err := c.CallTool("search", nil); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if got := c.next.Load(); got != 8 {
		t.Errorf("next = %d, want 8", got)
	}
}

func TestBadResponse(t *testing.T) {
	c := serve(t, "not json", nil)
	if _, err := c.ListTools(); err == nil || !strings.Contains(err.Error(), "bad response") {
		t.Errorf("err = %v", err)
	}
}
