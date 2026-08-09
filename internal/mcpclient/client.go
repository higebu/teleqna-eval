// Package mcpclient is a minimal MCP client speaking Streamable HTTP
// against a stateless server (no session, one POST per request).
package mcpclient

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type Client struct {
	URL  string
	HTTP *http.Client
	next int
}

func New(url string, timeout time.Duration) *Client {
	return &Client{URL: url, HTTP: &http.Client{Timeout: timeout}}
}

type Tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"inputSchema"`
}

func (c *Client) rpc(method string, params any) (json.RawMessage, error) {
	c.next++
	body, _ := json.Marshal(map[string]any{
		"jsonrpc": "2.0", "id": c.next, "method": method, "params": params,
	})
	req, _ := http.NewRequest("POST", c.URL, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	payload := sseData(data)
	var rpcResp struct {
		Result json.RawMessage `json:"result"`
		Error  *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(payload, &rpcResp); err != nil {
		return nil, fmt.Errorf("mcp: bad response (%s): %.200s", err, payload)
	}
	if rpcResp.Error != nil {
		return nil, fmt.Errorf("mcp: %s", rpcResp.Error.Message)
	}
	return rpcResp.Result, nil
}

// sseData unwraps an SSE body ("event: message\ndata: {...}"), which the server
// may send instead of plain JSON; a non-SSE body is returned unchanged.
func sseData(data []byte) []byte {
	if !bytes.HasPrefix(bytes.TrimSpace(data), []byte("event:")) && !bytes.Contains(data, []byte("\ndata: ")) {
		return data
	}
	payload := data
	sc := bufio.NewScanner(bytes.NewReader(data))
	sc.Buffer(make([]byte, 0, 1<<20), 64<<20)
	for sc.Scan() {
		if line := sc.Text(); strings.HasPrefix(line, "data: ") {
			payload = []byte(strings.TrimPrefix(line, "data: "))
		}
	}
	return payload
}

func (c *Client) ListTools() ([]Tool, error) {
	res, err := c.rpc("tools/list", map[string]any{})
	if err != nil {
		return nil, err
	}
	var out struct {
		Tools []Tool `json:"tools"`
	}
	if err := json.Unmarshal(res, &out); err != nil {
		return nil, err
	}
	return out.Tools, nil
}

// CallTool returns the tool's text output and whether the tool itself reported
// an error (isError), which is distinct from a transport failure.
func (c *Client) CallTool(name string, args json.RawMessage) (string, bool, error) {
	res, err := c.rpc("tools/call", map[string]any{"name": name, "arguments": args})
	if err != nil {
		return "", true, err
	}
	var out struct {
		IsError bool `json:"isError"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	}
	if err := json.Unmarshal(res, &out); err != nil {
		return "", true, err
	}
	var sb strings.Builder
	for _, blk := range out.Content {
		if blk.Type == "text" {
			sb.WriteString(blk.Text)
			sb.WriteString("\n")
		} else {
			fmt.Fprintf(&sb, "[%s content omitted]\n", blk.Type)
		}
	}
	return sb.String(), out.IsError, nil
}
