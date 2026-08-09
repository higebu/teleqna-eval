// teleqna-eval runs TeleQnA multiple-choice questions against an
// OpenAI-compatible model API, optionally bridging tools from a
// 3gpp-mcp server so the model can consult 3GPP specifications while
// answering. Results are written as JSONL plus a summary line.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ---------- TeleQnA ----------

type Question struct {
	ID       string
	Text     string
	Options  map[int]string // option number -> text
	Answer   int            // expected option number
	Category string
}

func loadQuestions(path, categoryPrefix, textFilter string) ([]Question, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m map[string]map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, err
	}
	optRe := regexp.MustCompile(`^option (\d+)$`)
	ansRe := regexp.MustCompile(`^option (\d+)\s*:`)
	var qs []Question
	for id, fields := range m {
		cat, _ := fields["category"].(string)
		if categoryPrefix != "" && !strings.HasPrefix(cat, categoryPrefix) {
			continue
		}
		if qText, _ := fields["question"].(string); textFilter != "" && !strings.Contains(qText, textFilter) {
			continue
		}
		q := Question{ID: id, Category: cat, Options: map[int]string{}}
		q.Text, _ = fields["question"].(string)
		for k, v := range fields {
			if mm := optRe.FindStringSubmatch(k); mm != nil {
				n, _ := strconv.Atoi(mm[1])
				q.Options[n], _ = v.(string)
			}
		}
		ans, _ := fields["answer"].(string)
		mm := ansRe.FindStringSubmatch(ans)
		if mm == nil || q.Text == "" || len(q.Options) < 2 {
			continue
		}
		q.Answer, _ = strconv.Atoi(mm[1])
		qs = append(qs, q)
	}
	// Deterministic order: sort by numeric suffix of "question N".
	sort.Slice(qs, func(i, j int) bool {
		ni, _ := strconv.Atoi(strings.TrimPrefix(qs[i].ID, "question "))
		nj, _ := strconv.Atoi(strings.TrimPrefix(qs[j].ID, "question "))
		return ni < nj
	})
	return qs, nil
}

// ---------- MCP bridge (Streamable HTTP, stateless) ----------

type MCPClient struct {
	URL  string
	HTTP *http.Client
	next int
}

type mcpTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"inputSchema"`
}

func (c *MCPClient) rpc(method string, params any) (json.RawMessage, error) {
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
	payload := data
	// The server may answer as SSE ("event: message\ndata: {...}").
	if bytes.HasPrefix(bytes.TrimSpace(data), []byte("event:")) || bytes.Contains(data, []byte("\ndata: ")) {
		sc := bufio.NewScanner(bytes.NewReader(data))
		sc.Buffer(make([]byte, 0, 1<<20), 64<<20)
		for sc.Scan() {
			if line := sc.Text(); strings.HasPrefix(line, "data: ") {
				payload = []byte(strings.TrimPrefix(line, "data: "))
			}
		}
	}
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

func (c *MCPClient) ListTools() ([]mcpTool, error) {
	res, err := c.rpc("tools/list", map[string]any{})
	if err != nil {
		return nil, err
	}
	var out struct {
		Tools []mcpTool `json:"tools"`
	}
	return out.Tools, json.Unmarshal(res, &out)
}

func (c *MCPClient) CallTool(name string, args json.RawMessage) (string, bool, error) {
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

// ---------- OpenAI-compatible chat ----------

type chatMessage struct {
	Role string `json:"role"`
	// Content is what we send back; ReasoningContent is only ever received
	// (DeepSeek-style reasoning models) and is cleared before echoing.
	Content          string     `json:"content,omitempty"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []toolCall `json:"tool_calls,omitempty"`
	ToolCallID       string     `json:"tool_call_id,omitempty"`
}

type toolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type chatUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

type chatClient struct {
	BaseURL, APIKey, Model string
	MaxTokens              int
	MaxTokensField         string         // "max_tokens", or "max_completion_tokens" for newer OpenAI models
	Extra                  map[string]any // extra request fields, e.g. {"reasoning_effort": "none"}
	HTTP                   *http.Client
}

// postJSON POSTs a JSON body with Bearer auth, retrying 429/5xx and network
// errors with exponential backoff, and returns the response body.
func postJSON(httpc *http.Client, url, apiKey string, reqBody map[string]any) ([]byte, error) {
	body, _ := json.Marshal(reqBody)
	var lastErr error
	for attempt := 0; attempt < 5; attempt++ {
		if attempt > 0 {
			time.Sleep(time.Duration(1<<attempt) * 2 * time.Second)
		}
		req, _ := http.NewRequest("POST", url, bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+apiKey)
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

func (c *chatClient) complete(messages []chatMessage, tools []map[string]any) (*chatMessage, *chatUsage, error) {
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
	data, err := postJSON(c.HTTP, c.BaseURL+"/chat/completions", c.APIKey, reqBody)
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

// ---------- backend abstraction ----------

// turn is one assistant step, backend-independent.
type turn struct {
	Content   string
	Reasoning string
	ToolCalls []toolCall
	Usage     chatUsage
}

// convo is one question's conversation with a model; each backend keeps the
// history in its own wire format.
type convo interface {
	step(withTools bool) (*turn, error)
	addUser(text string)
	addToolResult(tc toolCall, content string)
}

type backend interface {
	newConvo(system, user string) convo
}

type chatBackend struct {
	client *chatClient
	tools  []map[string]any
}

func (b *chatBackend) newConvo(system, user string) convo {
	return &chatConvo{b: b, messages: []chatMessage{
		{Role: "system", Content: system},
		{Role: "user", Content: user},
	}}
}

type chatConvo struct {
	b        *chatBackend
	messages []chatMessage
}

func (c *chatConvo) step(withTools bool) (*turn, error) {
	var tools []map[string]any
	if withTools {
		tools = c.b.tools
	}
	msg, usage, err := c.b.client.complete(c.messages, tools)
	if err != nil {
		return nil, err
	}
	t := &turn{Content: msg.Content, Reasoning: msg.ReasoningContent, ToolCalls: msg.ToolCalls}
	if usage != nil {
		t.Usage = *usage
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

func (c *chatConvo) addUser(text string) {
	c.messages = append(c.messages, chatMessage{Role: "user", Content: text})
}

func (c *chatConvo) addToolResult(tc toolCall, content string) {
	c.messages = append(c.messages, chatMessage{Role: "tool", ToolCallID: tc.ID, Content: content})
}

// ---------- OpenAI Responses API ----------

type responsesClient struct {
	BaseURL, APIKey, Model string
	MaxTokens              int
	Extra                  map[string]any
	HTTP                   *http.Client
}

type responsesBackend struct {
	client *responsesClient
	tools  []map[string]any
}

func (b *responsesBackend) newConvo(system, user string) convo {
	return &respConvo{b: b, instructions: system,
		input: []any{map[string]any{"role": "user", "content": user}}}
}

// respConvo replays the full item history on every request (store:false).
// Output items — including reasoning items with encrypted_content — are
// echoed back verbatim, as the migration guide requires for stateless use.
type respConvo struct {
	b            *responsesBackend
	instructions string
	input        []any
}

func (c *respConvo) addUser(text string) {
	c.input = append(c.input, map[string]any{"role": "user", "content": text})
}

func (c *respConvo) addToolResult(tc toolCall, content string) {
	c.input = append(c.input, map[string]any{
		"type": "function_call_output", "call_id": tc.ID, "output": content,
	})
}

func (c *respConvo) step(withTools bool) (*turn, error) {
	cl := c.b.client
	reqBody := map[string]any{
		"model":        cl.Model,
		"instructions": c.instructions,
		"input":        c.input,
		"store":        false,
		"include":      []string{"reasoning.encrypted_content"},
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
	data, err := postJSON(cl.HTTP, cl.BaseURL+"/responses", cl.APIKey, reqBody)
	if err != nil {
		return nil, fmt.Errorf("responses: %w", err)
	}
	var out struct {
		Status string            `json:"status"`
		Output []json.RawMessage `json:"output"`
		Usage  struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, fmt.Errorf("responses: bad response: %.300s", data)
	}
	t := &turn{Usage: chatUsage{PromptTokens: out.Usage.InputTokens, CompletionTokens: out.Usage.OutputTokens}}
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
			var tc toolCall
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

// ---------- Evaluation ----------

const systemPrompt = `You are a telecommunications standards expert answering multiple-choice questions about 3GPP specifications.

You have tools that let you consult the actual 3GPP specification documents (full-text search, table of contents, section text). Use them to verify your answer before responding: search for the key terms of the question, read the relevant section, and only then answer. Prefer exact wording from the specification over memory.

When you are ready to answer, reply with your reasoning followed by a final line in exactly this format:
ANSWER: <option number>

The final line must contain only one option number.`

const systemPromptNoTools = `You are a telecommunications standards expert answering multiple-choice questions about 3GPP specifications.

Answer from your own knowledge. Reply with brief reasoning followed by a final line in exactly this format:
ANSWER: <option number>

The final line must contain only one option number.`

var (
	answerLineRe = regexp.MustCompile(`(?mi)^[\s>*#]*ANSWER\s*[::]\s*\**\s*(?:option\s*)?(\d)`)
	answerAnyRe  = regexp.MustCompile(`(?i)ANSWER\s*[::]\s*\**\s*(?:option\s*)?(\d)`)
	optionAnyRe  = regexp.MustCompile(`(?i)\boption\s*(\d)\b`)
)

// extractAnswer pulls the chosen option number out of a model reply,
// trying strict format first and progressively looser fallbacks.
func extractAnswer(s string) int {
	if mm := answerLineRe.FindStringSubmatch(s); mm != nil {
		n, _ := strconv.Atoi(mm[1])
		return n
	}
	if mm := answerAnyRe.FindStringSubmatch(s); mm != nil {
		n, _ := strconv.Atoi(mm[1])
		return n
	}
	if all := optionAnyRe.FindAllStringSubmatch(s, -1); len(all) > 0 {
		n, _ := strconv.Atoi(all[len(all)-1][1])
		return n
	}
	return 0
}

type toolCallLog struct {
	Name string `json:"name"`
	Args string `json:"args"`
}

type result struct {
	ID           string        `json:"id"`
	Category     string        `json:"category"`
	Question     string        `json:"question"`
	Expected     int           `json:"expected"`
	Predicted    int           `json:"predicted"`
	Correct      bool          `json:"correct"`
	Rounds       int           `json:"rounds"`
	ToolCalls    []toolCallLog `json:"tool_calls"`
	PromptTok    int           `json:"prompt_tokens"`
	CompleteTok  int           `json:"completion_tokens"`
	DurationSec  float64       `json:"duration_sec"`
	FinalMessage string        `json:"final_message"`
	Error        string        `json:"error,omitempty"`
}

func formatQuestion(q Question) string {
	var sb strings.Builder
	sb.WriteString(q.Text)
	sb.WriteString("\n\n")
	nums := make([]int, 0, len(q.Options))
	for n := range q.Options {
		nums = append(nums, n)
	}
	sort.Ints(nums)
	for _, n := range nums {
		fmt.Fprintf(&sb, "option %d: %s\n", n, q.Options[n])
	}
	return sb.String()
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + fmt.Sprintf("\n...[truncated %d bytes; refine the query or use offset to read more]", len(s)-max)
}

func evalOne(be backend, mcp *MCPClient, q Question, maxRounds, toolResultMax int) result {
	start := time.Now()
	r := result{ID: q.ID, Category: q.Category, Question: q.Text, Expected: q.Answer}
	sys := systemPrompt
	if mcp == nil {
		sys = systemPromptNoTools
	}
	c := be.newConvo(sys, formatQuestion(q))
	answerRetries := 0
	for round := 0; ; round++ {
		r.Rounds = round + 1
		withTools := mcp != nil
		if round >= maxRounds {
			// Force a final answer: drop tools and ask for the answer.
			withTools = false
			c.addUser("Tool budget exhausted. Based on what you have read so far, give your final answer now as 'ANSWER: <option number>'.")
		}
		t, err := c.step(withTools)
		if err != nil {
			r.Error = err.Error()
			break
		}
		r.PromptTok += t.Usage.PromptTokens
		r.CompleteTok += t.Usage.CompletionTokens
		if len(t.ToolCalls) == 0 {
			pred := extractAnswer(t.Content)
			if pred == 0 {
				pred = extractAnswer(t.Reasoning)
			}
			if pred == 0 && answerRetries < 2 {
				answerRetries++
				c.addUser("Your reply did not contain a readable answer. Reply now with one line only: ANSWER: <option number>")
				continue
			}
			r.FinalMessage = t.Content
			r.Predicted = pred
			break
		}
		for _, tc := range t.ToolCalls {
			r.ToolCalls = append(r.ToolCalls, toolCallLog{Name: tc.Function.Name, Args: truncate(tc.Function.Arguments, 300)})
			text, isErr, err := mcp.CallTool(tc.Function.Name, json.RawMessage(tc.Function.Arguments))
			if err != nil {
				text = "tool error: " + err.Error()
			} else if isErr {
				text = "tool error: " + text
			}
			c.addToolResult(tc, truncate(text, toolResultMax))
		}
	}
	r.Correct = r.Predicted == q.Answer && r.Predicted != 0
	r.DurationSec = time.Since(start).Seconds()
	return r
}

func main() {
	var (
		model          = flag.String("model", "", "model id on the OpenAI-compatible endpoint (required)")
		api            = flag.String("api", "chat", "API style: 'chat' (chat/completions) or 'responses' (OpenAI Responses API)")
		baseURL        = flag.String("base-url", os.Getenv("OPENAI_BASE_URL"), "OpenAI-compatible base URL; defaults to $OPENAI_BASE_URL")
		keyEnv         = flag.String("key-env", "OPENAI_API_KEY", "environment variable holding the API key")
		mcpURL         = flag.String("mcp", os.Getenv("THREEGPP_MCP_URL"), "3gpp-mcp streamable HTTP endpoint; defaults to $THREEGPP_MCP_URL ('' disables tools)")
		dataPath       = flag.String("data", "data/TeleQnA.json", "path to TeleQnA JSON")
		category       = flag.String("category", "Standards specifications", "category prefix filter")
		filter         = flag.String("filter", "", "substring the question text must contain (e.g. '3GPP')")
		n              = flag.Int("n", 10, "number of questions")
		ids            = flag.String("ids", "", "comma-separated question ids to run (overrides -n/-seed sampling)")
		seed           = flag.Int64("seed", 42, "sampling seed")
		workers        = flag.Int("workers", 1, "concurrent questions")
		maxRounds      = flag.Int("max-rounds", 8, "max tool-calling rounds per question")
		maxTokens      = flag.Int("max-tokens", 8192, "max_tokens per completion (0 = provider default)")
		maxTokensField = flag.String("max-tokens-field", "max_tokens", "request field name for the token cap (some providers use max_completion_tokens)")
		extraBody      = flag.String("extra-body", "", "JSON object merged into every chat request, e.g. '{\"reasoning_effort\":\"none\"}'")
		resultMax      = flag.Int("tool-result-max", 16000, "max bytes of a tool result passed to the model")
		outPath        = flag.String("out", "", "JSONL output path (default results/<model>-<n>q-seed<seed>.jsonl)")
	)
	flag.Parse()

	if *model == "" {
		log.Fatal("-model is required")
	}
	if *baseURL == "" {
		log.Fatal("no endpoint: set -base-url or $OPENAI_BASE_URL")
	}
	apiKey := os.Getenv(*keyEnv)
	if apiKey == "" {
		log.Fatalf("%s is not set (choose the variable with -key-env)", *keyEnv)
	}
	qs, err := loadQuestions(*dataPath, *category, *filter)
	if err != nil {
		log.Fatalf("load questions: %v", err)
	}
	if *ids != "" {
		want := map[string]bool{}
		for _, id := range strings.Split(*ids, ",") {
			want[strings.TrimSpace(id)] = true
		}
		var picked []Question
		for _, q := range qs {
			if want[q.ID] {
				picked = append(picked, q)
			}
		}
		qs = picked
	} else {
		rand.New(rand.NewSource(*seed)).Shuffle(len(qs), func(i, j int) { qs[i], qs[j] = qs[j], qs[i] })
		if *n < len(qs) {
			qs = qs[:*n]
		}
	}

	var extra map[string]any
	if *extraBody != "" {
		if err := json.Unmarshal([]byte(*extraBody), &extra); err != nil {
			log.Fatalf("-extra-body: %v", err)
		}
	}

	var mcp *MCPClient
	var tools []map[string]any
	if *mcpURL != "" {
		mcp = &MCPClient{URL: *mcpURL, HTTP: &http.Client{Timeout: 120 * time.Second}}
		mcpTools, err := mcp.ListTools()
		if err != nil {
			log.Fatalf("mcp tools/list: %v", err)
		}
		for _, t := range mcpTools {
			if *api == "responses" {
				// Responses API uses a flat tool shape; strict mode is
				// attempted by default, so disable it for MCP schemas.
				tools = append(tools, map[string]any{
					"type": "function", "name": t.Name, "description": t.Description,
					"parameters": t.InputSchema, "strict": false,
				})
			} else {
				tools = append(tools, map[string]any{
					"type": "function",
					"function": map[string]any{
						"name": t.Name, "description": t.Description, "parameters": t.InputSchema,
					},
				})
			}
		}
		log.Printf("bridged %d MCP tools from %s", len(tools), *mcpURL)
	}

	httpc := &http.Client{Timeout: 300 * time.Second}
	var be backend
	switch *api {
	case "chat":
		be = &chatBackend{tools: tools, client: &chatClient{
			BaseURL: *baseURL, APIKey: apiKey, Model: *model,
			MaxTokens: *maxTokens, MaxTokensField: *maxTokensField, Extra: extra, HTTP: httpc,
		}}
	case "responses":
		be = &responsesBackend{tools: tools, client: &responsesClient{
			BaseURL: *baseURL, APIKey: apiKey, Model: *model,
			MaxTokens: *maxTokens, Extra: extra, HTTP: httpc,
		}}
	default:
		log.Fatalf("-api must be 'chat' or 'responses', got %q", *api)
	}

	if *outPath == "" {
		suffix := "tools"
		if mcp == nil {
			suffix = "notools"
		}
		tag := ""
		if *filter != "" {
			tag = "-" + strings.ToLower(strings.ReplaceAll(*filter, " ", "_"))
		}
		*outPath = fmt.Sprintf("results/%s-%s%s-%dq-seed%d.jsonl", *model, suffix, tag, *n, *seed)
	}
	if err := os.MkdirAll("results", 0o755); err != nil {
		log.Fatal(err)
	}
	out, err := os.Create(*outPath)
	if err != nil {
		log.Fatal(err)
	}
	defer out.Close()
	enc := json.NewEncoder(out)

	var (
		mu                                                              sync.Mutex
		correct, answered, totalPrompt, totalComplete, totalCalls, done int
	)
	type job struct {
		idx int
		q   Question
	}
	jobs := make(chan job)
	var wg sync.WaitGroup
	for w := 0; w < *workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				r := evalOne(be, mcp, j.q, *maxRounds, *resultMax)
				mu.Lock()
				_ = enc.Encode(r)
				status := "WRONG"
				if r.Correct {
					correct++
					status = "ok"
				}
				if r.Predicted != 0 {
					answered++
				}
				if r.Error != "" {
					status = "ERROR " + r.Error
				}
				totalPrompt += r.PromptTok
				totalComplete += r.CompleteTok
				totalCalls += len(r.ToolCalls)
				done++
				log.Printf("[%d/%d] %s: pred=%d exp=%d %s (%d tool calls, %.0fs)",
					done, len(qs), j.q.ID, r.Predicted, j.q.Answer, status, len(r.ToolCalls), r.DurationSec)
				mu.Unlock()
			}
		}()
	}
	for i, q := range qs {
		jobs <- job{i, q}
	}
	close(jobs)
	wg.Wait()
	fmt.Printf("\nmodel=%s tools=%v questions=%d answered=%d correct=%d accuracy=%.1f%%\n",
		*model, mcp != nil, len(qs), answered, correct, 100*float64(correct)/float64(len(qs)))
	fmt.Printf("tokens: prompt=%d completion=%d, tool calls=%d (avg %.1f/question)\n",
		totalPrompt, totalComplete, totalCalls, float64(totalCalls)/float64(len(qs)))
	fmt.Printf("results: %s\n", *outPath)
}
