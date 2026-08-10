// Command specbench runs the spec-grounded benchmark: free-form answers about
// 3GPP specifications, scored together with the citation that backs them.
//
// It follows the same rule as the TeleQnA harness — both conditions send the
// identical prompt, and only the tool attachment differs — so the delta is the
// value of being able to consult the specifications, not a prompt difference.
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	"teleqna-eval/internal/llm"
	"teleqna-eval/internal/mcpclient"
	"teleqna-eval/internal/specbench"
)

// The one prompt both conditions use. It never mentions tools: whether tools
// are attached is the only thing that differs between the two runs.
const systemPrompt = `You are answering questions about 3GPP specifications.

Reply with a JSON object and nothing else:
{"answer": <answer>, "spec_id": "<specification>", "section": "<section>"}

- For a list of ASN.1 field names, "answer" is a JSON array of the names, in the order they are defined.
- For an equation, "answer" is a JSON string holding the LaTeX of the equation, without $ delimiters.
- "spec_id" is the specification the answer is defined in, e.g. "TS 38.331".
- "section" is the clause number, or the clause heading when the specification numbers clauses by name.

The citation is part of the answer: it must be where this is actually defined.`

const retryPrompt = `Your reply did not contain the JSON object. Reply now with the JSON object only, in the format given above.`

type record struct {
	ID        string            `json:"id"`
	Type      string            `json:"type"`
	Question  string            `json:"question"`
	Gold      json.RawMessage   `json:"gold"`
	GoldSpec  string            `json:"gold_spec_id"`
	GoldSec   string            `json:"gold_section"`
	Score     specbench.Score   `json:"score"`
	PredSpec  string            `json:"predicted_spec_id"`
	PredSec   string            `json:"predicted_section"`
	Rounds    int               `json:"rounds"`
	ToolCalls []toolCall        `json:"tool_calls"`
	Usage     map[string]int    `json:"usage"`
	Duration  float64           `json:"duration_sec"`
	Final     string            `json:"final_message"`
	Error     string            `json:"error,omitempty"`
	Meta      map[string]string `json:"meta"`
	Retrieved map[string]int    `json:"retrieved,omitempty"`
}

type toolCall struct {
	Name   string `json:"name"`
	Args   string `json:"args"`
	Result string `json:"result,omitempty"`
}

func main() {
	var (
		tasksPath = flag.String("tasks", "bench/tasks-asn1.json", "task JSON from bench/generate.py")
		model     = flag.String("model", "", "model id (required)")
		api       = flag.String("api", "chat", "chat, responses or anthropic")
		baseURL   = flag.String("base-url", os.Getenv("OPENAI_BASE_URL"), "API base URL")
		keyEnv    = flag.String("key-env", "OPENAI_API_KEY", "env var holding the API key")
		mcpURL    = flag.String("mcp", os.Getenv("THREEGPP_MCP_URL"), "MCP endpoint; '' disables tools")
		workers   = flag.Int("workers", 4, "concurrent tasks")
		maxRounds = flag.Int("max-rounds", 20, "tool rounds before the answer is forced")
		maxTokens = flag.Int("max-tokens", 0, "token cap per completion (0 = provider default)")
		temp      = flag.String("temperature", "", "sampling temperature; empty sends none")
		timeout   = flag.Int("http-timeout", 900, "per-request timeout in seconds")
		resultMax = flag.Int("tool-result-max", 16000, "max bytes of a tool result")
		dbMan     = flag.String("db-manifest", "", "identifier of the pinned database")
		out       = flag.String("out", "", "JSONL output path")
	)
	flag.Parse()

	if *model == "" || *baseURL == "" {
		log.Fatal("-model and -base-url are required")
	}
	key := os.Getenv(*keyEnv)
	if key == "" {
		log.Fatalf("%s is not set", *keyEnv)
	}
	tasks, err := specbench.Load(*tasksPath)
	if err != nil {
		log.Fatal(err)
	}
	specbench.SortTasks(tasks)

	var temperature *float64
	if *temp != "" {
		v, err := strconv.ParseFloat(*temp, 64)
		if err != nil {
			log.Fatalf("-temperature: %v", err)
		}
		temperature = &v
	}

	var mcp *mcpclient.Client
	var tools []llm.ToolDef
	if *mcpURL != "" {
		mcp = mcpclient.New(*mcpURL, 120*time.Second)
		mt, err := mcp.ListTools()
		if err != nil {
			log.Fatalf("mcp tools/list: %v", err)
		}
		for _, t := range mt {
			tools = append(tools, llm.ToolDef{Name: t.Name, Description: t.Description, InputSchema: t.InputSchema})
		}
		log.Printf("bridged %d MCP tools from %s", len(tools), *mcpURL)
	}

	be, _, err := llm.New(llm.Config{
		API: *api, BaseURL: *baseURL, APIKey: key, Model: *model,
		MaxTokens: *maxTokens, MaxTokensField: "max_tokens", Temperature: temperature,
		HTTP: &http.Client{Timeout: time.Duration(*timeout) * time.Second}, Tools: tools,
	})
	if err != nil {
		log.Fatal(err)
	}

	if *out == "" {
		suffix := "notools"
		if mcp != nil {
			suffix = "tools"
		}
		base := filepath.Base(*tasksPath)
		*out = fmt.Sprintf("results/bench-%s-%s-%s.jsonl", *model, base[:len(base)-len(filepath.Ext(base))], suffix)
	}
	if dir := filepath.Dir(*out); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			log.Fatal(err)
		}
	}
	f, err := os.Create(*out)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	meta := map[string]string{
		"model": *model, "api": *api, "mcp": *mcpURL, "db_manifest": *dbMan,
		"tasks": *tasksPath, "started_at": time.Now().UTC().Format(time.RFC3339),
		"prompt_sha256": sha(systemPrompt),
	}
	enc := json.NewEncoder(f)
	var (
		mu   sync.Mutex
		wg   sync.WaitGroup
		sum  = map[string]*specbench.Summary{}
		done int
	)
	ch := make(chan specbench.Task)
	for w := 0; w < max(*workers, 1); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for t := range ch {
				r := run(be, mcp, t, *maxRounds, *resultMax)
				r.Meta = meta
				mu.Lock()
				_ = enc.Encode(r)
				if sum[t.Type] == nil {
					sum[t.Type] = &specbench.Summary{}
				}
				sum[t.Type].Add(r.Score, r.Final != "")
				done++
				log.Printf("[%d/%d] %s: answer=%v citation=%v (%d tool calls, %.0fs)",
					done, len(tasks), t.ID, r.Score.Answer, r.Score.Citation, len(r.ToolCalls), r.Duration)
				mu.Unlock()
			}
		}()
	}
	for _, t := range tasks {
		ch <- t
	}
	close(ch)
	wg.Wait()

	fmt.Printf("\nmodel=%s tools=%v tasks=%s\n", *model, mcp != nil, *tasksPath)
	for typ, s := range sum {
		fmt.Printf("  %-8s %s\n", typ, s)
	}
	fmt.Printf("results: %s\n", *out)
}

func run(be llm.Backend, mcp *mcpclient.Client, t specbench.Task, maxRounds, resultMax int) record {
	start := time.Now()
	r := record{ID: t.ID, Type: t.Type, Question: t.Question, Gold: t.Gold,
		GoldSpec: t.SpecID, GoldSec: t.Section, Usage: map[string]int{}, Retrieved: map[string]int{}}
	c := be.NewConvo(systemPrompt, t.Question)
	retries := 0
	for round := 0; ; round++ {
		r.Rounds = round + 1
		withTools := mcp != nil
		if withTools && round >= maxRounds {
			withTools = false
			c.AddUser("Tool budget exhausted. Answer now. " + retryPrompt)
		}
		turn, err := c.Step(withTools)
		if err != nil {
			r.Error = err.Error()
			break
		}
		r.Usage["prompt"] += turn.Usage.Prompt
		r.Usage["completion"] += turn.Usage.Completion
		r.Usage["cache_read"] += turn.Usage.CacheRead

		if len(turn.ToolCalls) == 0 {
			a, ok := specbench.ParseAnswer(turn.Content)
			if !ok && turn.Reasoning != "" {
				a, ok = specbench.ParseAnswer(turn.Reasoning)
			}
			if !ok && retries < 2 {
				retries++
				c.AddUser(retryPrompt)
				continue
			}
			r.Final = turn.Content
			r.Score = specbench.Grade(t, a)
			r.PredSpec, r.PredSec = a.SpecID, a.Section
			break
		}
		for _, tc := range turn.ToolCalls {
			text, isErr, err := mcp.CallTool(tc.Function.Name, json.RawMessage(tc.Function.Arguments))
			if err != nil {
				text = "tool error: " + err.Error()
			} else if isErr {
				text = "tool error: " + text
			}
			if len(text) > resultMax {
				text = text[:resultMax] + "\n...[truncated]"
			}
			c.AddToolResult(tc, text)
			r.ToolCalls = append(r.ToolCalls, toolCall{Name: tc.Function.Name, Args: tc.Function.Arguments, Result: text})
			r.Retrieved[tc.Function.Name]++
		}
	}
	r.Duration = time.Since(start).Seconds()
	return r
}

func sha(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}
