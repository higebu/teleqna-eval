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
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"teleqna-eval/internal/llm"
	"teleqna-eval/internal/mcpclient"
	"teleqna-eval/internal/retrieval"
	"teleqna-eval/internal/specbench"
)

// The one prompt both conditions use. It never mentions tools: whether tools
// are attached is the only thing that differs between the two runs.
const systemPrompt = `You are answering questions about 3GPP specifications.

Reply with a JSON object and nothing else:
{"answer": <answer>, "spec_id": "<specification>", "section": "<section>"}

- For a list of names — ASN.1 fields, required properties — "answer" is a JSON array of them. Give ASN.1 fields in the order they are defined.
- For an equation, "answer" is a JSON string holding the LaTeX of the equation, without $ delimiters.
- For a single value — a wire code, an element name, a data type — "answer" is a JSON string or number holding just that value.
- "spec_id" is the specification the answer is defined in, e.g. "TS 38.331".
- "section" is the clause number, or the clause heading when the specification numbers clauses by name. For an OpenAPI schema it is the name of the API definition that declares it, e.g. "Nnrf_NFManagement".

The citation is part of the answer: it must be where this is actually defined.`

const retryPrompt = `Your reply did not contain the JSON object. Reply now with the JSON object only, in the format given above.`

// goldCitation is the clause a task was generated from, or the API document for
// an OpenAPI schema, which is cited by name rather than by clause.
func goldCitation(t specbench.Task) string {
	if t.APIName != "" {
		return t.APIName
	}
	return t.Section
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
		fixedK    = flag.Int("fixedk", 0, "retrieval baseline: one search, top-k sections prepended, no tools (0 = let the model drive its own tool loop)")
		ctxFile   = flag.String("context", "", "retrieval baseline from a {task id: context} JSON, prepended with no tools attached (bench/openapi_retrieve.py writes these)")
		ctxLabel  = flag.String("context-label", "context", "retrieval name recorded for -context runs, e.g. openapi-bm25")
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
	// The names of the bridged tools, recorded in every record's meta. Which
	// tools were attached is part of what an answer means — two runs against
	// the same endpoint are not the same condition if the server gained a tool
	// in between — and the filename can only claim it.
	var toolNames []string
	if *mcpURL != "" {
		mcp = mcpclient.New(*mcpURL, 120*time.Second)
		mt, err := mcp.ListTools()
		if err != nil {
			log.Fatalf("mcp tools/list: %v", err)
		}
		for _, t := range mt {
			tools = append(tools, llm.ToolDef{Name: t.Name, Description: t.Description, InputSchema: t.InputSchema})
			toolNames = append(toolNames, t.Name)
		}
		sort.Strings(toolNames)
		log.Printf("bridged %d MCP tools from %s: %s", len(tools), *mcpURL, strings.Join(toolNames, " "))
	}

	be, _, err := llm.New(llm.Config{
		API: *api, BaseURL: *baseURL, APIKey: key, Model: *model,
		MaxTokens: *maxTokens, MaxTokensField: "max_tokens", Temperature: temperature,
		HTTP: &http.Client{Timeout: time.Duration(*timeout) * time.Second}, Tools: tools,
	})
	if err != nil {
		log.Fatal(err)
	}

	if *fixedK > 0 && mcp == nil {
		log.Fatal("-fixedk needs an MCP endpoint to retrieve from")
	}
	// A precomputed baseline is still a baseline: it attaches no tools, so the
	// two cannot both be in play.
	var contexts map[string]string
	if *ctxFile != "" {
		if *fixedK > 0 || mcp != nil {
			log.Fatal("-context is a baseline of its own: run it without -fixedk and without -mcp")
		}
		data, err := os.ReadFile(*ctxFile)
		if err != nil {
			log.Fatal(err)
		}
		if err := json.Unmarshal(data, &contexts); err != nil {
			log.Fatalf("%s: %v", *ctxFile, err)
		}
	}

	if *out == "" {
		suffix := "notools"
		switch {
		case contexts != nil:
			suffix = *ctxLabel
		case *fixedK > 0:
			suffix = fmt.Sprintf("fixedk%d", *fixedK)
		case mcp != nil && *maxRounds == 1:
			suffix = "round1"
		case mcp != nil:
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

	retrievalMode := "none"
	switch {
	case contexts != nil:
		retrievalMode = *ctxLabel
	case *fixedK > 0:
		retrievalMode = "fixedk"
	case mcp != nil && *maxRounds == 1:
		// One round of tools and then the answer is forced. It is a different
		// condition from the 20-round one, not a smaller version of it: what it
		// measures is whether a single query of the model's own construction
		// reaches the answer, so it must not be recorded under the same name.
		retrievalMode = "agentic-1"
	case mcp != nil:
		retrievalMode = "agentic"
	}
	meta := map[string]string{
		"model": *model, "api": *api, "mcp": *mcpURL, "db_manifest": *dbMan,
		"tasks": *tasksPath, "started_at": time.Now().UTC().Format(time.RFC3339),
		"prompt_sha256": sha(systemPrompt), "retrieval": retrievalMode,
		"fixed_k": strconv.Itoa(*fixedK), "context_file": *ctxFile,
		"mcp_tools": strings.Join(toolNames, ","),
		// The round budget is part of what a run means, and until now only the
		// filename carried it.
		"max_rounds": strconv.Itoa(*maxRounds),
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
				r := run(be, mcp, t, *maxRounds, *resultMax, *fixedK, contexts[t.ID])
				r.Meta = meta
				r.Retrieval = retrievalMode
				mu.Lock()
				_ = enc.Encode(r)
				if sum[t.Type] == nil {
					sum[t.Type] = &specbench.Summary{}
				}
				sum[t.Type].Add(r.Score, r.Final != "")
				done++
				log.Printf("[%d/%d] %s: answer=%v (%d tool calls, %.0fs)",
					done, len(tasks), t.ID, r.Score.Answer, len(r.ToolCalls), r.Duration)
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

func run(be llm.Backend, mcp *mcpclient.Client, t specbench.Task, maxRounds, resultMax, fixedK int, prepared string) specbench.Record {
	start := time.Now()
	r := specbench.Record{ID: t.ID, Type: t.Type, Question: t.Question, Gold: t.Gold,
		GoldSpec: t.SpecID, GoldSec: goldCitation(t), Usage: map[string]int{}, Retrieved: map[string]int{}}

	user := t.Question
	agentic := mcp != nil && fixedK == 0
	if prepared != "" {
		// A baseline retrieved ahead of time. It is prepended exactly as the
		// live one is, so the question itself is rendered identically in every
		// condition.
		user = prepared + user
		r.Retrieved["prepared_context"]++
	}
	if mcp != nil && fixedK > 0 {
		// The retrieved text is prepended, so the question itself is rendered
		// exactly as in every other condition.
		ctx, calls := retrieval.FixedK(mcp, t.Question, fixedK, resultMax)
		user = ctx + user
		for _, c := range calls {
			r.ToolCalls = append(r.ToolCalls, specbench.ToolCall{Name: c.Name, Args: c.Args, Result: c.Result})
			r.Retrieved[c.Name]++
		}
	}

	c := be.NewConvo(systemPrompt, user)
	retries := 0
	for round := 0; ; round++ {
		r.Rounds = round + 1
		withTools := agentic
		if agentic && round >= maxRounds {
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
			text = retrieval.Truncate(text, resultMax)
			c.AddToolResult(tc, text)
			r.ToolCalls = append(r.ToolCalls, specbench.ToolCall{Name: tc.Function.Name, Args: tc.Function.Arguments, Result: text})
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
