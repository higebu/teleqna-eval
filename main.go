// teleqna-eval runs TeleQnA multiple-choice questions against an
// OpenAI-compatible chat API, optionally bridging tools from a
// 3gpp-mcp server so the model can consult 3GPP specifications while
// answering. Results are written as JSONL plus a summary line.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime/debug"
	"strconv"
	"strings"
	"time"

	"teleqna-eval/internal/eval"
	"teleqna-eval/internal/llm"
	"teleqna-eval/internal/mcpclient"
	"teleqna-eval/internal/prompt"
	"teleqna-eval/internal/teleqna"
)

// meta is written next to the result file. It records everything needed to say
// which prompt, which model settings and which specification database produced
// a run — the provenance that earlier passes had to be reconstructed by hand.
type meta struct {
	RunID       string            `json:"run_id"`
	StartedAt   string            `json:"started_at"`
	HarnessSHA  string            `json:"harness_git_sha"`
	Args        []string          `json:"args"`
	Model       string            `json:"model"`
	API         string            `json:"api"`
	BaseURL     string            `json:"base_url"`
	Temperature *float64          `json:"temperature"`
	MaxTokens   int               `json:"max_tokens"`
	MaxRounds   int               `json:"max_rounds"`
	FixedK      int               `json:"fixed_k"`
	Retrieval   string            `json:"retrieval"`
	PromptID    string            `json:"prompt_id"`
	PromptSHA   string            `json:"prompt_sha256"`
	PromptText  string            `json:"prompt_system"`
	Questions   int               `json:"questions"`
	Repeat      int               `json:"repeat"`
	Seed        int64             `json:"seed"`
	Filter      string            `json:"filter"`
	Category    string            `json:"category"`
	MCPURL      string            `json:"mcp_url"`
	MCPServer   map[string]string `json:"mcp_server,omitempty"`
	MCPTools    []string          `json:"mcp_tools,omitempty"`
	DBManifest  string            `json:"db_manifest,omitempty"`
	Out         string            `json:"out"`
	Trace       string            `json:"trace,omitempty"`
}

func main() {
	var (
		model          = flag.String("model", "", "model id on the OpenAI-compatible endpoint (required)")
		api            = flag.String("api", "chat", "API style: 'chat' (chat/completions), 'responses' (OpenAI Responses API) or 'anthropic' (Anthropic Messages API)")
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
		temperature    = flag.String("temperature", "", "sampling temperature; empty sends no temperature field at all")
		httpTimeout    = flag.Int("http-timeout", 300, "per-request timeout in seconds; raise it when running without a token cap")
		extraBody      = flag.String("extra-body", "", "JSON object merged into every chat request, e.g. '{\"reasoning_effort\":\"none\"}'")
		resultMax      = flag.Int("tool-result-max", 16000, "max bytes of a tool result passed to the model")
		promptID       = flag.String("prompt", "teleqna", "prompt variant: "+strings.Join(prompt.IDs(), ", "))
		fixedK         = flag.Int("fixedk", 0, "retrieval baseline: run one search, prepend the top k sections and attach no tools (0 = let the model drive its own tool loop)")
		repeat         = flag.Int("repeat", 1, "run every question this many times (repeated measurements of one condition)")
		resume         = flag.Bool("resume", false, "append to an existing -out file, re-running only the questions that errored or are missing")
		runID          = flag.String("run-id", "", "identifier recorded on every record (default: timestamp)")
		dbManifest     = flag.String("db-manifest", "", "identifier of the pinned specification database the MCP server serves, recorded in the metadata")
		tracePath      = flag.String("trace", "auto", "path for the full message/tool-result trace ('auto' = <out>.trace.jsonl, '' = no trace)")
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
	p, err := prompt.Get(*promptID)
	if err != nil {
		log.Fatal(err)
	}
	var temp *float64
	if *temperature != "" {
		v, err := strconv.ParseFloat(*temperature, 64)
		if err != nil {
			log.Fatalf("-temperature: %v", err)
		}
		temp = &v
	}
	if *repeat < 1 {
		log.Fatal("-repeat must be at least 1")
	}
	qs, err := teleqna.Load(*dataPath, *category, *filter)
	if err != nil {
		log.Fatalf("load questions: %v", err)
	}
	qs = teleqna.Select(qs, *ids, *n, *seed)
	if len(qs) == 0 {
		log.Fatal("no questions matched: check -data, -category, -filter and -ids")
	}

	var extra map[string]any
	if *extraBody != "" {
		if err := json.Unmarshal([]byte(*extraBody), &extra); err != nil {
			log.Fatalf("-extra-body: %v", err)
		}
	}

	// caller stays a nil interface for the no-tools baseline: eval switches
	// tool attachment on mcp == nil. The prompt never depends on it.
	var caller eval.ToolCaller
	var tools []llm.ToolDef
	var toolNames []string
	var serverInfo map[string]string
	if *mcpURL != "" {
		mcp := mcpclient.New(*mcpURL, 120*time.Second)
		mcpTools, err := mcp.ListTools()
		if err != nil {
			log.Fatalf("mcp tools/list: %v", err)
		}
		for _, t := range mcpTools {
			tools = append(tools, llm.ToolDef{Name: t.Name, Description: t.Description, InputSchema: t.InputSchema})
			toolNames = append(toolNames, t.Name)
		}
		serverInfo = mcp.ServerInfo()
		caller = mcp
		log.Printf("bridged %d MCP tools from %s", len(tools), *mcpURL)
	}
	if *fixedK > 0 && caller == nil {
		log.Fatal("-fixedk needs an MCP endpoint to retrieve from")
	}

	be, effMaxTokens, err := llm.New(llm.Config{
		API: *api, BaseURL: *baseURL, APIKey: apiKey, Model: *model,
		MaxTokens: *maxTokens, MaxTokensField: *maxTokensField, Temperature: temp, Extra: extra,
		HTTP:  &http.Client{Timeout: time.Duration(*httpTimeout) * time.Second},
		Tools: tools,
	})
	if err != nil {
		log.Fatal(err)
	}
	if effMaxTokens != *maxTokens {
		log.Printf("max_tokens=%d (%s ceiling)", effMaxTokens, *model)
	}

	if *outPath == "" {
		suffix := "tools"
		switch {
		case caller == nil:
			suffix = "notools"
		case *fixedK > 0:
			suffix = fmt.Sprintf("fixedk%d", *fixedK)
		}
		tag := ""
		if *filter != "" {
			tag = "-" + strings.ToLower(strings.ReplaceAll(*filter, " ", "_"))
		}
		*outPath = fmt.Sprintf("results/%s-%s-%s%s-%dq-seed%d.jsonl", *model, *promptID, suffix, tag, *n, *seed)
	}
	if dir := filepath.Dir(*outPath); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			log.Fatal(err)
		}
	}
	if *tracePath == "auto" {
		*tracePath = strings.TrimSuffix(*outPath, ".jsonl") + ".trace.jsonl"
	}
	if *runID == "" {
		*runID = time.Now().UTC().Format("20060102T150405Z")
	}

	jobs, err := plan(qs, *repeat, *outPath, *resume)
	if err != nil {
		log.Fatal(err)
	}
	if len(jobs) == 0 {
		log.Fatalf("-resume: %s already has a record for every question", *outPath)
	}
	if *resume {
		log.Printf("resuming: %d of %d jobs still to run", len(jobs), len(qs)**repeat)
	}

	out, err := openOut(*outPath, *resume)
	if err != nil {
		log.Fatal(err)
	}
	defer out.Close()
	var trace *os.File
	if *tracePath != "" {
		if trace, err = openOut(*tracePath, *resume); err != nil {
			log.Fatal(err)
		}
		defer trace.Close()
	}

	retrieval := eval.RetrievalNone
	switch {
	case *fixedK > 0:
		retrieval = eval.RetrievalFixedK
	case caller != nil:
		retrieval = eval.RetrievalAgentic
	}
	m := meta{
		RunID: *runID, StartedAt: time.Now().UTC().Format(time.RFC3339), HarnessSHA: harnessSHA(),
		Args: os.Args[1:], Model: *model, API: *api, BaseURL: *baseURL,
		Temperature: temp, MaxTokens: effMaxTokens, MaxRounds: *maxRounds, FixedK: *fixedK,
		Retrieval: retrieval, PromptID: p.ID, PromptSHA: p.SHA256(), PromptText: p.System,
		Questions: len(qs), Repeat: *repeat, Seed: *seed, Filter: *filter, Category: *category,
		MCPURL: *mcpURL, MCPServer: serverInfo, MCPTools: toolNames, DBManifest: *dbManifest,
		Out: *outPath, Trace: *tracePath,
	}
	metaPath := strings.TrimSuffix(*outPath, ".jsonl") + ".meta.json"
	if err := writeMeta(metaPath, m); err != nil {
		log.Fatal(err)
	}

	s := eval.Run(be, caller, jobs, eval.Options{
		MaxRounds: *maxRounds, ToolResultMax: *resultMax, Workers: *workers,
		Prompt: p, RunID: *runID, FixedK: *fixedK,
	}, out, trace)

	fmt.Printf("\nmodel=%s prompt=%s retrieval=%s jobs=%d answered=%d correct=%d accuracy=%.1f%%\n",
		*model, p.ID, retrieval, s.Questions, s.Answered, s.Correct, 100*float64(s.Correct)/float64(s.Questions))
	fmt.Printf("tokens: prompt=%d completion=%d, tool calls=%d (avg %.1f/question)\n",
		s.Prompt, s.Completion, s.ToolCalls, float64(s.ToolCalls)/float64(s.Questions))
	if s.Prompt > 0 {
		fmt.Printf("cache: read=%d (%.0f%% of prompt) write=%d\n",
			s.CacheRead, 100*float64(s.CacheRead)/float64(s.Prompt), s.CacheWrite)
	}
	fmt.Printf("errors=%d answers needing a fallback parse=%d\n", s.Errors, s.LooseParse)
	fmt.Printf("results: %s\nmetadata: %s\n", *outPath, metaPath)
	if *tracePath != "" {
		fmt.Printf("trace: %s\n", *tracePath)
	}
}

// plan expands the questions into jobs. Without -resume that is every question
// once per repeat; with it, only the (question, repeat) pairs that are missing
// or errored, carrying an attempt number one higher than the record they
// replace so a spliced file shows which of its records were re-executed.
func plan(qs []teleqna.Question, repeat int, outPath string, resume bool) ([]eval.Job, error) {
	type key struct {
		id  string
		idx int
	}
	done := map[key]bool{}
	attempts := map[key]int{}
	if resume {
		f, err := os.Open(outPath)
		if err != nil && !os.IsNotExist(err) {
			return nil, err
		}
		if err == nil {
			defer f.Close()
			dec := json.NewDecoder(f)
			for {
				var r eval.Result
				if err := dec.Decode(&r); err != nil {
					break
				}
				k := key{r.ID, r.RepeatIdx}
				if r.Attempt > attempts[k] {
					attempts[k] = r.Attempt
				}
				if r.Error == "" {
					done[k] = true
				}
			}
		}
	}
	var jobs []eval.Job
	for idx := 0; idx < repeat; idx++ {
		for _, q := range qs {
			k := key{q.ID, idx}
			if done[k] {
				continue
			}
			jobs = append(jobs, eval.Job{Q: q, RepeatIdx: idx, Attempt: attempts[k] + 1})
		}
	}
	return jobs, nil
}

func openOut(path string, appendTo bool) (*os.File, error) {
	if appendTo {
		return os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	}
	return os.Create(path)
}

func writeMeta(path string, m meta) error {
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o644)
}

// harnessSHA reports the commit the binary was built from, so a result file can
// be tied to the harness that produced it.
func harnessSHA() string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return ""
	}
	for _, s := range info.Settings {
		if s.Key == "vcs.revision" {
			return s.Value
		}
	}
	return ""
}
