// teleqna-eval runs TeleQnA multiple-choice questions against an
// OpenAI-compatible chat API, optionally bridging tools from a
// 3gpp-mcp server so the model can consult 3GPP specifications while
// answering. Results are written as JSONL plus a summary line.
package main

import (
	"bytes"
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
	if *resume {
		if err := checkResume(metaPath, m); err != nil {
			log.Fatal(err)
		}
	}
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
		records, err := readRecords(outPath)
		if err != nil {
			return nil, err
		}
		for _, r := range records {
			k := key{r.ID, r.RepeatIdx}
			if r.Attempt > attempts[k] {
				attempts[k] = r.Attempt
			}
			if r.Error == "" {
				done[k] = true
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

// readRecords reads a results file for -resume, and repairs a torn tail.
//
// A run killed mid-write leaves a partial final line. Appending to the file
// then glues new records onto those bytes, and every reader that parses a line
// at a time stops there — the resumed results are written but cannot be read.
// A malformed *last* line is therefore truncated away: it holds no answer, and
// the question it belonged to is replanned like any other missing one. A
// malformed line anywhere else is not a torn write, so it is an error rather
// than something to repair silently.
func readRecords(path string) ([]eval.Result, error) {
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var (
		out  []eval.Result
		good int // bytes through the last record that parsed
	)
	lines := bytes.SplitAfter(data, []byte{'\n'})
	for i, line := range lines {
		if len(bytes.TrimSpace(line)) == 0 {
			good += len(line)
			continue
		}
		var r eval.Result
		if err := json.Unmarshal(line, &r); err != nil {
			if i != len(lines)-1 {
				return nil, fmt.Errorf("%s: line %d is not a record: %w", path, i+1, err)
			}
			log.Printf("%s: discarding a torn final record of %d bytes", path, len(line))
			return out, os.Truncate(path, int64(good))
		}
		good += len(line)
		out = append(out, r)
	}
	return out, nil
}

// checkResume refuses to append to a file measured under different settings.
//
// plan matches records by (question, repeat) alone, so a resume pointed at
// another run's output would reuse its answers and then stamp them with this
// invocation's metadata — a file that reads as one measurement and is two. The
// fields compared are the ones that decide what an answer means; the run id,
// timestamp, harness commit, base URL and repeat count are free to differ,
// since -repeat is how a resume adds passes in the first place.
func checkResume(metaPath string, m meta) error {
	data, err := os.ReadFile(metaPath)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	var old meta
	if err := json.Unmarshal(data, &old); err != nil {
		return fmt.Errorf("%s: %w", metaPath, err)
	}
	temp := func(t *float64) string {
		if t == nil {
			return "unset"
		}
		return fmt.Sprint(*t)
	}
	var diff []string
	for _, f := range []struct{ name, was, now string }{
		{"model", old.Model, m.Model},
		{"api", old.API, m.API},
		{"retrieval", old.Retrieval, m.Retrieval},
		{"fixed_k", fmt.Sprint(old.FixedK), fmt.Sprint(m.FixedK)},
		{"prompt_id", old.PromptID, m.PromptID},
		{"prompt_sha256", old.PromptSHA, m.PromptSHA},
		{"max_tokens", fmt.Sprint(old.MaxTokens), fmt.Sprint(m.MaxTokens)},
		{"max_rounds", fmt.Sprint(old.MaxRounds), fmt.Sprint(m.MaxRounds)},
		{"temperature", temp(old.Temperature), temp(m.Temperature)},
		{"filter", old.Filter, m.Filter},
		{"category", old.Category, m.Category},
		{"seed", fmt.Sprint(old.Seed), fmt.Sprint(m.Seed)},
		{"questions", fmt.Sprint(old.Questions), fmt.Sprint(m.Questions)},
		{"db_manifest", old.DBManifest, m.DBManifest},
	} {
		if f.was != f.now {
			diff = append(diff, fmt.Sprintf("  %s: %q -> %q", f.name, f.was, f.now))
		}
	}
	if len(diff) > 0 {
		return fmt.Errorf("-resume: %s was measured with different settings, so its records "+
			"are not this run's:\n%s\nwrite to a different -out, or drop -resume to start over",
			m.Out, strings.Join(diff, "\n"))
	}
	return nil
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
