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
	"strings"
	"time"

	"teleqna-eval/internal/eval"
	"teleqna-eval/internal/llm"
	"teleqna-eval/internal/mcpclient"
	"teleqna-eval/internal/teleqna"
)

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
		httpTimeout    = flag.Int("http-timeout", 300, "per-request timeout in seconds; raise it when running without a token cap")
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
	// prompts and tool use on mcp == nil.
	var caller eval.ToolCaller
	var tools []llm.ToolDef
	if *mcpURL != "" {
		mcp := mcpclient.New(*mcpURL, 120*time.Second)
		mcpTools, err := mcp.ListTools()
		if err != nil {
			log.Fatalf("mcp tools/list: %v", err)
		}
		for _, t := range mcpTools {
			tools = append(tools, llm.ToolDef{Name: t.Name, Description: t.Description, InputSchema: t.InputSchema})
		}
		caller = mcp
		log.Printf("bridged %d MCP tools from %s", len(tools), *mcpURL)
	}

	be, effMaxTokens, err := llm.New(llm.Config{
		API: *api, BaseURL: *baseURL, APIKey: apiKey, Model: *model,
		MaxTokens: *maxTokens, MaxTokensField: *maxTokensField, Extra: extra,
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
		if caller == nil {
			suffix = "notools"
		}
		tag := ""
		if *filter != "" {
			tag = "-" + strings.ToLower(strings.ReplaceAll(*filter, " ", "_"))
		}
		*outPath = fmt.Sprintf("results/%s-%s%s-%dq-seed%d.jsonl", *model, suffix, tag, *n, *seed)
	}
	if dir := filepath.Dir(*outPath); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			log.Fatal(err)
		}
	}
	out, err := os.Create(*outPath)
	if err != nil {
		log.Fatal(err)
	}
	defer out.Close()

	s := eval.Run(be, caller, qs, eval.Options{
		MaxRounds: *maxRounds, ToolResultMax: *resultMax, Workers: *workers,
	}, out)

	fmt.Printf("\nmodel=%s tools=%v questions=%d answered=%d correct=%d accuracy=%.1f%%\n",
		*model, caller != nil, s.Questions, s.Answered, s.Correct, 100*float64(s.Correct)/float64(s.Questions))
	fmt.Printf("tokens: prompt=%d completion=%d, tool calls=%d (avg %.1f/question)\n",
		s.Prompt, s.Completion, s.ToolCalls, float64(s.ToolCalls)/float64(s.Questions))
	if s.Prompt > 0 {
		fmt.Printf("cache: read=%d (%.0f%% of prompt) write=%d\n",
			s.CacheRead, 100*float64(s.CacheRead)/float64(s.Prompt), s.CacheWrite)
	}
	fmt.Printf("results: %s\n", *outPath)
}
