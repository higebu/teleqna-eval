// Package eval runs TeleQnA questions against a model, letting it call MCP
// tools until it answers or the round budget runs out.
//
// The prompt does not depend on whether tools are available: both conditions of
// a pair send the identical system and user messages, and differ only in
// whether tool definitions are attached to the request. Anything that would
// reintroduce that asymmetry belongs in a separate prompt variant, not here.
package eval

import (
	"encoding/json"
	"strings"
	"time"

	"3gpp-mcp-bench/internal/llm"
	"3gpp-mcp-bench/internal/prompt"
	"3gpp-mcp-bench/internal/retrieval"
	"3gpp-mcp-bench/internal/teleqna"
)

// ToolCaller is the MCP side of the loop; a nil ToolCaller is the no-tools
// baseline. Keep it a nil interface value, not a typed nil pointer.
type ToolCaller interface {
	CallTool(name string, args json.RawMessage) (text string, isErr bool, err error)
}

// Retrieval modes recorded on every result.
const (
	RetrievalNone    = "none"    // no tools attached
	RetrievalAgentic = "agentic" // the model drives its own tool loop
	RetrievalFixedK  = "fixedk"  // one search, top-k prepended, no tool loop
)

type Options struct {
	MaxRounds     int
	ToolResultMax int
	Workers       int
	Prompt        *prompt.Prompt
	RunID         string
	// FixedK > 0 replaces the tool loop with a single search whose top-k hits
	// are prepended to the user message.
	FixedK int
}

// Result is one JSONL record; the field names are the wire format the analysis
// scripts read, so do not rename them.
type Result struct {
	ID            string        `json:"id"`
	Category      string        `json:"category"`
	Question      string        `json:"question"`
	Release       int           `json:"release"`
	Expected      int           `json:"expected"`
	Predicted     int           `json:"predicted"`
	Correct       bool          `json:"correct"`
	StrictMatch   bool          `json:"strict_match"`
	Rounds        int           `json:"rounds"`
	ToolCalls     []ToolCallLog `json:"tool_calls"`
	PromptTok     int           `json:"prompt_tokens"`
	CompleteTok   int           `json:"completion_tokens"`
	CacheReadTok  int           `json:"cache_read_tokens"`
	CacheWriteTok int           `json:"cache_write_tokens"`
	DurationSec   float64       `json:"duration_sec"`
	FinalMessage  string        `json:"final_message"`
	Error         string        `json:"error,omitempty"`

	// Provenance: which run, prompt and parsing step produced this record.
	RunID           string `json:"run_id"`
	Attempt         int    `json:"attempt"`
	RepeatIdx       int    `json:"repeat_idx"`
	PromptID        string `json:"prompt_id"`
	PromptSHA       string `json:"prompt_sha256"`
	Retrieval       string `json:"retrieval"`
	ParseTier       string `json:"parse_tier"`
	AnswerRaw       string `json:"answer_raw"`
	AnswerRetries   int    `json:"answer_retries"`
	BudgetExhausted bool   `json:"budget_exhausted"`
}

type ToolCallLog struct {
	Name string `json:"name"`
	Args string `json:"args"`
}

// Trace is the full record of one question: every message and every tool result
// exactly as the model saw it. It goes to a sidecar file because tool results
// dominate its size — a full-pool tools run is hundreds of megabytes.
type Trace struct {
	RunID     string      `json:"run_id"`
	ID        string      `json:"id"`
	RepeatIdx int         `json:"repeat_idx"`
	Attempt   int         `json:"attempt"`
	PromptID  string      `json:"prompt_id"`
	System    string      `json:"system"`
	User      string      `json:"user"`
	Steps     []TraceStep `json:"steps"`
}

type TraceStep struct {
	Round     int             `json:"round"`
	User      string          `json:"user,omitempty"`
	Content   string          `json:"content,omitempty"`
	Reasoning string          `json:"reasoning,omitempty"`
	Tools     []TraceToolCall `json:"tools,omitempty"`
}

type TraceToolCall struct {
	Name   string `json:"name"`
	Args   string `json:"args"`
	Result string `json:"result"`
	IsErr  bool   `json:"is_error,omitempty"`
}

// One evaluates a single question and returns both the result record and the
// full trace behind it.
func One(be llm.Backend, mcp ToolCaller, q teleqna.Question, opts Options) (Result, *Trace) {
	start := time.Now()
	p := opts.Prompt
	r := Result{
		ID: q.ID, Category: q.Category, Question: q.Text, Release: q.Release,
		Expected: q.Answer, RunID: opts.RunID, Attempt: 1,
		PromptID: p.ID, PromptSHA: p.SHA256(), Retrieval: RetrievalNone,
	}

	user := p.Format(q)
	agentic := mcp != nil && opts.FixedK == 0
	var retrieved []TraceToolCall
	switch {
	case agentic:
		r.Retrieval = RetrievalAgentic
	case mcp != nil:
		r.Retrieval = RetrievalFixedK
		ctx, calls := retrieval.FixedK(mcp, q.Text, opts.FixedK, opts.ToolResultMax)
		for _, c := range calls {
			retrieved = append(retrieved, TraceToolCall{Name: c.Name, Args: c.Args, Result: c.Result, IsErr: c.IsErr})
		}
		// The retrieved text is prepended, so the question itself is still
		// rendered by the same prompt variant as in every other condition.
		user = ctx + user
		for _, c := range retrieved {
			r.ToolCalls = append(r.ToolCalls, ToolCallLog{Name: c.Name, Args: c.Args})
		}
	}

	tr := &Trace{
		RunID: opts.RunID, ID: q.ID, PromptID: p.ID,
		System: p.System, User: user,
	}
	if len(retrieved) > 0 {
		tr.Steps = append(tr.Steps, TraceStep{Round: -1, Tools: retrieved})
	}

	c := be.NewConvo(p.System, user)
	for round := 0; ; round++ {
		r.Rounds = round + 1
		step := TraceStep{Round: round}
		withTools := agentic
		if agentic && round >= opts.MaxRounds {
			// Force a final answer: drop tools and ask for the answer.
			withTools = false
			r.BudgetExhausted = true
			step.User = budgetExhaustedMessage(p)
			c.AddUser(step.User)
		}
		t, err := c.Step(withTools)
		if err != nil {
			r.Error = err.Error()
			tr.Steps = append(tr.Steps, step)
			break
		}
		r.PromptTok += t.Usage.Prompt
		r.CompleteTok += t.Usage.Completion
		r.CacheReadTok += t.Usage.CacheRead
		r.CacheWriteTok += t.Usage.CacheWrite
		step.Content, step.Reasoning = t.Content, t.Reasoning

		if len(t.ToolCalls) == 0 {
			parsed := p.Parse(t.Content)
			if parsed.Option == 0 && t.Reasoning != "" {
				parsed = p.Parse(t.Reasoning)
			}
			if parsed.Option == 0 && r.AnswerRetries < 2 {
				r.AnswerRetries++
				step.User = p.Retry
				c.AddUser(p.Retry)
				tr.Steps = append(tr.Steps, step)
				continue
			}
			r.FinalMessage = t.Content
			r.Predicted = parsed.Option
			r.ParseTier = parsed.Tier
			r.AnswerRaw = parsed.Raw
			tr.Steps = append(tr.Steps, step)
			break
		}
		for _, tc := range t.ToolCalls {
			r.ToolCalls = append(r.ToolCalls, ToolCallLog{Name: tc.Function.Name, Args: tc.Function.Arguments})
			text, isErr, err := mcp.CallTool(tc.Function.Name, json.RawMessage(tc.Function.Arguments))
			if err != nil {
				text, isErr = "tool error: "+err.Error(), true
			} else if isErr {
				text = "tool error: " + text
			}
			sent := retrieval.Truncate(text, opts.ToolResultMax)
			c.AddToolResult(tc, sent)
			step.Tools = append(step.Tools, TraceToolCall{
				Name: tc.Function.Name, Args: tc.Function.Arguments, Result: sent, IsErr: isErr,
			})
		}
		tr.Steps = append(tr.Steps, step)
	}

	r.Correct = r.Predicted == q.Answer && r.Predicted != 0
	r.StrictMatch = strings.TrimSpace(r.AnswerRaw) == teleqna.GoldAnswer(q)
	r.DurationSec = time.Since(start).Seconds()
	return r, tr
}

// budgetExhaustedMessage asks for the answer in the format the running prompt
// established, so the forced answer parses the same way a voluntary one does.
func budgetExhaustedMessage(p *prompt.Prompt) string {
	return "Tool budget exhausted. Based on what you have read so far, give your final answer now. " + p.Retry
}
