// Package eval runs TeleQnA questions against a model, letting it call MCP
// tools until it answers or the round budget runs out.
package eval

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"time"

	"teleqna-eval/internal/llm"
	"teleqna-eval/internal/teleqna"
)

const systemPrompt = `You are a telecommunications standards expert answering multiple-choice questions about 3GPP specifications.

You have tools that let you consult the actual 3GPP specification documents (full-text search, table of contents, section text). Use them to verify your answer before responding: search for the key terms of the question, read the relevant section, and only then answer. Prefer exact wording from the specification over memory.

When you are ready to answer, reply with your reasoning followed by a final line in exactly this format:
ANSWER: <option number>

The final line must contain only one option number.`

const systemPromptNoTools = `You are a telecommunications standards expert answering multiple-choice questions about 3GPP specifications.

Answer from your own knowledge. Reply with brief reasoning followed by a final line in exactly this format:
ANSWER: <option number>

The final line must contain only one option number.`

// ToolCaller is the MCP side of the loop; a nil ToolCaller is the no-tools
// baseline. Keep it a nil interface value, not a typed nil pointer.
type ToolCaller interface {
	CallTool(name string, args json.RawMessage) (text string, isErr bool, err error)
}

type Options struct {
	MaxRounds     int
	ToolResultMax int
	Workers       int
}

// Result is one JSONL record; the field names are consumed by the analysis
// scripts in teleqna-eval-results, so do not rename them.
type Result struct {
	ID            string        `json:"id"`
	Category      string        `json:"category"`
	Question      string        `json:"question"`
	Expected      int           `json:"expected"`
	Predicted     int           `json:"predicted"`
	Correct       bool          `json:"correct"`
	Rounds        int           `json:"rounds"`
	ToolCalls     []ToolCallLog `json:"tool_calls"`
	PromptTok     int           `json:"prompt_tokens"`
	CompleteTok   int           `json:"completion_tokens"`
	CacheReadTok  int           `json:"cache_read_tokens"`
	CacheWriteTok int           `json:"cache_write_tokens"`
	DurationSec   float64       `json:"duration_sec"`
	FinalMessage  string        `json:"final_message"`
	Error         string        `json:"error,omitempty"`
}

type ToolCallLog struct {
	Name string `json:"name"`
	Args string `json:"args"`
}

// The colon class holds ASCII ":" and full-width "："; models occasionally
// answer with the latter.
var (
	answerLineRe = regexp.MustCompile(`(?mi)^[\s>*#]*ANSWER\s*[:：]\s*\**\s*(?:option\s*)?(\d)`)
	answerAnyRe  = regexp.MustCompile(`(?i)ANSWER\s*[:：]\s*\**\s*(?:option\s*)?(\d)`)
	optionAnyRe  = regexp.MustCompile(`(?i)\boption\s*(\d)\b`)
)

// extractAnswer tries the strict answer line first, then progressively looser
// fallbacks; 0 means no option number was found.
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

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + fmt.Sprintf("\n...[truncated %d bytes; refine the query or use offset to read more]", len(s)-max)
}

// One evaluates a single question.
func One(be llm.Backend, mcp ToolCaller, q teleqna.Question, opts Options) Result {
	start := time.Now()
	r := Result{ID: q.ID, Category: q.Category, Question: q.Text, Expected: q.Answer}
	sys := systemPrompt
	if mcp == nil {
		sys = systemPromptNoTools
	}
	c := be.NewConvo(sys, teleqna.Format(q))
	answerRetries := 0
	for round := 0; ; round++ {
		r.Rounds = round + 1
		withTools := mcp != nil
		if round >= opts.MaxRounds {
			// Force a final answer: drop tools and ask for the answer.
			withTools = false
			c.AddUser("Tool budget exhausted. Based on what you have read so far, give your final answer now as 'ANSWER: <option number>'.")
		}
		t, err := c.Step(withTools)
		if err != nil {
			r.Error = err.Error()
			break
		}
		r.PromptTok += t.Usage.Prompt
		r.CompleteTok += t.Usage.Completion
		r.CacheReadTok += t.Usage.CacheRead
		r.CacheWriteTok += t.Usage.CacheWrite
		if len(t.ToolCalls) == 0 {
			pred := extractAnswer(t.Content)
			if pred == 0 {
				pred = extractAnswer(t.Reasoning)
			}
			if pred == 0 && answerRetries < 2 {
				answerRetries++
				c.AddUser("Your reply did not contain a readable answer. Reply now with one line only: ANSWER: <option number>")
				continue
			}
			r.FinalMessage = t.Content
			r.Predicted = pred
			break
		}
		for _, tc := range t.ToolCalls {
			r.ToolCalls = append(r.ToolCalls, ToolCallLog{Name: tc.Function.Name, Args: truncate(tc.Function.Arguments, 300)})
			text, isErr, err := mcp.CallTool(tc.Function.Name, json.RawMessage(tc.Function.Arguments))
			if err != nil {
				text = "tool error: " + err.Error()
			} else if isErr {
				text = "tool error: " + text
			}
			c.AddToolResult(tc, truncate(text, opts.ToolResultMax))
		}
	}
	r.Correct = r.Predicted == q.Answer && r.Predicted != 0
	r.DurationSec = time.Since(start).Seconds()
	return r
}
