package eval

import (
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"

	"teleqna-eval/internal/llm"
	"teleqna-eval/internal/prompt"
	"teleqna-eval/internal/teleqna"
)

// fakeBackend replays scripted turns and records how it was driven.
type fakeBackend struct {
	turns  []*llm.Turn // one per Step call; the last one repeats
	err    error       // returned by every Step when set
	mu     sync.Mutex
	system string
	user   string
	steps  int
	tools  []bool   // withTools per step
	users  []string // AddUser texts
}

func (b *fakeBackend) NewConvo(system, user string) llm.Convo {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.system, b.user = system, user
	return b
}

func (b *fakeBackend) Step(withTools bool) (*llm.Turn, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.tools = append(b.tools, withTools)
	if b.err != nil {
		return nil, b.err
	}
	t := b.turns[min(b.steps, len(b.turns)-1)]
	b.steps++
	return t, nil
}

func (b *fakeBackend) AddUser(text string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.users = append(b.users, text)
}

func (b *fakeBackend) AddToolResult(tc llm.ToolCall, content string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.users = append(b.users, content)
}

type fakeTools struct {
	text  string
	isErr bool
	err   error
	calls []string
	// byName lets a test answer different tools differently.
	byName map[string]string
}

func (f *fakeTools) CallTool(name string, args json.RawMessage) (string, bool, error) {
	f.calls = append(f.calls, name+" "+string(args))
	if text, ok := f.byName[name]; ok {
		return text, f.isErr, f.err
	}
	return f.text, f.isErr, f.err
}

func toolTurn(name, args string) *llm.Turn {
	var tc llm.ToolCall
	tc.ID = "t1"
	tc.Function.Name = name
	tc.Function.Arguments = args
	return &llm.Turn{ToolCalls: []llm.ToolCall{tc}, Usage: llm.Usage{Prompt: 10, Completion: 2}}
}

var question = teleqna.Question{ID: "question 1", Category: "c", Text: "Q?", Answer: 2,
	Options: map[int]string{1: "a", 2: "b"}}

func mustPrompt(t *testing.T, id string) *prompt.Prompt {
	t.Helper()
	p, err := prompt.Get(id)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func opts(t *testing.T, id string, maxRounds, resultMax int) Options {
	return Options{MaxRounds: maxRounds, ToolResultMax: resultMax, Prompt: mustPrompt(t, id)}
}

// The whole point of the prompt registry: a pair's two conditions must send the
// same bytes. If this test ever fails, the measured effect of the tools has
// picked up a prompt difference again.
func TestPromptIdenticalAcrossConditions(t *testing.T) {
	for _, id := range prompt.IDs() {
		t.Run(id, func(t *testing.T) {
			answer := &llm.Turn{Content: "ANSWER: 2\n{\"question 1\": {\"answer\": \"option 2: b\"}}"}

			withTools := &fakeBackend{turns: []*llm.Turn{answer}}
			One(withTools, &fakeTools{text: "x"}, question, opts(t, id, 8, 100))

			noTools := &fakeBackend{turns: []*llm.Turn{answer}}
			One(noTools, nil, question, opts(t, id, 8, 100))

			if withTools.system != noTools.system {
				t.Errorf("system prompts differ:\n tools: %q\n base:  %q", withTools.system, noTools.system)
			}
			if withTools.user != noTools.user {
				t.Errorf("user messages differ:\n tools: %q\n base:  %q", withTools.user, noTools.user)
			}
			if !withTools.tools[0] || noTools.tools[0] {
				t.Errorf("tool attachment = %v / %v, want true / false", withTools.tools, noTools.tools)
			}
		})
	}
}

func TestOneToolRoundThenAnswer(t *testing.T) {
	be := &fakeBackend{turns: []*llm.Turn{
		toolTurn("search", `{"q":"5G"}`),
		{Content: "reasoning\nANSWER: 2", Usage: llm.Usage{Prompt: 30, Completion: 5, CacheRead: 20, CacheWrite: 3}},
	}}
	tools := &fakeTools{text: "spec text"}
	r, tr := One(be, tools, question, opts(t, "ansline", 8, 100))

	if !r.Correct || r.Predicted != 2 || r.Rounds != 2 {
		t.Errorf("result = %+v", r)
	}
	if r.Retrieval != RetrievalAgentic || r.ParseTier != prompt.TierAnswerLine {
		t.Errorf("retrieval = %q, tier = %q", r.Retrieval, r.ParseTier)
	}
	if len(r.ToolCalls) != 1 || r.ToolCalls[0].Name != "search" || r.ToolCalls[0].Args != `{"q":"5G"}` {
		t.Errorf("tool calls = %+v", r.ToolCalls)
	}
	if r.PromptTok != 40 || r.CompleteTok != 7 || r.CacheReadTok != 20 || r.CacheWriteTok != 3 {
		t.Errorf("tokens = %+v", r)
	}
	if len(tools.calls) != 1 || be.users[0] != "spec text" {
		t.Errorf("tool calls = %v, fed back = %v", tools.calls, be.users)
	}
	// The trace must carry what the model actually saw, not just the call.
	if len(tr.Steps) != 2 || len(tr.Steps[0].Tools) != 1 || tr.Steps[0].Tools[0].Result != "spec text" {
		t.Errorf("trace = %+v", tr.Steps)
	}
	if tr.System != be.system || tr.User != be.user {
		t.Error("trace does not record the messages that were sent")
	}
}

func TestOneNoTools(t *testing.T) {
	be := &fakeBackend{turns: []*llm.Turn{{Content: "ANSWER: 1"}}}
	r, _ := One(be, nil, question, opts(t, "ansline", 8, 100))
	if r.Predicted != 1 || r.Correct || r.Retrieval != RetrievalNone {
		t.Errorf("result = %+v", r)
	}
	if be.tools[0] {
		t.Errorf("withTools = %v", be.tools)
	}
}

func TestOneAnswerRetries(t *testing.T) {
	be := &fakeBackend{turns: []*llm.Turn{{Content: "no idea"}}}
	r, _ := One(be, nil, question, opts(t, "ansline", 8, 100))
	if r.Predicted != 0 || r.Correct || r.Rounds != 3 || r.AnswerRetries != 2 {
		t.Errorf("result = %+v", r)
	}
	if r.ParseTier != prompt.TierNone {
		t.Errorf("tier = %q", r.ParseTier)
	}
	if len(be.users) != 2 || !strings.Contains(be.users[0], "readable answer") {
		t.Errorf("reprompts = %v", be.users)
	}
}

// The answer may only be in the reasoning field of a reasoning model.
func TestOneAnswerFromReasoning(t *testing.T) {
	be := &fakeBackend{turns: []*llm.Turn{{Reasoning: "so ANSWER: 2"}}}
	if r, _ := One(be, nil, question, opts(t, "ansline", 8, 0)); r.Predicted != 2 || !r.Correct {
		t.Errorf("result = %+v", r)
	}
}

func TestOneMaxRounds(t *testing.T) {
	be := &fakeBackend{turns: []*llm.Turn{
		toolTurn("search", "{}"), toolTurn("search", "{}"), {Content: "ANSWER: 2"},
	}}
	r, _ := One(be, &fakeTools{text: "x"}, question, opts(t, "ansline", 2, 100))
	if r.Rounds != 3 || r.Predicted != 2 || !r.BudgetExhausted {
		t.Errorf("result = %+v", r)
	}
	if be.tools[2] {
		t.Error("tools still offered in the forced-answer round")
	}
	if last := be.users[len(be.users)-1]; !strings.Contains(last, "Tool budget exhausted") {
		t.Errorf("last prompt = %q", last)
	}
}

func TestOneStepError(t *testing.T) {
	be := &fakeBackend{err: errors.New("boom")}
	r, _ := One(be, nil, question, opts(t, "ansline", 8, 0))
	if r.Error != "boom" || r.Predicted != 0 || r.Correct {
		t.Errorf("result = %+v", r)
	}
}

func TestOneToolError(t *testing.T) {
	be := &fakeBackend{turns: []*llm.Turn{toolTurn("search", "{}"), {Content: "ANSWER: 2"}}}
	r, tr := One(be, &fakeTools{err: errors.New("timeout")}, question, opts(t, "ansline", 8, 100))
	if r.Error != "" || !r.Correct {
		t.Errorf("result = %+v", r)
	}
	if !strings.Contains(be.users[0], "tool error: timeout") {
		t.Errorf("tool result = %q", be.users[0])
	}
	if !tr.Steps[0].Tools[0].IsErr {
		t.Error("trace does not mark the failed tool call")
	}
}

// The strict scoring TeleQnA itself uses: the answer string, not just its number.
func TestOneStrictMatch(t *testing.T) {
	be := &fakeBackend{turns: []*llm.Turn{{Content: `{"question 1": {"answer": "option 2: b"}}`}}}
	r, _ := One(be, nil, question, opts(t, "teleqna", 8, 0))
	if !r.Correct || !r.StrictMatch || r.ParseTier != prompt.TierJSON {
		t.Errorf("result = %+v", r)
	}

	be = &fakeBackend{turns: []*llm.Turn{{Content: `{"question 1": {"answer": "option 2: something else"}}`}}}
	r, _ = One(be, nil, question, opts(t, "teleqna", 8, 0))
	if !r.Correct || r.StrictMatch {
		t.Errorf("option number should match while the string does not: %+v", r)
	}
}

func TestOneFixedK(t *testing.T) {
	be := &fakeBackend{turns: []*llm.Turn{{Content: `{"answer": "option 2: b"}`}}}
	tools := &fakeTools{byName: map[string]string{
		"search":      `{"results":[{"spec_id":"TS 23.501","number":"5.1","title":"Overview"}]}`,
		"get_section": "the section text",
	}}
	o := opts(t, "teleqna", 8, 1000)
	o.FixedK = 1
	r, _ := One(be, tools, question, o)

	if r.Retrieval != RetrievalFixedK || !r.Correct {
		t.Errorf("result = %+v", r)
	}
	if len(tools.calls) != 2 || !strings.HasPrefix(tools.calls[1], "get_section ") {
		t.Errorf("calls = %v", tools.calls)
	}
	if !strings.Contains(be.user, "the section text") || !strings.HasSuffix(be.user, mustPrompt(t, "teleqna").Format(question)) {
		t.Errorf("user message = %q", be.user)
	}
	// No tool loop: the model is never offered tools in this condition.
	for i, withTools := range be.tools {
		if withTools {
			t.Errorf("step %d was offered tools", i)
		}
	}
}

func jobsOf(qs []teleqna.Question) []Job {
	jobs := make([]Job, 0, len(qs))
	for _, q := range qs {
		jobs = append(jobs, Job{Q: q, Attempt: 1})
	}
	return jobs
}

func TestRun(t *testing.T) {
	be := &fakeBackend{turns: []*llm.Turn{{Content: "ANSWER: 2", Usage: llm.Usage{Prompt: 7, Completion: 1}}}}
	qs := []teleqna.Question{question, {ID: "question 2", Text: "Q2?", Answer: 1, Options: map[int]string{1: "a", 2: "b"}}}
	var buf, trace strings.Builder
	o := opts(t, "ansline", 8, 0)
	o.Workers = 2
	s := Run(be, nil, jobsOf(qs), o, &buf, &trace)

	if s.Questions != 2 || s.Correct != 1 || s.Answered != 2 || s.Prompt != 14 || s.Completion != 2 {
		t.Errorf("summary = %+v", s)
	}
	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("wrote %d JSONL lines, want 2", len(lines))
	}
	for _, line := range lines {
		var r Result
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			t.Fatalf("%v: %s", err, line)
		}
		if r.Predicted != 2 || r.Attempt != 1 || r.PromptID != "ansline" || r.PromptSHA == "" {
			t.Errorf("record = %+v", r)
		}
	}
	if n := len(strings.Split(strings.TrimSpace(trace.String()), "\n")); n != 2 {
		t.Errorf("wrote %d trace lines, want 2", n)
	}
}

func TestRunZeroWorkers(t *testing.T) {
	be := &fakeBackend{turns: []*llm.Turn{{Content: "ANSWER: 2"}}}
	var buf strings.Builder
	o := opts(t, "ansline", 8, 0)
	o.Workers = 0
	if s := Run(be, nil, jobsOf([]teleqna.Question{question}), o, &buf, nil); s.Correct != 1 {
		t.Errorf("summary = %+v", s)
	}
}

// A fallback parse is counted separately so it can be re-scored as a failure.
func TestRunCountsLooseParses(t *testing.T) {
	be := &fakeBackend{turns: []*llm.Turn{{Content: "I would pick option 2"}}}
	var buf strings.Builder
	o := opts(t, "ansline", 8, 0)
	if s := Run(be, nil, jobsOf([]teleqna.Question{question}), o, &buf, nil); s.LooseParse != 1 {
		t.Errorf("summary = %+v", s)
	}
}
