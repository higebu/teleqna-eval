package eval

import (
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"

	"teleqna-eval/internal/llm"
	"teleqna-eval/internal/teleqna"
)

// fakeBackend replays scripted turns and records how it was driven.
type fakeBackend struct {
	turns  []*llm.Turn // one per Step call; the last one repeats
	err    error       // returned by every Step when set
	mu     sync.Mutex
	system string
	steps  int
	tools  []bool   // withTools per step
	users  []string // AddUser texts
}

func (b *fakeBackend) NewConvo(system, user string) llm.Convo {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.system = system
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
}

func (f *fakeTools) CallTool(name string, args json.RawMessage) (string, bool, error) {
	f.calls = append(f.calls, name+" "+string(args))
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

func TestOneToolRoundThenAnswer(t *testing.T) {
	be := &fakeBackend{turns: []*llm.Turn{
		toolTurn("search", `{"q":"5G"}`),
		{Content: "reasoning\nANSWER: 2", Usage: llm.Usage{Prompt: 30, Completion: 5, CacheRead: 20, CacheWrite: 3}},
	}}
	tools := &fakeTools{text: "spec text"}
	r := One(be, tools, question, Options{MaxRounds: 8, ToolResultMax: 100})

	if !r.Correct || r.Predicted != 2 || r.Rounds != 2 {
		t.Errorf("result = %+v", r)
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
	if be.system != systemPrompt || !be.tools[0] {
		t.Errorf("system = %q, withTools = %v", be.system, be.tools)
	}
}

func TestOneNoTools(t *testing.T) {
	be := &fakeBackend{turns: []*llm.Turn{{Content: "ANSWER: 1"}}}
	r := One(be, nil, question, Options{MaxRounds: 8, ToolResultMax: 100})
	if r.Predicted != 1 || r.Correct {
		t.Errorf("result = %+v", r)
	}
	if be.system != systemPromptNoTools || be.tools[0] {
		t.Errorf("system = %q, withTools = %v", be.system, be.tools)
	}
}

func TestOneAnswerRetries(t *testing.T) {
	be := &fakeBackend{turns: []*llm.Turn{{Content: "no idea"}}}
	r := One(be, nil, question, Options{MaxRounds: 8, ToolResultMax: 100})
	if r.Predicted != 0 || r.Correct || r.Rounds != 3 {
		t.Errorf("result = %+v", r)
	}
	if len(be.users) != 2 || !strings.Contains(be.users[0], "readable answer") {
		t.Errorf("reprompts = %v", be.users)
	}
}

// The answer may only be in the reasoning field of a reasoning model.
func TestOneAnswerFromReasoning(t *testing.T) {
	be := &fakeBackend{turns: []*llm.Turn{{Reasoning: "so ANSWER: 2"}}}
	if r := One(be, nil, question, Options{MaxRounds: 8}); r.Predicted != 2 || !r.Correct {
		t.Errorf("result = %+v", r)
	}
}

func TestOneMaxRounds(t *testing.T) {
	be := &fakeBackend{turns: []*llm.Turn{
		toolTurn("search", "{}"), toolTurn("search", "{}"), {Content: "ANSWER: 2"},
	}}
	r := One(be, &fakeTools{text: "x"}, question, Options{MaxRounds: 2, ToolResultMax: 100})
	if r.Rounds != 3 || r.Predicted != 2 {
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
	r := One(be, nil, question, Options{MaxRounds: 8})
	if r.Error != "boom" || r.Predicted != 0 || r.Correct {
		t.Errorf("result = %+v", r)
	}
}

func TestOneToolError(t *testing.T) {
	be := &fakeBackend{turns: []*llm.Turn{toolTurn("search", "{}"), {Content: "ANSWER: 2"}}}
	r := One(be, &fakeTools{err: errors.New("timeout")}, question, Options{MaxRounds: 8, ToolResultMax: 100})
	if r.Error != "" || !r.Correct {
		t.Errorf("result = %+v", r)
	}
	if !strings.Contains(be.users[0], "tool error: timeout") {
		t.Errorf("tool result = %q", be.users[0])
	}
}

func TestRun(t *testing.T) {
	be := &fakeBackend{turns: []*llm.Turn{{Content: "ANSWER: 2", Usage: llm.Usage{Prompt: 7, Completion: 1}}}}
	qs := []teleqna.Question{question, {ID: "question 2", Text: "Q2?", Answer: 1, Options: map[int]string{1: "a", 2: "b"}}}
	var buf strings.Builder
	s := Run(be, nil, qs, Options{MaxRounds: 8, Workers: 2}, &buf)

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
		if r.Predicted != 2 {
			t.Errorf("record = %+v", r)
		}
	}
}

func TestRunZeroWorkers(t *testing.T) {
	be := &fakeBackend{turns: []*llm.Turn{{Content: "ANSWER: 2"}}}
	var buf strings.Builder
	if s := Run(be, nil, []teleqna.Question{question}, Options{MaxRounds: 8, Workers: 0}, &buf); s.Correct != 1 {
		t.Errorf("summary = %+v", s)
	}
}

func TestExtractAnswer(t *testing.T) {
	tests := []struct {
		in   string
		want int
	}{
		{"reasoning\nANSWER: 3", 3},
		{"> **ANSWER:** 4", 4},
		{"ANSWER:2", 2},
		{"ANSWER：2", 2}, // full-width colon
		{"ANSWER: option 1", 1},
		{"the answer: 5 is right", 5},
		{"option 1 is wrong, option 2 is right", 2},
		{"no answer here", 0},
		{"", 0},
	}
	for _, tt := range tests {
		if got := extractAnswer(tt.in); got != tt.want {
			t.Errorf("extractAnswer(%q) = %d, want %d", tt.in, got, tt.want)
		}
	}
}

func TestTruncate(t *testing.T) {
	if got := truncate("abcde", 5); got != "abcde" {
		t.Errorf("got %q", got)
	}
	got := truncate("abcde", 3)
	if !strings.HasPrefix(got, "abc\n") || !strings.Contains(got, "truncated 2 bytes") {
		t.Errorf("got %q", got)
	}
}
