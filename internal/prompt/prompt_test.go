package prompt

import (
	"encoding/json"
	"strings"
	"testing"

	"teleqna-eval/internal/teleqna"
)

var q = teleqna.Question{
	ID:       "question 42",
	Text:     "What is the purpose of the AMF? [3GPP Release 18]",
	Options:  map[int]string{2: "b option", 1: "a option", 3: "c option"},
	Answer:   2,
	Category: "Standards specifications",
}

func TestGetUnknown(t *testing.T) {
	if _, err := Get("nope"); err == nil {
		t.Fatal("want an error for an unknown prompt")
	}
	for _, id := range IDs() {
		if _, err := Get(id); err != nil {
			t.Errorf("Get(%q): %v", id, err)
		}
	}
}

// The system prompt must be the upstream text, byte for byte, so the run cannot
// be waved away as "they used their own wording".
func TestTeleQnASystemIsUpstreamText(t *testing.T) {
	p, _ := Get("teleqna")
	// Byte for byte from netop-team/TeleQnA evaluation_tools.py.
	const want = "\nPlease provide the answers to the following telecommunications related multiple choice " +
		"questions. The questions will be in a JSON format, the answers must also be in a JSON " +
		"format as follows:\n {\n\"question 1\": {\n\"question\": question,\n" +
		"\"answer\": \"option {answer id}: {answer string}\"\n},\n...\n}\n"
	if p.System != want {
		t.Errorf("system prompt drifted from upstream:\n got %q\nwant %q", p.System, want)
	}
	// A variant may only add to it, never edit it.
	search, _ := Get("search")
	if !strings.HasPrefix(search.System, want) {
		t.Error("search does not build on the upstream prompt")
	}
}

// The user message must be valid JSON in the upstream shape, with the options
// in numeric order regardless of Go's map iteration order.
func TestFormatTeleQnA(t *testing.T) {
	got := formatTeleQnA(q)
	const lead = "Here are the questions: \n "
	if !strings.HasPrefix(got, lead) {
		t.Fatalf("missing lead-in: %q", got)
	}
	body := strings.TrimPrefix(got, lead)

	var outer map[string]map[string]string
	if err := json.Unmarshal([]byte(body), &outer); err != nil {
		t.Fatalf("%v: %s", err, body)
	}
	inner, ok := outer["question 42"]
	if !ok {
		t.Fatalf("keyed by %v, want the question id", outer)
	}
	if inner["question"] != q.Text || inner["option 1"] != "a option" || inner["option 3"] != "c option" {
		t.Errorf("body = %v", inner)
	}
	if _, leaked := inner["answer"]; leaked {
		t.Error("the answer was sent to the model")
	}
	// Upstream's category pop tests the wrong dict, so the category is sent.
	if inner["category"] != q.Category {
		t.Errorf("category = %q, want %q (upstream sends it)", inner["category"], q.Category)
	}
	if i1, i2 := strings.Index(body, `"option 1"`), strings.Index(body, `"option 2"`); i1 > i2 {
		t.Error("options are not in numeric order")
	}
	// Python's json.dumps separators, so the bytes match the upstream harness.
	if !strings.Contains(body, `", "option 1": "`) || !strings.Contains(body, `{"question": "`) {
		t.Errorf("separators do not match json.dumps: %s", body)
	}
	if got != formatTeleQnA(q) {
		t.Error("formatting is not deterministic")
	}
}

func TestParseTeleQnA(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want int
		tier string
		raw  string
	}{
		{"wrapper", `{"question 42": {"question": "q", "answer": "option 2: b option"}}`, 2, TierJSON, "option 2: b option"},
		{"bare", `{"answer": "option 3: c"}`, 3, TierJSON, "option 3: c"},
		{"fenced", "here you go\n```json\n{\"answer\": \"option 1: a\"}\n```", 1, TierJSON, "option 1: a"},
		{"after reasoning", "long reasoning about option 5\n{\"answer\": \"option 2: b\"}", 2, TierJSON, "option 2: b"},
		{"broken json", `{"answer": "option 4: d",`, 4, TierAnswerField, ""},
		{"prose only", "I would pick option 3 here", 3, TierOptionScan, ""},
		{"nothing", "no idea", 0, TierNone, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := parseTeleQnA(tt.in)
			if got.Option != tt.want || got.Tier != tt.tier {
				t.Errorf("got %+v, want option %d tier %s", got, tt.want, tt.tier)
			}
			if tt.raw != "" && got.Raw != tt.raw {
				t.Errorf("raw = %q, want %q", got.Raw, tt.raw)
			}
		})
	}
}

// Double-digit options must survive: the previous parser matched a single digit
// and would have read "option 10" as option 1.
func TestParseTwoDigitOption(t *testing.T) {
	if got := parseTeleQnA(`{"answer": "option 10: j"}`); got.Option != 10 {
		t.Errorf("got %+v", got)
	}
	if got := parseAnswerLine("ANSWER: 12"); got.Option != 12 {
		t.Errorf("got %+v", got)
	}
}

func TestParseAnswerLine(t *testing.T) {
	tests := []struct {
		in   string
		want int
		tier string
	}{
		{"reasoning\nANSWER: 3", 3, TierAnswerLine},
		{"> **ANSWER:** 4", 4, TierAnswerLine},
		{"ANSWER:2", 2, TierAnswerLine},
		{"ANSWER：2", 2, TierAnswerLine}, // full-width colon
		{"ANSWER: option 1", 1, TierAnswerLine},
		{"the answer: 5 is right", 5, TierAnswerAny},
		{"option 1 is wrong, option 2 is right", 2, TierOptionScan},
		{"no answer here", 0, TierNone},
		{"", 0, TierNone},
	}
	for _, tt := range tests {
		got := parseAnswerLine(tt.in)
		if got.Option != tt.want || got.Tier != tt.tier {
			t.Errorf("parseAnswerLine(%q) = %+v, want option %d tier %s", tt.in, got, tt.want, tt.tier)
		}
	}
}

func TestSHA256IdentifiesTheWording(t *testing.T) {
	a, _ := Get("teleqna")
	b, _ := Get("search")
	if a.SHA256() == b.SHA256() {
		t.Error("variants with different wording share a hash")
	}
	if a.SHA256() != a.SHA256() {
		t.Error("hash is not stable")
	}
}
