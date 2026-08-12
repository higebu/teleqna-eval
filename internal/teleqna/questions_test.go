package teleqna

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

const sample = `{
  "question 12": {
    "question": "What does 3GPP TS 23.501 define?",
    "option 1": "architecture",
    "option 2": "codecs",
    "answer": "option 1: architecture",
    "category": "Standards specifications"
  },
  "question 3": {
    "question": "Which IEEE 802.11 amendment adds HE?",
    "option 1": "ax",
    "option 2": "ac",
    "answer": "option 1: ax",
    "category": "Standards specifications"
  },
  "question 7": {
    "question": "Overview question about 3GPP",
    "option 1": "a",
    "option 2": "b",
    "answer": "option 2: b",
    "category": "Standards overview"
  },
  "question 8": {
    "question": "Only one option, 3GPP",
    "option 1": "a",
    "answer": "option 1: a",
    "category": "Standards specifications"
  },
  "question 9": {
    "question": "Unparseable answer, 3GPP",
    "option 1": "a",
    "option 2": "b",
    "answer": "a",
    "category": "Standards specifications"
  }
}`

func writeSample(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "TeleQnA.json")
	if err := os.WriteFile(path, []byte(sample), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func ids(qs []Question) []string {
	out := make([]string, len(qs))
	for i, q := range qs {
		out[i] = q.ID
	}
	return out
}

func TestLoad(t *testing.T) {
	path := writeSample(t)
	tests := []struct {
		name     string
		category string
		filter   string
		want     []string
	}{
		{"no filters", "", "", []string{"question 3", "question 7", "question 12"}},
		{"category prefix", "Standards specifications", "", []string{"question 3", "question 12"}},
		{"text filter", "", "3GPP", []string{"question 7", "question 12"}},
		{"both", "Standards specifications", "3GPP", []string{"question 12"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			qs, err := Load(path, tt.category, tt.filter)
			if err != nil {
				t.Fatal(err)
			}
			// "question 8" (one option) and "question 9" (bad answer) never load.
			if got := ids(qs); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ids = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestLoadFields(t *testing.T) {
	qs, err := Load(writeSample(t), "", "23.501")
	if err != nil {
		t.Fatal(err)
	}
	if len(qs) != 1 {
		t.Fatalf("got %d questions, want 1", len(qs))
	}
	q := qs[0]
	if q.Answer != 1 || q.Category != "Standards specifications" {
		t.Errorf("answer=%d category=%q", q.Answer, q.Category)
	}
	if !reflect.DeepEqual(q.Options, map[int]string{1: "architecture", 2: "codecs"}) {
		t.Errorf("options = %v", q.Options)
	}
}

func TestSelect(t *testing.T) {
	all := func() []Question {
		var qs []Question
		for _, id := range []string{"question 1", "question 2", "question 3", "question 4"} {
			qs = append(qs, Question{ID: id})
		}
		return qs
	}

	if got := ids(Select(all(), "question 3, question 1", 1, 42)); !reflect.DeepEqual(got, []string{"question 1", "question 3"}) {
		t.Errorf("-ids result = %v (must ignore n and keep load order)", got)
	}
	first := ids(Select(all(), "", 2, 42))
	if len(first) != 2 {
		t.Fatalf("got %d questions, want 2", len(first))
	}
	if second := ids(Select(all(), "", 2, 42)); !reflect.DeepEqual(first, second) {
		t.Errorf("same seed gave %v then %v", first, second)
	}
	if n := len(Select(all(), "", 99, 42)); n != 4 {
		t.Errorf("n larger than the pool returned %d questions, want 4", n)
	}
}

func TestFormat(t *testing.T) {
	got := Format(Question{Text: "Q?", Options: map[int]string{2: "b", 1: "a", 10: "j"}})
	want := "Q?\n\noption 1: a\noption 2: b\noption 10: j\n"
	if got != want {
		t.Errorf("Format() = %q, want %q", got, want)
	}
}

func TestParseRelease(t *testing.T) {
	tests := map[string]int{
		"What is the AMF? [3GPP Release 18]": 18,
		"Something [3GPP Release 14]":        14,
		"No tag here":                        0,
		"[Release 18]":                       0,
	}
	for text, want := range tests {
		if got := ParseRelease(text); got != want {
			t.Errorf("ParseRelease(%q) = %d, want %d", text, got, want)
		}
	}
}

func TestGoldAnswer(t *testing.T) {
	q := Question{Answer: 2, Options: map[int]string{1: "a", 2: "b"}}
	if got := GoldAnswer(q); got != "option 2: b" {
		t.Errorf("got %q", got)
	}
}
