package specbench

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// Record is one line of a results file: what the model was asked, what it
// answered, and how that scored. The runner writes it and the grader reads it,
// from this one definition — a results file that two programs describe
// differently is a results file that can be rewritten without anyone noticing.
type Record struct {
	ID        string            `json:"id"`
	Type      string            `json:"type"`
	Question  string            `json:"question"`
	Gold      json.RawMessage   `json:"gold"`
	GoldSpec  string            `json:"gold_spec_id"`
	GoldSec   string            `json:"gold_section"`
	Score     Score             `json:"score"`
	PredSpec  string            `json:"predicted_spec_id"`
	PredSec   string            `json:"predicted_section"`
	Rounds    int               `json:"rounds"`
	ToolCalls []ToolCall        `json:"tool_calls"`
	Usage     map[string]int    `json:"usage"`
	Duration  float64           `json:"duration_sec"`
	Final     string            `json:"final_message"`
	Error     string            `json:"error,omitempty"`
	Retrieval string            `json:"retrieval"`
	Meta      map[string]string `json:"meta"`
	Retrieved map[string]int    `json:"retrieved,omitempty"`
}

type ToolCall struct {
	Name   string `json:"name"`
	Args   string `json:"args"`
	Result string `json:"result,omitempty"`
}

// LoadRecords reads a JSONL results file.
func LoadRecords(path string) ([]Record, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []Record
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<26)
	for line := 1; sc.Scan(); line++ {
		if len(sc.Bytes()) == 0 {
			continue
		}
		var r Record
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, line, err)
		}
		out = append(out, r)
	}
	return out, sc.Err()
}

// WriteRecords writes a JSONL results file.
func WriteRecords(w io.Writer, recs []Record) error {
	enc := json.NewEncoder(w)
	for _, r := range recs {
		if err := enc.Encode(r); err != nil {
			return err
		}
	}
	return nil
}
