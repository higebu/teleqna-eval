package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"teleqna-eval/internal/eval"
	"teleqna-eval/internal/teleqna"
)

var planQuestions = []teleqna.Question{
	{ID: "question 1", Text: "a", Answer: 1, Options: map[int]string{1: "x", 2: "y"}},
	{ID: "question 2", Text: "b", Answer: 1, Options: map[int]string{1: "x", 2: "y"}},
}

func writeRecords(t *testing.T, path string, rs ...eval.Result) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for _, r := range rs {
		if err := enc.Encode(r); err != nil {
			t.Fatal(err)
		}
	}
}

func TestPlanWithoutResume(t *testing.T) {
	jobs, err := plan(planQuestions, 3, filepath.Join(t.TempDir(), "out.jsonl"), false)
	if err != nil {
		t.Fatal(err)
	}
	if len(jobs) != 6 {
		t.Fatalf("got %d jobs, want 6", len(jobs))
	}
	for _, j := range jobs {
		if j.Attempt != 1 {
			t.Errorf("job %+v: want attempt 1", j)
		}
	}
}

// Resuming must re-run only what is missing or errored, and must say in the
// data which records were re-executed — the failure that made an earlier
// Sonnet run a splice of two sessions without the report noticing.
func TestPlanResumeSkipsCompleteRecords(t *testing.T) {
	path := filepath.Join(t.TempDir(), "out.jsonl")
	writeRecords(t, path,
		eval.Result{ID: "question 1", Attempt: 1},
		eval.Result{ID: "question 2", Attempt: 1, Error: "boom"},
	)

	jobs, err := plan(planQuestions, 1, path, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(jobs) != 1 {
		t.Fatalf("got %+v, want only the errored question", jobs)
	}
	if jobs[0].Q.ID != "question 2" || jobs[0].Attempt != 2 {
		t.Errorf("job = %+v, want question 2 attempt 2", jobs[0])
	}
}

func TestPlanResumeCountsRepeatsSeparately(t *testing.T) {
	path := filepath.Join(t.TempDir(), "out.jsonl")
	writeRecords(t, path,
		eval.Result{ID: "question 1", RepeatIdx: 0},
		eval.Result{ID: "question 2", RepeatIdx: 0},
		eval.Result{ID: "question 1", RepeatIdx: 1},
	)

	jobs, err := plan(planQuestions, 2, path, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(jobs) != 1 || jobs[0].Q.ID != "question 2" || jobs[0].RepeatIdx != 1 {
		t.Errorf("jobs = %+v, want only question 2 of repeat 1", jobs)
	}
}

func TestPlanResumeWithNoExistingFile(t *testing.T) {
	jobs, err := plan(planQuestions, 1, filepath.Join(t.TempDir(), "missing.jsonl"), true)
	if err != nil {
		t.Fatal(err)
	}
	if len(jobs) != 2 {
		t.Errorf("got %d jobs, want 2", len(jobs))
	}
}

func TestWriteMeta(t *testing.T) {
	path := filepath.Join(t.TempDir(), "run.meta.json")
	if err := writeMeta(path, meta{RunID: "r1", PromptID: "teleqna", Model: "m"}); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var m meta
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatal(err)
	}
	if m.RunID != "r1" || m.PromptID != "teleqna" || m.Model != "m" {
		t.Errorf("meta = %+v", m)
	}
}
