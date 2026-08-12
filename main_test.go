package main

import (
	"bytes"
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

// A run killed mid-write leaves a partial final line. Appending to it would
// glue the next record onto those bytes and every line-at-a-time reader would
// stop there, so the torn record is truncated away and its question replanned.
func TestPlanResumeRepairsATornTail(t *testing.T) {
	path := filepath.Join(t.TempDir(), "out.jsonl")
	writeRecords(t, path, eval.Result{ID: "question 1", Attempt: 1})
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString(`{"id":"question 2","corr`); err != nil {
		t.Fatal(err)
	}
	f.Close()

	jobs, err := plan(planQuestions, 1, path, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(jobs) != 1 || jobs[0].Q.ID != "question 2" {
		t.Fatalf("jobs = %+v, want only question 2", jobs)
	}
	// The file must now end at the last whole record, so what is appended next
	// is readable.
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if n := bytes.Count(data, []byte{'\n'}); n != 1 || !bytes.HasSuffix(data, []byte{'\n'}) {
		t.Errorf("file did not end at the last whole record: %q", data)
	}
}

// A malformed line that is not the last one is not a torn write; repairing it
// would throw away good records after it.
func TestPlanResumeRejectsAMalformedMiddle(t *testing.T) {
	path := filepath.Join(t.TempDir(), "out.jsonl")
	if err := os.WriteFile(path, []byte("{\"id\":\"question 1\"}\nnot json\n{\"id\":\"question 2\"}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := plan(planQuestions, 1, path, true); err == nil {
		t.Error("want an error for a malformed line in the middle")
	}
}

// plan matches records by (question, repeat) alone, so a resume pointed at
// another run's output would reuse its answers under this run's metadata.
func TestCheckResume(t *testing.T) {
	dir := t.TempDir()
	base := meta{Model: "m", API: "chat", Retrieval: "agentic", PromptID: "teleqna",
		PromptSHA: "abc", MaxRounds: 20, Seed: 42, Questions: 1509, DBManifest: "v2",
		MCPServer: map[string]string{"name": "3gpp-mcp", "version": "dev"},
		MCPTools:  []string{"search", "get_section"},
		Out:       "results/out.jsonl"}

	path := filepath.Join(dir, "none.meta.json")
	if err := checkResume(path, base); err != nil {
		t.Errorf("no existing metadata should be allowed: %v", err)
	}

	path = filepath.Join(dir, "out.meta.json")
	if err := writeMeta(path, base); err != nil {
		t.Fatal(err)
	}
	same := base
	same.RunID, same.StartedAt, same.HarnessSHA, same.Repeat = "later", "now", "deadbeef", 3
	same.MCPURL = "http://localhost:9999/mcp/"        // the server moved port
	same.MCPTools = []string{"get_section", "search"} // ...and listed its tools in another order
	if err := checkResume(path, same); err != nil {
		t.Errorf("only the free fields differ, want no error: %v", err)
	}
	for name, m := range map[string]meta{
		"model":       func() meta { m := base; m.Model = "other"; return m }(),
		"prompt":      func() meta { m := base; m.PromptSHA = "def"; return m }(),
		"retrieval":   func() meta { m := base; m.Retrieval = "fixedk"; return m }(),
		"db_manifest": func() meta { m := base; m.DBManifest = "v3"; return m }(),
		// The retrieval environment decides what an answer means as much as
		// the model does, and -db-manifest is optional, so it cannot be the
		// only thing standing for the corpus.
		"mcp_server": func() meta { m := base; m.MCPServer = map[string]string{"name": "other"}; return m }(),
		"mcp_tools":  func() meta { m := base; m.MCPTools = []string{"search"}; return m }(),
	} {
		if err := checkResume(path, m); err == nil {
			t.Errorf("%s differs, want an error", name)
		}
	}
}

// Metadata missing is not the same as metadata matching: the settings behind
// those answers are unknown, and reusing them would label them with this run's.
func TestCheckResumeRefusesRecordsWithoutMetadata(t *testing.T) {
	dir := t.TempDir()
	out := filepath.Join(dir, "out.jsonl")
	m := meta{Model: "m", Out: out}

	if err := checkResume(filepath.Join(dir, "out.meta.json"), m); err != nil {
		t.Errorf("no results file yet, want no error: %v", err)
	}
	if err := os.WriteFile(out, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := checkResume(filepath.Join(dir, "out.meta.json"), m); err != nil {
		t.Errorf("empty results file, want no error: %v", err)
	}

	writeRecords(t, out, eval.Result{ID: "question 1", Attempt: 1})
	if err := checkResume(filepath.Join(dir, "out.meta.json"), m); err == nil {
		t.Error("records with no metadata, want an error")
	}
}

// A complete record whose trailing newline never reached the disk parses fine,
// so it must be kept — but the next append would otherwise land on the same
// line and glue two objects together.
func TestPlanResumeTerminatesAnUnterminatedRecord(t *testing.T) {
	path := filepath.Join(t.TempDir(), "out.jsonl")
	if err := os.WriteFile(path, []byte(`{"id":"question 1","attempt":1}`), 0o644); err != nil {
		t.Fatal(err)
	}

	jobs, err := plan(planQuestions, 1, path, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(jobs) != 1 || jobs[0].Q.ID != "question 2" {
		t.Fatalf("jobs = %+v, want only question 2: the whole record must be kept", jobs)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.HasSuffix(data, []byte{'\n'}) {
		t.Errorf("file does not end in a newline, the next append would glue onto it: %q", data)
	}
}

// Two servers that both refuse to identify themselves compare equal as "none",
// which is not a match but an unanswered question.
func TestCheckResumeRefusesAnUnidentifiedServer(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "out.meta.json")
	anon := meta{Model: "m", Retrieval: "agentic", MCPURL: "http://localhost:8082/mcp/",
		MCPTools: []string{"search"}, Out: filepath.Join(dir, "out.jsonl")}
	if err := writeMeta(path, anon); err != nil {
		t.Fatal(err)
	}
	if err := checkResume(path, anon); err == nil {
		t.Error("neither side identified the server, want an error")
	}

	// The same run against a server that does identify itself is fine.
	named := anon
	named.MCPServer = map[string]string{"name": "3gpp-mcp", "version": "dev"}
	if err := writeMeta(path, named); err != nil {
		t.Fatal(err)
	}
	if err := checkResume(path, named); err != nil {
		t.Errorf("identified server, want no error: %v", err)
	}

	// A run with no tools has no server to identify.
	base := meta{Model: "m", Retrieval: "none", Out: filepath.Join(dir, "none.jsonl")}
	p2 := filepath.Join(dir, "none.meta.json")
	if err := writeMeta(p2, base); err != nil {
		t.Fatal(err)
	}
	if err := checkResume(p2, base); err != nil {
		t.Errorf("no MCP endpoint, want no error: %v", err)
	}
}
