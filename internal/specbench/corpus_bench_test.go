package specbench

import (
	"os"
	"testing"
)

// These benchmarks run against a real specification database, because the cost
// that matters here is not in any Go function — it is in how the corpus is
// asked for a row. Grading once took 25 minutes because every lookup matched
// with UPPER(spec_id)=?, which puts a function on the indexed column and scans
// all 545,003 sections. A benchmark of the normalisers would have reported
// microseconds and found nothing; BenchmarkTitleOf reports the truth in one
// line.
//
//	SPECBENCH_DB=/path/to/3gpp-latest.db go test ./internal/specbench -bench . -benchtime 20x
//
// They skip when SPECBENCH_DB is unset, so `go test ./...` and CI are
// unaffected.

func benchCorpus(tb testing.TB) *Corpus {
	path := os.Getenv("SPECBENCH_DB")
	if path == "" {
		tb.Skip("set SPECBENCH_DB to a specification database to run this")
	}
	c, err := OpenCorpus(path)
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() { c.Close() })
	return c
}

const (
	benchSpec    = "TS 38.211"
	benchSection = "6.3.1"
)

// BenchmarkTitleOf is the one that would have caught the regression. A single
// row by primary key is microseconds; anything near a millisecond means the
// query is not on an index.
func BenchmarkTitleOf(b *testing.B) {
	c := benchCorpus(b)
	c.titleOf(benchSpec, benchSection) // warm the per-spec index
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if c.titleOf(benchSpec, benchSection) == "" {
			b.Fatal("no title")
		}
	}
}

// BenchmarkContentOf measures the one query still issued per record. Clause
// bodies are too large to hold, so this stays a round trip; it must stay an
// indexed one.
func BenchmarkContentOf(b *testing.B) {
	c := benchCorpus(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if c.contentOf(benchSpec, benchSection) == "" {
			b.Fatal("no content")
		}
	}
}

// BenchmarkSectionsIndex measures building one specification's index from
// cold, which is the only full read grading still does — once per
// specification, not once per record.
func BenchmarkSectionsIndex(b *testing.B) {
	c := benchCorpus(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		c.bySpec = map[string]*specIndex{}
		b.StartTimer()
		if len(c.sections(benchSpec).byNumber) == 0 {
			b.Fatal("empty index")
		}
	}
}

// BenchmarkAccessPath is here to keep the trap documented rather than to
// measure the shipped code. Both queries return the same row; the first is
// what the grader used to issue, and it reads the whole table to do it. If
// these two ever come out close, the index has stopped being used.
func BenchmarkAccessPath(b *testing.B) {
	c := benchCorpus(b)
	ids := c.rawIDs[benchSpec]
	if len(ids) == 0 {
		b.Skipf("%s is not in this database", benchSpec)
	}

	b.Run("upper-scan", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			var title string
			err := c.db.QueryRow(
				"SELECT title FROM sections WHERE UPPER(spec_id)=? AND number=?",
				benchSpec, benchSection).Scan(&title)
			if err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("indexed", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			var title string
			err := c.db.QueryRow(
				"SELECT title FROM sections WHERE spec_id=? AND number=?",
				ids[0], benchSection).Scan(&title)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

// BenchmarkCitation grades real records end to end, which is what a change to
// the corpus access pattern ultimately has to be judged on.
func BenchmarkCitation(b *testing.B) {
	path := os.Getenv("SPECBENCH_RECORDS")
	if path == "" {
		b.Skip("set SPECBENCH_RECORDS to a results JSONL to run this")
	}
	c := benchCorpus(b)
	recs, err := LoadRecords(path)
	if err != nil {
		b.Fatal(err)
	}
	if len(recs) == 0 {
		b.Fatal("no records")
	}
	var tasks []Task
	if dir := os.Getenv("SPECBENCH_TASKS"); dir != "" {
		if tasks, err = LoadTaskDir(dir); err != nil {
			b.Fatal(err)
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g := NewGrader(c, tasks)
		for _, r := range recs {
			g.Citation(r)
		}
	}
	b.ReportMetric(float64(len(recs)), "records/op")
}
