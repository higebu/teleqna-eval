// Command grade scores benchmark citations against the specification
// database.
//
// Grading is a pass of its own, and it never writes over its input. The run
// records what the model answered; this records what that was worth. Keeping
// them apart is what makes a results file re-gradable: the earlier scorer
// rewrote the files in place, so improving it silently changed the numbers a
// report had already quoted, with nothing left to compare against.
//
//	go run ./bench/grade -db 3gpp-latest.db -tasks-dir bench results/*.jsonl
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"

	"3gpp-mcp-bench/internal/specbench"
)

func main() {
	var (
		db       = flag.String("db", "", "pinned specification database (required)")
		tasksDir = flag.String("tasks-dir", "bench", "directory of tasks-*.json, read for the probe each gold was built from")
		outDir   = flag.String("out-dir", "", "write graded copies here (default: alongside the input)")
		suffix   = flag.String("suffix", ".graded.jsonl", "replaces .jsonl on each output name")
	)
	flag.Parse()
	if *db == "" || flag.NArg() == 0 {
		flag.Usage()
		os.Exit(2)
	}

	corpus, err := specbench.OpenCorpus(*db)
	if err != nil {
		log.Fatal(err)
	}
	defer corpus.Close()

	tasks, err := specbench.LoadTaskDir(*tasksDir)
	if err != nil {
		log.Fatal(err)
	}
	grader := specbench.NewGrader(corpus, tasks)
	version := graderVersion()

	fmt.Println("| run | n | answer | citation | answer+citation | exact | ancestor | contains | not found | wrong |")
	fmt.Println("|---|---|---|---|---|---|---|---|---|---|")
	for _, path := range flag.Args() {
		recs, err := specbench.LoadRecords(path)
		if err != nil {
			log.Fatal(err)
		}
		if len(recs) == 0 {
			continue
		}
		var tally specbench.Tally
		for i := range recs {
			v := grader.Citation(recs[i])
			recs[i].Score.Verdict = v
			recs[i].Score.Citation = v.OK()
			recs[i].Score.Both = recs[i].Score.Answer && v.OK()
			if recs[i].Meta == nil {
				recs[i].Meta = map[string]string{}
			}
			recs[i].Meta["grader"] = version
			tally.Add(recs[i].Score.Answer, v)
		}
		out := outPath(path, *outDir, *suffix)
		if out == path {
			log.Fatalf("%s: refusing to overwrite the input; set -out-dir or -suffix", path)
		}
		f, err := os.Create(out)
		if err != nil {
			log.Fatal(err)
		}
		if err := specbench.WriteRecords(f, recs); err != nil {
			log.Fatal(err)
		}
		if err := f.Close(); err != nil {
			log.Fatal(err)
		}
		fmt.Println(tally.Row(runName(path)))
	}
}

func outPath(path, dir, suffix string) string {
	name := strings.TrimSuffix(filepath.Base(path), ".jsonl") + suffix
	if dir == "" {
		dir = filepath.Dir(path)
	}
	return filepath.Join(dir, name)
}

func runName(path string) string {
	return strings.TrimSuffix(strings.TrimPrefix(filepath.Base(path), "bench-"), ".jsonl")
}

// graderVersion stamps each record with the revision that graded it, so a
// number in a report can be traced to the code that produced it.
func graderVersion() string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return "unknown"
	}
	rev, dirty := "", ""
	for _, s := range info.Settings {
		switch s.Key {
		case "vcs.revision":
			rev = s.Value
		case "vcs.modified":
			if s.Value == "true" {
				dirty = "-dirty"
			}
		}
	}
	if rev == "" {
		return "unknown"
	}
	if len(rev) > 12 {
		rev = rev[:12]
	}
	return rev + dirty
}
