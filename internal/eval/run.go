package eval

import (
	"encoding/json"
	"io"
	"log"
	"sync"

	"teleqna-eval/internal/llm"
	"teleqna-eval/internal/teleqna"
)

type Summary struct {
	Questions  int
	Answered   int
	Correct    int
	ToolCalls  int
	Prompt     int
	Completion int
	CacheRead  int
	CacheWrite int
	Errors     int
	LooseParse int // answers that needed a fallback below the prompt's own format
}

// Job is one question to evaluate. RepeatIdx separates repeated measurements of
// the same question; Attempt counts how many times a resumed run has had to
// re-execute it, so a spliced file says so in its own records.
type Job struct {
	Q         teleqna.Question
	RepeatIdx int
	Attempt   int
}

// Run evaluates every job with opts.Workers goroutines, writing one JSONL
// record per job to out as each finishes. When trace is non-nil the full
// message and tool-result history is written there.
func Run(be llm.Backend, mcp ToolCaller, jobs []Job, opts Options, out, trace io.Writer) Summary {
	enc := json.NewEncoder(out)
	var traceEnc *json.Encoder
	if trace != nil {
		traceEnc = json.NewEncoder(trace)
	}
	s := Summary{Questions: len(jobs)}
	var (
		mu   sync.Mutex
		done int
		wg   sync.WaitGroup
	)
	ch := make(chan Job)
	workers := max(opts.Workers, 1) // 0 workers would block on the first send
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range ch {
				r, tr := One(be, mcp, j.Q, opts)
				r.RepeatIdx, r.Attempt = j.RepeatIdx, j.Attempt
				tr.RepeatIdx, tr.Attempt = j.RepeatIdx, j.Attempt
				mu.Lock()
				_ = enc.Encode(r)
				if traceEnc != nil {
					_ = traceEnc.Encode(tr)
				}
				status := "WRONG"
				if r.Correct {
					s.Correct++
					status = "ok"
				}
				if r.Predicted != 0 {
					s.Answered++
				}
				if r.Error != "" {
					status = "ERROR " + r.Error
					s.Errors++
				}
				if isLooseTier(r.ParseTier) {
					s.LooseParse++
				}
				s.Prompt += r.PromptTok
				s.Completion += r.CompleteTok
				s.CacheRead += r.CacheReadTok
				s.CacheWrite += r.CacheWriteTok
				s.ToolCalls += len(r.ToolCalls)
				done++
				log.Printf("[%d/%d] %s: pred=%d exp=%d %s (%d tool calls, %s, %.0fs)",
					done, len(jobs), j.Q.ID, r.Predicted, j.Q.Answer, status,
					len(r.ToolCalls), r.ParseTier, r.DurationSec)
				mu.Unlock()
			}
		}()
	}
	for _, j := range jobs {
		ch <- j
	}
	close(ch)
	wg.Wait()
	return s
}

// isLooseTier reports whether an answer came from a fallback rather than from
// the format the prompt asked for. These are the records a sensitivity analysis
// re-scores as failures.
func isLooseTier(tier string) bool {
	switch tier {
	case "", "json", "answer_line":
		return false
	}
	return true
}
