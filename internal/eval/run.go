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
}

// Run evaluates every question with opts.Workers goroutines, writing one JSONL
// record per question to out as each finishes.
func Run(be llm.Backend, mcp ToolCaller, qs []teleqna.Question, opts Options, out io.Writer) Summary {
	enc := json.NewEncoder(out)
	s := Summary{Questions: len(qs)}
	var (
		mu   sync.Mutex
		done int
		wg   sync.WaitGroup
	)
	jobs := make(chan teleqna.Question)
	for w := 0; w < opts.Workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for q := range jobs {
				r := One(be, mcp, q, opts)
				mu.Lock()
				_ = enc.Encode(r)
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
				}
				s.Prompt += r.PromptTok
				s.Completion += r.CompleteTok
				s.CacheRead += r.CacheReadTok
				s.CacheWrite += r.CacheWriteTok
				s.ToolCalls += len(r.ToolCalls)
				done++
				log.Printf("[%d/%d] %s: pred=%d exp=%d %s (%d tool calls, %.0fs)",
					done, len(qs), q.ID, r.Predicted, q.Answer, status, len(r.ToolCalls), r.DurationSec)
				mu.Unlock()
			}
		}()
	}
	for _, q := range qs {
		jobs <- q
	}
	close(jobs)
	wg.Wait()
	return s
}
