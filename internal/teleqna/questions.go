// Package teleqna loads and samples TeleQnA multiple-choice questions.
package teleqna

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

type Question struct {
	ID       string
	Text     string
	Options  map[int]string // option number -> text
	Answer   int            // expected option number
	Category string
}

var (
	optionKeyRe = regexp.MustCompile(`^option (\d+)$`)
	answerRe    = regexp.MustCompile(`^option (\d+)\s*:`)
)

// Load reads the TeleQnA JSON, keeping questions whose category starts with
// categoryPrefix and whose text contains textFilter (both optional).
func Load(path, categoryPrefix, textFilter string) ([]Question, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m map[string]map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, err
	}
	var qs []Question
	for id, fields := range m {
		cat, _ := fields["category"].(string)
		if categoryPrefix != "" && !strings.HasPrefix(cat, categoryPrefix) {
			continue
		}
		if qText, _ := fields["question"].(string); textFilter != "" && !strings.Contains(qText, textFilter) {
			continue
		}
		q := Question{ID: id, Category: cat, Options: map[int]string{}}
		q.Text, _ = fields["question"].(string)
		for k, v := range fields {
			if mm := optionKeyRe.FindStringSubmatch(k); mm != nil {
				n, _ := strconv.Atoi(mm[1])
				q.Options[n], _ = v.(string)
			}
		}
		ans, _ := fields["answer"].(string)
		mm := answerRe.FindStringSubmatch(ans)
		if mm == nil || q.Text == "" || len(q.Options) < 2 {
			continue
		}
		q.Answer, _ = strconv.Atoi(mm[1])
		qs = append(qs, q)
	}
	// Map iteration is random, so sort by the numeric suffix of "question N":
	// -seed sampling must pick the same questions across runs.
	sort.Slice(qs, func(i, j int) bool {
		ni, _ := strconv.Atoi(strings.TrimPrefix(qs[i].ID, "question "))
		nj, _ := strconv.Atoi(strings.TrimPrefix(qs[j].ID, "question "))
		return ni < nj
	})
	return qs, nil
}

// Select picks the questions to run: the comma-separated ids when given,
// otherwise the first n of a seeded shuffle.
func Select(qs []Question, ids string, n int, seed int64) []Question {
	if ids != "" {
		want := map[string]bool{}
		for _, id := range strings.Split(ids, ",") {
			want[strings.TrimSpace(id)] = true
		}
		var picked []Question
		for _, q := range qs {
			if want[q.ID] {
				picked = append(picked, q)
			}
		}
		return picked
	}
	rand.New(rand.NewSource(seed)).Shuffle(len(qs), func(i, j int) { qs[i], qs[j] = qs[j], qs[i] })
	if n < len(qs) {
		qs = qs[:n]
	}
	return qs
}

// Format renders the question and its options as the user message.
func Format(q Question) string {
	var sb strings.Builder
	sb.WriteString(q.Text)
	sb.WriteString("\n\n")
	nums := make([]int, 0, len(q.Options))
	for n := range q.Options {
		nums = append(nums, n)
	}
	sort.Ints(nums)
	for _, n := range nums {
		fmt.Fprintf(&sb, "option %d: %s\n", n, q.Options[n])
	}
	return sb.String()
}
