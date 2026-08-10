// Package prompt holds the prompt variants the harness can run. A variant owns
// the system message, the rendering of a question into the user message and the
// parsing of the model's reply, so that a run is fully described by its id.
//
// The same variant is used for both conditions of a pair: the only difference
// between "with tools" and "no tools" is whether tool definitions are attached
// to the request. Nothing in this package may branch on tool availability.
package prompt

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"teleqna-eval/internal/teleqna"
)

// Parsed is one reply turned into an option number. Tier records which
// extraction step produced it, so a run can be re-scored later with the looser
// tiers counted as failures.
type Parsed struct {
	Option int
	Raw    string // the answer as the model wrote it, before the number was extracted
	Tier   string
}

// Tiers, ordered from the format the prompt asked for to the last resort.
const (
	TierJSON        = "json"         // a JSON object with an "answer" field
	TierAnswerField = "answer_field" // "answer": "option N" recovered by regex
	TierAnswerLine  = "answer_line"  // an ANSWER: line at the start of a line
	TierAnswerAny   = "answer_any"   // ANSWER: anywhere in the text
	TierOptionScan  = "option_scan"  // the last "option N" mentioned anywhere
	TierNone        = "none"
)

type Prompt struct {
	ID     string
	System string
	// Format renders the user message. Retry renders the message sent when a
	// reply could not be parsed.
	Format func(teleqna.Question) string
	Retry  string
	Parse  func(text string) Parsed
}

// SHA256 identifies the exact prompt text a run used, so a result file can be
// tied to its wording even after the source changes.
func (p *Prompt) SHA256() string {
	sum := sha256.Sum256([]byte(p.ID + "\x00" + p.System + "\x00" + p.Retry))
	return hex.EncodeToString(sum[:])
}

// teleqnaSystem is the system prompt of netop-team/TeleQnA's
// evaluation_tools.py, reproduced byte for byte: the leading and trailing
// newlines and the trailing spaces that end the first two lines are all
// upstream's, and are kept so the wording is not silently ours.
const teleqnaSystem = "\n" +
	"Please provide the answers to the following telecommunications related \n" +
	"multiple choice questions. The questions will be in a JSON format, the \n" +
	"answers must also be in a JSON format as follows:\n" +
	" {\n" +
	"\"question 1\": {\n" +
	"\"question\": question,\n" +
	"\"answer\": \"option {answer id}: {answer string}\"\n" +
	"},\n" +
	"...\n" +
	"}\n"

// cotSuffix is the only text separating the "cot" variant from "teleqna". It is
// appended to the shared system prompt for both conditions alike, so the pair
// stays symmetric while the reasoning budget changes.
const cotSuffix = "\nBefore you produce the JSON, work through the question step by step and " +
	"explain your reasoning. The JSON object must be the last thing in your reply.\n"

const ansLineSystem = `You are a telecommunications standards expert answering multiple-choice questions about 3GPP specifications.

Reply with your reasoning followed by a final line in exactly this format:
ANSWER: <option number>

The final line must contain only one option number.`

var registry = map[string]*Prompt{}

func register(p *Prompt) { registry[p.ID] = p }

func init() {
	register(&Prompt{
		ID:     "teleqna",
		System: teleqnaSystem,
		Format: formatTeleQnA,
		Retry: "Your reply did not contain the JSON object. Reply now with the JSON object only, " +
			"in the format given above.",
		Parse: parseTeleQnA,
	})
	register(&Prompt{
		ID:     "cot",
		System: teleqnaSystem + cotSuffix,
		Format: formatTeleQnA,
		Retry: "Your reply did not contain the JSON object. Reply now with the JSON object only, " +
			"in the format given above.",
		Parse: parseTeleQnA,
	})
	register(&Prompt{
		ID:     "ansline",
		System: ansLineSystem,
		Format: teleqna.Format,
		Retry:  "Your reply did not contain a readable answer. Reply now with one line only: ANSWER: <option number>",
		Parse:  parseAnswerLine,
	})
}

// Get returns the variant with the given id.
func Get(id string) (*Prompt, error) {
	p, ok := registry[id]
	if !ok {
		return nil, fmt.Errorf("unknown prompt %q (have %s)", id, strings.Join(IDs(), ", "))
	}
	return p, nil
}

// IDs lists the registered variants in a stable order.
func IDs() []string {
	ids := make([]string, 0, len(registry))
	for id := range registry {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

// formatTeleQnA renders the user message the way evaluation_tools.py does:
// the fixed lead-in followed by json.dumps of the question with its answer,
// explanation and category removed. The object is assembled by hand rather than
// with json.Marshal so the key order and the ", "/": " separators match Python's
// json.dumps, and so the output does not depend on Go's map iteration order.
func formatTeleQnA(q teleqna.Question) string {
	nums := make([]int, 0, len(q.Options))
	for n := range q.Options {
		nums = append(nums, n)
	}
	sort.Ints(nums)

	var inner strings.Builder
	inner.WriteString(`{"question": `)
	inner.Write(jsonString(q.Text))
	for _, n := range nums {
		fmt.Fprintf(&inner, `, "option %d": `, n)
		inner.Write(jsonString(q.Options[n]))
	}
	inner.WriteString("}")

	var sb strings.Builder
	sb.WriteString("Here are the questions: \n {")
	sb.Write(jsonString(q.ID))
	sb.WriteString(": ")
	sb.WriteString(inner.String())
	sb.WriteString("}")
	return sb.String()
}

func jsonString(s string) []byte {
	b, err := json.Marshal(s)
	if err != nil { // a Go string always marshals
		return []byte(`""`)
	}
	return b
}

var (
	answerFieldRe = regexp.MustCompile(`(?is)"answer"\s*:\s*"\s*(?:option\s*)?(\d+)`)
	answerLineRe  = regexp.MustCompile(`(?mi)^[\s>*#]*ANSWER\s*[:：]\s*\**\s*(?:option\s*)?(\d+)`)
	answerAnyRe   = regexp.MustCompile(`(?i)ANSWER\s*[:：]\s*\**\s*(?:option\s*)?(\d+)`)
	optionAnyRe   = regexp.MustCompile(`(?i)\boption\s*(\d+)\b`)
	optionNumRe   = regexp.MustCompile(`(?i)option\s*(\d+)`)
)

// parseTeleQnA reads the JSON object the prompt asked for, then falls back to
// progressively looser scans. Every fallback is recorded in Tier so a later
// re-scoring can treat it as a failure.
func parseTeleQnA(text string) Parsed {
	if raw, ok := answerFromJSON(text); ok {
		if mm := optionNumRe.FindStringSubmatch(raw); mm != nil {
			n, _ := strconv.Atoi(mm[1])
			return Parsed{Option: n, Raw: raw, Tier: TierJSON}
		}
		// A JSON answer that names no option number is still the model's answer;
		// keep it as the raw string so it can be inspected.
		return Parsed{Raw: raw, Tier: TierNone}
	}
	if mm := answerFieldRe.FindStringSubmatch(text); mm != nil {
		n, _ := strconv.Atoi(mm[1])
		return Parsed{Option: n, Raw: mm[0], Tier: TierAnswerField}
	}
	return scanOption(text)
}

func parseAnswerLine(text string) Parsed {
	if mm := answerLineRe.FindStringSubmatch(text); mm != nil {
		n, _ := strconv.Atoi(mm[1])
		return Parsed{Option: n, Raw: strings.TrimSpace(mm[0]), Tier: TierAnswerLine}
	}
	if mm := answerAnyRe.FindStringSubmatch(text); mm != nil {
		n, _ := strconv.Atoi(mm[1])
		return Parsed{Option: n, Raw: strings.TrimSpace(mm[0]), Tier: TierAnswerAny}
	}
	return scanOption(text)
}

func scanOption(text string) Parsed {
	if all := optionAnyRe.FindAllStringSubmatch(text, -1); len(all) > 0 {
		last := all[len(all)-1]
		n, _ := strconv.Atoi(last[1])
		return Parsed{Option: n, Raw: strings.TrimSpace(last[0]), Tier: TierOptionScan}
	}
	return Parsed{Tier: TierNone}
}

// answerFromJSON returns the "answer" field of the last JSON object in text.
// The reply may be wrapped in a code fence or preceded by reasoning, and the
// object may be either the answer object itself or the {"question N": {...}}
// wrapper the prompt asks for; scanning backwards finds the inner object first.
func answerFromJSON(text string) (string, bool) {
	for i := len(text) - 1; i >= 0; i-- {
		if text[i] != '{' {
			continue
		}
		var m map[string]json.RawMessage
		if err := json.NewDecoder(strings.NewReader(text[i:])).Decode(&m); err != nil {
			continue
		}
		if raw, ok := answerOf(m); ok {
			return raw, true
		}
	}
	return "", false
}

func answerOf(m map[string]json.RawMessage) (string, bool) {
	if v, ok := m["answer"]; ok {
		var s string
		if err := json.Unmarshal(v, &s); err == nil && s != "" {
			return s, true
		}
		return "", false
	}
	// The wrapper form: one entry per question, each holding an answer object.
	for _, v := range m {
		var inner map[string]json.RawMessage
		if err := json.Unmarshal(v, &inner); err != nil {
			continue
		}
		if raw, ok := answerOf(inner); ok {
			return raw, true
		}
	}
	return "", false
}
