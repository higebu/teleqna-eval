package eval

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"teleqna-eval/internal/teleqna"
)

// The fixed-k condition is the non-agentic retrieval baseline: one BM25 search
// over the same database the tools condition uses (3gpp-mcp ranks FTS5 hits
// with a weighted bm25), then the text of the top k hits, prepended to the
// user message. The model never chooses a query and never sees a tool, so the
// difference against the agentic condition is the value of letting the model
// drive its own search rather than the value of the corpus.

const fixedKContextHeader = "Excerpts retrieved from the 3GPP specifications:\n\n"

var (
	fixedKTokenRe   = regexp.MustCompile(`[A-Za-z0-9][A-Za-z0-9._-]*`)
	fixedKReleaseRe = regexp.MustCompile(`\s*\[3GPP Release \d+\]\s*$`)
)

// fixedKStopwords are dropped from the query. The list is deliberately small:
// FTS5 scores rare terms higher anyway, and an aggressive list would be a
// tuning knob that the agentic condition does not get.
var fixedKStopwords = map[string]bool{
	"the": true, "and": true, "for": true, "are": true, "what": true, "which": true,
	"does": true, "can": true, "with": true, "from": true, "that": true, "this": true,
	"how": true, "when": true, "why": true, "who": true, "was": true, "were": true,
	"has": true, "have": true, "will": true, "its": true, "not": true, "but": true,
	"you": true, "your": true, "there": true, "their": true, "used": true, "use": true,
	"following": true, "about": true, "into": true, "between": true, "during": true,
}

// fixedKMaxTerms bounds the FTS5 query; long OR queries match almost everything
// and are slower without ranking better.
const fixedKMaxTerms = 24

// fixedKQuery turns a question into an FTS5 OR query. The release tag is
// stripped: it is dataset metadata, and "3GPP Release 18" matches nearly every
// document in the corpus.
func fixedKQuery(q teleqna.Question) string {
	text := fixedKReleaseRe.ReplaceAllString(q.Text, "")
	seen := map[string]bool{}
	var terms []string
	for _, tok := range fixedKTokenRe.FindAllString(text, -1) {
		low := strings.ToLower(tok)
		if len(low) < 3 || fixedKStopwords[low] || seen[low] {
			continue
		}
		seen[low] = true
		terms = append(terms, tok)
		if len(terms) == fixedKMaxTerms {
			break
		}
	}
	if len(terms) == 0 {
		return text
	}
	return strings.Join(terms, " OR ")
}

type fixedKHit struct {
	SpecID string `json:"spec_id"`
	Number string `json:"number"`
	Title  string `json:"title"`
}

// retrieveFixedK runs the search and the per-hit section reads, and returns the
// context block to prepend plus the tool calls it made, so the trace shows the
// retrieval exactly as it happened.
func retrieveFixedK(mcp ToolCaller, q teleqna.Question, k, resultMax int) (string, []TraceToolCall) {
	var calls []TraceToolCall
	call := func(name string, args any) (string, bool) {
		raw, err := json.Marshal(args)
		if err != nil {
			return "", false
		}
		text, isErr, err := mcp.CallTool(name, raw)
		if err != nil {
			text, isErr = "tool error: "+err.Error(), true
		} else if isErr {
			text = "tool error: " + text
		}
		calls = append(calls, TraceToolCall{Name: name, Args: string(raw), Result: text, IsErr: isErr})
		return text, !isErr
	}

	searchOut, ok := call("search", map[string]any{"query": fixedKQuery(q), "limit": k})
	if !ok {
		return "", calls
	}
	var results struct {
		Results []fixedKHit `json:"results"`
	}
	if err := json.Unmarshal([]byte(searchOut), &results); err != nil {
		return "", calls
	}

	// Split the byte budget evenly across the hits so one long section cannot
	// crowd out the rest.
	per := resultMax
	if n := len(results.Results); n > 0 {
		per = resultMax / n
	}
	var sb strings.Builder
	sb.WriteString(fixedKContextHeader)
	for _, hit := range results.Results {
		text, ok := call("get_section", map[string]any{
			"spec_id": hit.SpecID, "section_number": hit.Number,
		})
		if !ok {
			continue
		}
		fmt.Fprintf(&sb, "--- %s section %s: %s ---\n%s\n\n", hit.SpecID, hit.Number, hit.Title, truncate(text, per))
	}
	if len(results.Results) == 0 {
		return "", calls
	}
	return sb.String(), calls
}
