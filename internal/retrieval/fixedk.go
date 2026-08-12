// Package retrieval holds the non-agentic retrieval baseline shared by the
// TeleQnA harness and the spec-grounded benchmark.
//
// The condition is one BM25 search over the same database the tool loop uses —
// 3gpp-mcp ranks FTS5 hits with a weighted bm25 — followed by the text of the
// top k hits, prepended to the user message. The model never writes a query and
// never sees a tool, so the difference against the agentic condition is the
// value of letting the model drive its own search rather than the value of the
// corpus.
//
// Note what this cannot reach: the OpenAPI definitions live in their own table
// with no full-text index, so no fixed-k run can retrieve them. That is a
// property of the corpus worth measuring, not a defect of this baseline.
package retrieval

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

// ToolCaller is the MCP side of the retrieval.
type ToolCaller interface {
	CallTool(name string, args json.RawMessage) (text string, isErr bool, err error)
}

// Call is one tool invocation, recorded so a trace shows the retrieval exactly
// as it happened.
type Call struct {
	Name   string
	Args   string
	Result string
	IsErr  bool
}

const ContextHeader = "Excerpts retrieved from the 3GPP specifications:\n\n"

var (
	tokenRe   = regexp.MustCompile(`[A-Za-z0-9][A-Za-z0-9._-]*`)
	releaseRe = regexp.MustCompile(`\s*\[3GPP Release \d+\]\s*`)
)

// stopwords are dropped from the query. The list is deliberately small: FTS5
// scores rare terms higher anyway, and an aggressive list would be a tuning
// knob that the agentic condition does not get.
var stopwords = map[string]bool{
	"the": true, "and": true, "for": true, "are": true, "what": true, "which": true,
	"does": true, "can": true, "with": true, "from": true, "that": true, "this": true,
	"how": true, "when": true, "why": true, "who": true, "was": true, "were": true,
	"has": true, "have": true, "will": true, "its": true, "not": true, "but": true,
	"you": true, "your": true, "there": true, "their": true, "used": true, "use": true,
	"following": true, "about": true, "into": true, "between": true, "during": true,
}

// maxTerms bounds the FTS5 query; long OR queries match almost everything and
// are slower without ranking better.
const maxTerms = 24

// Query turns a question into an FTS5 OR query. The TeleQnA release tag is
// stripped: it is dataset metadata, and "3GPP Release 18" matches nearly every
// document in the corpus.
func Query(text string) string {
	text = releaseRe.ReplaceAllString(text, " ")
	seen := map[string]bool{}
	var terms []string
	for _, tok := range tokenRe.FindAllString(text, -1) {
		low := strings.ToLower(tok)
		if len(low) < 3 || stopwords[low] || seen[low] {
			continue
		}
		seen[low] = true
		terms = append(terms, tok)
		if len(terms) == maxTerms {
			break
		}
	}
	if len(terms) == 0 {
		return strings.TrimSpace(text)
	}
	return strings.Join(terms, " OR ")
}

type hit struct {
	SpecID string `json:"spec_id"`
	Number string `json:"number"`
	Title  string `json:"title"`
}

// FixedK runs the search and the per-hit section reads, and returns the context
// block to prepend plus the calls it made. An empty block means the search
// found nothing to read.
func FixedK(mcp ToolCaller, question string, k, resultMax int) (string, []Call) {
	var calls []Call
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
		calls = append(calls, Call{Name: name, Args: string(raw), Result: text, IsErr: isErr})
		return text, !isErr
	}

	searchOut, ok := call("search", map[string]any{"query": Query(question), "limit": k})
	if !ok {
		return "", calls
	}
	var results struct {
		Results []hit `json:"results"`
	}
	if err := json.Unmarshal([]byte(searchOut), &results); err != nil || len(results.Results) == 0 {
		return "", calls
	}

	// Split the byte budget evenly across the hits so one long section cannot
	// crowd out the rest.
	per := resultMax / len(results.Results)
	var sb strings.Builder
	sb.WriteString(ContextHeader)
	for _, h := range results.Results {
		text, ok := call("get_section", map[string]any{
			"spec_id": h.SpecID, "section_number": h.Number,
		})
		if !ok {
			continue
		}
		fmt.Fprintf(&sb, "--- %s section %s: %s ---\n%s\n\n", h.SpecID, h.Number, h.Title, Truncate(text, per))
	}
	return sb.String(), calls
}

// Truncate cuts a tool result to the budget, telling the model that it did.
func Truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + fmt.Sprintf("\n...[truncated %d bytes; refine the query or use offset to read more]", len(s)-max)
}
