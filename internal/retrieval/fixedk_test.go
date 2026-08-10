package retrieval

import (
	"encoding/json"
	"strings"
	"testing"
)

type fakeTools struct {
	byName map[string]string
	calls  []string
}

func (f *fakeTools) CallTool(name string, args json.RawMessage) (string, bool, error) {
	f.calls = append(f.calls, name+" "+string(args))
	return f.byName[name], false, nil
}

func TestQuery(t *testing.T) {
	got := Query("What is the purpose of the AMF in TS 23.501? [3GPP Release 18]")
	if strings.Contains(got, "Release") || strings.Contains(got, "3GPP") {
		t.Errorf("release tag leaked into the query: %q", got)
	}
	if !strings.Contains(got, "AMF") || !strings.Contains(got, "23.501") || !strings.Contains(got, " OR ") {
		t.Errorf("query = %q", got)
	}
	if strings.Contains(got, "the OR") {
		t.Errorf("stopword kept: %q", got)
	}
}

func TestQueryFallsBackToTheText(t *testing.T) {
	if got := Query("a b c"); got != "a b c" {
		t.Errorf("got %q, want the original text when no term survives", got)
	}
}

func TestFixedK(t *testing.T) {
	tools := &fakeTools{byName: map[string]string{
		"search":      `{"results":[{"spec_id":"TS 23.501","number":"5.1","title":"Overview"}]}`,
		"get_section": "the section text",
	}}
	ctx, calls := FixedK(tools, "purpose of the AMF", 1, 1000)

	if !strings.HasPrefix(ctx, ContextHeader) || !strings.Contains(ctx, "the section text") {
		t.Errorf("context = %q", ctx)
	}
	if !strings.Contains(ctx, "TS 23.501 section 5.1: Overview") {
		t.Errorf("context does not name its source: %q", ctx)
	}
	if len(calls) != 2 || calls[0].Name != "search" || calls[1].Name != "get_section" {
		t.Errorf("calls = %+v", calls)
	}
}

// A search that finds nothing must add no context at all, rather than a header
// that would differ between the two conditions' prompts for no reason.
func TestFixedKNoHits(t *testing.T) {
	tools := &fakeTools{byName: map[string]string{"search": `{"results":[]}`}}
	ctx, calls := FixedK(tools, "nothing matches this", 5, 1000)
	if ctx != "" {
		t.Errorf("context = %q, want empty", ctx)
	}
	if len(calls) != 1 {
		t.Errorf("calls = %+v", calls)
	}
}

func TestTruncate(t *testing.T) {
	if got := Truncate("abcde", 5); got != "abcde" {
		t.Errorf("got %q", got)
	}
	got := Truncate("abcde", 3)
	if !strings.HasPrefix(got, "abc\n") || !strings.Contains(got, "truncated 2 bytes") {
		t.Errorf("got %q", got)
	}
}
