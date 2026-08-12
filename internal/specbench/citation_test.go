package specbench

import (
	"encoding/json"
	"reflect"
	"regexp"
	"testing"
)

func TestNormSec(t *testing.T) {
	for _, c := range []struct{ in, want string }{
		{"Clause 6.3.2.", "6.3.2"},
		{"section 5.4", "5.4"},
		{"Annex B.1", "b.1"},
		{"  6.1.6.2.23  ", "6.1.6.2.23"},
		{"PDCCH-ServingCellConfig", "pdcch-servingcellconfig"},
	} {
		if got := normSec(c.in); got != c.want {
			t.Errorf("normSec(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestLeadingNumber(t *testing.T) {
	// The converter leaves some headings unsplit, with the title still in the
	// number column. A citation names the number, so it has to resolve.
	if got := leadingNumber("7.2.160aA\tQuota-Indicator AVP"); got != "7.2.160aA" {
		t.Errorf("got %q", got)
	}
	if got := leadingNumber("5.4  Something Else"); got != "5.4" {
		t.Errorf("got %q", got)
	}
	if got := leadingNumber("6.3.2"); got != "6.3.2" {
		t.Errorf("got %q", got)
	}
}

func TestHTMLRows(t *testing.T) {
	// Cells hold <p> runs with no separator between them, which is how the
	// registry tables read back.
	rows := htmlRows(`<table><tbody><tr><td><p>45</p></td><td><p>Bearer Context</p></td></tr>` +
		`<tr><td><p>46</p></td><td><p>A &amp; B</p></td></tr></tbody></table>`)
	want := [][]string{{"45", "Bearer Context"}, {"46", "A & B"}}
	if !reflect.DeepEqual(rows, want) {
		t.Errorf("got %q, want %q", rows, want)
	}
}

func TestDeclaresSchema(t *testing.T) {
	doc := "components:\n  schemas:\n    UeContext:\n      type: object\n    Ncgi:\n      type: object\n"
	g := &Grader{corpus: newTestCorpus()}
	if !g.declares(doc, "schema", "Ncgi") {
		t.Error("Ncgi should be declared")
	}
	if g.declares(doc, "schema", "Missing") {
		t.Error("Missing should not be declared")
	}
	// A property of a schema is indented deeper and is not a declaration.
	if g.declares(doc, "schema", "type") {
		t.Error("a property is not a schema declaration")
	}
}

func TestDeclaresOperation(t *testing.T) {
	// Paths carrying a {placeholder} are quoted in these files, which an
	// earlier scorer missed — every such operation graded as undeclared.
	doc := "paths:\n  '/subscribers/{ueId}/update':\n    post:\n      summary: x\n" +
		"  /plain:\n    get:\n      summary: y\n"
	g := &Grader{corpus: newTestCorpus()}
	for _, c := range []struct {
		name string
		want bool
	}{
		{"POST /subscribers/{ueId}/update", true},
		{"GET /subscribers/{ueId}/update", false}, // wrong method on a real path
		{"GET /plain", true},
		{"POST /absent", false},
	} {
		if got := g.declares(doc, "operation", c.name); got != c.want {
			t.Errorf("declares(%q) = %v, want %v", c.name, got, c.want)
		}
	}
}

func TestCitationTargetsAcceptsBothEnds(t *testing.T) {
	// An object question is posed of one schema and answered by another. Both
	// documents are true citations; accepting only the first marked every
	// condition that followed the reference as wrong.
	task := Task{ID: "t1", Probe: &Probe{Kind: "object", Owner: "EventSubsc", Schema: "TargetUeInformation"}}
	g := NewGrader(newTestCorpus(), []Task{task})
	got := g.citationTargets(Record{ID: "t1"})
	want := []target{{"schema", "EventSubsc"}, {"schema", "TargetUeInformation"}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestCitationTargetsRequest(t *testing.T) {
	task := Task{ID: "t2", Probe: &Probe{Kind: "request", Method: "post", Path: "/ims-sessions", Schema: "ImsSession"}}
	g := NewGrader(newTestCorpus(), []Task{task})
	got := g.citationTargets(Record{ID: "t2"})
	want := []target{{"operation", "POST /ims-sessions"}, {"schema", "ImsSession"}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestCitationTargetsOneOfIncludesAlternatives(t *testing.T) {
	task := Task{ID: "t3", Probe: &Probe{Kind: "oneof", Schema: "AuthenticationVector"}}
	g := NewGrader(newTestCorpus(), []Task{task})
	gold, _ := json.Marshal([]string{"AvEapAkaPrime", "Av5GHeAka"})
	got := g.citationTargets(Record{ID: "t3", Gold: gold})
	want := []target{
		{"schema", "AuthenticationVector"},
		{"schema", "AvEapAkaPrime"},
		{"schema", "Av5GHeAka"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestCitationTargetsWithoutProbe(t *testing.T) {
	// Tasks generated before the probe existed fall back to the id, whose last
	// segment is the schema name.
	g := NewGrader(newTestCorpus(), nil)
	got := g.citationTargets(Record{ID: "openapi-Nudm_PP-PpMaximumResponseTime"})
	want := []target{{"schema", "PpMaximumResponseTime"}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestElementName(t *testing.T) {
	gold, _ := json.Marshal("Bearer Context")
	if got := elementName(Record{ID: "gtpv2c-name-93", Gold: gold}); got != "Bearer Context" {
		t.Errorf("got %q", got)
	}
	r := Record{ID: "gtpv2c-code-93", Question: "What is the type value of the 'Bearer Context' IE?"}
	if got := elementName(r); got != "Bearer Context" {
		t.Errorf("got %q", got)
	}
}

func TestVerdictOK(t *testing.T) {
	for v, want := range map[Verdict]bool{
		Exact: true, Ancestor: true, Contains: true, NotFound: false, Wrong: false,
	} {
		if v.OK() != want {
			t.Errorf("%s.OK() = %v", v, v.OK())
		}
	}
}

func TestTallyRow(t *testing.T) {
	var tally Tally
	tally.Add(true, Exact)
	tally.Add(false, Wrong)
	tally.Add(true, NotFound)
	tally.Add(true, Contains)
	got := tally.Row("run")
	want := "| run | 4 | 75.0% | 50.0% | 50.0% | 1 | 0 | 1 | 1 | 1 |"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

// newTestCorpus is a Corpus with no database behind it, for the parts of
// grading that read only the record and the task.
func newTestCorpus() *Corpus {
	return &Corpus{
		bySpec:  map[string]*specIndex{},
		rawIDs:  map[string][]string{},
		apiDocs: map[string][]string{},
		res:     map[string]*regexp.Regexp{},
	}
}
