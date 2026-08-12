package specbench

import (
	"encoding/json"
	"fmt"
	"html"
	"regexp"
	"strings"
)

// Verdict is how a citation relates to the clause the task came from.
//
// Comparing the cited clause to the gold clause as strings is too strict to
// mean anything. The corpus stores each 38.331 information element as a
// section whose number *is* the element's name, so a model that answers
// "TS 38.331 clause 6.3.2" — the clause those elements live under, and what an
// engineer would write — scored zero against a gold of
// "PDCCH-ServingCellConfig". And the RF test specifications copy clauses
// between each other, so more than one document really does define the same
// equation. A citation is therefore graded against the documents.
type Verdict string

const (
	// Exact is the section the task was generated from.
	Exact Verdict = "exact"
	// Ancestor contains that section, via parent_number — a coarser citation,
	// still true.
	Ancestor Verdict = "ancestor"
	// Contains is a different section that genuinely holds the same answer.
	Contains Verdict = "contains"
	// NotFound is a clause that does not exist: a fabricated reference, which
	// is a different failure from citing the wrong one and is counted apart.
	NotFound Verdict = "not_found"
	// Wrong is a real clause that does not hold the answer.
	Wrong Verdict = "wrong"
)

// OK reports whether the citation can be followed to the answer.
func (v Verdict) OK() bool { return v == Exact || v == Ancestor || v == Contains }

// Grader scores citations against the corpus. It holds the tasks because an
// OpenAPI citation is graded against the probe a task was generated from, not
// against anything that can be parsed out of the record.
type Grader struct {
	corpus *Corpus
	tasks  map[string]Task
	allof  map[string][]string
}

func NewGrader(corpus *Corpus, tasks []Task) *Grader {
	byID := make(map[string]Task, len(tasks))
	for _, t := range tasks {
		byID[t.ID] = t
	}
	return &Grader{corpus: corpus, tasks: byID, allof: map[string][]string{}}
}

// Citation grades one record. Only Wrong and NotFound are failures.
func (g *Grader) Citation(r Record) Verdict {
	if r.Type == "openapi" {
		return g.openapi(r)
	}
	spec := normSpec(r.PredSpec)
	sec := normSec(r.PredSec)
	goldSpec := normSpec(r.GoldSpec)
	if spec != goldSpec && sec == "" {
		return Wrong
	}
	index := g.corpus.sections(spec)
	if len(index.byKey) == 0 {
		return NotFound
	}
	hit, ok := index.byKey[sec]
	if !ok {
		return NotFound
	}

	if spec == goldSpec && normSec(hit.Number) == normSec(r.GoldSec) {
		return Exact
	}
	if spec == goldSpec {
		for _, a := range g.corpus.ancestors(goldSpec, r.GoldSec) {
			if a == hit.Number {
				return Ancestor
			}
		}
	}
	if holdsAnswer(g.corpus.titleOf(spec, hit.Number), g.corpus.contentOf(spec, hit.Number), r) {
		return Contains
	}
	return Wrong
}

// openapi grades a citation of an OpenAPI element.
//
// There is no clause number in an OpenAPI file, so the citation these tasks
// ask for is the specification and the API document. It counts when that
// document really declares the schema or operation the answer comes from —
// the same rule as everywhere else here: the citation has to hold the answer,
// not merely match a string.
func (g *Grader) openapi(r Record) Verdict {
	spec := normSpec(r.PredSpec)
	api := strings.TrimSpace(r.PredSec)
	if api == "" {
		return Wrong
	}
	targets := g.citationTargets(r)

	if content, ok := g.corpus.openapiDoc(spec, api); ok {
		for _, t := range targets {
			if g.declares(content, t.kind, t.name) {
				if api == r.GoldSec {
					return Exact
				}
				return Contains
			}
		}
		return Wrong
	}

	// The element named instead of the document that declares it.
	for _, t := range targets {
		if api != t.name {
			continue
		}
		for _, content := range g.corpus.openapiDocs(spec) {
			if g.declares(content, "schema", api) {
				return Contains
			}
		}
		break
	}

	// Not an API name. In the SBI specifications the normative definition is a
	// clause of the running text — "6.1.6.2.23  Type: UeContextTransferReqData",
	// "7.3.5  ThresholdCrossing <<dataType>>" — and the OpenAPI file in the
	// annex is generated from it. That clause is the citation an implementer
	// wants, so accept a section this specification titles after the schema.
	hit, ok := g.corpus.sections(spec).byKey[normSec(api)]
	if !ok {
		return NotFound
	}
	title := g.corpus.titleOf(spec, hit.Number)
	for _, t := range targets {
		if t.name == "" {
			continue
		}
		if g.corpus.re(`\b` + regexp.QuoteMeta(t.name) + `\b`).MatchString(title) {
			return Contains
		}
	}
	return Wrong
}

type target struct{ kind, name string }

// citationTargets lists every element whose declaration would justify this
// answer.
//
// An OpenAPI question points at one schema and its answer is defined in
// another: "the property `tgtUe` of `EventSubsc` holds an object, list its
// fields" is asked of TS 29.530, and answered by the definition of
// `TargetUeInformation`, which lives in TS 29.571 CommonData. Both documents
// are true citations — one is where the question is posed, the other is where
// the answer is written — and a scorer that accepts only the first marks the
// condition that actually followed the reference as wrong.
//
// Read from the task's own probe, never parsed out of the id: an id holds an
// API name that may itself contain hyphens, and deriving a verdict from a
// string split is how this scorer was wrong before.
func (g *Grader) citationTargets(r Record) []target {
	p := g.tasks[r.ID].Probe
	if p == nil {
		// A task from before the probe existed: its id ends in the schema name.
		parts := strings.Split(r.ID, "-")
		return withName([]target{{"schema", parts[len(parts)-1]}})
	}
	switch p.Kind {
	case "request":
		// The operation, and the schema its body carries.
		return withName([]target{
			{"operation", strings.ToUpper(p.Method) + " " + p.Path},
			{"schema", p.Schema},
		})
	case "object":
		return withName([]target{{"schema", p.Owner}, {"schema", p.Schema}})
	case "oneof":
		// The schema that declares the choice, and the alternatives, which are
		// the answer.
		out := []target{{"schema", p.Schema}}
		for _, alt := range r.GoldList() {
			out = append(out, target{"schema", alt})
		}
		return withName(out)
	case "allof":
		out := []target{{"schema", p.Schema}}
		for _, m := range g.allofMembers(r) {
			out = append(out, target{"schema", m})
		}
		return withName(out)
	}
	return withName([]target{{"schema", p.Schema}})
}

func withName(in []target) []target {
	out := in[:0]
	for _, t := range in {
		if t.name != "" {
			out = append(out, t)
		}
	}
	return out
}

var allofRefRe = regexp.MustCompile(`\$ref:\s*'?"?[^'"\s#]*#/components/schemas/([\w.-]+)`)

// allofMembers reads the schemas an allOf composes out of the gold document.
func (g *Grader) allofMembers(r Record) []string {
	if m, ok := g.allof[r.ID]; ok {
		return m
	}
	var members []string
	task := g.tasks[r.ID]
	if task.Probe != nil && task.Probe.Schema != "" {
		if content, ok := g.corpus.openapiDocExact(task.SpecID, task.APIName); ok {
			body := g.corpus.re(`(?m)^    ` + regexp.QuoteMeta(task.Probe.Schema) +
				`:\n((?:      .*\n|\n)*)`).FindStringSubmatch(content)
			if body != nil {
				for _, ref := range allofRefRe.FindAllStringSubmatch(body[1], -1) {
					members = append(members, ref[1])
				}
			}
		}
	}
	g.allof[r.ID] = members
	return members
}

// declares reports whether an OpenAPI document defines that schema, or that
// operation.
func (g *Grader) declares(content, kind, name string) bool {
	if name == "" {
		return false
	}
	if kind != "operation" {
		return g.corpus.re(`(?m)^    ` + regexp.QuoteMeta(name) + `:`).MatchString(content)
	}
	method, path, _ := strings.Cut(name, " ")
	// Paths that carry a {placeholder} are usually quoted in these files.
	loc := g.corpus.re(`(?m)^  ['"]?` + regexp.QuoteMeta(path) + `['"]?:`).FindStringIndex(content)
	if loc == nil {
		return false
	}
	rest := content[loc[1]:]
	if end := nextTopKeyRe.FindStringIndex(rest); end != nil {
		rest = rest[:end[0]]
	}
	return g.corpus.re(`(?m)^    ` + strings.ToLower(method) + `:`).MatchString(rest)
}

var nextTopKeyRe = regexp.MustCompile(`(?m)^  \S`)

// holdsAnswer reports whether a section actually documents what the task asks
// about, which is what makes a citation other than the gold one acceptable.
func holdsAnswer(title, content string, r Record) bool {
	switch r.Type {
	case "asn1":
		name := strings.TrimPrefix(r.ID, "asn1-")
		if !strings.Contains(content, name+" ::=") {
			return false
		}
		for _, g := range r.GoldList() {
			if !strings.Contains(content, g) {
				return false
			}
		}
		return true
	case "code":
		parts := strings.Split(r.ID, "-")
		code := parts[len(parts)-1]
		gold := r.GoldString()
		name := elementName(r)
		// A registry row: the code and the element in one row, in any column —
		// these tables carry merged cells, so the code is not always first.
		for _, row := range htmlRows(content) {
			hasCode, hasName := false, false
			for _, cell := range row {
				cell = strings.TrimSpace(cell)
				if cell == code {
					hasCode = true
				}
				if normName(cell) == normName(gold) || (name != "" && normName(cell) == normName(name)) {
					hasName = true
				}
			}
			if hasCode && hasName {
				return true
			}
		}
		// Or the clause that defines the element. 3GPP writes the code once,
		// in the registry, and the defining clause carries only the name and
		// the semantics — so a clause titled after the element is the citation
		// an implementer wants, whether or not it repeats the number.
		return name != "" && strings.Contains(normName(title), normName(name))
	case "subtreerefs":
		// The answer is a property of a whole clause tree, so no other clause
		// states it: not a subclause, which holds a part of it, and not a
		// sibling. The two citations that do hold it — the tree's root, and any
		// clause containing the root — are decided as exact and ancestor before
		// this is reached. Without this case the default below compares against
		// GoldString(), which is empty for a list-shaped gold, so every clause
		// of the right specification would count as holding the answer.
		return false
	case "ngapies", "ngapasn1":
		// The walk these tasks describe passes through clauses that do not hold
		// the answer — a message clause names an IE and points at another
		// clause for its ASN.1 — so naming one of them is not a citation. The
		// rule is the same as everywhere else here: the clause has to state the
		// value. Which of the two notations a clause writes it in is exactly
		// what separates the IE clause from the ASN.1 one, and the question
		// asked for one of them.
		flat := flatten(content)
		for _, g := range goldValues(r) {
			if !strings.Contains(flat, flatten(g)) {
				return false
			}
		}
		return true
	}
	return strings.Contains(normLatex(content), normLatex(r.GoldString()))
}

var quotedRe = regexp.MustCompile(`'([^']+)'`)

// flatten reduces a clause, or one gold value, to the text they can be compared
// on: the HTML the DOCX conversion leaves around every table cell removed, the
// entities it writes resolved, and the non-breaking spaces those entities
// become treated as the spaces they are printed as.
func flatten(s string) string {
	s = html.UnescapeString(tagRe.ReplaceAllString(s, " "))
	s = strings.Map(func(r rune) rune {
		switch r {
		case '\u00a0', '\u202f': // the non-breaking spaces &nbsp; resolves to
			return ' '
		case '\u2011': // and the non-breaking hyphen, which prints as one
			return '-'
		}
		return r
	}, s)
	return strings.ToLower(spaceRe.ReplaceAllString(strings.TrimSpace(s), " "))
}

// goldValues reads a gold of either shape as the list of values a citing clause
// has to state.
func goldValues(r Record) []string {
	if list := r.GoldList(); len(list) > 0 {
		return list
	}
	if s := r.GoldString(); s != "" {
		return []string{s}
	}
	return nil
}

// elementName is the protocol element a code task is about, however the task
// asks for it.
func elementName(r Record) string {
	if strings.Contains(r.ID, "-name-") {
		return r.GoldString()
	}
	if m := quotedRe.FindStringSubmatch(r.Question); m != nil {
		return m[1]
	}
	return ""
}

// GoldList reads a list-shaped gold off a record.
func (r Record) GoldList() []string {
	var f []string
	_ = json.Unmarshal(r.Gold, &f)
	return f
}

// GoldString reads a scalar or equation gold off a record.
func (r Record) GoldString() string {
	var s string
	if err := json.Unmarshal(r.Gold, &s); err == nil {
		return s
	}
	return string(r.Gold)
}

// Tally counts the verdicts of one run.
type Tally struct {
	N        int
	Answer   int
	Citation int
	Both     int
	Verdicts map[Verdict]int
}

func (t *Tally) Add(answerCorrect bool, v Verdict) {
	if t.Verdicts == nil {
		t.Verdicts = map[Verdict]int{}
	}
	t.N++
	t.Verdicts[v]++
	if answerCorrect {
		t.Answer++
	}
	if v.OK() {
		t.Citation++
	}
	if answerCorrect && v.OK() {
		t.Both++
	}
}

func (t Tally) Row(name string) string {
	pct := func(k int) float64 {
		if t.N == 0 {
			return 0
		}
		return 100 * float64(k) / float64(t.N)
	}
	return fmt.Sprintf("| %s | %d | %.1f%% | %.1f%% | %.1f%% | %d | %d | %d | %d | %d |",
		name, t.N, pct(t.Answer), pct(t.Citation), pct(t.Both),
		t.Verdicts[Exact], t.Verdicts[Ancestor], t.Verdicts[Contains],
		t.Verdicts[NotFound], t.Verdicts[Wrong])
}
