// Package specbench scores free-form answers about 3GPP specifications
// together with the citation that backs them.
//
// TeleQnA asks for an option number, which a model can reach by elimination
// and which says nothing about where the answer came from. These tasks ask for
// the value itself — an ASN.1 field list, an equation — and for the
// specification and section it is defined in, and score the two separately.
// An answer that is right without a correct citation is not usable for
// checking a specification, and the score says so.
package specbench

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"
)

// Task is one generated question. The gold comes out of the specification
// text mechanically, so it carries no human labelling error.
type Task struct {
	ID       string          `json:"id"`
	Type     string          `json:"type"`        // asn1, formula, code, openapi
	Kind     string          `json:"answer_kind"` // sequence, latex, scalar, set
	Question string          `json:"question"`
	Gold     json.RawMessage `json:"gold"`
	SpecID   string          `json:"spec_id"`
	Version  string          `json:"version"`
	Section  string          `json:"section"`
	// APIName is set for OpenAPI tasks, whose citation names an API document
	// rather than a clause.
	APIName      string `json:"api_name,omitempty"`
	SectionTitle string `json:"section_title"`
}

// GoldList returns a list-shaped gold: ASN.1 fields in definition order, or the
// required properties of an OpenAPI schema.
func (t Task) GoldList() []string {
	var f []string
	_ = json.Unmarshal(t.Gold, &f)
	return f
}

// GoldString returns the expected equation.
func (t Task) GoldString() string {
	var s string
	_ = json.Unmarshal(t.Gold, &s)
	return s
}

func Load(path string) ([]Task, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var tasks []Task
	if err := json.Unmarshal(data, &tasks); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return tasks, nil
}

// Answer is what the model is asked to return: the value plus where it is
// defined. Answer is left raw because its shape depends on the task type.
type Answer struct {
	Answer  json.RawMessage `json:"answer"`
	SpecID  string          `json:"spec_id"`
	Section string          `json:"section"`
}

// Fields reads an ASN.1 answer, accepting either a JSON array or a
// comma/newline separated list, since models drift between the two.
func (a Answer) Fields() []string {
	var arr []string
	if err := json.Unmarshal(a.Answer, &arr); err == nil {
		return trimAll(arr)
	}
	var s string
	if err := json.Unmarshal(a.Answer, &s); err != nil {
		return nil
	}
	return trimAll(regexp.MustCompile(`[,\n]`).Split(s, -1))
}

// Text reads a scalar answer, tolerating a model that wrapped it in an array.
func (a Answer) Text() string {
	var s string
	if err := json.Unmarshal(a.Answer, &s); err == nil {
		return s
	}
	var arr []string
	if err := json.Unmarshal(a.Answer, &arr); err == nil && len(arr) > 0 {
		return arr[0]
	}
	return strings.TrimSpace(string(a.Answer))
}

func trimAll(in []string) []string {
	var out []string
	for _, s := range in {
		s = strings.TrimSpace(strings.Trim(strings.TrimSpace(s), "\"'`-*"))
		if s != "" {
			out = append(out, s)
		}
	}
	return out
}

// ParseAnswer extracts the JSON object the prompt asked for. The reply may be
// fenced or preceded by reasoning, so the last valid object wins — the same
// rule the TeleQnA prompt variant uses.
func ParseAnswer(text string) (Answer, bool) {
	for i := len(text) - 1; i >= 0; i-- {
		if text[i] != '{' {
			continue
		}
		var a Answer
		if err := json.NewDecoder(strings.NewReader(text[i:])).Decode(&a); err != nil {
			continue
		}
		if len(a.Answer) > 0 {
			return a, true
		}
	}
	return Answer{}, false
}

// Score is one task's outcome. Answer and Citation are deliberately separate:
// the interesting failure in practice is a plausible answer attributed to the
// wrong clause.
type Score struct {
	Answer    bool    `json:"answer_correct"`
	SpecID    bool    `json:"spec_correct"`
	Section   bool    `json:"section_correct"`
	Citation  bool    `json:"citation_correct"` // both of the above
	Both      bool    `json:"answer_and_citation"`
	Partial   float64 `json:"partial"` // F1 over ASN.1 fields, 0/1 elsewhere
	Predicted string  `json:"predicted"`
}

// kind falls back to the task type, so task files written before answer_kind
// existed still score the way they did.
func (t Task) kind() string {
	if t.Kind != "" {
		return t.Kind
	}
	switch t.Type {
	case "asn1":
		return "sequence"
	case "openapi":
		return "set"
	case "code":
		return "scalar"
	}
	return "latex"
}

func Grade(t Task, a Answer) Score {
	s := Score{
		SpecID:  normSpec(a.SpecID) == normSpec(t.SpecID),
		Section: normSection(a.Section) == normSection(t.Section),
	}
	switch t.kind() {
	case "sequence": // order matters: an ASN.1 definition is ordered
		got, want := a.Fields(), t.GoldList()
		s.Answer = equalSeq(got, want)
		s.Partial = f1(got, want)
		s.Predicted = strings.Join(got, ", ")
	case "set": // a required-properties list is a set
		got, want := a.Fields(), t.GoldList()
		s.Answer = equalSet(got, want)
		s.Partial = f1(got, want)
		s.Predicted = strings.Join(got, ", ")
	case "scalar":
		got, want := a.Text(), t.GoldString()
		s.Answer = scalarEqual(got, want)
		if s.Answer {
			s.Partial = 1
		}
		s.Predicted = got
	default: // latex
		got, want := a.Text(), t.GoldString()
		s.Answer = normLatex(got) == normLatex(want)
		if s.Answer {
			s.Partial = 1
		}
		s.Predicted = got
	}
	// An OpenAPI schema is cited by the API document that defines it.
	if t.APIName != "" {
		s.Section = normSection(a.Section) == normSection(t.APIName)
	}
	// A section title is an acceptable citation when the corpus numbers the
	// section by name, which 38.331 does for every IE definition.
	if !s.Section && normSection(a.Section) == normSection(t.SectionTitle) {
		s.Section = true
	}
	s.Citation = s.SpecID && s.Section
	s.Both = s.Answer && s.Citation
	return s
}

var (
	spaceRe   = regexp.MustCompile(`\s+`)
	latexCmd  = regexp.MustCompile(`\\(left|right|quad|qquad)\b|\\[,;!]`)
	textRe    = regexp.MustCompile(`\\(text|mathrm|mathit)\{([^}]*)\}`)
	braceOne  = regexp.MustCompile(`\{(\\?[A-Za-z0-9]+)\}`)
	specIDRe  = regexp.MustCompile(`(?i)\b(TS|TR)\s*([0-9]{2}\.[0-9]{3}(?:-[0-9]+)?)`)
	sectionRe = regexp.MustCompile(`(?i)^(clause|section|sec\.?)\s*`)
	parenRe   = regexp.MustCompile(`\s*\([^)]*\)\s*$`)
	numRe     = regexp.MustCompile(`-?\d+`)
)

// scalarEqual compares a wire code or an element name against its gold.
//
// When the gold is a number the answer is compared as one, so "45",
// "45 (decimal)" and "IE type 45" all agree. When it is a name the comparison
// ignores case, spacing and the trailing expansion the registry tables carry,
// so "International Mobile Subscriber Identity (IMSI)" matches either form.
func scalarEqual(got, want string) bool {
	want = strings.TrimSpace(want)
	if isNumeric(want) {
		m := numRe.FindString(got)
		return m != "" && trimZeros(m) == trimZeros(want)
	}
	return normName(got) == normName(want)
}

func normName(s string) string {
	s = parenRe.ReplaceAllString(strings.TrimSpace(s), "")
	s = spaceRe.ReplaceAllString(s, " ")
	return strings.ToLower(strings.Trim(s, " .\"'"))
}

func trimZeros(s string) string {
	s = strings.TrimLeft(s, "0")
	if s == "" {
		return "0"
	}
	return s
}

func isNumeric(s string) bool {
	s = strings.TrimSpace(s)
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

// equalSet compares two lists as sets, for answers whose order is not defined.
func equalSet(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	seen := map[string]bool{}
	for _, x := range b {
		seen[strings.ToLower(x)] = true
	}
	for _, x := range a {
		if !seen[strings.ToLower(x)] {
			return false
		}
	}
	return true
}

// normSpec reduces "ts 38.331", "TS38.331" and "TS 38.331 v19.3.0" to one form.
func normSpec(s string) string {
	if mm := specIDRe.FindStringSubmatch(s); mm != nil {
		return strings.ToUpper(mm[1]) + " " + mm[2]
	}
	return strings.ToUpper(spaceRe.ReplaceAllString(strings.TrimSpace(s), " "))
}

// normSection strips the "clause"/"section" prefix and a trailing dot, so
// "Clause 6.3.2." and "6.3.2" agree.
func normSection(s string) string {
	s = sectionRe.ReplaceAllString(strings.TrimSpace(s), "")
	s = strings.TrimRight(strings.TrimSpace(s), ".")
	return strings.ToLower(spaceRe.ReplaceAllString(s, " "))
}

// normLatex removes the differences that do not change an equation: spacing,
// sizing commands, \text wrappers and braces around a single token.
func normLatex(s string) string {
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(strings.TrimSuffix(s, "$$"), "$$")
	s = strings.TrimPrefix(strings.TrimSuffix(s, "$"), "$")
	s = textRe.ReplaceAllString(s, "$2")
	s = latexCmd.ReplaceAllString(s, "")
	s = spaceRe.ReplaceAllString(s, "")
	s = strings.ReplaceAll(s, "\\cdot", "·")
	s = strings.ReplaceAll(s, "\\ast", "*")
	// The converter escapes angle brackets in maths, so \lt and < are the same
	// equation written twice.
	for _, r := range [][2]string{{"\\leq", "≤"}, {"\\geq", "≥"}, {"\\le", "≤"}, {"\\ge", "≥"},
		{"\\lt", "<"}, {"\\gt", ">"}, {"\\neq", "≠"}} {
		s = strings.ReplaceAll(s, r[0], r[1])
	}
	for i := 0; i < 3; i++ { // nested braces around single tokens
		s = braceOne.ReplaceAllString(s, "$1")
	}
	return strings.ToLower(s)
}

func equalSeq(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !strings.EqualFold(a[i], b[i]) {
			return false
		}
	}
	return true
}

// f1 is the overlap of the two field-name sets, so a mostly-right list scores
// above a wrong one.
func f1(got, want []string) float64 {
	if len(got) == 0 || len(want) == 0 {
		return 0
	}
	set := map[string]bool{}
	for _, w := range want {
		set[strings.ToLower(w)] = true
	}
	hit := 0
	seen := map[string]bool{}
	for _, g := range got {
		g = strings.ToLower(g)
		if set[g] && !seen[g] {
			hit++
			seen[g] = true
		}
	}
	p := float64(hit) / float64(len(got))
	r := float64(hit) / float64(len(want))
	if p+r == 0 {
		return 0
	}
	return 2 * p * r / (p + r)
}

// Summary aggregates scores for reporting.
type Summary struct {
	N        int
	Answer   int
	Citation int
	Both     int
	SpecID   int
	Section  int
	Partial  float64
	Answered int
}

func (s *Summary) Add(sc Score, answered bool) {
	s.N++
	if answered {
		s.Answered++
	}
	if sc.Answer {
		s.Answer++
	}
	if sc.Citation {
		s.Citation++
	}
	if sc.Both {
		s.Both++
	}
	if sc.SpecID {
		s.SpecID++
	}
	if sc.Section {
		s.Section++
	}
	s.Partial += sc.Partial
}

func (s Summary) String() string {
	if s.N == 0 {
		return "no tasks"
	}
	pct := func(n int) float64 { return 100 * float64(n) / float64(s.N) }
	return fmt.Sprintf(
		"n=%d answered=%d | answer %.1f%% | spec %.1f%% | section %.1f%% | citation %.1f%% | answer+citation %.1f%% | partial %.2f",
		s.N, s.Answered, pct(s.Answer), pct(s.SpecID), pct(s.Section), pct(s.Citation), pct(s.Both), s.Partial/float64(s.N))
}

// SortTasks keeps runs comparable regardless of file order.
func SortTasks(t []Task) { sort.Slice(t, func(i, j int) bool { return t[i].ID < t[j].ID }) }
