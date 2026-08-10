package specbench

import (
	"encoding/json"
	"testing"
)

func asn1Task() Task {
	return Task{
		ID: "asn1-X", Type: "asn1",
		Gold:         json.RawMessage(`["fieldA","fieldB","fieldC"]`),
		SpecID:       "TS 38.331",
		Section:      "PDCP-Config",
		SectionTitle: "PDCP-Config",
	}
}

func formulaTask() Task {
	return Task{
		ID: "formula-0", Type: "formula",
		Gold:    json.RawMessage(`"TRP=\\frac{1}{4\\pi }\\oint E d\\Omega"`),
		SpecID:  "TS 37.544",
		Section: "6.1.7.1",
	}
}

func answer(t *testing.T, body string) Answer {
	t.Helper()
	a, ok := ParseAnswer(body)
	if !ok {
		t.Fatalf("could not parse %s", body)
	}
	return a
}

func TestParseAnswerShapes(t *testing.T) {
	for _, body := range []string{
		`{"answer": ["a"], "spec_id": "TS 38.331", "section": "6.3.2"}`,
		"here you go\n```json\n{\"answer\": [\"a\"], \"spec_id\": \"TS 38.331\", \"section\": \"6.3.2\"}\n```",
		"long reasoning\n{\"answer\": [\"a\"], \"spec_id\": \"TS 38.331\", \"section\": \"6.3.2\"}",
	} {
		a := answer(t, body)
		if a.SpecID != "TS 38.331" || a.Section != "6.3.2" {
			t.Errorf("%s -> %+v", body, a)
		}
	}
	if _, ok := ParseAnswer("no json at all"); ok {
		t.Error("parsed an answer that is not there")
	}
}

// A model may return the field list as an array or as prose; both are the
// model's answer and neither should be scored as a formatting failure.
func TestFieldsAcceptsArrayOrList(t *testing.T) {
	a := answer(t, `{"answer": ["fieldA", "fieldB"], "spec_id": "x", "section": "y"}`)
	if got := a.Fields(); len(got) != 2 || got[0] != "fieldA" {
		t.Errorf("array form: %v", got)
	}
	a = answer(t, `{"answer": "fieldA, fieldB", "spec_id": "x", "section": "y"}`)
	if got := a.Fields(); len(got) != 2 || got[1] != "fieldB" {
		t.Errorf("string form: %v", got)
	}
}

func TestGradeASN1(t *testing.T) {
	task := asn1Task()

	right := answer(t, `{"answer": ["fieldA","fieldB","fieldC"], "spec_id": "TS 38.331", "section": "PDCP-Config"}`)
	if s := Grade(task, right); !s.Answer || !s.Citation || !s.Both || s.Partial != 1 {
		t.Errorf("exact answer scored %+v", s)
	}

	// Order is part of the definition, so a reordered list is not correct —
	// but it is not a hallucination either, and partial credit records that.
	shuffled := answer(t, `{"answer": ["fieldC","fieldB","fieldA"], "spec_id": "TS 38.331", "section": "PDCP-Config"}`)
	if s := Grade(task, shuffled); s.Answer || s.Partial != 1 {
		t.Errorf("reordered answer scored %+v", s)
	}

	partial := answer(t, `{"answer": ["fieldA","nope"], "spec_id": "TS 38.331", "section": "PDCP-Config"}`)
	s := Grade(task, partial)
	if s.Answer || s.Partial <= 0 || s.Partial >= 1 {
		t.Errorf("partly right answer scored %+v", s)
	}

	// The failure that matters most in practice: a right answer attributed to
	// the wrong clause must not count as a usable result.
	misplaced := answer(t, `{"answer": ["fieldA","fieldB","fieldC"], "spec_id": "TS 38.331", "section": "9.9.9"}`)
	if s := Grade(task, misplaced); !s.Answer || s.Citation || s.Both {
		t.Errorf("right answer with a wrong citation scored %+v", s)
	}
}

func TestCitationNormalisation(t *testing.T) {
	task := asn1Task()
	task.Section = "6.3.2"
	for _, cite := range []string{"6.3.2", "Clause 6.3.2", "section 6.3.2", " 6.3.2. "} {
		a := Answer{Answer: json.RawMessage(`[]`), SpecID: "ts38.331", Section: cite}
		if s := Grade(task, a); !s.SpecID || !s.Section {
			t.Errorf("citation %q scored spec=%v section=%v", cite, s.SpecID, s.Section)
		}
	}
	// A specification the model names with a version suffix is still that spec.
	a := Answer{Answer: json.RawMessage(`[]`), SpecID: "TS 38.331 v19.3.0", Section: "6.3.2"}
	if s := Grade(task, a); !s.SpecID {
		t.Error("version suffix broke the spec match")
	}
}

func TestGradeFormulaIgnoresCosmeticLatex(t *testing.T) {
	task := formulaTask()
	for _, eq := range []string{
		`TRP=\\frac{1}{4\\pi }\\oint E d\\Omega`,
		`TRP = \\frac{1}{4 \\pi} \\oint E d\\Omega`, // spacing
		`$$TRP=\\frac{1}{4\\pi}\\oint E d\\Omega$$`, // delimiters
		`TRP=\\frac{1}{4\\pi}\\oint E d\\Omega`,
	} {
		a := Answer{Answer: json.RawMessage(`"` + eq + `"`), SpecID: "TS 37.544", Section: "6.1.7.1"}
		if s := Grade(task, a); !s.Answer {
			t.Errorf("equation %q not accepted", eq)
		}
	}
	wrong := Answer{Answer: json.RawMessage(`"TRP=\\frac{1}{2\\pi}\\oint E d\\Omega"`), SpecID: "TS 37.544", Section: "6.1.7.1"}
	if s := Grade(task, wrong); s.Answer {
		t.Error("a different constant was accepted")
	}
}

func TestSummary(t *testing.T) {
	var s Summary
	s.Add(Score{Answer: true, SpecID: true, Section: true, Citation: true, Both: true, Partial: 1}, true)
	s.Add(Score{}, false)
	if s.N != 2 || s.Answer != 1 || s.Both != 1 || s.Answered != 1 {
		t.Errorf("summary = %+v", s)
	}
	if got := s.String(); got == "" {
		t.Error("empty summary line")
	}
}

func task(kind, gold, section, api string) Task {
	return Task{Type: "x", Kind: kind, Gold: json.RawMessage(gold),
		SpecID: "TS 29.274", Section: section, APIName: api}
}

// A wire code is a number however the model dresses it up, and an element name
// matches with or without the expansion the registry tables carry.
func TestGradeScalar(t *testing.T) {
	code := task("scalar", `"45"`, "8.1", "")
	for _, got := range []string{`"45"`, `45`, `"45 (decimal)"`, `"IE type 45"`, `"045"`} {
		if s := Grade(code, Answer{Answer: json.RawMessage(got), SpecID: "TS 29.274", Section: "8.1"}); !s.Answer {
			t.Errorf("answer %s scored wrong", got)
		}
	}
	if s := Grade(code, Answer{Answer: json.RawMessage(`"46"`), SpecID: "TS 29.274", Section: "8.1"}); s.Answer {
		t.Error("46 accepted for gold 45")
	}

	name := task("scalar", `"International Mobile Subscriber Identity (IMSI)"`, "8.1", "")
	for _, got := range []string{
		`"International Mobile Subscriber Identity (IMSI)"`,
		`"International Mobile Subscriber Identity"`,
		`"international mobile subscriber identity"`,
	} {
		if s := Grade(name, Answer{Answer: json.RawMessage(got), SpecID: "TS 29.274", Section: "8.1"}); !s.Answer {
			t.Errorf("answer %s scored wrong", got)
		}
	}
}

// Required properties are a set, so their order must not decide the score.
func TestGradeSet(t *testing.T) {
	tk := task("set", `["a","b","c"]`, "Nnrf_NFManagement", "Nnrf_NFManagement")
	tk.SpecID = "TS 29.510"
	ans := func(a string) Answer {
		return Answer{Answer: json.RawMessage(a), SpecID: "TS 29.510", Section: "Nnrf_NFManagement"}
	}
	if s := Grade(tk, ans(`["c","a","b"]`)); !s.Answer || !s.Both {
		t.Errorf("reordered set scored %+v", s)
	}
	if s := Grade(tk, ans(`["a","b"]`)); s.Answer || s.Partial == 0 {
		t.Errorf("subset scored %+v, want wrong with partial credit", s)
	}
	// An OpenAPI citation names the API document, not a clause.
	if s := Grade(tk, Answer{Answer: json.RawMessage(`["a","b","c"]`), SpecID: "TS 29.510", Section: "6.1.6.2.2"}); s.Citation {
		t.Error("a clause number was accepted as an OpenAPI citation")
	}
}

// Task files written before answer_kind existed must score as they did.
func TestKindFallsBackToType(t *testing.T) {
	for typ, want := range map[string]string{"asn1": "sequence", "openapi": "set", "code": "scalar", "formula": "latex"} {
		if got := (Task{Type: typ}).kind(); got != want {
			t.Errorf("%s -> %s, want %s", typ, got, want)
		}
	}
	if got := (Task{Type: "asn1", Kind: "scalar"}).kind(); got != "scalar" {
		t.Errorf("explicit kind ignored: %s", got)
	}
}

// The converter escapes angle brackets inside maths, so the corpus writes
// "\lt" where the source shows "<". A model answering with either form has
// given the same equation and must score the same.
func TestGradeLatexEscapedRelations(t *testing.T) {
	tk := Task{Type: "formula", Kind: "latex", Gold: json.RawMessage(`"N\\leq a\\lt N+1"`),
		SpecID: "TS 23.032", Section: "Altitude"}
	for _, got := range []string{`"N\\leq a\\lt N+1"`, `"N \\le a < N+1"`, `"$$N\\leq a<N+1$$"`} {
		s := Grade(tk, Answer{Answer: json.RawMessage(got), SpecID: "TS 23.032", Section: "Altitude"})
		if !s.Answer {
			t.Errorf("answer %s scored wrong", got)
		}
	}
	if s := Grade(tk, Answer{Answer: json.RawMessage(`"N\\geq a\\gt N+1"`), SpecID: "TS 23.032", Section: "Altitude"}); s.Answer {
		t.Error("a different relation was accepted")
	}
}
