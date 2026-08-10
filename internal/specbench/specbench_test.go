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
