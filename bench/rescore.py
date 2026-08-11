#!/usr/bin/env python3
"""Re-grade benchmark citations against the specification database.

Comparing the cited clause to the gold clause as strings is too strict to mean
anything. The corpus stores each 38.331 information element as a section whose
number *is* the element's name, so a model that answers "TS 38.331 clause
6.3.2" — the clause those elements live under, and what an engineer would
write — scored zero against a gold of "PDCCH-ServingCellConfig". And the RF
test specifications copy clauses between each other, so more than one document
really does define the same equation.

So a citation is graded against the documents instead:

  exact     the cited section is the one the task was generated from
  ancestor  the cited section contains that one, via parent_number — a
            coarser citation, still true
  contains  a different section that genuinely holds the same answer
  wrong     none of the above, including a section that does not exist

Only "wrong" is a failure. "not_found" is reported separately because citing a
clause that does not exist is a different failure from citing the wrong one.

    python3 bench/rescore.py --db 3gpp-latest.db results/bench-*.jsonl
"""

import argparse
import collections
import glob
import json
import os
import re
import sqlite3
import sys
from html.parser import HTMLParser

SPEC_RE = re.compile(r"(?i)\b(TS|TR)\s*([0-9]{2}\.[0-9]{3}(?:-[0-9]+)?)")
SEC_PREFIX = re.compile(r"(?i)^(clause|section|sec\.?|annex)\s*")
WS = re.compile(r"\s+")


def norm_spec(s):
    m = SPEC_RE.search(s or "")
    return f"{m.group(1).upper()} {m.group(2)}" if m else WS.sub(" ", (s or "").strip()).upper()


def leading_number(s):
    """The clause number at the start of a heading, e.g. "7.2.160aA" from
    "7.2.160aA\tQuota-Indicator AVP"."""
    return re.split(r"[\t\n]|\s{2,}", (s or "").strip(), 1)[0].strip()


def norm_sec(s):
    s = SEC_PREFIX.sub("", (s or "").strip())
    return WS.sub(" ", s.strip().rstrip(".")).lower()


def norm_latex(s):
    s = (s or "").strip().strip("$")
    s = re.sub(r"\\(text|mathrm|mathit)\{([^}]*)\}", r"\2", s)
    s = re.sub(r"\\(left|right|quad|qquad)\b|\\[,;!]", "", s)
    s = WS.sub("", s).replace("\\cdot", "·")
    # The converter escapes angle brackets in maths: \lt and < are the same.
    for a, b in (("\\leq", "≤"), ("\\geq", "≥"), ("\\le", "≤"), ("\\ge", "≥"),
                 ("\\lt", "<"), ("\\gt", ">"), ("\\neq", "≠")):
        s = s.replace(a, b)
    for _ in range(3):
        s = re.sub(r"\{(\\?[A-Za-z0-9]+)\}", r"\1", s)
    return s.lower()


class Corpus:
    def __init__(self, path):
        self.conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.by_spec = {}

    def sections(self, spec):
        """{normalised number or title: (number, parent_number, content)}"""
        if spec not in self.by_spec:
            index = {}
            rows = self.conn.execute(
                "SELECT number,title,parent_number,content FROM sections WHERE UPPER(spec_id)=?",
                (spec,),
            )
            for number, title, parent, content in rows:
                entry = (number, parent, content)
                index.setdefault(norm_sec(number), entry)
                index.setdefault(norm_sec(title), entry)
                # 9222 sections (1.7%) have the whole heading in the number
                # column because the converter did not split it — including
                # real clauses such as TS 32.299 7.2.160aA. Index the leading
                # token as well so the number an engineer would write resolves.
                index.setdefault(norm_sec(leading_number(number)), entry)
            self.by_spec[spec] = index
        return self.by_spec[spec]

    def title_of(self, spec, number):
        row = self.conn.execute(
            "SELECT title FROM sections WHERE UPPER(spec_id)=? AND number=?", (spec, number)
        ).fetchone()
        return row[0] if row else ""

    def ancestors(self, spec, number):
        """Numbers of the sections that contain `number`, innermost first."""
        out, seen = [], set()
        row = self.conn.execute(
            "SELECT parent_number FROM sections WHERE UPPER(spec_id)=? AND number=?",
            (spec, number),
        ).fetchone()
        parent = row[0] if row else None
        while parent and parent not in seen:
            seen.add(parent)
            out.append(parent)
            row = self.conn.execute(
                "SELECT parent_number FROM sections WHERE UPPER(spec_id)=? AND number=?",
                (spec, parent),
            ).fetchone()
            parent = row[0] if row else None
        return out


def gold_of(rec):
    g = rec["gold"]
    return json.loads(g) if isinstance(g, (str, bytes)) and rec["type"] != "formula" else g


QUOTED = re.compile(r"'([^']+)'")


def element_name(rec):
    """The protocol element a code task is about, however the task asks for it."""
    if "-name-" in rec["id"]:
        return rec["gold"] if isinstance(rec["gold"], str) else json.loads(rec["gold"])
    m = QUOTED.search(rec.get("question", ""))
    return m.group(1) if m else None


def holds_answer(number, title, content, rec):
    """Does this section actually document what the task asks about?"""
    kind = rec["type"]
    if kind == "asn1":
        gold = gold_of(rec)
        name = rec["id"][len("asn1-"):]
        if f"{name} ::=" not in content:
            return False
        return all(g in content for g in gold)
    if kind == "code":
        code = rec["id"].rsplit("-", 1)[-1]
        gold = rec["gold"] if isinstance(rec["gold"], str) else json.loads(rec["gold"])
        name = element_name(rec)
        # A registry row: the code and the element in one row, in any column —
        # these tables carry merged cells, so the code is not always first.
        for row in html_rows(content):
            cells = [cell.strip() for cell in row]
            if code in cells and (norm_name(gold) in map(norm_name, cells) or
                                  (name and norm_name(name) in map(norm_name, cells))):
                return True
        # Or the clause that defines the element. 3GPP writes the code once, in
        # the registry, and the defining clause carries only the name and the
        # semantics — so a clause titled after the element is the citation an
        # implementer wants, whether or not it repeats the number.
        return bool(name) and norm_name(name) in norm_name(title)
    gold = rec["gold"] if isinstance(rec["gold"], str) else json.loads(rec["gold"])
    return norm_latex(gold) in norm_latex(content)


def norm_name(s):
    s = re.sub(r"\s*\([^)]*\)\s*$", "", (s or "").strip())
    return WS.sub(" ", s).strip(" .").lower()


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self._row, self._cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []

    def handle_endtag(self, tag):
        if tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None
        elif tag in ("td", "th") and self._cell is not None:
            if self._row is not None:
                self._row.append(WS.sub(" ", "".join(self._cell)).strip())
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def html_rows(content):
    p = _TableParser()
    p.feed(content)
    return p.rows




# {task id: task}, for the probes. Populated by load_tasks.
TASKS = {}
ALLOF_CACHE = {}
ALLOF_DB = None


def load_tasks(tasks_dir):
    for path in sorted(glob.glob(os.path.join(tasks_dir, "tasks-*.json"))):
        for t in json.load(open(path)):
            TASKS[t["id"]] = t


def citation_targets(rec):
    """Every element whose declaration would justify this answer.

    An OpenAPI question points at one schema and its answer is defined in
    another: "the property `tgtUe` of `EventSubsc` holds an object, list its
    fields" is asked of TS 29.530, and answered by the definition of
    `TargetUeInformation`, which lives in TS 29.571 CommonData. Both documents
    are true citations — one is where the question is posed, the other is where
    the answer is written — and a scorer that accepts only the first marks the
    condition that actually followed the reference as wrong.

    Read from the task's own `probe`, never parsed out of the id: an id holds
    an API name that may itself contain hyphens, and deriving a verdict from a
    string split is how this scorer was wrong before.
    """
    p = TASKS.get(rec["id"], {}).get("probe") or {}
    kind = p.get("kind")
    if kind == "request":
        # The operation, and the schema its body carries.
        return [("operation", f"{p['method'].upper()} {p['path']}"), ("schema", p.get("schema"))]
    if kind == "object":
        return [("schema", p.get("owner")), ("schema", p.get("schema"))]
    if kind == "oneof":
        # The schema that declares the choice, and the alternatives themselves,
        # which are the answer.
        return [("schema", p.get("schema"))] + [("schema", g) for g in gold_of(rec)]
    if kind == "allof":
        return [("schema", p.get("schema"))] + [("schema", m) for m in allof_members(rec)]
    # A task from before the probe existed: its id ends in the schema name.
    return [("schema", rec["id"].rsplit("-", 1)[-1])]


ALLOF_REF = re.compile(r"\$ref:\s*'?\"?[^'\"\s#]*#/components/schemas/([\w.-]+)")


def allof_members(rec):
    """The schemas an allOf composes, read out of the gold document."""
    task = TASKS.get(rec["id"], {})
    row = ALLOF_CACHE.get(rec["id"], ...)
    if row is not ...:
        return row
    members = []
    r = ALLOF_DB.execute(
        "SELECT content FROM openapi_specs WHERE spec_id=? AND api_name=?",
        (task.get("spec_id"), task.get("api_name")),
    ).fetchone() if ALLOF_DB else None
    if r:
        m = re.search(rf"^    {re.escape((task.get('probe') or {}).get('schema', ''))}:\n"
                      r"((?:      .*\n|\n)*)", r[0], re.M)
        if m:
            members = ALLOF_REF.findall(m.group(1))
    ALLOF_CACHE[rec["id"]] = members
    return members


def declares(content, kind, name):
    """Does this OpenAPI document define that schema, or that operation?"""
    if not name:
        return False
    if kind == "operation":
        method, _, path = name.partition(" ")
        # Paths that carry a {placeholder} are usually quoted in these files.
        m = re.search(rf"^  ['\"]?{re.escape(path)}['\"]?:", content, re.M)
        if not m:
            return False
        rest = content[m.end():]
        end = re.search(r"^  \S", rest, re.M)
        return re.search(rf"^    {method.lower()}:", rest[:end.start() if end else len(rest)],
                         re.M) is not None
    return re.search(rf"^    {re.escape(name)}:", content, re.M) is not None


def element_names(rec):
    """Names that identify the thing a task is about, besides its document.

    A model that answers "TS 29.518 / UeRegStatusUpdateReqData" has named the
    schema rather than the API document that declares it. That is a precise and
    checkable attribution of the same element, and rejecting it would be the
    same mistake this scorer made four times before: pinning the citation to
    one location and failing every other true one.
    """
    p = TASKS.get(rec["id"], {}).get("probe") or {}
    return {n for n in (p.get("schema"), p.get("owner")) if n}


def grade_openapi(corpus, rec):
    """An OpenAPI element is cited by the API document that declares it.

    There is no clause number in an OpenAPI file, so the citation these tasks
    ask for is the specification and the API document. It counts when that
    document really declares the schema or operation the question was built
    from — the same rule as everywhere else in this scorer: the citation has to
    hold the answer, not merely match a string.
    """
    spec = norm_spec(rec.get("predicted_spec_id"))
    api = (rec.get("predicted_section") or "").strip()
    if not api:
        return "wrong"
    targets = [(k, n) for k, n in citation_targets(rec) if n]

    row = corpus.conn.execute(
        "SELECT content FROM openapi_specs WHERE UPPER(spec_id)=? AND api_name=?",
        (spec, api),
    ).fetchone()
    if row:
        if any(declares(row[0], k, n) for k, n in targets):
            return "exact" if api == rec["gold_section"] else "contains"
        return "wrong"

    # The element named instead of the document that declares it.
    if any(api == n for _, n in targets):
        for (content,) in corpus.conn.execute(
            "SELECT content FROM openapi_specs WHERE UPPER(spec_id)=?", (spec,)
        ):
            if declares(content, "schema", api):
                return "contains"

    # Not an API name. In the SBI specifications the normative definition is a
    # clause of the running text — "6.1.6.2.23  Type: UeContextTransferReqData",
    # "7.3.5  ThresholdCrossing <<dataType>>" — and the OpenAPI file in the
    # annex is generated from it. That clause is the citation an implementer
    # wants, so accept a section this specification titles after the schema.
    hit = corpus.sections(spec).get(norm_sec(api))
    if hit is None:
        return "not_found"
    number, _, _ = hit
    if any(n and re.search(rf"\b{re.escape(n)}\b", corpus.title_of(spec, number))
           for _, n in targets):
        return "contains"
    return "wrong"


def grade(corpus, rec):
    if rec["type"] == "openapi":
        return grade_openapi(corpus, rec)
    spec = norm_spec(rec.get("predicted_spec_id"))
    sec = norm_sec(rec.get("predicted_section"))
    gold_spec = norm_spec(rec["gold_spec_id"])
    if spec != gold_spec and not sec:
        return "wrong"

    index = corpus.sections(spec)
    if not index:
        return "not_found"
    hit = index.get(sec)
    if hit is None:
        return "not_found"
    number, _, content = hit
    title = corpus.title_of(spec, number)

    if spec == gold_spec and norm_sec(number) == norm_sec(rec["gold_section"]):
        return "exact"
    if spec == gold_spec and number in corpus.ancestors(gold_spec, rec["gold_section"]):
        return "ancestor"
    if holds_answer(number, title, content, rec):
        return "contains"
    return "wrong"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("--tasks-dir", default="bench",
                    help="task files, read for the probe each OpenAPI gold was built from")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    load_tasks(args.tasks_dir)
    corpus = Corpus(args.db)
    global ALLOF_DB
    ALLOF_DB = corpus.conn
    print("| run | n | answer | citation | answer+citation | exact | ancestor | contains | not found | wrong |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for path in args.files:
        recs = [json.loads(line) for line in open(path) if line.strip()]
        if not recs:
            continue
        verdicts = collections.Counter()
        cited = ans = both = 0
        for r in recs:
            v = grade(corpus, r)
            verdicts[v] += 1
            ok = v in ("exact", "ancestor", "contains")
            cited += ok
            ans += r["score"]["answer_correct"]
            both += ok and r["score"]["answer_correct"]
            r["score"]["citation_verdict"] = v
            r["score"]["citation_correct"] = ok
            r["score"]["answer_and_citation"] = ok and r["score"]["answer_correct"]
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        n = len(recs)
        pct = lambda k: 100 * k / n
        print("| %s | %d | %.1f%% | %.1f%% | %.1f%% | %d | %d | %d | %d | %d |" % (
            path.split("/")[-1].replace("bench-", "").replace(".jsonl", ""), n,
            pct(ans), pct(cited), pct(both),
            verdicts["exact"], verdicts["ancestor"], verdicts["contains"],
            verdicts["not_found"], verdicts["wrong"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
