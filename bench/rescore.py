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
import json
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


def norm_sec(s):
    s = SEC_PREFIX.sub("", (s or "").strip())
    return WS.sub(" ", s.strip().rstrip(".")).lower()


def norm_latex(s):
    s = (s or "").strip().strip("$")
    s = re.sub(r"\\(text|mathrm|mathit)\{([^}]*)\}", r"\2", s)
    s = re.sub(r"\\(left|right|quad|qquad)\b|\\[,;!]", "", s)
    s = WS.sub("", s).replace("\\cdot", "·")
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
            self.by_spec[spec] = index
        return self.by_spec[spec]

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


def holds_answer(content, rec):
    """Does this section actually contain the answer the task asks for?"""
    kind = rec["type"]
    if kind == "asn1":
        gold = gold_of(rec)
        name = rec["id"][len("asn1-"):]
        if f"{name} ::=" not in content:
            return False
        return all(g in content for g in gold)
    if kind == "code":
        # The registry row must be there: the code and the name, in one row of
        # one of the tables this section holds.
        gold = rec["gold"] if isinstance(rec["gold"], str) else json.loads(rec["gold"])
        code = rec["id"].rsplit("-", 1)[-1]
        for row in html_rows(content):
            if len(row) >= 2 and row[0].strip() == code:
                return norm_name(gold) in (norm_name(row[0]), norm_name(row[1])) or \
                    (len(row) > 2 and norm_name(gold) == norm_name(row[2]))
        return False
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


SCHEMA_RE = re.compile(r"^    ([A-Za-z][\w]*):\n((?:      .*\n|\n)*)", re.M)
REQUIRED_RE = re.compile(r"^      required:\n((?:        - \w+\n)+)", re.M)


def grade_openapi(corpus, rec):
    """OpenAPI schemas are cited by API document, and the same schema may be
    declared by more than one of them."""
    spec = norm_spec(rec.get("predicted_spec_id"))
    api = (rec.get("predicted_section") or "").strip()
    if not api:
        return "wrong"
    name = rec["id"].rsplit("-", 1)[-1]
    gold = gold_of(rec)

    row = corpus.conn.execute(
        "SELECT content FROM openapi_specs WHERE UPPER(spec_id)=? AND api_name=?",
        (spec, api),
    ).fetchone()
    if not row:
        return "not_found"
    start = row[0].find("  schemas:")
    for m in SCHEMA_RE.finditer(row[0][start:]):
        if m.group(1) != name:
            continue
        rm = REQUIRED_RE.search(m.group(2))
        if not rm:
            continue
        props = [l.strip("- \n") for l in rm.group(1).strip().splitlines()]
        if props == gold:
            return "exact" if api == rec["gold_section"] else "contains"
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

    if spec == gold_spec and norm_sec(number) == norm_sec(rec["gold_section"]):
        return "exact"
    if spec == gold_spec and number in corpus.ancestors(gold_spec, rec["gold_section"]):
        return "ancestor"
    if holds_answer(content, rec):
        return "contains"
    return "wrong"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    corpus = Corpus(args.db)
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
