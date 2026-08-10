#!/usr/bin/env python3
"""Generate spec-grounded benchmark tasks from a pinned 3gpp-mcp database.

TeleQnA does not test what a 3GPP tool is actually used for. Its 3GPP subset
contains no equations, no ASN.1 structure questions (1 of 1509 questions even
mentions ASN.1, none mention PFCP, AVP, TLV or SEQUENCE), and it scores a
multiple-choice option rather than a value or a citation. A model can pick the
right option by elimination without knowing the number, and nothing checks
whether it looked at the right clause.

These tasks are the opposite: free-form answers whose gold comes mechanically
out of the specification text, plus the specification and section the answer
was taken from. There is no human labelling step, so there are no label errors
of the kind TeleQnA carries — the gold is the document.

    python3 bench/generate.py --db 3gpp-latest.db --out bench/ --n 30

Two task types:

  asn1     given an ASN.1 type name, name its fields.       gold = field names
  formula  given the sentence that introduces an equation,   gold = LaTeX
           reproduce the equation.

Both also require the answer to say which specification and section it came
from, which is scored separately: being right for the wrong reason is not
useful when the point is to check a specification.
"""

import argparse
import collections
import json
import os
import random
import re
import sqlite3
import sys
from html.parser import HTMLParser

ASN1_FENCE = re.compile(r"```asn1\n(.*?)```", re.S)
# The head of a definition at column 0, e.g. "PDCP-Config ::= SEQUENCE {".
ASN1_HEAD = re.compile(r"^([A-Za-z][\w-]*)\s*::=\s*SEQUENCE\s*\{", re.M)
# A member line: indented, a lower-case identifier followed by its type.
ASN1_FIELD = re.compile(r"^\s+([a-z][\w-]*)\s+\S")


def top_level_fields(block, start):
    """Members of the SEQUENCE opening at `start`, ignoring nested ones.

    A CHOICE or an inner SEQUENCE brings its own members, and counting those
    as fields of the outer type produced golds that were simply wrong — a
    model answering the two real top-level fields looked like a failure. Only
    lines seen while the brace depth is 1 are members.
    """
    depth, fields = 0, []
    for line in block[start:].splitlines():
        opens, closes = line.count("{"), line.count("}")
        if depth == 1:
            m = ASN1_FIELD.match(line)
            if m and m.group(1) not in fields:
                fields.append(m.group(1))
        depth += opens - closes
        if depth <= 0:
            break
    return fields

DISPLAY_MATH = re.compile(r"\$\$(.+?)\$\$", re.S)
# An equation states a relation, and is worth asking about only if it has
# structure a model has to reproduce rather than a number it could guess.
RELATION = re.compile(r"=|\\\\leq|\\\\geq|\\\\le\\b|\\\\ge\\b|<|>")
STRUCTURE = re.compile(r"\\\\[a-zA-Z]{2,}|[_^]")

# Specifications whose text is current enough to be worth asking about. The
# corpus also holds GSM-era documents whose section numbering is a plain title.
MODERN = re.compile(r"^T[SR] (2[1-9]|3[0-9])\.")


def asn1_tasks(conn, n, rng):
    """One task per ASN.1 SEQUENCE type that the whole corpus defines exactly
    once, so the citation has a single correct answer."""
    defs = collections.defaultdict(list)
    rows = conn.execute(
        "SELECT spec_id,version,number,title,content FROM sections WHERE content LIKE '%```asn1%'"
    )
    for spec, version, number, title, content in rows:
        for block in ASN1_FENCE.findall(content):
            if "/example/" in block:  # the illustrative block in 38.331 6.1.2
                continue
            for m in ASN1_HEAD.finditer(block):
                fields = top_level_fields(block, m.end() - 1)
                if 3 <= len(fields) <= 12:
                    defs[m.group(1)].append(
                        dict(spec_id=spec, version=version, section=number,
                             section_title=title, fields=fields)
                    )
    unique = [(k, v[0]) for k, v in defs.items() if len(v) == 1 and MODERN.match(v[0]["spec_id"])]
    unique.sort()
    rng.shuffle(unique)

    tasks = []
    for name, d in unique[:n]:
        tasks.append({
            "id": f"asn1-{name}",
            "type": "asn1",
            "question": (
                f"The 3GPP specifications define an ASN.1 type named {name} as a SEQUENCE. "
                f"List the names of its top-level fields — the immediate members of that "
                f"SEQUENCE, not the members of any nested CHOICE or SEQUENCE — in the order "
                f"they appear in the definition."
            ),
            "gold": d["fields"],
            "spec_id": d["spec_id"],
            "version": d["version"],
            "section": d["section"],
            "section_title": d["section_title"],
        })
    return tasks


def formula_tasks(conn, n, rng):
    """One task per display equation that a sentence introduces, using that
    sentence as the stem so the task is stated in the document's own words."""
    cands = []
    rows = conn.execute(
        "SELECT spec_id,version,number,title,content FROM sections WHERE content LIKE '%$$%'"
    )
    for spec, version, number, title, content in rows:
        if not MODERN.match(spec):
            continue
        for m in DISPLAY_MATH.finditer(content):
            eq = m.group(1).strip()
            if not 8 <= len(eq) <= 120 or "\n" in eq:
                continue
            stem = content[: m.start()].rstrip().split("\n")[-1].strip()
            if len(stem) < 60 or stem.startswith("#") or "$$" in stem:
                continue
            # The stem has to read as the document's own sentence introducing
            # the equation. Tables and figure captions surround equations too,
            # and make an unanswerable task.
            if "<" in stem or "|" in stem or not stem.endswith((":", ".", "by", "as")):
                continue
            # And the equation has to be one: a relation, with some structure
            # to reproduce rather than a bare arithmetic result.
            if not RELATION.search(eq) or not STRUCTURE.search(eq):
                continue
            # Only equations that are the single display equation of their
            # section, so "the equation introduced by this sentence" is unique.
            if len(DISPLAY_MATH.findall(content)) > 6:
                continue
            cands.append(dict(spec_id=spec, version=version, section=number,
                              section_title=title, stem=stem, eq=eq))
    rng.shuffle(cands)

    # The RF test specifications copy whole measurement clauses between each
    # other, so the same sentence introduces the same equation in several
    # documents. Such a task has more than one correct citation; drop it.
    stem_count = collections.Counter(d["stem"] for d in cands)
    eq_count = collections.Counter(d["eq"] for d in cands)

    tasks, seen = [], set()
    for d in cands:
        key = (d["spec_id"], d["section"])
        if key in seen or stem_count[d["stem"]] > 1 or eq_count[d["eq"]] > 1:
            continue
        seen.add(key)
        tasks.append({
            "id": f"formula-{len(tasks):02d}",
            "type": "formula",
            "question": (
                "A 3GPP specification contains this sentence:\n\n"
                f"    {d['stem']}\n\n"
                "Give the equation it introduces, in LaTeX."
            ),
            "gold": d["eq"],
            "spec_id": d["spec_id"],
            "version": d["version"],
            "section": d["section"],
            "section_title": d["section_title"],
        })
        if len(tasks) == n:
            break
    return tasks


class _TableParser(HTMLParser):
    """Rows of the HTML tables the DOCX conversion produces."""

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
                self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def html_rows(content):
    p = _TableParser()
    p.feed(content)
    return p.rows


# The registries a protocol defines its wire codes in. Each is one HTML table
# whose first two columns are the code and the name; Diameter adds a data type.
# These are the values an implementation has to get exactly right, and the ones
# a model has no way to recall — which is what makes them worth asking.
CODE_TABLES = [
    dict(key="gtpv2c", spec="TS 29.274", section="8.1", protocol="GTPv2-C",
         unit="information element", label="IE Type value", typed=False),
    dict(key="gtpv2c", spec="TS 29.274", section="6.1.0", protocol="GTPv2-C",
         unit="message", label="Message Type value", typed=False),
    dict(key="pfcp", spec="TS 29.244", section="8.1.2", protocol="PFCP",
         unit="information element", label="IE Type value", typed=False),
    dict(key="pfcp", spec="TS 29.244", section="7.3", protocol="PFCP",
         unit="message", label="Message Type value", typed=False),
    dict(key="diameter", spec="TS 29.230", section="7.1", protocol="Diameter",
         unit="AVP", label="AVP Code", typed=True),
]

# Placeholder rows carry no information and would make an unanswerable task.
SKIP_NAME = re.compile(r"(?i)^\s*(reserved|spare|unassigned|for future use|.*reserved for)")
DATA_TYPES = {"OctetString", "Integer32", "Integer64", "Unsigned32", "Unsigned64",
              "Float32", "Float64", "Grouped", "Address", "Time", "UTF8String",
              "DiameterIdentity", "DiameterURI", "Enumerated", "IPFilterRule",
              "QoSFilterRule"}


def _code_rows(conn, entry):
    """(code, name, data type) for the usable rows of one registry table."""
    row = conn.execute(
        "SELECT version, title, content FROM sections WHERE spec_id=? AND number=?",
        (entry["spec"], entry["section"]),
    ).fetchone()
    if not row:
        return None, []
    version, title, content = row
    out = []
    for cells in html_rows(content):
        if len(cells) < 2:
            continue
        code, name = cells[0].strip(), cells[1].strip()
        if not re.fullmatch(r"\d+", code) or not name or SKIP_NAME.match(name):
            continue
        dtype = cells[2].strip() if entry["typed"] and len(cells) > 2 else ""
        if entry["typed"] and dtype not in DATA_TYPES:
            dtype = ""
        out.append((code, name, dtype))
    # An ambiguous code or name has more than one right answer; drop both sides.
    by_code = collections.Counter(c for c, _, _ in out)
    by_name = collections.Counter(n.lower() for _, n, _ in out)
    out = [r for r in out if by_code[r[0]] == 1 and by_name[r[1].lower()] == 1]
    return (version, title), out


def code_tasks(conn, n, rng, key=None):
    """Wire codes, asked in both directions so neither is a one-way lookup."""
    cands = []
    for entry in CODE_TABLES:
        if key and entry["key"] != key:
            continue
        meta, rows = _code_rows(conn, entry)
        if not rows:
            continue
        version, title = meta
        for code, name, dtype in rows:
            base = dict(spec_id=entry["spec"], version=version, section=entry["section"],
                        section_title=title, answer_kind="scalar", type="code")
            cands.append(dict(base, id=f"{entry['key']}-code-{code}", gold=code, question=(
                f"In {entry['protocol']}, what is the {entry['label']} (decimal) of the "
                f"{name!r} {entry['unit']}?")))
            cands.append(dict(base, id=f"{entry['key']}-name-{code}", gold=name, question=(
                f"In {entry['protocol']}, which {entry['unit']} has {entry['label']} "
                f"{code} (decimal)? Give its name as the specification writes it.")))
            if dtype:
                cands.append(dict(base, id=f"{entry['key']}-type-{code}", gold=dtype, question=(
                    f"What is the data type of the Diameter AVP {name!r}?")))
    rng.shuffle(cands)
    return cands[:n]


# A schema body indented four spaces under "  schemas:", and its required list.
SCHEMA_RE = re.compile(r"^    ([A-Za-z][\w]*):\n((?:      .*\n|\n)*)", re.M)
REQUIRED_RE = re.compile(r"^      required:\n((?:        - \w+\n)+)", re.M)


def _openapi_schemas(conn):
    """{schema name: [(spec, api, required properties)]} over every API file."""
    owners = collections.defaultdict(list)
    for spec, api, content in conn.execute("SELECT spec_id, api_name, content FROM openapi_specs"):
        start = content.find("  schemas:")
        if start < 0:
            continue
        for m in SCHEMA_RE.finditer(content[start:]):
            rm = REQUIRED_RE.search(m.group(2))
            if not rm:
                continue
            props = [line.strip("- \n") for line in rm.group(1).strip().splitlines()]
            if 2 <= len(props) <= 8:
                owners[m.group(1)].append((spec, api, props))
    return owners


def openapi_tasks(conn, n, rng):
    """Required properties of a 5G SBI schema.

    These definitions are not in the full-text index at all — they live in their
    own table, reachable only through list_openapi/get_openapi — so this is the
    one task type a search-only baseline cannot reach.
    """
    owners = _openapi_schemas(conn)
    unique = sorted((k, v[0]) for k, v in owners.items() if len(v) == 1)
    rng.shuffle(unique)

    tasks = []
    for name, (spec, api, props) in unique[:n]:
        tasks.append({
            "id": f"openapi-{api}-{name}",
            "type": "openapi",
            "answer_kind": "set",
            "question": (
                f"In the 3GPP 5G service based interface APIs, the OpenAPI schema {name!r} "
                f"declares a list of required properties. List them."
            ),
            "gold": props,
            "spec_id": spec,
            "version": "",
            "section": api,
            "section_title": api,
            "api_name": api,
        })
    return tasks


def verify(conn, tasks):
    """Re-derive every gold from the database and report what does not match.

    An earlier version of the ASN.1 generator captured nested CHOICE members as
    fields of the outer type, so a model answering correctly was scored wrong
    and the benchmark understated the tool by tens of points. Gold that cannot
    be re-derived from the document is not gold.
    """
    bad = []
    for t in tasks:
        kind, gold = t["type"], t["gold"]
        if kind == "code":
            entry = next(e for e in CODE_TABLES
                         if e["spec"] == t["spec_id"] and e["section"] == t["section"])
            _, rows = _code_rows(conn, entry)
            code = t["id"].rsplit("-", 1)[-1]
            row = next((r for r in rows if r[0] == code), None)
            want = {"code": code, "name": row[1] if row else None,
                    "type": row[2] if row else None}["-".join(t["id"].split("-")[1:2])]
            if row is None or want != gold:
                bad.append((t["id"], gold, want))
        elif kind == "openapi":
            row = conn.execute(
                "SELECT content FROM openapi_specs WHERE spec_id=? AND api_name=?",
                (t["spec_id"], t["api_name"]),
            ).fetchone()
            ok = False
            if row:
                start = row[0].find("  schemas:")
                name = t["id"].rsplit("-", 1)[-1]
                for m in SCHEMA_RE.finditer(row[0][start:]):
                    if m.group(1) != name:
                        continue
                    rm = REQUIRED_RE.search(m.group(2))
                    ok = bool(rm) and [l.strip("- \n") for l in rm.group(1).strip().splitlines()] == gold
                if not ok:
                    bad.append((t["id"], gold, "not re-derivable"))
        else:
            row = conn.execute(
                "SELECT content FROM sections WHERE spec_id=? AND number=?",
                (t["spec_id"], t["section"]),
            ).fetchone()
            if not row:
                bad.append((t["id"], gold, "section missing"))
                continue
            if kind == "formula":
                if re.sub(r"\s+", "", gold) not in re.sub(r"\s+", "", row[0]):
                    bad.append((t["id"], gold, "equation not in the section"))
            elif kind == "asn1":
                name = t["id"][len("asn1-"):]
                found = None
                for block in ASN1_FENCE.findall(row[0]):
                    for m in ASN1_HEAD.finditer(block):
                        if m.group(1) == name:
                            found = top_level_fields(block, m.end() - 1)
                if found != gold:
                    bad.append((t["id"], gold, found))
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True, help="pinned 3gpp-mcp database")
    ap.add_argument("--out", default="bench", help="directory to write the task files to")
    ap.add_argument("--n", type=int, default=30, help="tasks per type")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify", action="store_true",
                    help="re-derive every gold from the database and fail if any does not match")
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    os.makedirs(args.out, exist_ok=True)
    generators = [("asn1", asn1_tasks), ("formula", formula_tasks), ("openapi", openapi_tasks)]
    # One file per protocol: a single "code" pool would be swamped by the
    # Diameter registry, which alone has 1719 usable rows.
    for key in sorted({e["key"] for e in CODE_TABLES}):
        generators.append((key, lambda c, n, r, k=key: code_tasks(c, n, r, key=k)))
    failures = 0
    for name, fn in generators:
        tasks = fn(conn, args.n, random.Random(args.seed))
        for t in tasks:
            t.setdefault("answer_kind", {"asn1": "sequence", "formula": "latex"}[t["type"]]
                         if t["type"] in ("asn1", "formula") else "scalar")
        path = os.path.join(args.out, f"tasks-{name}.json")
        with open(path, "w") as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"{path}: {len(tasks)} tasks", file=sys.stderr)
        specs = collections.Counter(t["spec_id"] for t in tasks)
        print(f"  specifications: {dict(specs.most_common(6))}", file=sys.stderr)
        if args.verify:
            bad = verify(conn, tasks)
            failures += len(bad)
            print(f"  verified: {len(tasks) - len(bad)}/{len(tasks)}", file=sys.stderr)
            for tid, gold, got in bad[:5]:
                print(f"    MISMATCH {tid}: gold={gold!r} re-derived={got!r}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
