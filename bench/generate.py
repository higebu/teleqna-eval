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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True, help="pinned 3gpp-mcp database")
    ap.add_argument("--out", default="bench", help="directory to write the task files to")
    ap.add_argument("--n", type=int, default=30, help="tasks per type")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    os.makedirs(args.out, exist_ok=True)
    for name, fn in (("asn1", asn1_tasks), ("formula", formula_tasks)):
        tasks = fn(conn, args.n, random.Random(args.seed))
        path = os.path.join(args.out, f"tasks-{name}.json")
        with open(path, "w") as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"{path}: {len(tasks)} tasks", file=sys.stderr)
        specs = collections.Counter(t["spec_id"] for t in tasks)
        print(f"  specifications: {dict(specs.most_common(6))}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
