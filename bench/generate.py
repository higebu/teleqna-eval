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

The task types:

  asn1      given an ASN.1 type name, name its fields.      gold = field names
  formula   given the sentence that introduces an equation,  gold = LaTeX
            reproduce the equation.
  gtpv2c    a wire code or the element that carries it, in   gold = code or name
  pfcp      either direction, from a protocol's registry.
  diameter
  ngapies   which IEs of a named NGAP/S1AP message are       gold = IE names
            mandatory.
  ngapasn1  a constraint the ASN.1 of NGAP/S1AP puts on an   gold = a bound,
            IE the question names through a message.         a size or a value set
  openapi-* the 5G SBI schemas, in openapi_tasks.py.

Every type also requires the answer to say which specification and section it
came from, which is scored separately: being right for the wrong reason is not
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openapi_tasks  # noqa: E402  the 5G SBI generators, which need a YAML parser

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
# Since 3gpp-mcp emits standalone equations as fenced latex, most display maths
# is a fence; $$...$$ survives where a fence cannot go, such as a table cell.
LATEX_FENCE = re.compile(r"```latex\n(.+?)\n```", re.S)


def display_equations(content):
    """(equation, offset) for every standalone equation, in either notation."""
    for pattern in (LATEX_FENCE, DISPLAY_MATH):
        for m in pattern.finditer(content):
            yield m.group(1).strip(), m.start()
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
        "SELECT spec_id,version,number,title,content FROM sections "
        "WHERE content LIKE '%```latex%' OR content LIKE '%$$%'"
    )
    for spec, version, number, title, content in rows:
        if not MODERN.match(spec):
            continue
        equations = list(display_equations(content))
        for eq, offset in equations:
            if not 8 <= len(eq) <= 120 or "\n" in eq:
                continue
            stem = content[:offset].rstrip().split("\n")[-1].strip()
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
            # Only sections with a handful of equations, so "the equation this
            # sentence introduces" has one answer.
            if len(equations) > 6:
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


# The radio-access application protocols. Their shape is the reason these tasks
# exist: a message is a table of information elements, each IE is its own
# clause, and the ASN.1 for every IE of the protocol sits in *one* clause of a
# few hundred kilobytes. So the document that holds an answer is not the
# document the question names, and the clause that holds it is far too large to
# read whole — 270 KB against the 16 KB a tool result is truncated to.
#
# That is the opposite of the `asn1` type, which is generated from RRC: TS 38.331
# gives each type its own small clause, so one retrieval holds the answer.
AP_PROTOCOLS = [
    dict(key="ngap", protocol="NGAP", spec="TS 38.413",
         messages="9.2.", ies="9.3.", asn1="9.4.5"),
    dict(key="s1ap", protocol="S1AP", spec="TS 36.413",
         messages="9.1.", ies="9.2.", asn1="9.3.4"),
]

# The head of an ASN.1 assignment at column 0. The body runs to the next one:
# stopping at the first line that starts at column 0 would cut an ENUMERATED
# off at its closing brace, which is where its values are.
AP_ASSIGN = re.compile(r"^([A-Za-z][\w-]*)\s*::=", re.M)
AP_INT = re.compile(r"^INTEGER\s*\(\s*(-?\d+)\s*\.\.\s*(\d+)")
AP_SIZE = re.compile(r"^(BIT STRING|OCTET STRING)\s*\(\s*SIZE\s*\(\s*(\d+)\s*\)\s*\)")
AP_ENUM = re.compile(r"^ENUMERATED\s*\{(.*?)\}", re.S)
AP_ENUM_VALUE = re.compile(r"[a-zA-Z][\w-]*")
AP_CLAUSE_REF = re.compile(r"\b\d+(?:\.\d+)+\b")
# The IE table every message clause and every IE clause opens with.
AP_TABLE_HEAD = "IE/Group Name"


def _ap_key(s):
    """A name reduced to what the two notations agree on: the IE clause titles
    it "AMF UE NGAP ID" and the ASN.1 calls it "AMF-UE-NGAP-ID"."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _ap_assignments(text):
    """{name: [body, ...]} for one ASN.1 clause. A name defined twice keeps both,
    so the generator can drop it rather than pick one."""
    out, heads = {}, list(AP_ASSIGN.finditer(text))
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        out.setdefault(m.group(1), []).append(text[m.end():end].strip())
    return out


def _ap_ie_tables(conn, entry, prefix):
    """Clauses under `prefix` that open with an IE table, as
    {number: (title, content, rows)}."""
    out = {}
    for num, title, content in conn.execute(
        "SELECT number,title,content FROM sections WHERE spec_id=? AND number LIKE ? "
        "ORDER BY number", (entry["spec"], prefix + "%")
    ):
        rows = html_rows(content)
        if rows and rows[0][:1] == [AP_TABLE_HEAD]:
            out[num] = (title, content, rows)
    return out


def _ap_row_name(cells):
    """The IE/Group Name of a table row, and whether it is a top-level row.

    Nesting is written with leading '>' characters, one per level, so a row's
    depth is in its name rather than in the markup.
    """
    name = re.sub(r"\s+", " ", cells[0]).strip()
    return name.lstrip("> ").strip(), not name.startswith(">")


def _ap_corpus(conn, entry):
    """Everything one protocol's tasks are derived from, read once."""
    messages = _ap_ie_tables(conn, entry, entry["messages"])
    ies = _ap_ie_tables(conn, entry, entry["ies"])
    row = conn.execute("SELECT version,title,content FROM sections WHERE spec_id=? AND number=?",
                       (entry["spec"], entry["asn1"])).fetchone()
    if row is None:
        raise SystemExit(f"{entry['spec']}: no clause {entry['asn1']} in this database")
    version, asn1_title, asn1 = row[0], row[1], _ap_assignments(row[2])
    # A message a model could not name unambiguously, or an IE title that two
    # clauses share, has more than one right answer. Drop both sides.
    msg_titles = collections.Counter(t for t, _, _ in messages.values())
    ie_titles = collections.Counter(_ap_key(t) for t, _, _ in ies.values())
    ie_by_title = {_ap_key(t): num for num, (t, _, _) in ies.items()
                   if ie_titles[_ap_key(t)] == 1}
    return dict(entry, messages=messages, ies=ies, asn1=asn1, version=version,
                asn1_title=asn1_title, msg_titles=msg_titles, ie_by_title=ie_by_title)


def _ap_mandatory(rows):
    """The top-level IEs a message table marks Presence 'M', in table order."""
    out = []
    for cells in rows[1:]:
        name, top = _ap_row_name(cells)
        if not top or len(cells) < 2 or cells[1].strip() != "M" or not name:
            continue
        out.append(name)
    return out


def ap_mandatory_tasks(conn, n, rng):
    """Which IEs of a named message are mandatory.

    One clause holds the answer, but it holds it as a table of up to 40 rows at
    several nesting depths, and what is asked for is the subset satisfying a
    predicate. Nothing here has to be composed across documents — which is the
    point of running it beside the ASN.1 type below: the two are designed to
    need a different number of hops, and the one-call condition is what says
    whether they got them.
    """
    cands, answered = [], set()
    for entry in AP_PROTOCOLS:
        c = _ap_corpus(conn, entry)
        for num, (title, content, rows) in sorted(c["messages"].items()):
            if c["msg_titles"][title] > 1:
                continue
            mandatory = _ap_mandatory(rows)
            # One mandatory IE is a lookup, and a table of forty is a
            # transcription exercise rather than a question.
            if not 2 <= len(mandatory) <= 10:
                continue
            if len(set(_ap_key(m) for m in mandatory)) != len(mandatory):
                continue
            # Most NGAP messages are mandatory in exactly the same three IEs, so
            # a pool drawn straight from the clause list asks one question
            # eighteen times and rewards a model that answers from the pattern
            # without opening anything. One task per distinct answer.
            key = (entry["key"], tuple(_ap_key(m) for m in mandatory))
            if key in answered:
                continue
            answered.add(key)
            cands.append({
                "id": f"{entry['key']}-mandatory-{num}",
                "type": "ngapies",
                "answer_kind": "set",
                "question": (
                    f"The {entry['protocol']} {title} message is specified as a table of "
                    f"information elements. List the IEs that table marks as mandatory — "
                    f"Presence \"M\" — counting only the rows at the top level of the "
                    f"table and not the members of any nested group or list. Give each "
                    f"one as the IE/Group Name column writes it."
                ),
                "gold": mandatory,
                "spec_id": entry["spec"],
                "version": c["version"],
                "section": num,
                "section_title": title,
                "probe": {"kind": "ngap-mandatory", "message": num},
            })
    rng.shuffle(cands)
    return cands[:n]


def _ap_asn1_shape(body):
    """What one ASN.1 assignment states, when it states something a question can
    have a single answer to: an upper bound, a fixed size, or a value set."""
    head = body.splitlines()[0].strip() if body else ""
    if (m := AP_INT.match(head)):
        return "int", m.group(2), None
    if (m := AP_SIZE.match(head)):
        return "size", m.group(2), m.group(1)
    if (m := AP_ENUM.match(body)):
        values = [v for v in (s.strip() for s in re.split(r"[,\n]", m.group(1)))
                  if AP_ENUM_VALUE.fullmatch(v)]
        # A single-valued ENUMERATED carries no information to ask for, and the
        # extension marker is not one of the values.
        if len(values) >= 2:
            return "enum", values, None
    return None, None, None


def ap_asn1_tasks(conn, n, rng):
    """An ASN.1 constraint on an IE, asked through a message that carries it.

    The question names a message and an IE by the words the message table uses;
    the answer is in the protocol's ASN.1 clause, under a name the question does
    not contain, in a notation the IE clause does not always use — the IE clause
    writes `INTEGER (0..2^40 -1)` where the ASN.1 writes `1099511627775`. The
    gold is the ASN.1 one, so the question asks for a decimal and reaching it
    means reaching that clause.
    """
    cands = []
    for entry in AP_PROTOCOLS:
        c = _ap_corpus(conn, entry)
        seen = set()
        for num, (title, content, rows) in sorted(c["messages"].items()):
            if c["msg_titles"][title] > 1:
                continue
            names = collections.Counter(_ap_row_name(cells)[0] for cells in rows[1:])
            for cells in rows[1:]:
                name, _ = _ap_row_name(cells)
                if not name or names[name] > 1 or _ap_key(name) in seen:
                    continue
                ref = cells[3] if len(cells) > 3 else ""
                m = AP_CLAUSE_REF.search(ref)
                if not m or m.group(0) not in c["ies"]:
                    continue
                ie_num = m.group(0)
                ie_title = c["ies"][ie_num][0]
                # The row has to name the IE the way its own clause titles it.
                # Where a message renames an IE, following the reference is a
                # step the question cannot state, and the task would be asking
                # about something it did not name.
                if _ap_key(ie_title) != _ap_key(name):
                    continue
                if c["ie_by_title"].get(_ap_key(ie_title)) != ie_num:
                    continue
                bodies = c["asn1"].get(_ap_typename(c, ie_title), [])
                if len(bodies) != 1:
                    continue
                kind, value, unit = _ap_asn1_shape(bodies[0])
                if not kind:
                    continue
                seen.add(_ap_key(name))
                cands.append(_ap_asn1_task(conn, entry, c, num, title, name,
                                           ie_num, kind, value, unit))
    rng.shuffle(cands)
    return cands[:n]


def _ap_typename(c, ie_title):
    """The ASN.1 assignment an IE clause title maps to, if exactly one does."""
    hits = [k for k in c["asn1"] if _ap_key(k) == _ap_key(ie_title)]
    return hits[0] if len(hits) == 1 else None


def _ap_asn1_task(conn, entry, c, msg_num, msg_title, ie_name, ie_num, kind, value, unit):
    proto = entry["protocol"]
    stem = (f"The {proto} {msg_title} message carries an information element named "
            f"\"{ie_name}\".")
    if kind == "int":
        question = (f"{stem} In the ASN.1 with which {proto} defines that IE, what is the "
                    f"largest value it may take? Answer with a decimal integer.")
        gold, answer_kind = value, "scalar"
    elif kind == "size":
        held = "bits" if unit == "BIT STRING" else "octets"
        question = (f"{stem} In the ASN.1 with which {proto} defines that IE, it is a "
                    f"{unit} of a fixed size. How many {held} is it? Answer with a "
                    f"decimal integer.")
        gold, answer_kind = value, "scalar"
    else:
        question = (f"{stem} In the ASN.1 with which {proto} defines that IE, it is an "
                    f"ENUMERATED. List its values, spelled as the ASN.1 spells them, "
                    f"leaving out the extension marker.")
        gold, answer_kind = value, "set"
    # Whether the IE's own clause already states the answer decides how far the
    # walk really has to go, and it is recorded rather than filtered on: a task
    # is not made multi-hop by asserting that it is.
    plain = re.sub(r"<[^>]+>", " ", c["ies"][ie_num][1])
    wanted = gold if isinstance(gold, list) else [gold]
    asn1_only = not all(re.search(r"\b" + re.escape(w) + r"\b", plain) for w in wanted)
    return {
        "id": f"{entry['key']}-asn1-{_ap_key(ie_name)}",
        "type": "ngapasn1",
        "answer_kind": answer_kind,
        "question": question,
        "gold": gold,
        "spec_id": entry["spec"],
        "version": c["version"],
        # The ASN.1 clause is where the value asked for is written, so it is the
        # citation the task was generated from. The IE clause counts too when it
        # states the same value, which the grader decides against the document.
        "section": entry["asn1"],
        "section_title": c["asn1_title"],
        "stratum": "asn1-only" if asn1_only else "ie-clause-too",
        "probe": {"kind": f"ngap-{kind}", "message": msg_num, "ie": ie_num,
                  "asn1": entry["asn1"], "type_name": _ap_typename(c, c["ies"][ie_num][0]),
                  "asn1_only": asn1_only},
    }


def ap_rederive(conn, task):
    """Re-derive an NGAP/S1AP gold from the database, for --verify."""
    entry = next(e for e in AP_PROTOCOLS if e["spec"] == task["spec_id"])
    c = _ap_corpus(conn, entry)
    probe = task["probe"]
    if probe["kind"] == "ngap-mandatory":
        msg = c["messages"].get(probe["message"])
        return _ap_mandatory(msg[2]) if msg else None
    bodies = c["asn1"].get(probe["type_name"], [])
    if len(bodies) != 1:
        return None
    return _ap_asn1_shape(bodies[0])[1]


_OPENAPI_STORE = None


def _openapi_store(conn):
    """Parsing 477 YAML documents takes a few seconds, so do it once."""
    global _OPENAPI_STORE
    if _OPENAPI_STORE is None:
        _OPENAPI_STORE = openapi_tasks.Store(conn)
    return _OPENAPI_STORE


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
            store = _openapi_store(conn)
            got = openapi_tasks.rederive(store, t)
            if got != gold:
                bad.append((t["id"], gold, got))
        elif kind in ("ngapies", "ngapasn1"):
            got = ap_rederive(conn, t)
            if got != gold:
                bad.append((t["id"], gold, got))
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
    ap.add_argument("--only", default="",
                    help="comma-separated task names to regenerate; the others are left "
                         "alone. The asn1 and formula scans read every clause in the "
                         "corpus, so iterating on one generator is much faster with this")
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    os.makedirs(args.out, exist_ok=True)
    generators = [("asn1", asn1_tasks), ("formula", formula_tasks),
                  ("ngapies", ap_mandatory_tasks), ("ngapasn1", ap_asn1_tasks)]
    for name, fn in openapi_tasks.GENERATORS.items():
        generators.append((name, lambda c, n, r, f=fn: f(_openapi_store(c), n, r)))
    # One file per protocol: a single "code" pool would be swamped by the
    # Diameter registry, which alone has 1719 usable rows.
    for key in sorted({e["key"] for e in CODE_TABLES}):
        generators.append((key, lambda c, n, r, k=key: code_tasks(c, n, r, key=k)))
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    if only:
        unknown = only - {name for name, _ in generators}
        if unknown:
            print(f"--only: no such task type: {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
        generators = [(name, fn) for name, fn in generators if name in only]
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
