#!/usr/bin/env python3
"""Task generators for the 5G service-based interface OpenAPI definitions.

The first version of this track asked for the required properties of a named
schema. That question is answered by opening one document and reading one
block, which is not what makes 5G SBI definitions hard: 78% of their $refs
point at *another* file, and the schema an engineer actually needs is usually
assembled out of several of them.

These generators ask the questions the structure poses:

  request   the top-level properties of an operation's request body
  object    the fields of the object a property holds, one $ref away
  allof     the property set a schema ends up with after its allOf is merged
  oneof     the alternatives a oneOf/anyOf schema may take
  describe  one property set, asked three ways over the same schemas: from a
  named     description alone, from the schema name, and from the name together
  lookup    with the document that declares it

All four of the first group hand the model an identifier *and* the document it
is in, so none of them requires the target to be found. The describe/named/
lookup ladder is what varies that; see its section below.

`object` and `allof` are stratified by whether resolving the reference stays
inside the file or crosses into another document, because that is the split a
single retrieval query cannot serve: it can prepend the chunk that was matched,
not the chunk that chunk points at.

Gold comes out of the parsed document, so there is no labelling step. Every
generator has a matching re-derivation used by `generate.py --verify`.

There is no clause number in an OpenAPI file, so the citation these tasks ask
for is the specification and the API document — `TS 29.510` /
`Nnrf_NFManagement` — and not a section.
"""

import collections
import math
import re

import yaml

# A schema is usable as gold when its property set is unambiguous and small
# enough to write out. oneOf/anyOf mean "one of these shapes", so the merged
# property set is not defined and those schemas are only used by oneof_tasks.
MIN_PROPS, MAX_PROPS = 3, 12
# oneOf/anyOf alternatives per task.
MIN_ALTS, MAX_ALTS = 2, 6
# At most this many tasks from one API document, so no single file dominates a
# task type. Two rather than one because the allOf and oneOf pools are spread
# over 53 and 38 documents respectively, and a cap of one would not fill 50.
MAX_PER_API = 2
# `anyOf: [X, NullValue]` is the removable-attribute idiom, present in 56 of the
# 104 usable oneOf/anyOf schemas and guessable from the schema's own "Rm" suffix
# without opening anything. It would flatter the no-tools baseline, so it is not
# asked about.
NULLABLE = "NullValue"

REF_RE = re.compile(r"^(?P<file>[^#]*)#/components/schemas/(?P<name>[\w.-]+)$")


class Store:
    """Every OpenAPI document in the pinned database, parsed and cross-linked."""

    def __init__(self, conn):
        self.by_file = {}
        for spec, api, filename, content in conn.execute(
            "SELECT spec_id, api_name, filename, content FROM openapi_specs ORDER BY filename"
        ):
            try:
                doc = yaml.safe_load(content)
            except yaml.YAMLError:
                continue
            if not isinstance(doc, dict):
                continue
            self.by_file[filename] = {"spec": spec, "api": api, "file": filename, "doc": doc}

    def __iter__(self):
        return iter(self.by_file.values())

    def schemas(self, entry):
        comp = entry["doc"].get("components") or {}
        s = comp.get("schemas")
        return s if isinstance(s, dict) else {}

    def resolve(self, entry, ref):
        """Follow a $ref to (entry, schema name, schema). None if it leaves the store."""
        m = REF_RE.match(ref or "")
        if not m:
            return None
        target = entry if not m["file"] else self.by_file.get(m["file"])
        if target is None:
            return None
        schema = self.schemas(target).get(m["name"])
        if not isinstance(schema, dict):
            return None
        return target, m["name"], schema

    def crosses_file(self, entry, ref):
        m = REF_RE.match(ref or "")
        return bool(m and m["file"] and m["file"] != entry["file"])


def _properties(schema):
    p = schema.get("properties")
    return p if isinstance(p, dict) else {}


def merged_properties(store, entry, schema, depth=0):
    """The top-level property names of a schema, with allOf merged.

    allOf is an AND, so the union is well defined; oneOf and anyOf are not, and
    a schema carrying either returns None rather than a guess.
    """
    if depth > 4 or not isinstance(schema, dict):
        return None
    if schema.get("oneOf") or schema.get("anyOf"):
        return None
    names = set(_properties(schema))
    for member in schema.get("allOf") or []:
        if not isinstance(member, dict):
            return None
        if "$ref" in member:
            got = store.resolve(entry, member["$ref"])
            if got is None:
                return None
            sub = merged_properties(store, got[0], got[2], depth + 1)
        else:
            sub = merged_properties(store, entry, member, depth + 1)
        if sub is None:
            return None
        names |= set(sub)
    return sorted(names)


def _usable(names):
    return names is not None and MIN_PROPS <= len(names) <= MAX_PROPS


def _task(entry, tid, question, gold, probe, kind="set", stratum=None):
    """One task. `probe` records what the gold was derived from, in fields
    rather than in the id, so --verify re-derives from data and never from a
    parse of the identifier."""
    t = {
        "id": tid,
        "type": "openapi",
        "answer_kind": kind,
        "question": question,
        "gold": gold,
        "spec_id": entry["spec"],
        "version": "",
        "section": entry["api"],
        "section_title": entry["api"],
        "api_name": entry["api"],
        "probe": probe,
    }
    if stratum:
        t["stratum"] = stratum
    return t


# --------------------------------------------------------------------------
# request: what a client actually sends

METHODS = ("get", "put", "post", "patch", "delete")


def _operations(store, entry):
    paths = entry["doc"].get("paths")
    if not isinstance(paths, dict):
        return
    for path, item in paths.items():
        if not isinstance(item, dict):
            continue
        for method in METHODS:
            op = item.get(method)
            if isinstance(op, dict):
                yield path, method, op


def _request_schema(store, entry, op):
    body = op.get("requestBody")
    if not isinstance(body, dict):
        return None
    if "$ref" in body:  # a shared requestBody object, not a schema
        return None
    for media in (body.get("content") or {}).values():
        schema = (media or {}).get("schema")
        if isinstance(schema, dict) and "$ref" in schema:
            return store.resolve(entry, schema["$ref"])
    return None


def request_tasks(store, n, rng):
    pool = []
    for entry in store:
        for path, method, op in _operations(store, entry):
            got = _request_schema(store, entry, op)
            if got is None:
                continue
            target, name, schema = got
            props = merged_properties(store, target, schema)
            if not _usable(props):
                continue
            pool.append((entry, path, method, name, props))
    pool.sort(key=lambda x: (x[0]["file"], x[1], x[2]))
    rng.shuffle(pool)

    seen, tasks = collections.Counter(), []
    for entry, path, method, name, props in pool:
        if seen[entry["api"]] >= MAX_PER_API:  # spread the sample over documents
            continue
        seen[entry["api"]] += 1
        tasks.append(_task(
            entry,
            f"openapi-request-{entry['api']}-{method}-{name}",
            f"In the 3GPP 5G service based interface API {entry['api']!r} "
            f"({entry['spec']}), the operation {method.upper()} {path} takes a request "
            f"body. List the top-level properties of the schema that body carries. "
            f"Do not list path, query or header parameters.",
            props,
            {"kind": "request", "path": path, "method": method, "schema": name},
        ))
        if len(tasks) == n:
            break
    return tasks


# --------------------------------------------------------------------------
# object: one hop through a $ref

def object_tasks(store, n, rng):
    local, cross = [], []
    for entry in store:
        for owner, schema in store.schemas(entry).items():
            if not isinstance(schema, dict):
                continue
            for prop, value in _properties(schema).items():
                ref = value.get("$ref") if isinstance(value, dict) else None
                if not ref:
                    continue
                got = store.resolve(entry, ref)
                if got is None:
                    continue
                target, name, sub = got
                props = merged_properties(store, target, sub)
                if not _usable(props):
                    continue
                item = (entry, owner, prop, name, props)
                (cross if store.crosses_file(entry, ref) else local).append(item)
    return _stratified(
        local, cross, n, rng,
        lambda e, owner, prop, name, props, stratum: _task(
            e,
            f"openapi-object-{e['api']}-{owner}-{prop}",
            f"In the 3GPP 5G service based interface API {e['api']!r} ({e['spec']}), the "
            f"schema {owner!r} has a property {prop!r} that holds an object. List the "
            f"top-level fields of that object.",
            props, {"kind": "object", "owner": owner, "property": prop, "schema": name},
            stratum=stratum,
        ),
    )


# --------------------------------------------------------------------------
# allof: the shape a schema ends up with

def allof_tasks(store, n, rng):
    local, cross = [], []
    for entry in store:
        for name, schema in store.schemas(entry).items():
            if not isinstance(schema, dict):
                continue
            members = schema.get("allOf")
            if not isinstance(members, list) or len(members) < 2:
                continue
            props = merged_properties(store, entry, schema)
            if not _usable(props):
                continue
            refs = [m.get("$ref") for m in members if isinstance(m, dict) and m.get("$ref")]
            item = (entry, name, len(members), props)
            (cross if any(store.crosses_file(entry, r) for r in refs) else local).append(item)
    return _stratified(
        local, cross, n, rng,
        lambda e, name, count, props, stratum: _task(
            e,
            f"openapi-allof-{e['api']}-{name}",
            f"In the 3GPP 5G service based interface API {e['api']!r} ({e['spec']}), the "
            f"schema {name!r} is composed with allOf of {count} members. List the "
            f"top-level properties the composed schema ends up with, merging every "
            f"member.",
            props, {"kind": "allof", "schema": name}, stratum=stratum,
        ),
    )


# --------------------------------------------------------------------------
# oneof: what the value may be

def oneof_tasks(store, n, rng):
    pool = []
    for entry in store:
        for name, schema in store.schemas(entry).items():
            if not isinstance(schema, dict):
                continue
            # A schema declares at most one of the two in practice; take the
            # first that is present rather than emitting the same schema twice.
            key = next((k for k in ("oneOf", "anyOf") if isinstance(schema.get(k), list)), None)
            if key is None:
                continue
            members = schema[key]
            if not MIN_ALTS <= len(members) <= MAX_ALTS:
                continue
            alts = []
            for m in members:
                if not isinstance(m, dict) or "$ref" not in m:
                    alts = []
                    break
                got = store.resolve(entry, m["$ref"])
                if got is None:
                    alts = []
                    break
                alts.append(got[1])
            if len(alts) != len(members) or len(set(alts)) != len(alts):
                continue
            if NULLABLE in alts:
                continue
            pool.append((entry, name, key, sorted(alts)))
    pool.sort(key=lambda x: (x[0]["file"], x[1]))
    rng.shuffle(pool)

    seen, tasks = collections.Counter(), []
    for entry, name, key, alts in pool:
        if seen[entry["api"]] >= MAX_PER_API:
            continue
        seen[entry["api"]] += 1
        tasks.append(_task(
            entry,
            f"openapi-oneof-{entry['api']}-{name}",
            f"In the 3GPP 5G service based interface API {entry['api']!r} "
            f"({entry['spec']}), the schema {name!r} is defined with {key}. Name every "
            f"schema it may be.",
            alts, {"kind": "oneof", "schema": name, "key": key},
        ))
        if len(tasks) == n:
            break
    return tasks


def _stratified(local, cross, n, rng, build):
    """Half the tasks from each stratum, spread over API documents.

    The cross-file half is the one a single retrieval query cannot serve: the
    chunk it matches names a schema in another document, and prepending the
    match does not bring that document with it.
    """
    out, seen = [], collections.Counter()
    for bucket, stratum in ((cross, "cross-file"), (local, "same-file")):
        bucket.sort(key=lambda x: (x[0]["file"], str(x[1]), str(x[2])))
        rng.shuffle(bucket)
        want = n // 2 if stratum == "cross-file" else n - len(out)
        taken = 0
        for item in bucket:
            entry = item[0]
            if seen[entry["api"]] >= MAX_PER_API:
                continue
            seen[entry["api"]] += 1
            out.append(build(*item, stratum))
            taken += 1
            if taken == want:
                break
    return out[:n]


# --------------------------------------------------------------------------
# describe: the target fixed by what it is for, not by what it is called
#
# Every type above hands the model an identifier *and* the document that holds
# it, so fetching that document is a correct route and searching for it is only
# a substitute. Nothing in the set makes the model *find* an element. These
# three types are the ladder that does, over one pool of schemas:
#
#   lookup    the API document and the schema name, as the types above ask
#   named     the schema name, and not the document
#   describe  neither: the words the document uses to say what the schema is for
#
# One pool, so all three ask for the same gold about the same schemas and differ
# only in what the question gives away. `named` is the rung that has to be
# there. Without it, `describe` against `lookup` confounds two changes; with it,
# the first measurement said what the middle rung is for — describing costs
# nothing over naming, because a name with no document still has to be searched
# for, and it is the document that was doing the work all along.
#
# The uniqueness test is the whole of the work here. `--verify` re-derives a
# gold and drops a task whose answer moved, which checks that the *answer* is
# derivable, not that the *question* has one answer. A description that fits two
# schemas marks a model wrong for naming the other one, which is exactly the
# grading failure this benchmark has already been wrong in six times, always in
# the direction of understating the tool. So a candidate survives only if every
# check in `_Descriptions.reject` passes, and `--verify` re-runs all of them.

# A description shorter than this designates nothing; one longer than this is a
# specification clause pasted into a YAML field, not a description.
DESC_MIN, DESC_MAX = 40, 400
# The similarity above which a second description is close enough to make its
# schema a defensible second answer. IDF-weighted Jaccard over the words of the
# two descriptions: shared boilerplate ("represents", "contains the") is nearly
# free, and a shared rare word ("MTLF", "jCard") is most of the score. The
# corpus is full of X/XPatch and Req/Resp pairs that sit at 0.7-0.86, and pairs
# a reader would have to think about — `DataSamplingRule` against
# `DataReportingRule` — sit at 0.54.
DESC_SIM = 0.4
# A cross-reference is not a description of what the element is for.
DESC_XREF = re.compile(r"(?i)^\s*(see|refer|as (defined|specified)|according)\b")

# A description word this rare is a term rather than English, so finding it
# inside the schema name is the name leaking and not a coincidence. log(N/df)
# over the 5664 descriptions: 2.5 is a word in at most 8% of them, which keeps
# "request" and "information" out of the test and lets "ACR" and "MTLF" in.
IDF_TERM = 2.5

_WORD_RE = re.compile(r"[a-z0-9]+")
# Runs of one case, so that every way of gluing them back together can be tried:
# "A2xContext" has to yield "a2x", which no camel-case split gets right on its
# own, and "ScMgmtProfile-Single" has to yield "scmgmtprofile". A capital run
# gives back the capital that starts the next word — `ECRData` is ECR and Data,
# not ECRD and ata — or the acronym its description expands is never formed.
_RUN_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]|[a-z]+|[0-9]+")


def _words(s):
    return _WORD_RE.findall(s.lower())


def _flat(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _name_runs(name):
    """Every contiguous join of the case-runs of a schema name."""
    parts = [p.lower() for p in _RUN_RE.findall(name)]
    return {"".join(parts[i:j]) for i in range(len(parts))
            for j in range(i + 1, len(parts) + 1)}


def _leaks_name(name, desc):
    """Does the description say the schema's name back in other words?

    If it does, the question carries its own answer key and a text search finds
    the target without anything being understood — which is the failure the
    description is meant to avoid, not a difficulty it poses. 3GPP writes most
    of these descriptions as the name expanded ("Default Unrelated Class" for
    `DefaultUnrelatedClass`), so this rejects the majority of the corpus.
    """
    runs = {r for r in _name_runs(name) if len(r) >= 3}
    if not runs:
        return True
    ws = _words(desc)
    flat = _flat(desc)
    for r in runs:
        # The run written out as a word, or as the stem of one.
        if any(w == r or w.startswith(r) or (r.startswith(w) and len(w) >= 3) for w in ws):
            return True
        # Or buried in the running text: "the ScMgmtProfile represents ...".
        if len(r) >= 5 and r in flat:
            return True
    # Or spelled as an acronym the description expands: "Network Slice Selection
    # Assistance Information" for `Nssai`, "Allocation and Retention Priority"
    # for `Arp`.
    for i in range(len(ws)):
        acronym = ""
        for w in ws[i:i + 8]:
            acronym += w[0]
            if len(acronym) >= 3 and acronym in runs:
                return True
    return False


class _Descriptions:
    """Every described schema in the store, indexed for the uniqueness test.

    Built once over all 477 documents rather than per candidate: the test asks
    whether a second element anywhere in the corpus would also answer the
    question, so it cannot be run against one document at a time.
    """

    def __init__(self, store):
        self.store = store
        self.entries = []            # (entry, name, description)
        self.by_key = {}             # (spec, api, name) -> index
        self.name_count = collections.Counter()
        self.verbatim = collections.Counter()
        props_of = {}
        for entry in store:
            for name, schema in store.schemas(entry).items():
                if not isinstance(schema, dict):
                    continue
                self.name_count[name] += 1
                desc = schema.get("description")
                if not isinstance(desc, str) or not desc.strip():
                    continue
                desc = " ".join(desc.split())
                i = len(self.entries)
                self.by_key[(entry["spec"], entry["api"], name)] = i
                self.entries.append((entry, name, desc))
                self.verbatim[_flat(desc)] += 1
                props_of[i] = merged_properties(store, entry, schema)

        self.props = props_of
        # Two schemas with the same property set answer the question the same
        # way and are cited differently, so neither can be asked about.
        self.same_props = collections.defaultdict(list)
        for i, p in props_of.items():
            if _usable(p):
                self.same_props[tuple(p)].append(i)

        n = len(self.entries)
        self.bags = [{w for w in _words(d) if len(w) >= 3} for _, _, d in self.entries]
        df = collections.Counter()
        for bag in self.bags:
            for w in bag:
                df[w] += 1
        self.idf = {w: math.log(n / c) for w, c in df.items()}
        self.weight = [sum(self.idf[w] for w in bag) for bag in self.bags]
        self.inverted = collections.defaultdict(list)
        for i, bag in enumerate(self.bags):
            for w in bag:
                self.inverted[w].append(i)

    def nearest(self, i):
        """The most similar other description, as (similarity, index)."""
        if self.weight[i] <= 0:
            return 1.0, None
        shared = collections.defaultdict(float)
        for w in self.bags[i]:
            for j in self.inverted[w]:
                if j != i:
                    shared[j] += self.idf[w]
        best, at = 0.0, None
        for j, w in shared.items():
            sim = w / (self.weight[i] + self.weight[j] - w)
            if sim > best:
                best, at = sim, j
        return best, at

    def reject(self, i):
        """Why this schema cannot be asked about by description. None if it can.

        Order matters only for the counts the generator prints; every check is
        independent, and `--verify` re-runs all of them against the database.
        """
        entry, name, desc = self.entries[i]
        props = self.props[i]
        if not _usable(props):
            return "property set is not usable as gold"
        if not DESC_MIN <= len(desc) <= DESC_MAX:
            return "description too short or too long"
        if DESC_XREF.match(desc):
            return "description is a cross-reference, not a description"
        if self.name_count[name] != 1:
            return f"{self.name_count[name]} schemas in the store are named {name!r}"
        if self.verbatim[_flat(desc)] != 1:
            return "another schema carries the same description verbatim"
        if _leaks_name(name, desc):
            return "description says the schema name back"
        # Gluing the case-runs of a name back together cannot cut inside an
        # all-caps stretch: `EELACRReq` is one run, so no join of it yields the
        # "ACR" its description writes. A rare word of the description found
        # anywhere in the name catches those.
        term = next((w for w in set(_words(desc))
                     if len(w) >= 3 and self.idf.get(w, 0.0) >= IDF_TERM
                     and w in _flat(name)), None)
        if term is not None:
            return f"description writes {term!r}, which is part of the schema name"
        low = desc.lower()
        leaked = [p for p in props if p.lower() in low]
        if leaked:
            return f"description names the gold properties {leaked}"
        # A *piece* of a property name is not filtered on, and the choice is
        # worth stating: a description often shares one word-component with a
        # gold property without naming it, and rejecting all of those was tried
        # and costs a third of the pool. It buys nothing, because the gold is
        # scored as an exact set of camel-cased names and one shared component
        # does not produce one. The names themselves, above, are different:
        # writing one out hands over part of the answer.
        #
        # No example is given here on purpose. Naming the schema this rule was
        # decided on would put a live task, its description and part of its gold
        # into a public repository, which is the one thing the task files are
        # kept out of git to prevent.
        others = [j for j in self.same_props[tuple(props)] if j != i]
        if others:
            twin = self.entries[others[0]]
            return f"{twin[1]!r} in {twin[0]['api']} has the same property set"
        sim, at = self.nearest(i)
        if sim >= DESC_SIM:
            twin = self.entries[at] if at is not None else (None, "?", "")
            return f"description is {sim:.2f} similar to {twin[1]!r} in {twin[0]['api']}"
        return None

    def check(self, entry, name):
        """The uniqueness test for one schema, for --verify."""
        i = self.by_key.get((entry["spec"], entry["api"], name))
        if i is None:
            return "schema has no description"
        return self.reject(i)


def _descriptions(store):
    if not hasattr(store, "_descriptions"):
        store._descriptions = _Descriptions(store)
    return store._descriptions


def _describe_pool(store, n, rng):
    """The elements both types are generated from, chosen once.

    `describe` and `named` have to ask about the *same* schemas or the pair
    measures the elements rather than the way they are designated, so the
    selection is made here and both generators read it.
    """
    cached = getattr(store, "_describe_selection", None)
    if cached is not None and cached[0] == n:
        return cached[1]

    index = _descriptions(store)
    reasons = collections.Counter()
    pool = []
    for i in range(len(index.entries)):
        why = index.reject(i)
        if why is not None:
            reasons[_reason_bucket(why)] += 1
            continue
        entry, name, desc = index.entries[i]
        pool.append((entry, name, desc, index.props[i]))
    pool.sort(key=lambda x: (x[0]["file"], x[1]))
    rng.shuffle(pool)

    seen, chosen = collections.Counter(), []
    for item in pool:
        if seen[item[0]["api"]] >= MAX_PER_API:
            continue
        seen[item[0]["api"]] += 1
        chosen.append(item)
        if len(chosen) == n:
            break
    store._describe_selection = (n, (chosen, reasons, len(pool)))
    return chosen, reasons, len(pool)


_BUCKETS = [
    ("property set is not usable", "property set is not usable as gold"),
    ("description too short or too long", "description too short or too long"),
    ("cross-reference", "description is a cross-reference"),
    ("are named", "schema name is not unique in the store"),
    ("same description verbatim", "description repeated verbatim elsewhere"),
    ("says the schema name back", "description says the schema name back"),
    ("part of the schema name", "description writes a term the schema name contains"),
    ("names the gold properties", "description names the gold properties"),
    ("has the same property set", "another schema has the same property set"),
    ("similar to", "another description is too similar"),
]


def _reason_bucket(why):
    for needle, label in _BUCKETS:
        if needle in why:
            return label
    return why


def describe_tasks(store, n, rng):
    chosen, _, _ = _describe_pool(store, n, rng)
    return [_task(
        entry,
        f"openapi-describe-{entry['api']}-{name}",
        "Across the OpenAPI documents of the 3GPP 5G service based interfaces, "
        "exactly one schema is described as follows:\n\n"
        f"    {desc}\n\n"
        "List the top-level properties of that schema, merging every allOf "
        "member. The question does not name the schema or the document that "
        "declares it.",
        props,
        {"kind": "describe", "schema": name},
    ) for entry, name, desc, props in chosen]


def named_tasks(store, n, rng):
    chosen, _, _ = _describe_pool(store, n, rng)
    return [_task(
        entry,
        f"openapi-named-{entry['api']}-{name}",
        "Across the OpenAPI documents of the 3GPP 5G service based interfaces, "
        f"exactly one schema is named {name!r}. List the top-level properties of "
        "that schema, merging every allOf member. The question does not name the "
        "document that declares it.",
        props,
        {"kind": "named", "schema": name},
    ) for entry, name, desc, props in chosen]


def lookup_tasks(store, n, rng):
    """The third rung: the document *and* the name, which is what every other
    type in this file gives.

    Without it the pair above measures the wrong thing. `named` withholds the
    document, so finding the schema still takes a search — the same search
    `describe` takes — and the two come out level, which says nothing about
    naming and everything about the question both were asked. This asks the same
    21 schemas the way `run-openapi.sh` asks its four types, and it is the
    condition a lookup tool is enough for.
    """
    chosen, _, _ = _describe_pool(store, n, rng)
    return [_task(
        entry,
        f"openapi-lookup-{entry['api']}-{name}",
        f"In the 3GPP 5G service based interface API {entry['api']!r} "
        f"({entry['spec']}), the schema {name!r} is declared. List its top-level "
        "properties, merging every allOf member.",
        props,
        {"kind": "lookup", "schema": name},
    ) for entry, name, desc, props in chosen]


def describe_report(store, n, rng):
    """What the uniqueness test removed, for the generator to print.

    A filter that is never shown to have removed anything is a filter nobody
    can check.
    """
    chosen, reasons, pool = _describe_pool(store, n, rng)
    total = len(_descriptions(store).entries)
    lines = [f"  described schemas: {total}; usable after the uniqueness test: {pool}; "
             f"selected: {len(chosen)}"]
    for label, count in reasons.most_common():
        lines.append(f"    dropped {count:5d}  {label}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# re-derivation, for generate.py --verify

def rederive(store, task):
    """Recompute a task's gold from the documents. Returns the gold or a reason."""
    entry = next((e for e in store if e["spec"] == task["spec_id"]
                  and e["api"] == task["api_name"]), None)
    if entry is None:
        return "API document missing"
    probe = task.get("probe") or {}
    kind = probe.get("kind")

    if kind == "request":
        for path, method, op in _operations(store, entry):
            if path != probe["path"] or method != probe["method"]:
                continue
            got = _request_schema(store, entry, op)
            if got is None:
                return "operation has no request body schema"
            target, name, schema = got
            if name != probe["schema"]:
                return f"request body is now {name!r}"
            return merged_properties(store, target, schema)
        return "operation not found"

    if kind == "object":
        schema = store.schemas(entry).get(probe["owner"])
        if not isinstance(schema, dict):
            return "schema missing"
        value = _properties(schema).get(probe["property"])
        got = store.resolve(entry, (value or {}).get("$ref"))
        if got is None:
            return "property is not a reference"
        if got[1] != probe["schema"]:
            return f"property now points at {got[1]!r}"
        return merged_properties(store, got[0], got[2])

    if kind == "allof":
        schema = store.schemas(entry).get(probe["schema"])
        if not isinstance(schema, dict):
            return "schema missing"
        return merged_properties(store, entry, schema)

    if kind in ("describe", "named", "lookup"):
        # Two things have to hold, and re-deriving the gold only shows the
        # first: that the answer still comes out of the document, and that the
        # question still has one answer. A description that has become a fit for
        # a second schema makes the task score a correct answer as wrong, so the
        # uniqueness test runs again here rather than only at generation.
        schema = store.schemas(entry).get(probe["schema"])
        if not isinstance(schema, dict):
            return "schema missing"
        why = _descriptions(store).check(entry, probe["schema"])
        if why is not None:
            return f"no longer unique: {why}"
        return merged_properties(store, entry, schema)

    if kind == "oneof":
        schema = store.schemas(entry).get(probe["schema"])
        if not isinstance(schema, dict):
            return "schema missing"
        members = schema.get(probe["key"])
        if not isinstance(members, list):
            return f"no {probe['key']}"
        alts = []
        for m in members:
            got = store.resolve(entry, (m or {}).get("$ref"))
            if got is None:
                return "alternative does not resolve"
            alts.append(got[1])
        return sorted(alts)

    return f"unknown openapi task kind {kind!r}"


GENERATORS = collections.OrderedDict([
    ("openapi-request", request_tasks),
    ("openapi-object", object_tasks),
    ("openapi-allof", allof_tasks),
    ("openapi-oneof", oneof_tasks),
    ("openapi-describe", describe_tasks),
    ("openapi-named", named_tasks),
    ("openapi-lookup", lookup_tasks),
])
