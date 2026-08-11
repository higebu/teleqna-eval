#!/usr/bin/env python3
"""Task generators for the 5G service-based interface OpenAPI definitions.

The first version of this track asked for the required properties of a named
schema. That question is answered by opening one document and reading one
block, which is not what makes 5G SBI definitions hard: 78% of their $refs
point at *another* file, and the schema an engineer actually needs is usually
assembled out of several of them.

These four generators ask the questions the structure poses:

  request   the top-level properties of an operation's request body
  object    the fields of the object a property holds, one $ref away
  allof     the property set a schema ends up with after its allOf is merged
  oneof     the alternatives a oneOf/anyOf schema may take

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
])
