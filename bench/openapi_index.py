#!/usr/bin/env python3
"""Build a retrieval index over the OpenAPI store, for the fair baselines.

The fixed-k baseline searches the FTS5 index 3gpp-mcp builds over clause text.
OpenAPI definitions are not in that index — they live in their own table — so
on the 5G SBI tasks the baseline is not a weaker retrieval strategy, it is no
retrieval at all. Any gap measured against it says "one condition can reach the
data", which is not the question the other task types answer.

This builds the index that baseline is missing: one row per schema and per
operation, with an FTS5 table over it. It is a *side* index, written to its own
file, so the pinned corpus and 3gpp-mcp itself are untouched.

    python3 bench/openapi_index.py --db 3gpp-latest.db --out bench/openapi-index.db

Two conditions read it, and they bracket what a non-agentic retriever can do:

  bm25    one query built from the question, top-k chunks prepended
  exact   the chunk whose name the question names, prepended

`exact` is deliberately an oracle: it is the best a single lookup could ever
do. If the agent still wins against it, the win is not about query quality.

The chunk is the unit that decides the answer, so it is fixed here and not
tuned afterwards: a schema chunk carries its own YAML with one level of $ref
expanded, which is what a retriever could reasonably index without resolving
the whole graph.
"""

import argparse
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openapi_tasks  # noqa: E402

SCHEMA = """
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    spec_id TEXT NOT NULL,
    api_name TEXT NOT NULL,
    filename TEXT NOT NULL,
    kind TEXT NOT NULL,      -- schema | operation
    name TEXT NOT NULL,      -- schema name, or "METHOD /path"
    body TEXT NOT NULL
);
CREATE INDEX chunks_name ON chunks(name);
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    name, body, content='chunks', content_rowid='id', tokenize='unicode61'
);
"""


def render_schema(store, entry, name, schema, depth=0):
    """A schema as YAML-ish text, with one level of $ref expanded.

    Expanding one level is what makes the chunk self-contained enough to answer
    "what fields does this have" without the retriever resolving the whole
    reference graph. It is also the limit: a task whose answer sits two hops
    away is not served by any single chunk, which is exactly what the
    cross-file stratum is there to measure.
    """
    out = [f"{name}:"]
    for key in ("type", "description"):
        if isinstance(schema.get(key), str):
            out.append(f"  {key}: {schema[key]}")
    for key in ("allOf", "oneOf", "anyOf"):
        members = schema.get(key)
        if not isinstance(members, list):
            continue
        out.append(f"  {key}:")
        for m in members:
            if not isinstance(m, dict):
                continue
            ref = m.get("$ref")
            if ref and depth == 0:
                got = store.resolve(entry, ref)
                if got:
                    out.append(f"    - $ref: {ref}")
                    out.append("      # expanded:")
                    body = render_schema(store, got[0], got[1], got[2], depth + 1)
                    out += ["      " + line for line in body.splitlines()]
                    continue
            if ref:
                out.append(f"    - $ref: {ref}")
            else:
                out += [f"    - {k}: {v}" for k, v in m.items() if isinstance(v, (str, int))]
                for p in (m.get("properties") or {}):
                    out.append(f"      property: {p}")
    props = schema.get("properties")
    if isinstance(props, dict):
        out.append("  properties:")
        for p, v in props.items():
            out.append(f"    {p}:")
            if not isinstance(v, dict):
                continue
            ref = v.get("$ref")
            if ref and depth == 0:
                got = store.resolve(entry, ref)
                if got:
                    out.append(f"      $ref: {ref}")
                    out.append("      # expanded:")
                    body = render_schema(store, got[0], got[1], got[2], depth + 1)
                    out += ["      " + line for line in body.splitlines()]
                    continue
            for k, val in v.items():
                if isinstance(val, (str, int, bool)):
                    out.append(f"      {k}: {val}")
    return "\n".join(out)


def build(conn, out_path):
    store = openapi_tasks.Store(conn)
    if os.path.exists(out_path):
        os.remove(out_path)
    idx = sqlite3.connect(out_path)
    idx.executescript(SCHEMA)

    rows = []
    for entry in store:
        for name, schema in store.schemas(entry).items():
            if not isinstance(schema, dict):
                continue
            rows.append((entry["spec"], entry["api"], entry["file"], "schema", name,
                         render_schema(store, entry, name, schema)))
        for path, method, op in openapi_tasks._operations(store, entry):
            body = [f"{method.upper()} {path}"]
            if isinstance(op.get("summary"), str):
                body.append(f"  summary: {op['summary']}")
            for p in op.get("parameters") or []:
                if isinstance(p, dict) and isinstance(p.get("name"), str):
                    body.append(f"  parameter: {p['name']}")
            got = openapi_tasks._request_schema(store, entry, op)
            if got:
                body.append("  requestBody:")
                rendered = render_schema(store, got[0], got[1], got[2])
                body += ["    " + line for line in rendered.splitlines()]
            rows.append((entry["spec"], entry["api"], entry["file"], "operation",
                         f"{method.upper()} {path}", "\n".join(body)))

    idx.executemany(
        "INSERT INTO chunks(spec_id,api_name,filename,kind,name,body) VALUES (?,?,?,?,?,?)", rows)
    idx.execute("INSERT INTO chunks_fts(rowid,name,body) "
                "SELECT id,name,body FROM chunks")
    idx.commit()
    stats = {
        "chunks": len(rows),
        "schemas": sum(1 for r in rows if r[3] == "schema"),
        "operations": sum(1 for r in rows if r[3] == "operation"),
        "documents": len(store.by_file),
        "bytes": sum(len(r[5]) for r in rows),
    }
    idx.close()
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True, help="pinned 3gpp-mcp database")
    ap.add_argument("--out", default="bench/openapi-index.db", help="index to write")
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    stats = build(conn, args.out)
    print(f"{args.out}: " + json.dumps(stats), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
