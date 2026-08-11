#!/usr/bin/env python3
"""Precompute the OpenAPI baselines' retrieved context, one block per task.

A fixed-k baseline is non-adaptive by definition: one query, decided by the
question, and no second look at what came back. Nothing about it depends on the
model, so running it ahead of time and handing specbench the block to prepend
is the same measurement as running it inline — and it keeps the harness free of
a SQLite dependency, which it has deliberately never had.

    python3 bench/openapi_index.py --db 3gpp-latest.db --out bench/openapi-index.db
    python3 bench/openapi_retrieve.py --index bench/openapi-index.db \\
        --tasks bench/tasks-openapi-allof.json --mode bm25 \\
        --out bench/context-openapi-allof-bm25.json

Two modes:

  bm25    one FTS5 query built from the question exactly as internal/retrieval
          builds it for the clause-text baseline, top-k chunks prepended
  exact   the chunk the question names, prepended. An oracle: no query
          formulation can beat naming the right chunk, so a gap that survives
          this condition is not a gap in query quality

The query builder and the byte budget are copied from internal/retrieval so the
two baselines differ in what they search, not in how they are assembled.
"""

import argparse
import json
import os
import re
import sqlite3
import sys

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
RELEASE_RE = re.compile(r"\s*\[3GPP Release \d+\]\s*")
MAX_TERMS = 24
CONTEXT_HEADER = "Excerpts retrieved from the 3GPP specifications:\n\n"

# internal/retrieval/fixedk.go's list, kept identical on purpose.
STOPWORDS = {
    "the", "and", "for", "are", "what", "which", "does", "can", "with", "from",
    "that", "this", "how", "when", "why", "who", "was", "were", "has", "have",
    "will", "its", "not", "but", "you", "your", "there", "their", "used", "use",
    "following", "about", "into", "between", "during",
}


def query(text):
    text = RELEASE_RE.sub(" ", text)
    seen, terms = set(), []
    for tok in TOKEN_RE.findall(text):
        low = tok.lower()
        if len(low) < 3 or low in STOPWORDS or low in seen:
            continue
        seen.add(low)
        terms.append(tok)
        if len(terms) == MAX_TERMS:
            break
    return " OR ".join(terms) if terms else text.strip()


def truncate(s, limit):
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n...[truncated {len(s) - limit} bytes; refine the query or use offset to read more]"


def fts_query(text):
    """FTS5 needs bare terms quoted; the tokens here can carry dots and dashes."""
    return " OR ".join(f'"{t}"' for t in query(text).split(" OR ") if t)


def retrieve_bm25(idx, task, k, result_max):
    rows = idx.execute(
        "SELECT c.spec_id, c.api_name, c.kind, c.name, c.body FROM chunks_fts f "
        "JOIN chunks c ON c.id = f.rowid WHERE chunks_fts MATCH ? "
        "ORDER BY bm25(chunks_fts, 5.0, 1.0) LIMIT ?",
        (fts_query(task["question"]), k),
    ).fetchall()
    return rows


def retrieve_exact(idx, task, k, result_max):
    p = task.get("probe") or {}
    if p.get("kind") == "request":
        name = f"{p['method'].upper()} {p['path']}"
    elif p.get("kind") == "object":
        name = p["owner"]
    else:
        name = p.get("schema", "")
    rows = idx.execute(
        "SELECT spec_id, api_name, kind, name, body FROM chunks "
        "WHERE name = ? AND api_name = ? LIMIT ?",
        (name, task["api_name"], k),
    ).fetchall()
    if not rows:  # the oracle still has to find something to be an oracle
        rows = idx.execute(
            "SELECT spec_id, api_name, kind, name, body FROM chunks WHERE name = ? LIMIT ?",
            (name, k),
        ).fetchall()
    return rows


def block(rows, result_max):
    if not rows:
        return ""
    per = result_max // len(rows)
    out = [CONTEXT_HEADER]
    for spec, api, kind, name, body in rows:
        out.append(f"--- {spec} API {api} {kind} {name} ---\n{truncate(body, per)}\n\n")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True, help="index from bench/openapi_index.py")
    ap.add_argument("--tasks", required=True, help="task JSON from bench/generate.py")
    ap.add_argument("--mode", choices=("bm25", "exact"), required=True)
    ap.add_argument("--k", type=int, default=5, help="chunks to prepend")
    ap.add_argument("--result-max", type=int, default=16000,
                    help="byte budget for the whole block, as in the clause-text baseline")
    ap.add_argument("--out", required=True, help="{task id: context block} JSON")
    args = ap.parse_args()

    idx = sqlite3.connect(f"file:{args.index}?mode=ro", uri=True)
    tasks = json.load(open(args.tasks))
    fn = retrieve_bm25 if args.mode == "bm25" else retrieve_exact

    contexts, empty, holds = {}, 0, 0
    for t in tasks:
        rows = fn(idx, t, args.k, args.result_max)
        contexts[t["id"]] = block(rows, args.result_max)
        if not rows:
            empty += 1
        # Did the retrieved text actually contain the answer? Reported so the
        # baseline's ceiling is visible rather than inferred from its score.
        gold = t["gold"] if isinstance(t["gold"], list) else [t["gold"]]
        if gold and all(str(g) in contexts[t["id"]] for g in gold):
            holds += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(contexts, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"{args.out}: {len(contexts)} contexts, {empty} empty, "
          f"{holds}/{len(tasks)} contain every gold token", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
