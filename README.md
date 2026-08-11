# teleqna-eval

Evaluation harness that measures how much [3gpp-mcp](https://github.com/higebu/3gpp-mcp)
improves LLM accuracy on [TeleQnA](https://github.com/netop-team/TeleQnA)
multiple-choice questions.

The harness runs each question through any OpenAI-compatible chat API with
the tools of a running 3gpp-mcp server bridged into the request, so the model
can search and read 3GPP specifications before answering. Running the same
questions with `-mcp ''` gives the no-tools baseline, and the delta between
the two runs is the effect of 3gpp-mcp.

## Results

> **These numbers come from the superseded protocol.** They were measured
> before the two conditions were made to share one prompt (see
> [Protocol](#protocol)): the baseline was told to keep its reasoning brief and
> the tools condition was not, so the effect below mixes retrieval with
> reasoning length. They are kept here until the re-run under the current
> protocol replaces them.

All 1,509 `Standards specifications` TeleQnA questions tagged `3GPP`, same
question set for every run, each model on its vendor's own API, evaluated
2026-08-09:

| Model | No tools | With 3gpp-mcp | Δ | Win/loss pairs¹ | McNemar |
|---|---|---|---|---|---|
| DeepSeek V4 Flash | 75.3% (1137/1509) | **86.5%** (1305/1509) | +11.1pt | 225 / 57 | χ²=98.9, p<10⁻²² |
| Claude Sonnet 5 | 73.5% (1109/1509) | **84.5%** (1275/1509) | +11.0pt | 241 / 75 | χ²=86.2, p<10⁻¹⁹ |
| GPT 5.6 Luna | 73.6% (1110/1509) | **85.0%** (1282/1509) | **+11.4pt** | 242 / 70 | χ²=93.7, p<10⁻²¹ |

¹ questions only the tools run answered correctly / only the baseline answered correctly.

Observations:

- The effect reproduces across three unrelated model families, each on its
  own vendor's API, so it is a property of the tool access, not of one model
  or one serving stack. The three deltas land within 0.4pt of each other.
- 68 of the 1,509 questions were answered correctly by all three models with
  tools and by none of them without tools, against 16.2 expected if the six
  runs were independent — so such questions exist. Which questions they are is
  mostly noise: the 8-round pass produces a set of 66 that shares only 41
  members with this one (Jaccard 0.44), so the individual ids are not listed.
- Tool-call efficiency differs sharply: Claude Sonnet 5 averaged 3.3
  calls/question, GPT 5.6 Luna 5.7, DeepSeek V4 Flash 10.1 — Sonnet reaches
  the same gain with a third of the searches.
- Questions that still hit the 20-round tool budget: DeepSeek 61, Sonnet 47,
  Luna 7.

### Usage per run (tools / baseline)

| Run | Prompt tokens | Completion tokens | Tool calls |
|---|---|---|---|
| DeepSeek V4 Flash | 164.7M / 0.33M | 5.97M / 4.83M | 15,222 |
| Claude Sonnet 5 | 76.6M / 0.33M | 1.22M / 0.14M | 4,967 |
| GPT 5.6 Luna | 67.0M / 0.22M | 0.83M / 0.52M | 8,652 |

Wall-clock per pair was 25–60 minutes at 8–32 concurrent questions; a
single-instance 3gpp-mcp server absorbed 32 parallel tool streams without
errors.

## Method

- **Dataset**: TeleQnA category `Standards specifications`, filtered to
  questions whose text contains `3GPP` (the category also contains
  IEEE 802.11 etc., which a 3GPP tool cannot help with): 1,509 questions.
  All 1,509 carry a `[3GPP Release N]` tag, and for 1,471 of them that tag is
  the only occurrence of `3GPP` in the text — so the filter is in practice
  "questions carrying a release tag". Releases: 18 (780), 17 (641), 14 (55),
  19 (17), 16 (16).
- **Tools condition**: the 11 MCP tools of a 3gpp-mcp server (database:
  latest version of every spec) are bridged into the model API as function
  tools; the model may call up to 20 rounds of tools per question, then is
  asked to answer. Every model runs on its vendor's own API: GPT 5.6 Luna on
  OpenAI's Responses API (`-api responses`, default reasoning effort),
  Claude Sonnet 5 on Anthropic's OpenAI-compatible chat completions
  endpoint, DeepSeek V4 Flash on `api.deepseek.com`.
- **Baseline condition**: the same prompt with no tools attached. In the runs
  reported above this was *not* true — the two conditions used different system
  prompts — which is the reason those numbers are superseded. See
  [Protocol](#protocol).
- **Answering**: the answer is extracted with strict-to-loose fallbacks and up
  to 2 re-prompts, identically in both conditions. Every record now carries the
  `parse_tier` that produced it, so a fallback parse can be re-scored as a
  failure. Questions that errored mid-run were re-executed; under the current
  protocol that is `-resume`, which records the `attempt` on each record. In
  the superseded runs it was an ad-hoc merge, and one of them — Claude Sonnet 5
  with tools — is a splice: 597 of its 1,509 questions were re-executed in a
  second session after the API key ran out of credit.
- **Generation**: no token limit and no temperature, matching the TeleQnA
  paper's own settings; every pair uses identical settings on both conditions,
  one run per condition. (Telco-RAG uses `max_tokens=4000`; GSMA evals sets
  `temperature=0`.) An earlier pass with `max_tokens=8192` and an 8-round
  budget measured +12.4/+10.9/+9.3pt; the cap suppressed only DeepSeek's
  baseline (146 of its 1,509 baseline questions used at least 8192 completion
  tokens, against 1 for Sonnet and 0 for Luna). The result files record tokens
  summed over a question's rounds, not per generation, so this counts questions
  that reached the cap rather than individual truncated completions.
- **Scoring**: exact match of the option number; an unanswered question counts
  as wrong. `strict_match` additionally records whether the answer string
  equals TeleQnA's own `option N: text`, which is how the upstream harness
  scores. Significance via McNemar's test with continuity correction on paired
  outcomes.

Caveats:

- Absolute numbers are **not** directly comparable to Telco-RAG / TelcoAI /
  GSMA leaderboard figures: model generations, prompt formats (numeric
  `option N` here vs letter-based elsewhere), and question subsets differ.
  The paired same-model delta is the measurement.
- TeleQnA questions are tagged with the release they were written against,
  while the server database held the latest version of every spec; a
  release-pinned database (`3gpp-mcp build --release 17 ...`) would remove a
  potential source of answer drift. The observed gains occur despite it.
- The superseded runs read from a deployed server whose database is rebuilt
  weekly, so they cannot be reproduced exactly. Runs under the current protocol
  serve a pinned database and record its identifier with `-db-manifest`.
- Per-question outputs are not published: TeleQnA is deliberately
  distributed as a password-protected archive to keep it out of crawled
  training corpora, and raw run logs embed question content.
- Each figure comes from a single run per condition, and re-running an
  unchanged condition moves it by up to about a point (Luna's baseline moved
  75.0% → 73.6% across two identical runs, with 151 individual questions
  flipping). The ~11pt effect is far larger than that, but do not read the
  differences *between* models as meaningful.

## Protocol

**One prompt per pair.** Both conditions send the identical system and user
message; the only difference is whether tool definitions are attached to the
request. `internal/prompt` owns the wording and nothing in the harness may
branch on tool availability — `TestPromptIdenticalAcrossConditions` fails the
build if that ever changes. This is the rule the published runs broke: their
baseline was told to give *brief* reasoning while the tools condition was not,
and on Claude Sonnet 5 the baseline answered in a median of 10 completion
tokens, so its deficit is partly a missing chain of thought rather than missing
retrieval.

**Prompt variants** (`-prompt`):

| ID | Wording | Answer format |
|---|---|---|
| `teleqna` (default) | the system prompt of TeleQnA's own `evaluation_tools.py`, byte for byte | JSON, `"answer": "option N: text"` |
| `cot` | the same text plus one sentence asking for step-by-step reasoning first | same |
| `ansline` | the harness's original wording | `ANSWER: <n>` |

`teleqna` sends the bytes upstream sends: `TestFormatMatchesUpstreamBytes`
compares the rendered user message against a golden file generated by
evaluation_tools.py's own logic, and `TestTeleQnASystemIsUpstreamText` pins the
system prompt. That includes reproducing an upstream quirk — its
`if 'category' in questions_only` tests the outer dict of questions rather than
the question itself, so the category field is never removed and every question
upstream sends carries it. The field is one constant string across the filtered
pool, so it cannot separate the two conditions.

It deviates from upstream in two documented ways: one question per request
(upstream batches five into one JSON object, which cannot host a tool-calling
loop), and up to two re-prompts when the reply does not parse (upstream retries
the whole batch up to five times). Both apply equally to both conditions.

Regenerate the golden file after extracting the dataset:

```bash
python3 - <<'EOF'
from copy import deepcopy
import collections, json
all_q = json.load(open('data/TeleQnA.json'), object_pairs_hook=collections.OrderedDict)
def upstream(qid):
    d = collections.OrderedDict({qid: all_q[qid]})
    only = deepcopy(d)
    for q in d:
        only[q].pop("answer")
        if 'explanation' in only[q]:
            only[q].pop('explanation')
        if 'category' in only:      # upstream's bug: never fires
            only[q].pop('category')
    return "Here are the questions: \n " + json.dumps(only)
ids = [k for k, v in all_q.items()
       if v.get('category','').startswith('Standards specifications') and '3GPP' in v['question']]
picks = [ids[0]] + [k for k in ids if 'option 5' in all_q[k]][:1]
json.dump([{"id": p, "fields": all_q[p], "want": upstream(p)} for p in picks],
          open('internal/prompt/testdata/upstream_user_prompt.json','w'),
          indent=2, ensure_ascii=False)
EOF
```

**Retrieval conditions**: no tools (`-mcp ''`), the model's own tool loop
(default), or the non-agentic baseline `-fixedk N` — one BM25 search over the
same database, the text of the top N sections prepended, no tools attached.
3gpp-mcp ranks FTS5 hits with a weighted bm25, so `-fixedk` is a fixed-k RAG
baseline over exactly the corpus the agentic condition searches.

**Provenance.** Every run writes `<out>.meta.json` (harness commit, every flag,
prompt id and sha256, model, temperature, MCP server identity and tool list,
`-db-manifest`) and, unless disabled, `<out>.trace.jsonl` with every message and
every tool result as the model saw it. Each record carries `run_id`, `attempt`,
`repeat_idx`, `parse_tier` and `answer_raw`. Traces are large — hundreds of
megabytes for a full-pool tools run — and are not meant to be committed.

**Repeats and resume.** `-repeat N` measures the same condition N times;
`-resume` appends to an existing file, re-running only what is missing or
errored and recording a higher `attempt` on those records, so a spliced file
says so in its own data.

## Setup

```bash
# 1. Dataset (password-protected zip; the password is published in the TeleQnA README)
git clone --depth 1 https://github.com/netop-team/TeleQnA
pip install pyzipper
python3 -c "
import pyzipper
with pyzipper.AESZipFile('TeleQnA/TeleQnA.zip') as z:
    z.setpassword(b'teleqnadataset')
    open('data/TeleQnA.json','wb').write(z.read('TeleQnA.txt'))
"

# 2. Any OpenAI-compatible chat completions endpoint + its API key
export OPENAI_BASE_URL=...   # e.g. https://api.openai.com/v1
export OPENAI_API_KEY=...

# 3. Your 3gpp-mcp server (streamable HTTP endpoint, e.g. https://<host>/mcp/)
export THREEGPP_MCP_URL=...

# 4. Build
go build -o eval .
```

The dataset is deliberately not committed to this repository: upstream
password-protects it to keep it out of GitHub-crawled training data.

## Usage

The harness talks to any OpenAI-compatible chat completions API: pick an
endpoint with `-base-url` (or `$OPENAI_BASE_URL`), an API key variable with
`-key-env` (default `OPENAI_API_KEY`), and a model with `-model`. No provider
is special-cased.

```bash
# Smoke test: 10 questions, tools bridged from $THREEGPP_MCP_URL
./eval -model gpt-5.2 -n 10 -seed 42

# 3GPP-only questions, full pool
./eval -model gpt-5.2 -filter 3GPP -n 1509 -seed 42 -workers 8

# No-tools baseline on the same set
./eval -model gpt-5.2 -filter 3GPP -n 1509 -seed 42 -workers 8 -mcp ''

# Other OpenAI-compatible providers are just a different base URL + key:
./eval -base-url https://api.deepseek.com/v1 -key-env DEEPSEEK_API_KEY -model deepseek-v4-flash ...

# OpenAI Platform reasoning models reject function tools on chat/completions
# unless reasoning is disabled; use the Responses API backend instead:
./eval -api responses -base-url https://api.openai.com/v1 -model gpt-5.6-luna ...

# Claude models: the OpenAI-compatible endpoint does not support prompt
# caching, so use the native Messages API backend:
./eval -api anthropic -base-url https://api.anthropic.com/v1 -key-env ANTHROPIC_API_KEY -model claude-sonnet-5 ...
```

The scripts in [`examples/`](examples/) (`run_{platform}_{model}.sh`)
reproduce the paired runs behind the results above.
Results are written as JSONL under `results/` (one record per question:
prediction, expectation, tool-call trace, token usage, duration) plus a
summary line on stdout.

## The specification-grounded benchmark

`bench/` generates questions from a pinned specification database and scores
the answer *and* the clause it is attributed to.

```bash
python3 bench/generate.py --db 3gpp-latest.db --verify   # tasks-*.json
go run ./bench -tasks bench/tasks-asn1.json -model ... -out results/asn1.jsonl
go run ./bench/grade -db 3gpp-latest.db -tasks-dir bench results/asn1.jsonl
```

**Running and grading are separate passes, and grading never writes over its
input.** The run records what the model answered and whether the answer is
right, which needs nothing but the reply. Whether a citation is right needs the
corpus — a clause that contains the gold one is a coarser citation rather than
a wrong one, and a clause that does not exist is a different failure from one
that does — so `bench/grade` decides it afterwards, writes a `.graded.jsonl`
beside the input, and stamps each record with the revision that graded it.

The grader reads the database directly rather than through the MCP server: the
tool under test must not be the one deciding whether its own citation exists.

## Flags

| Flag | Default | Description |
|---|---|---|
| `-model` | (required) | Model ID on the chat endpoint |
| `-api` | `chat` | `chat` (chat/completions), `responses` (OpenAI Responses API) or `anthropic` (Anthropic Messages API) |
| `-base-url` | `$OPENAI_BASE_URL` | OpenAI-compatible base URL |
| `-key-env` | `OPENAI_API_KEY` | Name of the environment variable holding the API key |
| `-mcp` | `$THREEGPP_MCP_URL` | Streamable HTTP MCP endpoint; `''` disables tools |
| `-data` | `data/TeleQnA.json` | Extracted TeleQnA JSON |
| `-category` | `Standards specifications` | TeleQnA category prefix filter |
| `-filter` | (none) | Substring the question text must contain, e.g. `3GPP` |
| `-n` | 10 | Number of questions (seeded sample) |
| `-ids` | (none) | Comma-separated question ids to run (overrides sampling) |
| `-seed` | 42 | Sampling seed — keep it fixed across compared runs |
| `-workers` | 1 | Concurrent questions |
| `-max-rounds` | 8 | Tool-calling rounds per question before forcing an answer |
| `-max-tokens` | 8192 | Token cap per completion (0 = provider default, or the model's ceiling on `-api anthropic`) |
| `-max-tokens-field` | `max_tokens` | Request field name for the cap (e.g. `max_completion_tokens`) |
| `-temperature` | (unset) | Sampling temperature; empty sends no temperature field at all. Some reasoning models reject it |
| `-http-timeout` | 300 | Per-request timeout in seconds; raise it when running without a token cap |
| `-tool-result-max` | 16000 | Max bytes of a tool result passed to the model |
| `-prompt` | `teleqna` | Prompt variant: `teleqna`, `cot` or `ansline` |
| `-fixedk` | 0 | Non-agentic baseline: one search, top-k sections prepended, no tools |
| `-repeat` | 1 | Run every question this many times |
| `-resume` | false | Append to an existing `-out` file, re-running only missing or errored questions |
| `-run-id` | timestamp | Identifier recorded on every record |
| `-db-manifest` | (none) | Identifier of the pinned database the MCP server serves, recorded in the metadata |
| `-trace` | auto | Full message/tool-result trace path; `''` disables |
| `-out` | auto | JSONL output path |

## License

MIT — see [LICENSE](LICENSE).

The TeleQnA dataset itself is not included; it is distributed separately by
[netop-team/TeleQnA](https://github.com/netop-team/TeleQnA) (MIT), whose
authors ask that you cite their paper when using the data:

```bibtex
@misc{maatouk2023teleqna,
  title={TeleQnA: A Benchmark Dataset to Assess Large Language Models Telecommunications Knowledge},
  author={Ali Maatouk and Fadhel Ayed and Nicola Piovesan and Antonio De Domenico and Merouane Debbah and Zhi-Quan Luo},
  year={2023},
  eprint={2310.15051},
  archivePrefix={arXiv},
  primaryClass={cs.IT}
}
```
