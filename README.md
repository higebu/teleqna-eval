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

All 1,509 `Standards specifications` TeleQnA questions tagged `3GPP`, same
question set for every run, evaluated 2026-08-09:

| Model | No tools | With 3gpp-mcp | Δ | Win/loss pairs¹ | McNemar |
|---|---|---|---|---|---|
| DeepSeek V4 Flash | 73.2% (1105/1509) | **85.6%** (1292/1509) | **+12.4pt** | 249 / 62 | χ²=111.2, p<10⁻²⁵ |
| Claude Sonnet 5 | 73.8% (1113/1509) | **84.7%** (1278/1509) | +10.9pt | 234 / 69 | χ²=88.8, p<10⁻²⁰ |
| GPT 5.6 Luna | 74.3% (1121/1509) | **82.6%** (1246/1509) | +8.3pt | 203 / 78 | χ²=54.7, p<10⁻¹² |

¹ questions only the tools run answered correctly / only the baseline answered correctly.

Observations:

- The effect reproduces across three unrelated model families, so it is a
  property of the tool access, not of one model.
- 60 of the 1,509 questions were answered correctly by all three models with
  tools and by none of them without tools — questions that effectively
  require reading the specification. Their TeleQnA IDs (`question N`):
  60, 558, 669, 812, 955, 1099, 1134, 1241, 1411, 1463, 1545, 1810, 1875,
  1877, 1918, 1988, 2488, 2608, 2771, 3189, 3527, 3630, 3673, 4184, 4242,
  4254, 4293, 4401, 4620, 4652, 4737, 4873, 4918, 5454, 5581, 5607, 5797,
  6023, 6139, 6216, 6379, 6393, 6706, 6956, 7194, 7657, 7956, 8063, 8138,
  8210, 8350, 8885, 8948, 9050, 9549, 9674, 9728, 9738, 9881, 9949.
- Tool-call efficiency differs sharply: Claude Sonnet 5 averaged 2.6
  calls/question, GPT 5.6 Luna 5.0, DeepSeek V4 Flash 8.2 — Sonnet reaches
  near-top accuracy with a third of the searches.

### Usage per run (tools / baseline)

| Run | Prompt tokens | Completion tokens | Tool calls |
|---|---|---|---|
| DeepSeek V4 Flash | 95.4M / 0.33M | 4.51M / 3.29M | 12,479 |
| Claude Sonnet 5 | 46.2M / 0.33M | 0.75M / 0.15M | 3,953 |
| GPT 5.6 Luna | 46.8M / 0.22M | 0.75M / 0.50M | 7,601 |

Wall-clock per pair was 25–50 minutes at 32 concurrent questions (8 for
Sonnet); a single-instance 3gpp-mcp server absorbed 32 parallel tool streams
without errors.

## Method

- **Dataset**: TeleQnA category `Standards specifications`, filtered to
  questions whose text contains `3GPP` (the category also contains
  IEEE 802.11 etc., which a 3GPP tool cannot help with): 1,509 questions.
- **Tools condition**: the 11 MCP tools of a 3gpp-mcp server (database:
  latest version of every spec) are bridged into the chat API as function
  tools; the model may call up to 8 rounds of tools per question, then is
  asked to answer.
- **Baseline condition**: same prompt, no tools.
- **Answering**: `ANSWER: <option number>` extracted with strict-to-loose
  fallbacks; up to 2 re-prompts if no answer is parseable. A handful of
  provider-side request errors (8 of 9,054 completions) were retried with
  `-ids`; final tallies contain an answer for every question.
- **Generation**: `max_tokens=8192`, identical across both conditions of
  every pair. Temperature unset. (Reference points: the TeleQnA paper code
  sets neither; Telco-RAG uses `max_tokens=4000`; GSMA evals sets
  `temperature=0`.)
- **Scoring**: exact match of the option number; an unanswered question
  counts as wrong. Significance via McNemar's test on paired outcomes.

Caveats:

- Absolute numbers are **not** directly comparable to Telco-RAG / TelcoAI /
  GSMA leaderboard figures: model generations, prompt formats (numeric
  `option N` here vs letter-based elsewhere), and question subsets differ.
  The paired same-model delta is the measurement.
- TeleQnA questions are tagged with the release they were written against,
  while the server database held the latest version of every spec; a
  release-pinned database (`3gpp-mcp build --release 17 ...`) would remove a
  potential source of answer drift. The observed gains occur despite it.
- Per-question outputs are not published: TeleQnA is deliberately
  distributed as a password-protected archive to keep it out of crawled
  training corpora, and raw run logs embed question content.

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
./eval -base-url https://api.anthropic.com/v1 -key-env ANTHROPIC_API_KEY -model claude-sonnet-5 ...

# OpenAI Platform reasoning models reject function tools on chat/completions
# unless reasoning is disabled; use the Responses API backend instead:
./eval -api responses -base-url https://api.openai.com/v1 -model gpt-5.6-luna ...
```

The scripts in [`examples/`](examples/) (`run_{platform}_{model}.sh`)
reproduce the paired runs behind the results above.
Results are written as JSONL under `results/` (one record per question:
prediction, expectation, tool-call trace, token usage, duration) plus a
summary line on stdout.

## Flags

| Flag | Default | Description |
|---|---|---|
| `-model` | (required) | Model ID on the chat endpoint |
| `-api` | `chat` | `chat` (chat/completions) or `responses` (OpenAI Responses API) |
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
| `-max-tokens` | 8192 | Token cap per completion (0 = provider default) |
| `-max-tokens-field` | `max_tokens` | Request field name for the cap (e.g. `max_completion_tokens`) |
| `-tool-result-max` | 16000 | Max bytes of a tool result passed to the model |
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
