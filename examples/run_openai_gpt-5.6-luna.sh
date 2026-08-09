#!/bin/bash
# GPT 5.6 Luna on the OpenAI Platform (Responses API): full 3GPP pool pair.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${OPENAI_API_KEY:?set OPENAI_API_KEY}"
: "${THREEGPP_MCP_URL:?set THREEGPP_MCP_URL (3gpp-mcp streamable HTTP endpoint)}"

MODEL=gpt-5.6-luna
COMMON=(-api responses -base-url https://api.openai.com/v1 -model "$MODEL"
  -filter 3GPP -n 1509 -seed 42 -workers 16
  -max-tokens 0 -max-rounds 20 -http-timeout 900)

./eval "${COMMON[@]}" -out "results/${MODEL}-tools-3gpp-1509q-seed42.jsonl"
./eval "${COMMON[@]}" -mcp '' -out "results/${MODEL}-notools-3gpp-1509q-seed42.jsonl"
