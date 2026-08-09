#!/bin/bash
# Claude Sonnet 5 on the Anthropic Messages API: full 3GPP pool pair.
# The published numbers were measured on the OpenAI-compatible endpoint,
# before this backend existed; -api anthropic additionally caches the prefix.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY}"
: "${THREEGPP_MCP_URL:?set THREEGPP_MCP_URL (3gpp-mcp streamable HTTP endpoint)}"

MODEL=claude-sonnet-5
COMMON=(-api anthropic -base-url https://api.anthropic.com/v1 -key-env ANTHROPIC_API_KEY -model "$MODEL"
  -filter 3GPP -n 1509 -seed 42 -workers 8
  -max-tokens 0 -max-rounds 20 -http-timeout 900)

./eval "${COMMON[@]}" -out "results/${MODEL}-tools-3gpp-1509q-seed42.jsonl"
./eval "${COMMON[@]}" -mcp '' -out "results/${MODEL}-notools-3gpp-1509q-seed42.jsonl"
