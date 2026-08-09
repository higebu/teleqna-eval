#!/bin/bash
# DeepSeek V4 Flash on the official DeepSeek API: full 3GPP pool pair.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DEEPSEEK_API_KEY:?set DEEPSEEK_API_KEY}"
: "${THREEGPP_MCP_URL:?set THREEGPP_MCP_URL (3gpp-mcp streamable HTTP endpoint)}"

MODEL=deepseek-v4-flash
COMMON=(-base-url https://api.deepseek.com -key-env DEEPSEEK_API_KEY -model "$MODEL"
  -filter 3GPP -n 1509 -seed 42 -workers 32)

./eval "${COMMON[@]}" -out "results/${MODEL}-official-tools-3gpp-1509q-seed42.jsonl"
./eval "${COMMON[@]}" -mcp '' -out "results/${MODEL}-official-notools-3gpp-1509q-seed42.jsonl"
