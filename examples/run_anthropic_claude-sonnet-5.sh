#!/bin/bash
# Claude Sonnet 5 full pair: all 1,509 3GPP-tagged Standards-specifications
# questions (tools, then baseline) on the Anthropic OpenAI-compatible endpoint.
set -ex
cd "$(dirname "$0")/.."
: "${THREEGPP_MCP_URL:?set THREEGPP_MCP_URL to your 3gpp-mcp endpoint}"
B=https://api.anthropic.com/v1
M=claude-sonnet-5
N=1509
./eval -base-url $B -key-env ANTHROPIC_API_KEY -model "$M" -filter 3GPP -n $N -seed 42 -workers 8 -out "results/$M-tools-3gpp-${N}q-seed42.jsonl"
./eval -base-url $B -key-env ANTHROPIC_API_KEY -model "$M" -filter 3GPP -n $N -seed 42 -workers 8 -mcp '' -out "results/$M-notools-3gpp-${N}q-seed42.jsonl"
