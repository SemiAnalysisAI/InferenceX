#!/usr/bin/env bash
# 8k-in / 1k-out random benchmark at concurrency 32 against the patched DEP8
# server. Standalone `vllm bench serve` (NOT the agentic aiperf replay) -- the
# quick throughput smoke the user asked for first.
set -uo pipefail

BASE=http://localhost:8000
SERVED=deepseek-ai/DeepSeek-V4-Pro
LOG=/home/jiacao/InferenceX/dsv4-bench-8k1k-c32.log

exec vllm bench serve \
    --backend openai-chat --endpoint /v1/chat/completions \
    --base-url "$BASE" --model "$SERVED" \
    --dataset-name random \
    --random-input-len 8000 --random-output-len 1000 \
    --max-concurrency 32 --num-prompts 128 \
    --ignore-eos --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --result-dir /home/jiacao/InferenceX \
    --result-filename dsv4-bench-8k1k-c32.json > "$LOG" 2>&1
