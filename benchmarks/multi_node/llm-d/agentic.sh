#!/usr/bin/env bash
# Client-only AgentX adapter for an already-ready llm-d Envoy frontend.
set -eo pipefail

: "${INFMAX_CONTAINER_WORKSPACE:?Set the repository mount path}"
export MODEL="$MODEL_NAME"
export SERVED_MODEL_NAME="$MODEL_NAME"
export PORT="$VLLM_PORT"
export AIPERF_SERVER_URL="http://localhost:$ENVOY_PORT"
export RESULT_DIR="$BENCHMARK_LOGS_DIR/agentic"
export AGENTIC_OUTPUT_DIR="$BENCHMARK_LOGS_DIR"
export CONC_LIST="${BENCH_MAX_CONCURRENCY//x/ }"
export CONC="${CONC_LIST%% *}"

# Envoy does not expose vLLM /metrics. Scrape each engine leader directly;
# headless members of multi-node engines do not run HTTP servers.
IFS=',' read -r -a ips <<< "$ALL_IPS"
metrics_urls=()
for ((i=0; i<PREFILL_NODES; i+=PREFILL_NODES/PREFILL_WORKERS)); do
    metrics_urls+=("http://${ips[$i]}:$VLLM_PORT/metrics")
done
for ((i=PREFILL_NODES; i<PREFILL_NODES+DECODE_NODES; i+=DECODE_NODES/DECODE_WORKERS)); do
    metrics_urls+=("http://${ips[$i]}:$VLLM_PORT/metrics")
done
export AIPERF_SERVER_METRICS_URLS="$(IFS=,; echo "${metrics_urls[*]}")"

exec bash "$INFMAX_CONTAINER_WORKSPACE/benchmarks/multi_node/agentic_srt.sh"
