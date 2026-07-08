#!/usr/bin/env bash
set -euo pipefail
set -x

# Client-only fixed-seq-len benchmark for srt-slurm multinode jobs, using
# InferenceX's own bench_serving client instead of srt-slurm's downstream
# sa-bench fork. srt-slurm owns server startup; this script runs as
# benchmark.type=custom against the already-ready frontend.
#
# Recipe wiring (see agentic recipes for a working example of the hook):
#
#   benchmark:
#     type: custom
#     command: bash /infmax-workspace/benchmarks/multi_node/bench_serving_srt.sh
#     env:
#       INFMAX_CONTAINER_WORKSPACE: /infmax-workspace
#       RESULT_DIR: /logs
#       PORT: "8000"
#
# Results are written as results_concurrency_<C>_gpus_<G>[_ctx_<P>_gen_<D>].json
# into a bench_isl_<ISL>_osl_<OSL>/ subdirectory of RESULT_DIR, which is the
# exact layout the srt_slurm/run.sh collector (and sa-bench before it)
# produces, so downstream result processing is unchanged.

INFMAX_CONTAINER_WORKSPACE="${INFMAX_CONTAINER_WORKSPACE:-/infmax-workspace}"
source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"

check_env_vars MODEL PORT ISL OSL RANDOM_RANGE_RATIO

read -r -a CONCURRENCIES <<< "${CONC_LIST:-${CONC:-}}"
if [ "${#CONCURRENCIES[@]}" -eq 0 ]; then
    echo "ERROR: CONC_LIST (or CONC) must contain at least one concurrency" >&2
    exit 1
fi
for concurrency in "${CONCURRENCIES[@]}"; do
    if ! [[ "$concurrency" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: invalid concurrency: $concurrency" >&2
        exit 1
    fi
done

# GPU counts feed the result filename that downstream tooling parses for
# perf-per-GPU. Prefer explicit values from the recipe's benchmark.env;
# fall back to the worker topology env the workflow exports.
PREFILL_GPUS="${PREFILL_GPUS:-$(( ${PREFILL_NUM_WORKERS:-0} * ${PREFILL_TP:-0} ))}"
DECODE_GPUS="${DECODE_GPUS:-$(( ${DECODE_NUM_WORKERS:-0} * ${DECODE_TP:-0} ))}"
TOTAL_GPUS="${TOTAL_GPUS:-$(( PREFILL_GPUS + DECODE_GPUS ))}"
if [ "$TOTAL_GPUS" -le 0 ]; then
    echo "ERROR: set TOTAL_GPUS (or PREFILL_*/DECODE_* worker topology env) so results can be attributed" >&2
    exit 1
fi

RESULT_ROOT="${RESULT_DIR:-/logs}/bench_isl_${ISL}_osl_${OSL}"
mkdir -p "$RESULT_ROOT"

# The frontend serves the model under its served-model-name, which may
# differ from the HF id in MODEL.
BENCH_MODEL="${SERVED_MODEL_NAME:-$MODEL}"
BENCH_BACKEND="${BENCH_BACKEND:-vllm}"

for concurrency in "${CONCURRENCIES[@]}"; do
    if [ "${DISAGG:-true}" == "true" ] && [ "$PREFILL_GPUS" -gt 0 ] && [ "$DECODE_GPUS" -gt 0 ]; then
        result_name="results_concurrency_${concurrency}_gpus_${TOTAL_GPUS}_ctx_${PREFILL_GPUS}_gen_${DECODE_GPUS}"
    else
        result_name="results_concurrency_${concurrency}_gpus_${TOTAL_GPUS}"
    fi

    echo "Running concurrency $concurrency of: ${CONCURRENCIES[*]}"
    extra_args=()
    [ "${BENCH_USE_CHAT_TEMPLATE:-0}" == "1" ] && extra_args+=(--use-chat-template)
    [ "${BENCH_DSV4:-0}" == "1" ] && extra_args+=(--dsv4)
    [ "${BENCH_TRUST_REMOTE_CODE:-0}" == "1" ] && extra_args+=(--trust-remote-code)
    [ -n "${BENCH_TOKENIZER:-}" ] && extra_args+=(--tokenizer "$BENCH_TOKENIZER")
    [ -n "${BENCH_ENDPOINT:-}" ] && extra_args+=(--endpoint "$BENCH_ENDPOINT")

    run_benchmark_serving \
        --model "$BENCH_MODEL" \
        --port "$PORT" \
        --backend "$BENCH_BACKEND" \
        --input-len "$ISL" \
        --output-len "$OSL" \
        --random-range-ratio "$RANDOM_RANGE_RATIO" \
        --num-prompts $(( concurrency * 10 )) \
        --max-concurrency "$concurrency" \
        --result-filename "$result_name" \
        --result-dir "$RESULT_ROOT" \
        --bench-serving-dir "$INFMAX_CONTAINER_WORKSPACE" \
        "${extra_args[@]}"
done
