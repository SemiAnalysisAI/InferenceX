#!/usr/bin/env bash
set -eo pipefail

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    EP_SIZE

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

echo "TP: $TP, CONC: $CONC, ISL: $ISL, OSL: $OSL, EP_SIZE: $EP_SIZE"

if [ "$EP_SIZE" -ne 1 ]; then
    echo "FATAL: DSv4 ATOM benchmark expects EP_SIZE=1, got $EP_SIZE" >&2
    exit 1
fi

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

export OMP_NUM_THREADS=1
export AITER_LOG_LEVEL=WARNING

# Keep the runtime overlay narrow: this benchmark uses the updated ATOM image
# from amd-master.yaml and only overlays ROCm/aiter#2998 for the DSv4 sparse
# MQA sink and Indexer top-k implementations until they are included in the
# image-provided AITER package.
if [ "${AITER_DSV4_PR2998:-1}" = "1" ]; then
    AITER_PR2998_REPO=${AITER_PR2998_REPO:-https://github.com/ROCm/aiter.git}
    AITER_PR2998_REF=${AITER_PR2998_REF:-pull/2998/head}
    AITER_PR2998_SHA=${AITER_PR2998_SHA:-aa0c5b6d97ffc6d4d11b8172dc848239f229c863}
    AITER_PR2998_DIR=${AITER_PR2998_DIR:-/tmp/aiter-dsv4-pr2998}

    rm -rf "$AITER_PR2998_DIR"
    git clone --filter=blob:none "$AITER_PR2998_REPO" "$AITER_PR2998_DIR"
    (
        cd "$AITER_PR2998_DIR"
        git fetch --depth=1 origin "$AITER_PR2998_REF"
        fetched_sha="$(git rev-parse FETCH_HEAD)"
        if [ "$fetched_sha" != "$AITER_PR2998_SHA" ]; then
            echo "FATAL: $AITER_PR2998_REF resolved to $fetched_sha, expected $AITER_PR2998_SHA" >&2
            exit 1
        fi
        git checkout --force FETCH_HEAD

        if [ ! -d 3rdparty/composable_kernel/include ]; then
            git submodule update --init --recursive --depth=1 3rdparty/composable_kernel \
                || git submodule update --init --recursive 3rdparty/composable_kernel
        fi

        PREBUILD_KERNELS=${AITER_PREBUILD_KERNELS:-0} \
        python3 -m pip install --no-deps --no-build-isolation --force-reinstall -e .
    )

    python3 - <<'PYEOF'
import inspect
from aiter.ops.triton.attention.dsv4_indexer import dsv4_indexer_topk
from aiter.ops.triton.attention.sparse_mqa_sink import sparse_mqa_sink

indexer_params = inspect.signature(dsv4_indexer_topk).parameters
missing = [name for name in ("seq_ids", "kv_lens") if name not in indexer_params]
if missing:
    raise SystemExit(f"FATAL: AITER PR2998 DSv4 Indexer API missing {missing}")
print("AITER PR2998 DSv4 sparse/indexer ops imported successfully")
PYEOF
else
    echo "WARN: AITER_DSV4_PR2998=0; using image-provided AITER"
fi

# DSv4-Pro advertises a 1M native context. Set the benchmark context
# explicitly so ATOM does not reserve KV cache for the full native length.
if [ "$ISL" = "1024" ] && [ "$OSL" = "1024" ]; then
    MAX_MODEL_LEN_VALUE=2304
else
    MAX_MODEL_LEN_VALUE=10240
fi
CALCULATED_MAX_MODEL_LEN=" --max-model-len $MAX_MODEL_LEN_VALUE "

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN_VALUE="$EVAL_MAX_MODEL_LEN"
    CALCULATED_MAX_MODEL_LEN=" --max-model-len $MAX_MODEL_LEN_VALUE "
fi

if [ "$EP_SIZE" -gt 1 ]; then
  EP=" --enable-expert-parallel"
else
  EP=" "
fi

start_gpu_monitor

set -x

BLOCK_SIZE=${BLOCK_SIZE:-128}
python3 -m atom.entrypoints.openai_server \
    --model "$MODEL" \
    --server-port "$PORT" \
    -tp "$TP" \
    --kv_cache_dtype fp8 $CALCULATED_MAX_MODEL_LEN $EP \
    --block-size "$BLOCK_SIZE" \
    --enforce-eager \
    --trust-remote-code > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --server-pid "$SERVER_PID" \
    --trust-remote-code

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
