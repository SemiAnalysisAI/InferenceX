#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

patch_sgl_components() {
    local aiter_ref="9046b6f446e35c5d712ca092b9d84a5db8319ef8"
    local sgl_kernel_ref="d40cb2f72551c4a597108dda16507e8188b666b7"

    if [[ ! -d /sgl-workspace ]]; then
        echo "/sgl-workspace not found; assuming image ships correct versions."
        return 0
    fi

    (
        set -e

        cd /sgl-workspace/aiter
        git fetch && git checkout "$aiter_ref"
        python setup.py develop
        echo "aiter ($aiter_ref) installed."

        cd /sgl-workspace/sgl-kernel
        git fetch && git checkout "$sgl_kernel_ref"
        python setup_rocm.py install
        echo "sgl-kernel ($sgl_kernel_ref) installed."

        cd /sgl-workspace/sglang
        rm -f python/pyproject.toml
        cp python/pyproject_other.toml python/pyproject.toml
        pip install -e "python[all_hip]"
        echo "sglang reinstalled."
    )
}
# Apply patch_sgl_components for lmsysorg/sglang:v0.5.8-rocm700-mi30x ONLY
patch_sgl_components

hf download "$MODEL"

# Reference
# https://rocm.docs.amd.com/en/docs-7.0-rc1/preview/benchmark-docker/inference-sglang-deepseek-r1-fp8.html#run-the-inference-benchmark

# If the machine runs a MEC FW older than 177, RCCL
# cannot reclaim some memory.
# Disable that features to avoid crashes.
# This is related to the changes in the driver at:
# https://rocm.docs.amd.com/en/docs-6.4.3/about/release-notes.html#amdgpu-driver-updates
version=`rocm-smi --showfw | grep MEC | head -n 1 |  awk '{print $NF}'`
if [[ "$version" == "" || $version -lt 177 ]]; then
  export HSA_NO_SCRATCH_RECLAIM=1
fi

export SGLANG_USE_AITER=1
export SGLANG_AITER_MLA_PERSIST=1

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

set -x
python3 -m sglang.launch_server \
--model-path=$MODEL --host=0.0.0.0 --port=$PORT --trust-remote-code \
--tensor-parallel-size=$TP \
--mem-fraction-static=0.8 \
--cuda-graph-max-bs=128 \
--chunked-prefill-size=131072 \
--num-continuous-decode-steps=4 \
--max-prefill-tokens=131072 \
--kv-cache-dtype fp8_e4m3 \
--attention-backend aiter \
--disable-radix-cache > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts $(( $CONC * 10 )) \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
    append_lm_eval_summary
fi
set +x
