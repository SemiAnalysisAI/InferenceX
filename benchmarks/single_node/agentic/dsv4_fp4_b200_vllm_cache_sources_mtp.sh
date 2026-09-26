#!/usr/bin/env bash
set -eo pipefail
set -x

# The original DeepSeek-V4-Pro TP8/MTP3 cache-source validation recipe.
# Preserve run 34420875287's cache pressure independently of the 0813 DSpark curve.
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars MODEL MODEL_PATH TP CONC KV_OFFLOADING KV_OFFLOAD_BACKEND
check_env_vars TOTAL_CPU_DRAM_GB RESULT_DIR DURATION PORT GPU_MEMORY_UTILIZATION
check_env_vars EP_SIZE DP_ATTENTION DCP_SIZE PCP_SIZE EVAL_ONLY THINKING_MODE

if [[ "$TP" != 8 || "$EP_SIZE" != 1 || "$DP_ATTENTION" != false || "$DCP_SIZE" != 1 || "$PCP_SIZE" != 1 ]]; then
    echo "Cache-source validation requires TP8 without EP, DP attention, or context parallelism" >&2
    exit 1
fi
export GPU_COUNT=8
[[ -d "$MODEL_PATH" ]] || { echo "Missing pre-staged model: $MODEL_PATH" >&2; exit 1; }
nvidia-smi
resolve_trace_source
install_agentic_deps

export AIPERF_SERVER_METRICS_URLS="http://localhost:$PORT/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:prompt_tokens_cached_by_source"
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=32768
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_RUST_FRONTEND=0
export VLLM_DSV4_MEGA_FP8_COMBINE=1
export VLLM_RPC_TIMEOUT=600000
export PYTHONHASHSEED=42
export TORCH_CUDA_ARCH_LIST=10.0
export PYTHONNOUSERSITE=1
export VLLM_FLOAT32_MATMUL_PRECISION=high

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"
case "$KV_OFFLOAD_BACKEND" in
    vllm-native)
        require_agentic_kv_offload_backend vllm-native "dram dram+nvme"
        SECONDARY_TIERS='[]'
        if [[ "$KV_OFFLOADING" == dram+nvme ]]; then
            check_env_vars NVME_OFFLOAD_DIR
            SECONDARY_TIERS="[{\"type\":\"fs\",\"root_dir\":\"$NVME_OFFLOAD_DIR\",\"locality\":\"LOCAL\"}]"
        fi
        OFFLOAD_CONFIG="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":$((TOTAL_CPU_DRAM_GB * 1000000000)),\"secondary_tiers\":$SECONDARY_TIERS}}"
        ;;
    vllm-simple)
        require_agentic_kv_offload_backend vllm-simple nvme
        check_env_vars NVME_OFFLOAD_DIR
        OFFLOAD_CONFIG="{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"kv_offload_backend\":\"disk\",\"disk_path\":\"$NVME_OFFLOAD_DIR/cache.bin\",\"disk_capacity_bytes\":$((1000000000000 / GPU_COUNT)),\"disk_buffer_slots\":4,\"lazy_offload\":false}}"
        ;;
    *) echo "Unsupported validation offload backend: $KV_OFFLOAD_BACKEND" >&2; exit 1 ;;
esac

# Use the same committed golden MTP curve as the shared SRT selector.
SPEC_CONFIG=$(python3 - <<'PY'
import json
import os
from infx.srt_slurm.synthetic_acceptance import GOLDEN_DIR, golden_length

spec = {"method": "mtp", "num_speculative_tokens": 3}
if os.environ["EVAL_ONLY"] != "true":
    spec["rejection_sample_method"] = "synthetic"
    spec["synthetic_acceptance_length"] = golden_length(
        "dsv4", spec, os.environ["THINKING_MODE"], GOLDEN_DIR
    )
print(json.dumps(spec))
PY
)
MAX_NUM_SEQS=$((2 * CONC))
CAPTURE_SIZE_LIST=()
for ((num_seqs = 1; num_seqs <= MAX_NUM_SEQS; num_seqs++)); do
    CAPTURE_SIZE_LIST+=("$((num_seqs * 4))")
done
CAPTURE_SIZE_LIST+=(100 200 300 400 500)
CUDA_GRAPH_CAPTURE_SIZES=$(printf '%s\n' "${CAPTURE_SIZE_LIST[@]}" | sort -n -u | paste -sd, -)
COMPILATION_CONFIG="{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[${CUDA_GRAPH_CAPTURE_SIZES}]}"

{ set +x; } 2>/dev/null
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0 --port "$PORT" --trust-remote-code
    --kv-cache-dtype fp8 --block-size 256 --max-model-len 1048576
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --numa-bind --enable-cumem-allocator --no-enable-flashinfer-autotune
    --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4
    --enable-auto-tool-choice --reasoning-parser deepseek_v4
    --attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV4","use_prefill_query_quantization":true,"indexer_kv_dtype":"mxfp4"}'
    --speculative-config "$SPEC_CONFIG" --no-disable-hybrid-kv-cache-manager
    --disable-uvicorn-access-log --compilation-config "$COMPILATION_CONFIG"
    --max-num-seqs "$MAX_NUM_SEQS" --tensor-parallel-size "$TP" --data-parallel-size 1
    --kv-transfer-config "$OFFLOAD_CONFIG"
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
sha256sum -c /opt/pr56318-validation/validation-overlay.sha256 | tee "$RESULT_DIR/overlay-verification.log"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "$EVAL_ONLY" == true ]]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    curl --fail --silent --show-error "$AIPERF_SERVER_METRICS_URLS" > "$RESULT_DIR/metrics-before.prom"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
    curl --fail --silent --show-error "$AIPERF_SERVER_METRICS_URLS" > "$RESULT_DIR/metrics-after.prom"
fi
