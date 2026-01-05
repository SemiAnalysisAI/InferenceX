#!/usr/bin/bash

# === Required Env Vars ===
# MODEL
# TP
# CONC
# ISL
# OSL
# RANDOM_RANGE_RATIO
# RESULT_FILENAME

echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"

hf download $MODEL

SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)
PORT=8888

# === Monkey Patch for MoE Debug Logging (optional) ===
# Enable by setting MOE_DEBUG=1. When enabled, we set MOE_DEBUG_LOG (if not provided)
PATCH_DIR=$(mktemp -d /tmp/moe_patch-XXXXXX)
cat << 'EOF' > "$PATCH_DIR/sitecustomize.py"
import os
import time
import atexit
import threading
import builtins

_original_import = builtins.__import__
_patched = False

def _get_rank() -> int:
    for k in ("SGLANG_TP_RANK", "TP_RANK", "LOCAL_RANK", "RANK"):
        v = os.environ.get(k)
        if v is not None:
            try:
                return int(v)
            except ValueError:
                pass
    return 0

_RANK = _get_rank()
_ONLY_RANK = int(os.environ.get("MOE_DEBUG_ONLY_RANK", "0"))
_ENABLED = (_RANK == _ONLY_RANK)

_LOG_PATH = os.environ.get("MOE_DEBUG_LOG", "/tmp/moe_debug.tp0.log")
_fh = None
_seq = 0
_lock = threading.Lock()

def _log(msg: str) -> None:
    if not _ENABLED:
        return
    global _fh
    global _seq
    if _fh is None:
        os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
        _fh = open(_LOG_PATH, "a", buffering=1)
        atexit.register(lambda: _fh and _fh.close())
    ts = time.time()
    with _lock:
        _seq += 1
        _fh.write(f"seq={_seq} ts={ts:.6f} [pid={os.getpid()} rank={_RANK}] {msg}\n")
        _fh.flush()

def _apply_moe_debug_patch(fused_moe_module):
    if not _ENABLED:
        return
    import torch
    original = fused_moe_module.fused_experts_impl
    def patched(hidden_states, w1, w2, topk_weights, topk_ids, *args, **kwargs):
        num_tokens = hidden_states.shape[0]
        topk = topk_ids.shape[1]
        num_activated = torch.unique(topk_ids).numel()
        _log(f"batch_size={num_tokens} top_k={topk} activated_experts={num_activated}")
        return original(hidden_states, w1, w2, topk_weights, topk_ids, *args, **kwargs)
    fused_moe_module.fused_experts_impl = patched
    _log("Patch applied successfully")

def _patching_import(name, globals=None, locals=None, fromlist=(), level=0):
    global _patched
    module = _original_import(name, globals, locals, fromlist, level)
    if not _patched and name == "sglang.srt.layers.moe.fused_moe_triton.fused_moe":
        _patched = True
        _apply_moe_debug_patch(module)
    return module

builtins.__import__ = _patching_import
_log("Import hook installed")
EOF

# Inject sitecustomize.py to patch fused_moe and log activations when enabled.
if [[ "${MOE_DEBUG:-}" == "1" ]]; then
    export MOE_DEBUG_LOG="${MOE_DEBUG_LOG:-/workspace/moe_debug.tp0.log}"
    export MOE_DEBUG_ONLY_RANK="${MOE_DEBUG_ONLY_RANK:-0}"
    pip3 install --no-deps --target "$PATCH_DIR" sentencepiece
    export PYTHONPATH="$PATCH_DIR:${PYTHONPATH:-}"
    echo "[MoE Debug] Patch directory: $PATCH_DIR"
    echo "[MoE Debug] PYTHONPATH: $PYTHONPATH"
else
    echo "[MoE Debug] Disabled (set MOE_DEBUG=1 to enable)"
fi

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

set -x
python3 -m sglang.launch_server \
--model-path=$MODEL --host=0.0.0.0 --port=$PORT --trust-remote-code \
--tensor-parallel-size=$TP \
--mem-fraction-static=0.8 \
--cuda-graph-max-bs=128 \
--chunked-prefill-size=196608 \
--num-continuous-decode-steps=4 \
--max-prefill-tokens=196608 \
--disable-radix-cache \
> $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Source benchmark utilities
source "$(dirname "$0")/benchmark_lib.sh"

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# If profiling is enabled, start profiling via SGLang HTTP API (managed by benchmark_serving)
if [[ "${PROFILE:-}" == "1" ]]; then
    SGLANG_TORCH_PROFILER_DIR="${SGLANG_TORCH_PROFILER_DIR:-/workspace}"
    mkdir -p "$SGLANG_TORCH_PROFILER_DIR"
    echo "[PROFILE] Using benchmark_serving managed profiling (--profile); dir=$SGLANG_TORCH_PROFILER_DIR"
fi

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
    --result-dir /workspace/ \
    &
BENCH_PID=$!

wait "$BENCH_PID"

ls -lt "${SGLANG_TORCH_PROFILER_DIR:-/workspace}"

if [[ "${PROFILE:-}" == "1" ]]; then
  SGLANG_TORCH_PROFILER_DIR="${SGLANG_TORCH_PROFILER_DIR:-/workspace}"
  # Wait briefly for the file to appear (auto-stop writes it)
  TRACE_FILE=""
  for _ in {1..180}; do
    TRACE_FILE=$(ls -t "$SGLANG_TORCH_PROFILER_DIR"/*.trace.json* 2>/dev/null | head -n1)
    [[ -n "$TRACE_FILE" ]] && break
    sleep 1
  done

  if [[ -n "$TRACE_FILE" ]]; then
    DEST_TRACE="/workspace/profile_${RESULT_FILENAME}.trace.json.gz"
    MERGED_TRACE=$(ls -t "$SGLANG_TORCH_PROFILER_DIR"/merged-*.trace.json* 2>/dev/null | head -n1)
    if [[ -n "$MERGED_TRACE" ]]; then
      echo "[PROFILE] Running MFU analyzer on merged trace: $MERGED_TRACE (GPU=MI325X)"
      PYTHONNOUSERSITE=1 python3 utils/mfu_trace_analyzer.py "$MERGED_TRACE" "$MERGED_TRACE" --gpu MI325X --tp $TP --decode-batch-size 2 || echo "[PROFILE] MFU analyzer failed; continuing without modification"
    else
      echo "[PROFILE] No merged trace found; analyzing selected trace: $TRACE_FILE (GPU=MI325X)"
      PYTHONNOUSERSITE=1 python3 utils/mfu_trace_analyzer.py "$TRACE_FILE" "$TRACE_FILE" --gpu MI325X --tp $TP --decode-batch-size 2 || echo "[PROFILE] MFU analyzer failed; continuing without modification"
    fi
    echo "[PROFILE] Found trace: $TRACE_FILE -> $DEST_TRACE"
    cp "$TRACE_FILE" "$DEST_TRACE"
  else
    echo "[PROFILE] No trace found under $SGLANG_TORCH_PROFILER_DIR" >&2
  fi
fi
