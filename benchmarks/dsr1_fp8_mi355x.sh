#!/usr/bin/env bash

source "$(dirname "$0")/benchmark_lib.sh"

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

hf download "$MODEL"

# Reference
# https://rocm.docs.amd.com/en/docs-7.0-docker/benchmark-docker/inference-sglang-deepseek-r1-fp8.html

export SGLANG_USE_AITER=1
export RCCL_MSCCL_ENABLE=0
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

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

# ---- Rank gating (log from ONE TP process only) ----
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

# ---- File logger ----
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
        _fh = open(_LOG_PATH, "a", buffering=1)  # line-buffered
        atexit.register(lambda: _fh and _fh.close())
    ts = time.time()
    with _lock:
        _seq += 1
        _fh.write(f"seq={_seq} ts={ts:.6f} [pid={os.getpid()} rank={_RANK}] {msg}\n")
        _fh.flush()

def _patching_import(name, globals=None, locals=None, fromlist=(), level=0):
    global _patched
    module = _original_import(name, globals, locals, fromlist, level)

    # Patch after sglang's fused_moe module is loaded
    if not _patched and name == "sglang.srt.layers.moe.fused_moe_triton.fused_moe":
        _patched = True
        _apply_moe_debug_patch(module)

    return module

def _apply_moe_debug_patch(fused_moe_module):
    # If not the chosen rank, do nothing (no wrapper, zero overhead)
    if not _ENABLED:
        return

    import torch

    original_fused_experts_impl = fused_moe_module.fused_experts_impl

    def patched_fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        *args,
        **kwargs
    ):
        num_tokens = hidden_states.shape[0]
        topk = topk_ids.shape[1]
        num_activated = torch.unique(topk_ids).numel()

        _log(f"batch_size={num_tokens} top_k={topk} activated_experts={num_activated}")
        return original_fused_experts_impl(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            *args,
            **kwargs
        )

    fused_moe_module.fused_experts_impl = patched_fused_experts_impl
    _log("Patch applied successfully")

# Install the patching import hook (quiet; no stdout)
builtins.__import__ = _patching_import
_log("Import hook installed")
EOF

# and inject a sitecustomize.py that patches fused_moe to log activations.
if [[ "${MOE_DEBUG:-}" == "1" ]]; then
    # Write MoE debug to its own file (per run) if not already set
    export MOE_DEBUG_LOG="${MOE_DEBUG_LOG:-/workspace/moe_debug.tp0.log}"
    # Only emit logs from TP rank 0
    export MOE_DEBUG_ONLY_RANK="${MOE_DEBUG_ONLY_RANK:-0}"

    pip3 install --no-deps --target "$PATCH_DIR" sentencepiece

    # Create sitecustomize.py - Python automatically imports this at startup
    # Set PYTHONPATH so sitecustomize.py is found and loaded automatically
    export PYTHONPATH="$PATCH_DIR:${PYTHONPATH:-}"

    echo "[MoE Debug] Patch directory: $PATCH_DIR"
    echo "[MoE Debug] PYTHONPATH: $PYTHONPATH"
else
    echo "[MoE Debug] Disabled (set MOE_DEBUG=1 to enable)"
fi

python3 -m sglang.launch_server \
    --attention-backend aiter \
    --model-path $MODEL \
    --host=0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size $TP \
    --trust-remote-code \
    --chunked-prefill-size 196608 \
    --mem-fraction-static 0.8 --disable-radix-cache \
    --num-continuous-decode-steps 4 \
    --max-prefill-tokens 196608 \
    --cuda-graph-max-bs $CONC > $SERVER_LOG 2>&1 &

SERVER_PID=$!

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
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
    append_lm_eval_summary
fi
set +x
