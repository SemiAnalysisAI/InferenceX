#!/usr/bin/env bash

# === Required Env Vars ===
# MODEL
# TP
# CONC
# ISL
# OSL
# RANDOM_RANGE_RATIO
# RESULT_FILENAME
# PORT_OFFSET

echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"

pip3 install --user sentencepiece
hf download $MODEL
PORT=$(( 8888 + $PORT_OFFSET ))
SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)

export TORCH_CUDA_ARCH_LIST="9.0"


# === Monkey Patch for MoE Debug Logging ===
# Create a directory for our patch
PATCH_DIR=$(mktemp -d /tmp/moe_patch-XXXXXX)

# Create sitecustomize.py - Python automatically imports this at startup
cat << 'EOF' > "$PATCH_DIR/sitecustomize.py"
import sys
import builtins

_original_import = builtins.__import__

_patched = False

def _patching_import(name, globals=None, locals=None, fromlist=(), level=0):
    global _patched
    module = _original_import(name, globals, locals, fromlist, level)
    
    # Patch after sglang's fused_moe module is loaded
    if not _patched and name == "sglang.srt.layers.moe.fused_moe_triton.fused_moe":
        _patched = True
        _apply_moe_debug_patch(module)
    
    return module

def _apply_moe_debug_patch(fused_moe_module):
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
        E = w1.shape[0]
        topk = topk_ids.shape[1]
        unique_experts = torch.unique(topk_ids)
        num_activated = unique_experts.numel()
        
        print(f"[MoE Debug] batch_size={num_tokens}, "
              f"top_k={topk}, "
              f"total_experts={E}, "
              f"activated_experts={num_activated}, "
              f"expert_ids={unique_experts.tolist()}", 
              flush=True)
        
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
    print("[MoE Debug] Patch applied successfully", flush=True)

# Install the patching import hook
builtins.__import__ = _patching_import
print("[MoE Debug] Import hook installed", flush=True)
EOF

# Set PYTHONPATH so sitecustomize.py is found and loaded automatically
export PYTHONPATH="$PATCH_DIR:${PYTHONPATH:-}"

echo "[MoE Debug] Patch directory: $PATCH_DIR"
echo "[MoE Debug] PYTHONPATH: $PYTHONPATH"

set -x
if [[ $ISL -eq 1024 && $OSL -eq 1024 ]]; then
    python3 -m sglang.launch_server --model-path $MODEL --tokenizer-path $MODEL \
    --host 0.0.0.0 --port $PORT --trust-remote-code \
    --tensor-parallel-size=$TP --data-parallel-size=1 \
    --disable-radix-cache --max-running-requests 512 --cuda-graph-max-bs 0 \
    --chunked-prefill-size 32768 --max-prefill-tokens 32768 --mem-fraction-static 0.82 \
    --attention-backend flashinfer --stream-interval 10 \
    --decode-log-interval 1 --disable-cuda-graph \
    > $SERVER_LOG 2>&1 &
else
    python3 -m sglang.launch_server --model-path $MODEL --tokenizer-path $MODEL \
    --host 0.0.0.0 --port $PORT --trust-remote-code \
    --tensor-parallel-size=$TP --data-parallel-size=1 \
    --disable-radix-cache --max-running-requests 256 --cuda-graph-max-bs 256 \
    --chunked-prefill-size 32768 --max-prefill-tokens 32768 --mem-fraction-static 0.82 \
    --attention-backend flashinfer --stream-interval 10 \
    --decode-log-interval 1 \
    > $SERVER_LOG 2>&1 &
fi

SERVER_PID=$!

# Source benchmark utilities
source "$(dirname "$0")/benchmark_lib.sh"

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# If profiling is enabled, start profiling via SGLang HTTP API
if [[ "${PROFILE:-}" == "1" ]]; then
    SGLANG_TORCH_PROFILER_DIR="${SGLANG_TORCH_PROFILER_DIR:-/workspace}"
    mkdir -p "$SGLANG_TORCH_PROFILER_DIR"
fi

if [[ "${PROFILE:-}" == "1" ]]; then
  echo "[PROFILE] Will start mid-run; dir=$SGLANG_TORCH_PROFILER_DIR"

  # Wait until the run has ramped up (tune this)
  #sleep "${PROFILE_DELAY_SECS:-60}"

  # Start a SMALL bounded capture (this auto-stops; do NOT call stop_profile)
  curl -sf -X POST "http://127.0.0.1:$PORT/start_profile" \
    -H "Content-Type: application/json" \
    -d "{
      \"output_dir\": \"$SGLANG_TORCH_PROFILER_DIR\",
      \"num_steps\": 5,
      \"start_step\": 0,
      \"activities\": [\"GPU\", \"CPU\"],
      \"merge_profiles\": true,
      \"profile_by_stage\": true,
      \"record_shapes\": true
    }" || true
fi

run_benchmark_serving \
  --model "$MODEL" \
  --port "$PORT" \
  --backend vllm \
  --input-len "$ISL" \
  --output-len "$OSL" \
  --random-range-ratio "$RANDOM_RANGE_RATIO" \
  --num-prompts 10 \
  --max-concurrency "$CONC" \
  --result-filename "$RESULT_FILENAME" \
  --result-dir /workspace/ \
  &
BENCH_PID=$!

wait "$BENCH_PID"

ls -lt "$SGLANG_TORCH_PROFILER_DIR"

if [[ "${PROFILE:-}" == "1" ]]; then
  # Wait briefly for the file to appear (auto-stop writes it)
  TRACE_FILE=""
  for _ in {1..180}; do
    TRACE_FILE=$(ls -t "$SGLANG_TORCH_PROFILER_DIR"/*.trace.json* 2>/dev/null | head -n1)
    [[ -n "$TRACE_FILE" ]] && break
    sleep 1
  done

  if [[ -n "$TRACE_FILE" ]]; then
    DEST_TRACE="/workspace/profile_${RESULT_FILENAME}.trace.json.gz"
    # If a merged profile exists, run MFU analyzer on it before copying
    MERGED_TRACE=$(ls -t "$SGLANG_TORCH_PROFILER_DIR"/merged-*.trace.json* 2>/dev/null | head -n1)
    if [[ -n "$MERGED_TRACE" ]]; then
      echo "[PROFILE] Running MFU analyzer on merged trace: $MERGED_TRACE"
      PYTHONNOUSERSITE=1 python3 utils/mfu_trace_analyzer.py "$MERGED_TRACE" "$MERGED_TRACE" --gpu H200 --tp $TP --decode-batch-size $CONC || echo "[PROFILE] MFU analyzer failed; continuing without modification"
    fi
    echo "[PROFILE] Found trace: $TRACE_FILE -> $DEST_TRACE"
    cp "$TRACE_FILE" "$DEST_TRACE"
  else
    echo "[PROFILE] No trace found under $SGLANG_TORCH_PROFILER_DIR" >&2
  fi
fi

# Archive server log to workspace for artifact upload
if [[ -n "${SERVER_LOG:-}" ]]; then
  DEST_SERVER_LOG="/workspace/server_${RESULT_FILENAME}.log"
  echo "[INFO] Copying server log to ${DEST_SERVER_LOG}"
  cp "$SERVER_LOG" "$DEST_SERVER_LOG" || true
fi
