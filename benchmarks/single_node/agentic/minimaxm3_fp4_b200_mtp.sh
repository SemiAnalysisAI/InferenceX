#!/usr/bin/env bash
set -euo pipefail
set -x

# MiniMax-M3 NVFP4 B200 AgentX with EAGLE3-GQA and synthetic acceptance.
# DRAM KV offload uses vLLM's SimpleCPUOffloadConnector in lazy mode.
#
# Port of the validated agentic/minimaxm3_fp4_b300_mtp.sh. Spec-decode only,
# per the AgentX policy that agentic recipes are run and published with
# speculative decoding enabled rather than as an STP/MTP A/B (MODELS.md:
# MiniMax-M3 agentic non-EAGLE3 is deprecated after 2026-08-03, and the 8k1k
# scenario that carried the B200 MiniMax-M3 curves was removed in #2493 --
# without this recipe MiniMax-M3 has no active B200 config at all).
#
# The only B200 deltas are the two blocks marked "B200:" below -- the
# checkpoint-resolution guard and the draft staging path. Every engine flag is
# the B300 script unchanged so the two SKU curves stay directly comparable.
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION

source "$(dirname "$0")/../../benchmark_lib.sh"

export EVAL_FRAMEWORK="lm-eval"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION

DRAFT_MODEL="Inferact/MiniMax-M3-EAGLE3-GQA"
NUM_SPEC_TOKENS=3
# Golden AL for the GQA draft head: golden_al_distribution/minimaxm3_eagle3_gqa.yaml
# minimax-m3.thinking_on[3]. The non-GQA curve (minimaxm3_eagle3.yaml) reads 2.83
# at the same level -- that head is not what this script runs. The curve is a
# property of the target/draft pair, not of the SKU it was measured on, so B200
# pins the same 2.78 the B300 sibling does.
SYNTHETIC_ACCEPT_LEN=2.78

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# A non-empty directory is NOT a staged checkpoint: an aborted pull leaves
# config.json and friends behind with no weights, and an `ls -A` guard accepts
# it, so the cell skips the download and dies later inside the loader. Check
# that every shard the index names is actually present -- and accept the
# single-file layout the EAGLE3 draft head ships in, which has no index.
checkpoint_is_complete() {
    local dir="$1"
    [[ -d "$dir" && -f "$dir/config.json" ]] || return 1
    CKPT_DIR="$dir" python3 - <<'PYEOF'
import glob, json, os, sys

d = os.environ["CKPT_DIR"]
index = os.path.join(d, "model.safetensors.index.json")
if os.path.isfile(index):
    with open(index) as fh:
        shards = sorted(set(json.load(fh)["weight_map"].values()))
    missing = [s for s in shards if not os.path.isfile(os.path.join(d, s))]
    if missing:
        print(
            f"{len(missing)}/{len(shards)} shards missing, e.g. {missing[:3]}",
            file=sys.stderr,
        )
        sys.exit(1)
elif not glob.glob(os.path.join(d, "*.safetensors")):
    print("no shard index and no .safetensors present", file=sys.stderr)
    sys.exit(1)
PYEOF
}

# B200: runners/launch_b200-dgxc.sh resolves the checkpoint to a cluster-local
# path (/scratch/fsw/models/MiniMax-M3-NVFP4) and then rewrites MODEL to that
# path before handing off to this script, so `hf download "$MODEL"` cannot work
# on this runner the way it does on b300-nv, where MODEL stays the HF repo id.
# Keep the repo id separate for the case where the checkpoint is not staged.
HF_MODEL_ID="${HF_MODEL_ID:-nvidia/MiniMax-M3-NVFP4}"

if [[ -n "${MODEL_PATH:-}" ]]; then
    if ! checkpoint_is_complete "$MODEL_PATH"; then
        # Every concurrency of this sweep runs as its own allocation against
        # the same shared path, so serialize: one cell pulls the ~250 GB
        # checkpoint and the rest wait on it rather than racing as writers.
        # `hf download` resumes into a partially-populated --local-dir.
        mkdir -p "$MODEL_PATH"
        MODEL_DOWNLOAD_LOCK="${MODEL_PATH%/}.download.lock"
        echo "Checkpoint at $MODEL_PATH is incomplete; acquiring $MODEL_DOWNLOAD_LOCK"
        exec 9>"$MODEL_DOWNLOAD_LOCK"
        flock -w "${MODEL_DOWNLOAD_LOCK_TIMEOUT:-21600}" 9 || {
            echo "Error: timed out waiting for another cell to stage $MODEL_PATH" >&2
            exit 1
        }
        if checkpoint_is_complete "$MODEL_PATH"; then
            echo "Another cell staged $MODEL_PATH while we waited"
        else
            hf download "$HF_MODEL_ID" --local-dir "$MODEL_PATH"
        fi
        flock -u 9
        exec 9>&-
        checkpoint_is_complete "$MODEL_PATH" || {
            echo "Error: $MODEL_PATH is still incomplete after hf download $HF_MODEL_ID." >&2
            exit 1
        }
    fi
else
    hf download "$HF_MODEL_ID"
    export MODEL_PATH="$HF_MODEL_ID"
fi

# B200: the B300 sibling stages the draft under /data/models, which does not
# exist on b200-dgxc. The launcher bind-mounts only $MODEL_PATH itself
# (--container-mounts=...,$MODEL_PATH:$MODEL_PATH,...), so its parent exists
# solely inside the container overlay -- writable, but per-job. Stage the draft
# there rather than inside $MODEL_PATH, which is the shared read-mostly
# checkpoint directory and must not be polluted. The EAGLE3-GQA head is small
# next to the target, so the per-job pull is cheap; this is what the
# now-deprecated 8k1k B200 MiniMax-M3 MTP recipe did as well.
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-$(dirname "${MODEL_PATH%/}")/${DRAFT_MODEL##*/}}"
if ! checkpoint_is_complete "$DRAFT_MODEL_PATH"; then
    hf download "$DRAFT_MODEL" --local-dir "$DRAFT_MODEL_PATH"
    checkpoint_is_complete "$DRAFT_MODEL_PATH" || {
        echo "Error: $DRAFT_MODEL_PATH is incomplete after hf download $DRAFT_MODEL." >&2
        exit 1
    }
fi

nvidia-smi
resolve_trace_source
install_agentic_deps

OFFLOAD_ARGS=()
if require_agentic_kv_offload_backend vllm-simple; then
    CPU_OFFLOAD_BYTES=$((TOTAL_CPU_DRAM_GB * 1024 * 1024 * 1024))
    export VLLM_USE_SIMPLE_KV_OFFLOAD=1
    OFFLOAD_CONFIG=$(printf \
        '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":%d,"lazy_offload":true}}' \
        "$CPU_OFFLOAD_BYTES")
    OFFLOAD_ARGS=(--kv-transfer-config "$OFFLOAD_CONFIG")
fi

export PYTHONNOUSERSITE=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_FLOAT32_MATMUL_PRECISION=high
export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm

# Same 0.9 the B300 sibling and the deprecated 8k1k B200 MiniMax-M3 NVFP4 MTP
# recipe ran. Exposed as an override because B200's 180 GB leaves little beyond
# the ~250 GB checkpoint: TP2 could not host the 1M-context KV for even one
# request at this value and was dropped from the search space, so TP4 is the
# smallest topology this script is expected to serve.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

SERVER_PID=""
cleanup_agentic_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "vLLM server" 60
    exit "$exit_code"
}
trap cleanup_agentic_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [ "${EVAL_ONLY:-}" = "true" ]; then
    SPEC_CONFIG=$(printf \
        '{"method":"eagle3","model":"%s","num_speculative_tokens":%d,"attention_backend":"FLASH_ATTN"}' \
        "$DRAFT_MODEL_PATH" "$NUM_SPEC_TOKENS")
else
    SPEC_CONFIG=$(printf \
        '{"method":"eagle3","model":"%s","num_speculative_tokens":%d,"attention_backend":"FLASH_ATTN","rejection_sample_method":"synthetic","synthetic_acceptance_length":%.2f}' \
        "$DRAFT_MODEL_PATH" "$NUM_SPEC_TOKENS" "$SYNTHETIC_ACCEPT_LEN")
fi

{ set +x; } 2>/dev/null
VLLM_CMD=(
    vllm serve "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --tensor-parallel-size "$TP"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --block-size 128
    --language-model-only
    --enable-prefix-caching
    --no-enable-flashinfer-autotune
    --reasoning-parser minimax_m3
    --tool-call-parser minimax_m3
    --enable-auto-tool-choice
    --default-chat-template-kwargs '{"thinking_mode":"enabled"}'
    --attention-config '{"backend":"FLASHINFER","use_trtllm_attention":true,"indexer_kv_dtype":"fp8"}'
    --kv-cache-dtype fp8
    --max-cudagraph-capture-size 512
    --max-num-batched-tokens 16384
    --stream-interval 20
    --trust-remote-code
    --speculative-config "$SPEC_CONFIG"
    "${OFFLOAD_ARGS[@]}"
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"
set -x

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"
if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
