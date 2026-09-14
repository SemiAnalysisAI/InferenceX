#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for DeepSeek-V4-Pro-0813 FP4 on MI355X using
# ATOM DSpark K6. All throughput runs use golden AL 3.77; eval uses real
# acceptance. The historical _mtp filename is also routed from draft_model.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

require_agentic_kv_offload_none

echo "Attention mode: $([ "$DP_ATTENTION" = "true" ] && echo dp || echo tp) (DP_ATTENTION=$DP_ATTENTION, CONC=$CONC)"

if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

if [[ "$MODEL" != "deepseek-ai/DeepSeek-V4-Pro-0813" ]]; then
    echo "ERROR: DSpark requires the DeepSeek-V4-Pro-0813 checkpoint, got $MODEL" >&2
    exit 1
fi
export DSV4_MODEL_REVISION=72e1d3230f6c080a530b0a1d46f8eb4602340597
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --revision "$DSV4_MODEL_REVISION" --local-dir "$MODEL_PATH"
    fi
else
    # ATOM has no --revision flag. Serve the resolved immutable snapshot path.
    MODEL_PATH=$(python3 - "$MODEL" "$DSV4_MODEL_REVISION" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2]))
PY
    )
fi
export MODEL_PATH
export AGENTIC_TOKENIZER_PATH="$MODEL_PATH"
mkdir -p "$RESULT_DIR"
if [ "$DP_ATTENTION" != "true" ]; then
    # DSpark q7 reaches padded TP MoE buckets. The pinned AITER callback must
    # restore InferenceMode before updating its preallocated inference tensors.
    python3 "$(dirname "$0")/patch_aiter_comm_fused_inference.py" \
        --output "$RESULT_DIR/aiter_runtime_fix.json"
fi
python3 "$(dirname "$0")/check_dsv4_dspark_checkpoint.py" \
    --model-path "$MODEL_PATH" --revision "$DSV4_MODEL_REVISION" \
    --output "$RESULT_DIR/checkpoint_preflight.json"

rocm-smi || true
amd-smi || true

resolve_trace_source
install_agentic_deps

# ATOM runtime settings validated with the DeepSeek-V4-Pro AgentX baseline.
export AITER_BF16_FP8_MOE_BOUND=0
export AITER_LOG_LEVEL=WARNING
export ATOM_MOE_GU_ITLV=1
export ATOM_DISABLE_MMAP=true
export ATOM_DEBUG_PREFIX_HITS=1
export ATOM_PROFILER_MORE=0
export ATOM_PROFILER_TIMEOUT=1200

# EP is config-driven so the TP band remains TP-only while DEP uses one expert
# shard per GPU.
EP_ARGS=()
if [ "$EP_SIZE" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
fi

# The high-concurrency band uses ATOM's native RCCL DEP transport. Session
# affinity is required: otherwise consecutive turns can land on another DPA
# rank and lose access to the prefix KV produced by the previous turn.
DEP_ARGS=()
STATE_CHECKPOINT_INTERVAL_TOKENS=8192
if [ "$DP_ATTENTION" = "true" ]; then
    if [ "$EP_SIZE" -ne "$TP" ]; then
        echo "ERROR: native RCCL DEP requires EP_SIZE=$TP for TP=$TP, got EP_SIZE=$EP_SIZE" >&2
        exit 1
    fi
    # Keep only runtime controls that are not already expressed by DEP_ARGS.
    export ATOM_DP_SESSION_AFFINITY=1
    export ATOM_DP_LB_REQ_EQUIV=512
    export ATOM_ENABLE_PREFILL_DELAYER=1
    export ATOM_PREFILL_DECODE_INTERVAL=10
    # Client-side counterpart to session affinity: make AIPerf emit a stable
    # session id from its correlation id so the DPA router pins each
    # conversation to one rank.
    export AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=1
    export AGENTIC_WARMUP_GRACE_PERIOD=3600
    DEP_ARGS=(
        --enable-dp-attention
        --all2all-backend rccl
        --dp-load-balance least_tokens
        --moe-backend standard
    )
fi

# Raise the AIPerf HTTP TCP user timeout to 900000 ms (15 min), well above the
# aiperf default of 30000 ms (30 s), so long-stalling AgentX request
# connections are not torn down as dead during extended server-side pauses.
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_UI_REALTIME_METRICS_ENABLED=true

# Require ATOM Prometheus metrics in every official result.
export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="atom:"

wait_for_amd_gpu_clean

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

# Record the server interpreter's installed sources once, without importing GPU
# packages or resolving AITER's merged tuning table. The bundled CSV is evidence
# of image contents, not proof of which kernels a serving request executes.
python3 - "$RESULT_DIR/runtime_manifest.json" <<'PY'
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
from importlib import metadata, util
from pathlib import Path


def package_manifest(name: str, distribution: str) -> dict:
    info = {"distribution": distribution, "version": None, "origin": None,
            "package_dir": None, "git": {"head": None, "dirty": None}}
    try:
        info["version"] = metadata.version(distribution)
    except (metadata.PackageNotFoundError, OSError) as exc:
        info["version_error"] = str(exc)
    try:
        spec = util.find_spec(name)
        if spec is None or spec.origin is None:
            info["source_error"] = "Package source was not found"
            return info
        origin = Path(spec.origin).resolve()
        info["origin"] = str(origin)
        info["package_dir"] = str(origin.parent)

        def git(*args: str) -> str:
            return subprocess.run(
                ["git", "-C", str(origin.parent), *args], check=True,
                capture_output=True, text=True, timeout=5,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            ).stdout.strip()

        root = Path(git("rev-parse", "--show-toplevel"))
        # Do not mistake an unrelated enclosing checkout for the package repo.
        git("ls-files", "--error-unmatch", "--", str(origin))
        info["git"].update(root=str(root), head=git("rev-parse", "HEAD"))
        info["git"]["dirty"] = bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, ValueError, ImportError, subprocess.SubprocessError) as exc:
        info["source_error"] = str(exc)
    return info


packages = {name: package_manifest(name, dist)
            for name, dist in (("atom", "atom"), ("aiter", "amd-aiter"))}
bundled = {"path": None, "sha256": None, "ep48_rows": [],
           "scope": "Bundled CSV only; runtime overrides and kernel dispatch are not resolved"}
aiter_dir = packages["aiter"]["package_dir"]
if aiter_dir is not None:
    path = Path(aiter_dir) / "configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv"
    bundled["path"] = str(path)
    try:
        data = path.read_bytes()
        bundled["sha256"] = hashlib.sha256(data).hexdigest()
        for row in csv.DictReader(io.StringIO(data.decode("utf-8"))):
            if (row.get("gfx") == "gfx950" and row.get("cu_num") == "256"
                    and row.get("model_dim") == "7168"
                    and row.get("inter_dim") == "3072" and row.get("expert") == "48"
                    and row.get("topk") == "6"
                    and row.get("token") in {"16384", "32768", "131072"}):
                bundled["ep48_rows"].append({key: row.get(key) for key in (
                    "gfx", "cu_num", "token", "model_dim", "inter_dim", "expert", "topk",
                    "block_m", "kernelName1", "kernelName2",
                )})
    except (OSError, ValueError, csv.Error) as exc:
        bundled["error"] = str(exc)
else:
    bundled["error"] = "AITER package source was not found"

fix_path = Path(sys.argv[1]).parent / "aiter_runtime_fix.json"
manifest = {
    "requested_image": os.environ.get("IMAGE"),
    "python_executable": sys.executable,
    "server_command_file": "server_command.txt",
    "checkpoint": json.loads((Path(sys.argv[1]).parent / "checkpoint_preflight.json").read_text()),
    "speculation": {
        "method": "dspark", "num_speculative_tokens": 6, "target_verify_length": 7,
        "forced_acceptance_length": None if os.environ.get("EVAL_ONLY") == "true" else 3.77,
        "confidence_schedule": False, "ragged": False,
    },
    "graph_evidence": "Requested FULL q7; capture completion must be checked in server.log",
    "packages": packages,
    "runtime_fixes": [json.loads(fix_path.read_text())] if fix_path.exists() else [],
    "aiter_overrides": {key: os.environ.get(key) for key in (
        "AITER_CONFIG_FMOE", "AITER_BYPASS_TUNE_CONFIG",
    )},
    "bundled_dsv4_fmoe": bundled,
}
Path(sys.argv[1]).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

SERVER_PID=""
cleanup_atom_server() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "ATOM server" 60
    exit "$exit_code"
}
trap cleanup_atom_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# AgentX concurrency counts session trees. Keep 2x scheduler headroom for the
# request bursts produced by subagent fan-out.
MAX_NUM_SEQS=$((2 * CONC))

# DPA splits the C48+ workload across eight ranks, so real decode batches are
# commonly 3, 5-7, and 9-15. ATOM's default power-of-two ladder rounds those
# shapes up and runs unnecessary attention, MoE, and collective work. Capture
# every small shape for DEP, while retaining larger graphs for the C96+ arms.
CUDAGRAPH_ARGS=()
if [ "$DP_ATTENTION" = "true" ]; then
    CUDAGRAPH_CAPTURE_SIZES='[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,48,64,128]'
    if [ "$MAX_NUM_SEQS" -gt 128 ]; then
        CUDAGRAPH_CAPTURE_SIZES='[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,48,64,128,256,512]'
    fi
    CUDAGRAPH_ARGS=(--cudagraph-capture-sizes "$CUDAGRAPH_CAPTURE_SIZES")
fi

# golden_al_distribution/dsv4-pro-0813-dspark.yaml: thinking_on, K6 -> AL 3.77.
# K6 means six draft tokens plus one target token (q7), matching SGLang.
# Apply the golden value to both throughput bands; eval must verify real drafts.
NUM_SPEC_TOKENS=6
SPEC_DECODE_AL=3.77
SPEC_ARGS=(
    --method dspark
    --num-speculative-tokens "$NUM_SPEC_TOKENS"
)
if [ "${EVAL_ONLY:-false}" != "true" ]; then
    SPEC_ARGS+=(--spec-decode-acceptance-length "$SPEC_DECODE_AL")
fi

echo "Starting ATOM server with MAX_NUM_SEQS=$MAX_NUM_SEQS NUM_SPEC_TOKENS=$NUM_SPEC_TOKENS STATE_CHECKPOINT_INTERVAL_TOKENS=$STATE_CHECKPOINT_INTERVAL_TOKENS DP_ATTENTION=$DP_ATTENTION EP_SIZE=$EP_SIZE EVAL_ONLY=${EVAL_ONLY:-false}"
ATOM_CMD=(
    python3 -u -m atom.entrypoints.openai_server
    --model "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --server-port "$PORT"
    # uvicorn defaults to a 5s idle keep-alive; AIPerf pools sockets for far
    # longer (aiohttp ~15s) and warmup inter-turn gaps under backlog exceed 5s,
    # so the server closes an idle pooled socket and the reused write hits
    # 'Connection reset by peer' (errno 104). One such reset on a root AgentX
    # warmup request aborts the whole run. Outlast the client idle window.
    --timeout-keep-alive 900
    --tensor-parallel-size "$TP"
    --data-parallel-size 1
    --kv-cache-dtype fp8
    --index-cache-dtype fp4
    --enable-prefix-caching
    --gpu-memory-utilization 0.9
    --max-num-batched-tokens 16384
    --attn-prefill-chunk-size 16384
    --state-checkpoint-interval-tokens "$STATE_CHECKPOINT_INTERVAL_TOKENS"
    --level 3
    --cudagraph-mode FULL
    "${CUDAGRAPH_ARGS[@]}"
    "${SPEC_ARGS[@]}"
    "${EP_ARGS[@]}"
    "${DEP_ARGS[@]}"
    --max-num-seqs "$MAX_NUM_SEQS"
)
write_command "$RESULT_DIR/server_command.txt" "${ATOM_CMD[@]}"
"${ATOM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY:-false}" = "true" ]; then
    run_eval --port "$PORT"
else
    # AgentX DSv4 traces already carry fully formed chat payloads; do not apply
    # AIPerf's generic chat template on top of them.
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
