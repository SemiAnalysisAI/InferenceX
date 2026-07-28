#!/usr/bin/env bash
set -euo pipefail

KIMIK3_MODEL_CACHE_ROOT="${KIMIK3_MODEL_CACHE_ROOT:-/raid/hf-hub-cache/models--moonshotai--Kimi-K3}"
KIMIK3_SQUASH_DIR="${KIMIK3_SQUASH_DIR:-/raid/hf-hub-cache/inferencex/squash}"
KIMIK3_SLURM_TIME_MINUTES="${KIMIK3_SLURM_TIME_MINUTES:-480}"
KIMIK3_STARTUP_TIMEOUT_SECONDS="${KIMIK3_STARTUP_TIMEOUT_SECONDS:-7200}"
KIMIK3_HEALTH_POLL_SECONDS="${KIMIK3_HEALTH_POLL_SECONDS:-10}"
KIMIK3_CLEANUP_TIMEOUT_SECONDS="${KIMIK3_CLEANUP_TIMEOUT_SECONDS:-120}"
KIMIK3_CLEANUP_POLL_SECONDS="${KIMIK3_CLEANUP_POLL_SECONDS:-2}"
export KIMIK3_MODEL_CACHE_ROOT KIMIK3_SQUASH_DIR
export PORT="${PORT:-8888}"

HF_HUB_CACHE_MOUNT="${HF_HUB_CACHE_MOUNT:-/raid/hf-hub-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/hf-hub-cache}"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

require_exact() {
    local name="$1" expected="$2"
    if [[ "${!name:-}" != "$expected" ]]; then
        fail "the native MI300X path requires $name=$expected, got '${!name:-}'"
    fi
}

require_set() {
    local name
    for name in "$@"; do
        if [[ -z "${!name:-}" ]]; then
            fail "$name must be set"
        fi
    done
}

require_set GITHUB_WORKSPACE RUNNER_NAME IMAGE MODEL RESULT_FILENAME CONC_LIST

require_exact IS_MULTINODE true
require_exact IS_AGENTIC 1
require_exact SCENARIO_TYPE agentic-coding
require_exact FRAMEWORK vllm
require_exact MODEL_PREFIX kimik3
require_exact PRECISION fp4
require_exact SPEC_DECODING none
require_exact DISAGG false
require_exact PREFILL_EP 1
require_exact PREFILL_DP_ATTN false
require_exact DECODE_EP 1
require_exact DECODE_DP_ATTN false

if [[ "${PREFILL_TP:-}" != "8" || "${PREFILL_PP_SIZE:-}" != "2" ||
      "${DECODE_TP:-}" != "8" || "${DECODE_PP_SIZE:-}" != "2" ]]; then
    fail "the native MI300X path serves only TP8 x PP2, got prefill TP${PREFILL_TP:-?} x PP${PREFILL_PP_SIZE:-?} and decode TP${DECODE_TP:-?} x PP${DECODE_PP_SIZE:-?}"
fi
if [[ "${PREFILL_NUM_WORKERS:-}" != "1" || "${DECODE_NUM_WORKERS:-}" != "0" ]]; then
    fail "the native MI300X path is aggregated: it needs 1 prefill worker and 0 decode workers, got ${PREFILL_NUM_WORKERS:-?}P/${DECODE_NUM_WORKERS:-?}D"
fi

read -r -a CONCURRENCIES <<< "$CONC_LIST"
if [[ "${#CONCURRENCIES[@]}" -ne 1 ]]; then
    fail "one concurrency per allocation is required, got CONC_LIST='$CONC_LIST'"
fi
CONC_VALUE="${CONCURRENCIES[0]}"
case "$CONC_VALUE" in
    1|2|4|8) ;;
    *) fail "concurrency must be 1, 2, 4, or 8, got '$CONC_VALUE'" ;;
esac

if [[ -n "${AITER_SITUV2_A8W4+set}" ]]; then
    if [[ "$AITER_SITUV2_A8W4" != "0" && "$AITER_SITUV2_A8W4" != "1" ]]; then
        fail "AITER_SITUV2_A8W4 must be 0 or 1 when set, got '$AITER_SITUV2_A8W4'"
    fi
fi

JOB_ID=""
HEAD_NODE=""
SERVER_PID=""
CLIENT_PID=""
SERVER_LOG_DIR=""
SERVER_LOG=""
SERVER_RC_FILE=""
EXTRACT_DIR=""
HANDOFF_HOST=""
SCRATCH_HOST=""
CLEANUP_DONE=0

dump_server_log() {
    if [[ -n "$SERVER_LOG" && -s "$SERVER_LOG" ]]; then
        echo "=== last 200 lines of the vLLM server log ==="
        tail -n 200 "$SERVER_LOG"
        echo "============================================="
    fi
}

package_server_logs() {
    if [[ -n "$SERVER_LOG_DIR" && -d "$SERVER_LOG_DIR" ]]; then
        tar czf "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz" \
            -C "$SERVER_LOG_DIR" . 2>/dev/null ||
            echo "[cleanup] WARNING: could not package server logs"
    fi
}

run_bounded() {
    local timeout_seconds="$1"
    shift
    local marker waited pid
    marker=$(mktemp)
    ( "$@" >/dev/null 2>&1; printf 'done' > "$marker" ) &
    pid=$!
    waited=0
    while [[ ! -s "$marker" ]] && (( waited < timeout_seconds )); do
        sleep "$KIMIK3_CLEANUP_POLL_SECONDS"
        waited=$(( waited + KIMIK3_CLEANUP_POLL_SECONDS ))
    done
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    rm -f "$marker"
}

cleanup() {
    if [[ "$CLEANUP_DONE" == "1" ]]; then
        return 0
    fi
    CLEANUP_DONE=1
    set +e

    if [[ -n "$CLIENT_PID" ]]; then
        echo "[cleanup] stopping AgentX client"
        kill "$CLIENT_PID" 2>/dev/null
        wait "$CLIENT_PID" 2>/dev/null
        CLIENT_PID=""
    fi

    if [[ -n "$SERVER_PID" ]]; then
        echo "[cleanup] stopping server step"
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null
        SERVER_PID=""
    fi

    package_server_logs

    if [[ -n "$JOB_ID" && -n "$HEAD_NODE" && -n "$SCRATCH_HOST" ]]; then
        echo "[cleanup] removing node-local scratch $SCRATCH_HOST"
        run_bounded "$KIMIK3_CLEANUP_TIMEOUT_SECONDS" \
            srun --overlap --jobid="$JOB_ID" --nodes=1 --ntasks=1 \
            --nodelist="$HEAD_NODE" rm -rf "$SCRATCH_HOST"
    fi

    if [[ -n "$JOB_ID" ]]; then
        echo "[cleanup] cancelling allocation $JOB_ID"
        scancel "$JOB_ID" 2>/dev/null
        local waited=0
        while [[ -n "$(squeue -j "$JOB_ID" --noheader 2>/dev/null)" ]]; do
            if (( waited >= KIMIK3_CLEANUP_TIMEOUT_SECONDS )); then
                echo "[cleanup] WARNING: job $JOB_ID still present after ${waited}s"
                break
            fi
            sleep "$KIMIK3_CLEANUP_POLL_SECONDS"
            waited=$(( waited + KIMIK3_CLEANUP_POLL_SECONDS ))
        done
    fi

    [[ -n "$HANDOFF_HOST" ]] && rm -f "$HANDOFF_HOST"
    [[ -n "$EXTRACT_DIR" ]] && rm -rf "$EXTRACT_DIR"
    [[ -n "$SERVER_LOG_DIR" ]] && rm -rf "$SERVER_LOG_DIR"
    return 0
}

trap 'rc=$?; cleanup; exit "$rc"' EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM
trap 'cleanup; exit 129' HUP

salloc_stdout=$(
    salloc \
        --parsable \
        --partition=compute \
        --exclude=chi-mi300x-049,chi-mi300x-121 \
        --nodes=2 \
        --ntasks-per-node=1 \
        --gres=gpu:8 \
        --cpus-per-task=256 \
        --exclusive \
        --time="$KIMIK3_SLURM_TIME_MINUTES" \
        --no-shell \
        --job-name="$RUNNER_NAME"
)
JOB_ID=$(printf '%s' "$salloc_stdout" | tr -d '[:space:]' | sed -n 's/^\([0-9][0-9]*\).*/\1/p')
if [[ -z "$JOB_ID" ]]; then
    fail "salloc did not return a job ID (stdout: '$salloc_stdout')"
fi
echo "Allocated Slurm job $JOB_ID"

head_node_output=$(
    srun --jobid="$JOB_ID" --nodes=2 --ntasks=2 --ntasks-per-node=1 \
        bash -c 'if [[ "$SLURM_PROCID" == "0" ]]; then hostname; fi'
)
HEAD_NODE=$(printf '%s\n' "$head_node_output" | awk 'NF {print $1; exit}')
if [[ -z "$HEAD_NODE" ]]; then
    fail "could not resolve the rank-0 hostname for job $JOB_ID"
fi
echo "Rank 0 runs on $HEAD_NODE"

preflight_output=$(
    srun --jobid="$JOB_ID" --nodes=2 --ntasks=2 --ntasks-per-node=1 --export=ALL \
        bash runners/mi300x_native_node_preflight.sh
)
REVISION=$(printf '%s\n' "$preflight_output" | python3 -c '
import sys

records = []
for line in sys.stdin:
    if not line.startswith("INFERENCEX_KIMIK3_PREFLIGHT "):
        continue
    records.append(
        dict(item.split("=", 1) for item in line.split()[1:] if "=" in item)
    )

if len(records) != 2:
    sys.exit(f"ERROR: expected one preflight record per node, got {len(records)}")

hostnames = {record.get("hostname") for record in records}
if len(hostnames) != 2:
    sys.exit(f"ERROR: preflight covered {len(hostnames)} distinct node(s): {sorted(hostnames)}")

revisions = {record.get("revision") for record in records}
if len(revisions) != 1:
    sys.exit(f"ERROR: nodes hold different model revisions: {sorted(revisions)}")

for record in records:
    host = record.get("hostname")
    if record.get("gpu_count") != "8" or record.get("gpu_arch") != "gfx942":
        sys.exit(f"ERROR: node {host} is not 8x gfx942: {record}")
    if int(record.get("squash_size_bytes") or 0) <= 0:
        sys.exit(f"ERROR: node {host} has no valid container image")

print(revisions.pop())
')
echo "Both nodes verified at model revision $REVISION"

IMAGE_PATH="$KIMIK3_SQUASH_DIR/$(printf '%s' "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
MODEL_SNAPSHOT="$KIMIK3_MODEL_CACHE_ROOT/snapshots/$REVISION"
MODEL_CONTAINER_PATH="/models/Kimi-K3"

export MULTINODE_NODE_COUNT=2
export MULTINODE_GPUS_PER_NODE=8
export MULTINODE_MASTER_ADDR="$HEAD_NODE"
export MODEL_PATH="$MODEL_CONTAINER_PATH"

SERVER_LOG_DIR=$(mktemp -d)
SERVER_LOG="$SERVER_LOG_DIR/vllm_server.log"
SERVER_RC_FILE="$SERVER_LOG_DIR/server.rc"

{
    set +e
    srun --jobid="$JOB_ID" \
        --nodes=2 \
        --ntasks=2 \
        --ntasks-per-node=1 \
        --kill-on-bad-exit=1 \
        --container-image="$IMAGE_PATH" \
        --container-remap-root \
        --no-container-mount-home \
        --no-container-entrypoint \
        --container-workdir=/workspace \
        --container-mounts="$GITHUB_WORKSPACE:/workspace,$MODEL_SNAPSHOT:$MODEL_CONTAINER_PATH:ro,/dev/kfd:/dev/kfd,/dev/dri:/dev/dri" \
        --export=ALL \
        bash -c 'export MULTINODE_NODE_RANK="$SLURM_PROCID"; exec bash /workspace/benchmarks/multi_node/agentic/kimik3_fp4_mi300x_vllm.sh' \
        > "$SERVER_LOG" 2>&1
    printf '%s\n' "$?" > "$SERVER_RC_FILE"
} &
SERVER_PID=$!
echo "Started both server ranks; logging to $SERVER_LOG"

HEALTH_URL="http://${HEAD_NODE}:${PORT}/health"
startup_deadline=$(( $(date +%s) + KIMIK3_STARTUP_TIMEOUT_SECONDS ))
while true; do
    if [[ -f "$SERVER_RC_FILE" ]]; then
        server_rc=$(tr -d '[:space:]' < "$SERVER_RC_FILE")
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
        dump_server_log
        fail "the vLLM server step exited with code ${server_rc:-unknown} before becoming healthy"
    fi
    if curl -sf --max-time 10 "$HEALTH_URL" > /dev/null 2>&1; then
        echo "Server is healthy at $HEALTH_URL"
        break
    fi
    if (( $(date +%s) >= startup_deadline )); then
        dump_server_log
        fail "the vLLM server did not become healthy within ${KIMIK3_STARTUP_TIMEOUT_SECONDS}s"
    fi
    sleep "$KIMIK3_HEALTH_POLL_SECONDS"
done

SCRATCH_HOST="$KIMIK3_SQUASH_DIR/.runs/${JOB_ID}-conc${CONC_VALUE}"
srun --overlap --jobid="$JOB_ID" --nodes=1 --ntasks=1 --nodelist="$HEAD_NODE" \
    mkdir -p "$SCRATCH_HOST/output" "$SCRATCH_HOST/agentic"

HANDOFF_NAME="multinode_agentic_handoff.tar.gz"
HANDOFF_HOST="$GITHUB_WORKSPACE/$HANDOFF_NAME"
: > "$HANDOFF_HOST"

export KIMIK3_HANDOFF_PATH="/workspace/$HANDOFF_NAME"
export INFMAX_CONTAINER_WORKSPACE=/workspace
export RESULT_DIR=/results/agentic
export AGENTIC_OUTPUT_DIR=/results/output
export AIPERF_SERVER_METRICS_URLS="http://${HEAD_NODE}:${PORT}/metrics"

CLIENT_WORKER_SCRIPT='set -uo pipefail
mkdir -p /results/output /results/agentic
bash /workspace/benchmarks/multi_node/agentic_srt.sh
client_rc=$?
shopt -s nullglob
cd /results
aggregates=(output/*.json)
tar czf "$KIMIK3_HANDOFF_PATH" "${aggregates[@]}" agentic
exit "$client_rc"'

srun --overlap --jobid="$JOB_ID" --nodes=1 --ntasks=1 --nodelist="$HEAD_NODE" \
    --container-image="$IMAGE_PATH" \
    --container-remap-root \
    --no-container-mount-home \
    --no-container-entrypoint \
    --container-workdir=/workspace \
    --container-mounts="$GITHUB_WORKSPACE:/workspace,$SCRATCH_HOST:/results,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE,$MODEL_SNAPSHOT:$MODEL_CONTAINER_PATH:ro,/dev/kfd:/dev/kfd,/dev/dri:/dev/dri" \
    --export=ALL \
    bash -c "$CLIENT_WORKER_SCRIPT" &
CLIENT_PID=$!
client_rc=0
wait "$CLIENT_PID" || client_rc=$?
CLIENT_PID=""

if [[ ! -s "$HANDOFF_HOST" ]]; then
    fail "the AgentX client produced no handoff archive (client exit code $client_rc)"
fi

archive_entries=$(tar tzf "$HANDOFF_HOST")
if printf '%s\n' "$archive_entries" | grep -q '^/'; then
    fail "the handoff archive contains absolute paths"
fi
if printf '%s\n' "$archive_entries" | grep -Eq '(^|/)\.\.(/|$)'; then
    fail "the handoff archive contains parent-directory components"
fi

EXTRACT_DIR=$(mktemp -d)
tar xzf "$HANDOFF_HOST" -C "$EXTRACT_DIR"

AGGREGATE="$EXTRACT_DIR/output/${RESULT_FILENAME}_conc${CONC_VALUE}.json"
if [[ ! -f "$AGGREGATE" ]]; then
    fail "the handoff archive is missing ${RESULT_FILENAME}_conc${CONC_VALUE}.json"
fi
cp "$AGGREGATE" "$GITHUB_WORKSPACE/"

RAW_SOURCE="$EXTRACT_DIR/agentic/conc_${CONC_VALUE}"
RAW_DEST="$GITHUB_WORKSPACE/LOGS/agentic/conc_${CONC_VALUE}"
mkdir -p "$RAW_DEST"
if [[ -d "$RAW_SOURCE" ]]; then
    cp -R "$RAW_SOURCE/." "$RAW_DEST/"
fi

rm -f "$HANDOFF_HOST"
HANDOFF_HOST=""

if (( client_rc != 0 )); then
    fail "the AgentX client exited with code $client_rc"
fi

echo "Native MI300X Kimi K3 AgentX run complete at concurrency ${CONC_VALUE}"
