#!/usr/bin/env bash
set -euo pipefail

# Native two-node launcher for aggregated Kimi-K3 AgentX on MI300X.
#
# Reached only via NATIVE_MULTINODE=1 from launch_mi300x-amds.sh, so the default
# single-node path is untouched. Owns the whole lifecycle: validate the narrow
# topology contract, allocate two exclusive nodes, verify both of them, run one
# vLLM rank per node, drive AgentX from rank 0, hand a bounded set of artifacts
# back to the workspace, and cancel the allocation on success, failure, or
# signal.
#
# Model and image state stays node-local on /raid. Only host-owned result and
# log files cross into $GITHUB_WORKSPACE, because a root-owned file there breaks
# the next job's checkout.
#
# Operator runbook: docs/kimik3-mi300x-native-multinode.md

source "$(dirname "${BASH_SOURCE[0]}")/slurm_utils.sh"

KIMIK3_MODEL_CACHE_ROOT="${KIMIK3_MODEL_CACHE_ROOT:-/raid/hf-hub-cache/models--moonshotai--Kimi-K3}"
KIMIK3_SQUASH_DIR="${KIMIK3_SQUASH_DIR:-/raid/hf-hub-cache/inferencex/squash}"
KIMIK3_SLURM_TIME_MINUTES="${KIMIK3_SLURM_TIME_MINUTES:-480}"
KIMIK3_STARTUP_TIMEOUT_SECONDS="${KIMIK3_STARTUP_TIMEOUT_SECONDS:-7200}"
KIMIK3_HEALTH_POLL_SECONDS="${KIMIK3_HEALTH_POLL_SECONDS:-10}"
KIMIK3_CLEANUP_TIMEOUT_SECONDS="${KIMIK3_CLEANUP_TIMEOUT_SECONDS:-120}"
KIMIK3_CLEANUP_POLL_SECONDS="${KIMIK3_CLEANUP_POLL_SECONDS:-2}"
KIMIK3_SOCKET_IFNAME="${KIMIK3_SOCKET_IFNAME:-ens51f1np1}"
# Anything that has to finish *before* scancel gets its own short deadline: a
# cancelled CI job is SIGKILLed within seconds, and every second spent here is a
# second two exclusive nodes stay allocated.
KIMIK3_PRESCANCEL_TIMEOUT_SECONDS="${KIMIK3_PRESCANCEL_TIMEOUT_SECONDS:-15}"
export KIMIK3_MODEL_CACHE_ROOT KIMIK3_SQUASH_DIR
export PORT="${PORT:-8888}"

# The AgentX client pre-downloads its trace corpus into HF_HUB_CACHE. The
# multi-node workflow template does not set it (only the single-node one does),
# so give the client its own node-local cache; otherwise every job re-downloads
# the corpus into a throwaway container layer.
#
# Deliberately not /raid/hf-hub-cache itself. This mount is read-write, and that
# directory is the parent of both the pinned checkpoint and every job's squash
# image, so mounting it would hand the replay client delete rights over the
# ~1.5 TB snapshot the preflight refuses to re-download.
HF_HUB_CACHE_MOUNT="${HF_HUB_CACHE_MOUNT:-/raid/hf-hub-cache/inferencex/agentx-hub}"
# Not exported: only the client container has this path mounted.
HF_HUB_CACHE_CONTAINER="${HF_HUB_CACHE:-/hf-hub-cache}"

fail_with() {
    local rc="$1"
    shift
    echo "ERROR: $*" >&2
    exit "$rc"
}

fail() {
    fail_with 1 "$@"
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

# --- Contract validation, strictly before any allocation -------------------

require_set GITHUB_WORKSPACE RUNNER_NAME IMAGE MODEL RESULT_FILENAME CONC_LIST \
    KIMIK3_NODELIST

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

if [[ ! "$KIMIK3_SOCKET_IFNAME" =~ ^[[:alnum:]_.:-]+$ ]]; then
    fail "KIMIK3_SOCKET_IFNAME must name one network interface, got '$KIMIK3_SOCKET_IFNAME'"
fi

# Every timing knob feeds `waited=$(( waited + poll ))` deadline loops, where a
# zero, fractional or non-numeric value never advances the counter and spins
# forever -- once *before* scancel, where it would leak the allocation.
for knob in KIMIK3_STARTUP_TIMEOUT_SECONDS KIMIK3_HEALTH_POLL_SECONDS \
    KIMIK3_CLEANUP_TIMEOUT_SECONDS KIMIK3_CLEANUP_POLL_SECONDS \
    KIMIK3_PRESCANCEL_TIMEOUT_SECONDS KIMIK3_SLURM_TIME_MINUTES; do
    if [[ ! "${!knob}" =~ ^[1-9][0-9]*$ ]]; then
        fail "$knob must be a positive integer, got '${!knob}'"
    fi
done

# Passed through untouched; gfx942 has no committed Kimi-K3 tuned MoE tables yet.
if [[ -n "${AITER_SITUV2_A8W4+set}" ]]; then
    if [[ "$AITER_SITUV2_A8W4" != "0" && "$AITER_SITUV2_A8W4" != "1" ]]; then
        fail "AITER_SITUV2_A8W4 must be 0 or 1 when set, got '$AITER_SITUV2_A8W4'"
    fi
fi

canary_image_values=0
for name in KIMIK3_CANARY_SQUASH_URL KIMIK3_CANARY_SQUASH_SHA256 \
    KIMIK3_SQUASH_FILE_OVERRIDE; do
    [[ -n "${!name:-}" ]] && canary_image_values=$(( canary_image_values + 1 ))
done
if (( canary_image_values != 0 && canary_image_values != 3 )); then
    fail "KIMIK3_CANARY_SQUASH_URL, KIMIK3_CANARY_SQUASH_SHA256, and KIMIK3_SQUASH_FILE_OVERRIDE must be set together"
fi
if (( canary_image_values == 3 )); then
    if [[ ! "$KIMIK3_CANARY_SQUASH_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
        fail "KIMIK3_CANARY_SQUASH_SHA256 must be a lowercase sha256 digest"
    fi
    if [[ "$KIMIK3_SQUASH_FILE_OVERRIDE" != /*.sqsh ]]; then
        fail "KIMIK3_SQUASH_FILE_OVERRIDE must be an absolute .sqsh path"
    fi
fi

# --- Lifecycle state and cleanup -------------------------------------------

JOB_ID=""
HEAD_NODE=""
SERVER_PID=""
CLIENT_PID=""
SERVER_LOG_DIR=""
SERVER_LOG=""
SERVER_RC_FILE=""
SERVER_SRUN_PID_FILE=""
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
    if [[ -n "$SERVER_LOG_DIR" ]]; then
        echo "[cleanup] packaging server logs"
        bundle_server_logs "$SERVER_LOG_DIR" \
            "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz"
        # bundle_server_logs returns 0 on an empty directory, which otherwise
        # looks identical to a successful archive.
        if [[ ! -s "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz" ]]; then
            echo "[cleanup] WARNING: the server produced no log to package"
        fi
    fi
}

# Give a best-effort command a deadline without leaving it running afterwards.
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
        # Kill the srun itself, not just the wrapper subshell that waits on it:
        # killing the wrapper orphans both vLLM ranks, which then keep running
        # (and keep appending to the log we are about to package and delete)
        # until scancel lands.
        if [[ -n "$SERVER_SRUN_PID_FILE" && -s "$SERVER_SRUN_PID_FILE" ]]; then
            kill "$(cat "$SERVER_SRUN_PID_FILE")" 2>/dev/null
        fi
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null
        SERVER_PID=""
    fi

    # Only the scratch removal needs the allocation alive, so it is the only
    # thing allowed to delay scancel -- and only briefly. Packaging logs writes
    # host-local files and happens afterwards.
    if [[ -n "$JOB_ID" && -n "$HEAD_NODE" && -n "$SCRATCH_HOST" ]]; then
        echo "[cleanup] removing node-local scratch $SCRATCH_HOST"
        run_bounded "$KIMIK3_PRESCANCEL_TIMEOUT_SECONDS" \
            srun --overlap --jobid="$JOB_ID" --nodes=1 --ntasks=1 \
            --nodelist="$HEAD_NODE" rm -rf "$SCRATCH_HOST"
    fi

    if [[ -n "$JOB_ID" ]]; then
        echo "[cleanup] cancelling allocation $JOB_ID"
        scancel "$JOB_ID" 2>/dev/null
        local waited=0
        while slurm_job_is_active "$JOB_ID"; do
            if (( waited >= KIMIK3_CLEANUP_TIMEOUT_SECONDS )); then
                echo "[cleanup] WARNING: job $JOB_ID still present after ${waited}s"
                break
            fi
            sleep "$KIMIK3_CLEANUP_POLL_SECONDS"
            waited=$(( waited + KIMIK3_CLEANUP_POLL_SECONDS ))
        done
    fi

    package_server_logs

    [[ -n "$HANDOFF_HOST" ]] && rm -f "$HANDOFF_HOST"
    [[ -n "$EXTRACT_DIR" ]] && rm -rf "$EXTRACT_DIR"
    [[ -n "$SERVER_LOG_DIR" ]] && rm -rf "$SERVER_LOG_DIR"
    return 0
}

trap 'rc=$?; cleanup; exit "$rc"' EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM
trap 'cleanup; exit 129' HUP

# --- Kernel probe ----------------------------------------------------------

# Diagnostic mode for the gfx942 A16W4 SiTUv2 illegal access. It shares nothing
# with the canary below except the image and the preflight: one node, one GPU,
# no model, no server, no client. Deliberately its own allocation -- the canary
# holds two exclusive nodes for hours, and a probe that borrowed that would idle
# 15 of 16 GPUs on every iteration of what is meant to be a fast loop.
if [[ "${KIMIK3_PROBE_ONLY:-0}" == "1" ]]; then
    set -x
    PROBE_CASES="${KIMIK3_PROBE_CASES:-m2 seq m4096}"
    PROBE_NODE=$(scontrol show hostnames "$KIMIK3_NODELIST" | head -n 1)
    [[ -n "$PROBE_NODE" ]] || fail "could not resolve a probe node from KIMIK3_NODELIST='$KIMIK3_NODELIST'"

    set +e
    salloc_output=$(
        salloc \
            --partition=compute \
            --nodelist="$PROBE_NODE" \
            --nodes=1 \
            --ntasks-per-node=1 \
            --gres=gpu:8 \
            --cpus-per-task=32 \
            --time="${KIMIK3_PROBE_TIME_MINUTES:-60}" \
            --no-shell \
            --job-name="${RUNNER_NAME}-probe" \
            2>&1
    )
    salloc_rc=$?
    set -e
    printf '%s\n' "$salloc_output"
    [[ "$salloc_rc" -eq 0 ]] || fail "probe salloc failed with exit code $salloc_rc"
    JOB_ID=$(
        printf '%s\n' "$salloc_output" |
            sed -nE 's/.*(Pending|Granted) job allocation ([0-9]+).*/\2/p' |
            tail -n 1
    )
    [[ -n "$JOB_ID" ]] || fail "probe salloc did not report a job ID"
    HEAD_NODE="$PROBE_NODE"
    echo "Probe allocated Slurm job $JOB_ID on $PROBE_NODE"

    SERVER_LOG_DIR="$GITHUB_WORKSPACE/multinode_server_logs_live"
    rm -rf "$SERVER_LOG_DIR"
    mkdir -p "$SERVER_LOG_DIR"

    # The image lives on node-local /raid and the preflight is what imports it,
    # so the probe cannot skip it. Its model checks are satisfied for free: the
    # node comes from KIMIK3_NODELIST, which is already staged for the canary.
    probe_preflight=$(
        srun --jobid="$JOB_ID" --nodes=1 --ntasks=1 --export=ALL \
            bash runners/mi300x_native_node_preflight.sh
    )
    printf '%s\n' "$probe_preflight" | grep -q '^INFERENCEX_KIMIK3_PREFLIGHT ' ||
        fail "probe node $PROBE_NODE produced no preflight record"

    IMAGE_PATH="${KIMIK3_SQUASH_FILE_OVERRIDE:-$KIMIK3_SQUASH_DIR/$(printf '%s' "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh}"
    PROBE_LOG="$SERVER_LOG_DIR/kernel_probe.log"
    probe_rc=0

    set +e
    srun --overlap --jobid="$JOB_ID" \
        --nodes=1 \
        --ntasks=1 \
        --gres=gpu:1 \
        --container-image="$IMAGE_PATH" \
        --container-remap-root \
        --container-writable \
        --no-container-mount-home \
        --no-container-entrypoint \
        --container-workdir=/workspace \
        --container-mounts="$GITHUB_WORKSPACE:/workspace,/dev/kfd:/dev/kfd,/dev/dri:/dev/dri" \
        --export=ALL,KIMIK3_PROBE_CASES="$PROBE_CASES" \
        bash -c '
            set -x
            # Synchronous kernel error reporting: without it the fault surfaces
            # at an arbitrary later launch and names the wrong kernel.
            export AMD_SERIALIZE_KERNEL="${AMD_SERIALIZE_KERNEL:-3}"
            aiter_root=$(python -c "import pathlib, aiter; print(pathlib.Path(aiter.__file__).resolve().parent)")
            if [[ "${KIMIK3_PROBE_APPLY_OVERLAY:-1}" == "1" && -d "$KIMIK3_AITER_OVERLAY_DIR" ]]; then
                install -m 0644 "$KIMIK3_AITER_OVERLAY_DIR/fused_moe.py" "$aiter_root/fused_moe.py"
                for filename in mixed_moe_gemm_2stage.py mfma_preshuffle_pipeline.py \
                                lds_dma_policy.py mfma_policy.py; do
                    install -m 0644 "$KIMIK3_AITER_OVERLAY_DIR/$filename" \
                        "$aiter_root/ops/flydsl/kernels/$filename"
                done
                python -m compileall -q "$aiter_root/ops/flydsl/kernels"
                echo "PROBE_OVERLAY_APPLIED ref=${KIMIK3_AITER_OVERLAY_REF:-unknown}"
            else
                echo "PROBE_OVERLAY_SKIPPED (baseline image aiter)"
            fi
            export PROBE_EXPECT_AITER_ROOT="$aiter_root"

            worst=0
            for probe_case in ${KIMIK3_PROBE_CASES}; do
                echo "===== PROBE CASE ${probe_case} ====="
                # A fresh process per case, and caches that cannot carry a
                # tuned config or a compiled kernel across cases: two cases
                # sharing them would stop being independent observations.
                case_root="/tmp/k3-probe/${probe_case}"
                rm -rf "$case_root"
                mkdir -p "$case_root/jit" "$case_root/triton"
                # Artifacts stay container-local and reach the host only as
                # stdout. This step runs with --container-remap-root, so a JSON
                # written straight into /workspace would be a root-owned file in
                # $GITHUB_WORKSPACE and would break the next job checkout.
                PROBE_CASE="$probe_case" \
                PROBE_ARTIFACT_DIR="$case_root" \
                AITER_JIT_DIR="$case_root/jit" \
                TRITON_CACHE_DIR="$case_root/triton" \
                    python /workspace/benchmarks/kernel_probe/k3_moe_probe.py
                rc=$?
                echo "----- k3_probe_${probe_case}.json -----"
                cat "$case_root/k3_probe_${probe_case}.json" || echo "(no JSON produced)"
                echo "===== PROBE CASE ${probe_case} rc=${rc} ====="
                [[ "$rc" -gt "$worst" ]] && worst="$rc"
            done
            exit "$worst"
        ' 2>&1 | tee "$PROBE_LOG"
    probe_rc="${PIPESTATUS[0]}"
    set -e

    echo "kernel probe exited with $probe_rc"
    # cleanup() packages SERVER_LOG_DIR into the tarball the workflow uploads
    # with always(), so the probe output and its JSON survive this exit.
    exit "$probe_rc"
fi

# --- Allocation ------------------------------------------------------------

set -x

# Exclude the same known-bad nodes as the single-node launcher:
#   chi-mi300x-049: persistent /nvme_home disk-full
#   chi-mi300x-121: provisioning incomplete; missing /raid and Enroot storage
set +e
salloc_output=$(
    salloc \
        --partition=compute \
        --exclude=chi-mi300x-049,chi-mi300x-121 \
        --nodelist="$KIMIK3_NODELIST" \
        --nodes=2 \
        --ntasks-per-node=1 \
        --gres=gpu:8 \
        --cpus-per-task=256 \
        --exclusive \
        --time="$KIMIK3_SLURM_TIME_MINUTES" \
        --no-shell \
        --job-name="$RUNNER_NAME" \
        2>&1
)
salloc_rc=$?
set -e
printf '%s\n' "$salloc_output"
if [[ "$salloc_rc" -ne 0 ]]; then
    fail "salloc failed with exit code $salloc_rc"
fi
JOB_ID=$(
    printf '%s\n' "$salloc_output" |
        sed -nE 's/.*(Pending|Granted) job allocation ([0-9]+).*/\2/p' |
        tail -n 1
)
if [[ -z "$JOB_ID" ]]; then
    fail "salloc did not report a job ID"
fi
echo "Allocated Slurm job $JOB_ID"

# Rank 0 is defined by task rank, not by scontrol's node ordering. Slurm needs
# the node name for placement, while torch.distributed needs a cross-node
# routable address. On this cluster each hostname resolves locally to
# 127.0.1.1, so using the hostname as --master-addr strands the remote ranks.
head_endpoint_output=$(
    srun --jobid="$JOB_ID" --nodes=2 --ntasks=2 --ntasks-per-node=1 \
        bash -c '
            if [[ "$SLURM_PROCID" == "0" ]]; then
                private_ip=$(
                    hostname -I |
                        tr " " "\n" |
                        awk "/^10\\./ {print; exit}"
                )
                [[ -n "$private_ip" ]] || {
                    echo "rank 0 has no private 10/8 address" >&2
                    exit 1
                }
                printf "%s %s\n" "$(hostname)" "$private_ip"
            fi
        '
)
read -r HEAD_NODE HEAD_ADDR extra < <(
    printf '%s\n' "$head_endpoint_output" | awk 'NF {print; exit}'
)
if [[ -z "$HEAD_NODE" || -z "$HEAD_ADDR" || -n "${extra:-}" ]]; then
    fail "could not resolve the rank-0 hostname and private address for job $JOB_ID"
fi
echo "Rank 0 runs on $HEAD_NODE at $HEAD_ADDR"

# --- Canary image staging --------------------------------------------------

if (( canary_image_values == 3 )); then
    stage_output=$(
        srun --jobid="$JOB_ID" --nodes=2 --ntasks=2 --ntasks-per-node=1 \
            --export=ALL \
            bash runners/mi300x_stage_canary_squash.sh
    )
    printf '%s\n' "$stage_output"
    printf '%s\n' "$stage_output" | python3 -c '
import os
import sys

records = []
for line in sys.stdin:
    if line.startswith("INFERENCEX_KIMIK3_IMAGE_STAGE "):
        records.append(
            dict(item.split("=", 1) for item in line.split()[1:] if "=" in item)
        )
if len(records) != 2:
    sys.exit(f"ERROR: expected two image-stage records, got {len(records)}")
if len({record.get("hostname") for record in records}) != 2:
    sys.exit(f"ERROR: image staging did not cover two distinct nodes: {records}")
expected = os.environ["KIMIK3_CANARY_SQUASH_SHA256"]
if {record.get("sha256") for record in records} != {expected}:
    sys.exit(f"ERROR: staged image digest mismatch: {records}")
'
fi

# --- Per-node verification -------------------------------------------------

preflight_output=$(
    srun --jobid="$JOB_ID" --nodes=2 --ntasks=2 --ntasks-per-node=1 --export=ALL \
        bash runners/mi300x_native_node_preflight.sh
)
REVISION=$(printf '%s\n' "$preflight_output" | python3 -c '
import os
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

# The image tag is mutable, so two nodes can each hold a valid squash built from
# a different push. Size is the only cross-node identity signal in the record.
sizes = {record.get("squash_size_bytes") for record in records}
if len(sizes) != 1:
    sys.exit(f"ERROR: nodes imported different builds of the image: {sorted(sizes)} bytes")

expected_sha256 = os.environ.get("KIMIK3_CANARY_SQUASH_SHA256")
if expected_sha256:
    hashes = {record.get("squash_sha256") for record in records}
    if hashes != {expected_sha256}:
        sys.exit(
            f"ERROR: nodes hold a different canary image digest: {sorted(hashes)}"
        )

print(revisions.pop())
')
echo "Both nodes verified at model revision $REVISION"

IMAGE_PATH="${KIMIK3_SQUASH_FILE_OVERRIDE:-$KIMIK3_SQUASH_DIR/$(printf '%s' "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh}"
MODEL_CACHE_CONTAINER_PATH="/models-cache"
MODEL_CONTAINER_PATH="$MODEL_CACHE_CONTAINER_PATH/snapshots/$REVISION"

# --- Server ranks ----------------------------------------------------------

export MULTINODE_NODE_COUNT=2
export MULTINODE_GPUS_PER_NODE=8
export MULTINODE_MASTER_ADDR="$HEAD_ADDR"
# torch.distributed's initial TCPStore uses MULTINODE_MASTER_ADDR, but Gloo
# subsequently forms a full mesh and otherwise advertises each node's
# hostname. These hosts resolve their own names to 127.0.1.1, so pin both Gloo
# and RCCL/NCCL bootstrap traffic to the verified private interface.
export GLOO_SOCKET_IFNAME="$KIMIK3_SOCKET_IFNAME"
export NCCL_SOCKET_IFNAME="$KIMIK3_SOCKET_IFNAME"
export MODEL_PATH="$MODEL_CONTAINER_PATH"

# Keep the live log in the workspace from the first byte. GitHub cancellation
# can SIGKILL the launcher before its traps finish, but the workflow's
# `always()` upload step still runs. A /tmp-only log is therefore lost exactly
# on the failure path where it matters most.
SERVER_LOG_DIR="$GITHUB_WORKSPACE/multinode_server_logs_live"
rm -rf "$SERVER_LOG_DIR"
mkdir -p "$SERVER_LOG_DIR"
SERVER_LOG="$SERVER_LOG_DIR/vllm_server.log"
SERVER_RC_FILE="$SERVER_LOG_DIR/server.rc"
SERVER_SRUN_PID_FILE="$SERVER_LOG_DIR/server.srun.pid"

# The rc file, not the PID, is the liveness signal: a background child that has
# exited but not been reaped still answers `kill -0`. `set +e` is what lets a
# failing rank reach the printf below instead of killing this subshell silently.
# srun runs backgrounded inside the wrapper purely so its own PID can be
# recorded -- cleanup needs it, because killing the wrapper leaves srun running.
# --container-writable matches the other three AMD launchers: AITER JIT-compiles
# kernels into its package directory at runtime and --no-container-mount-home
# leaves it nowhere else to write.
{
    set +e
    srun --jobid="$JOB_ID" \
        --nodes=2 \
        --ntasks=2 \
        --ntasks-per-node=1 \
        --kill-on-bad-exit=1 \
        --container-image="$IMAGE_PATH" \
        --container-remap-root \
        --container-writable \
        --no-container-mount-home \
        --no-container-entrypoint \
        --container-workdir=/workspace \
        --container-mounts="$GITHUB_WORKSPACE:/workspace,$KIMIK3_MODEL_CACHE_ROOT:$MODEL_CACHE_CONTAINER_PATH:ro,/dev/kfd:/dev/kfd,/dev/dri:/dev/dri" \
        --export=ALL \
        bash -c 'export MULTINODE_NODE_RANK="$SLURM_PROCID"; exec bash /workspace/benchmarks/multi_node/agentic/kimik3_fp4_mi300x_vllm.sh' \
        > "$SERVER_LOG" 2>&1 &
    srun_pid=$!
    printf '%s\n' "$srun_pid" > "$SERVER_SRUN_PID_FILE"
    wait "$srun_pid"
    printf '%s\n' "$?" > "$SERVER_RC_FILE"
} &
SERVER_PID=$!
echo "Started both server ranks; logging to $SERVER_LOG"

HEALTH_URL="http://${HEAD_ADDR}:${PORT}/health"
startup_deadline=$(( $(date +%s) + KIMIK3_STARTUP_TIMEOUT_SECONDS ))
# Weight load can take the better part of an hour; tracing every poll would
# bury the server log in curl lines.
set +x
while true; do
    # -s, not -f: printf creates the file before it writes the code.
    if [[ -s "$SERVER_RC_FILE" ]]; then
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
set -x

# --- AgentX client on rank 0 ----------------------------------------------

# AgentX artifacts stay on node-local /raid; only the bounded archive below
# crosses into the workspace.
SCRATCH_HOST="$KIMIK3_SQUASH_DIR/.runs/${JOB_ID}-conc${CONC_VALUE}"
srun --overlap --jobid="$JOB_ID" --nodes=1 --ntasks=1 --nodelist="$HEAD_NODE" \
    mkdir -p "$SCRATCH_HOST/output" "$SCRATCH_HOST/agentic" "$HF_HUB_CACHE_MOUNT"

# Pre-created by the host user so the remapped-root container writes *into* an
# already host-owned file instead of creating a root-owned one.
HANDOFF_NAME="multinode_agentic_handoff.tar.gz"
HANDOFF_HOST="$GITHUB_WORKSPACE/$HANDOFF_NAME"
: > "$HANDOFF_HOST"

export KIMIK3_HANDOFF_PATH="/workspace/$HANDOFF_NAME"
export INFMAX_CONTAINER_WORKSPACE=/workspace
export RESULT_DIR=/results/agentic
export AGENTIC_OUTPUT_DIR=/results/output
export AIPERF_SERVER_METRICS_URLS="http://${HEAD_ADDR}:${PORT}/metrics"

# No `set -e`: a failing replay must still hand back whatever it produced, and
# the client's own exit code is what the launcher reports.
CLIENT_WORKER_SCRIPT='set -uo pipefail
mkdir -p /results/output /results/agentic
bash /workspace/benchmarks/multi_node/agentic_srt.sh
client_rc=$?
shopt -s nullglob
cd /results
aggregates=(output/*.json)
# nullglob plus set -u: an empty aggregates array is unbound on bash < 4.4.
tar czf "$KIMIK3_HANDOFF_PATH" ${aggregates[@]+"${aggregates[@]}"} agentic
exit "$client_rc"'

# HF_HUB_CACHE is set here rather than exported: only this container has the
# cache mounted, and pointing the server ranks at a path they cannot see would
# make them resolve a cache root that does not exist.
HF_HUB_CACHE="$HF_HUB_CACHE_CONTAINER" \
srun --overlap --jobid="$JOB_ID" --nodes=1 --ntasks=1 --nodelist="$HEAD_NODE" \
    --container-image="$IMAGE_PATH" \
    --container-remap-root \
    --container-writable \
    --no-container-mount-home \
    --no-container-entrypoint \
    --container-workdir=/workspace \
    --container-mounts="$GITHUB_WORKSPACE:/workspace,$SCRATCH_HOST:/results,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE_CONTAINER,/dev/kfd:/dev/kfd,/dev/dri:/dev/dri" \
    --export=ALL \
    bash -c "$CLIENT_WORKER_SCRIPT" &
CLIENT_PID=$!
client_rc=0
wait "$CLIENT_PID" || client_rc=$?
CLIENT_PID=""

# --- Bounded artifact handoff ---------------------------------------------

if [[ ! -s "$HANDOFF_HOST" ]]; then
    # A failing replay is the more informative status; fall back to 1 when the
    # client claimed success and still handed back nothing.
    fail_with "$(( client_rc == 0 ? 1 : client_rc ))" \
        "the AgentX client produced no handoff archive (client exit code $client_rc)"
fi

archive_entries=$(tar tzf "$HANDOFF_HOST")
if printf '%s\n' "$archive_entries" | grep -q '^/'; then
    fail "the handoff archive contains absolute paths"
fi
if printf '%s\n' "$archive_entries" | grep -Eq '(^|/)\.\.(/|$)'; then
    fail "the handoff archive contains parent-directory components"
fi

EXTRACT_DIR=$(mktemp -d)
# The name checks above cannot see link *targets*, so extract without honouring
# archived ownership or modes as well.
tar xzf "$HANDOFF_HOST" -C "$EXTRACT_DIR" --no-same-owner --no-same-permissions

AGGREGATE="$EXTRACT_DIR/output/${RESULT_FILENAME}_conc${CONC_VALUE}.json"
if [[ ! -f "$AGGREGATE" ]]; then
    fail_with "$(( client_rc == 0 ? 1 : client_rc ))" \
        "the handoff archive is missing ${RESULT_FILENAME}_conc${CONC_VALUE}.json (client exit code $client_rc)"
fi
copy_to_workspace "$AGGREGATE" "$GITHUB_WORKSPACE/$(basename "$AGGREGATE")"

RAW_SOURCE="$EXTRACT_DIR/agentic/conc_${CONC_VALUE}"
RAW_DEST="$GITHUB_WORKSPACE/LOGS/agentic/conc_${CONC_VALUE}"
mkdir -p "$RAW_DEST"
if [[ -d "$RAW_SOURCE" ]]; then
    cp -R "$RAW_SOURCE/." "$RAW_DEST/"
fi

rm -f "$HANDOFF_HOST"
HANDOFF_HOST=""

if (( client_rc != 0 )); then
    fail_with "$client_rc" "the AgentX client exited with code $client_rc"
fi

echo "Native MI300X Kimi K3 AgentX run complete at concurrency ${CONC_VALUE}"
