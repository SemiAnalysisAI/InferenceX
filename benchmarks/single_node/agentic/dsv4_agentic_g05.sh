#!/usr/bin/env bash
set -euo pipefail

# Host-side launcher for the FULL agentic trace replay on tw-g05.
#
# The sibling dsv4_quick_ab_g05.sh drives the 10-in/10-out probe; this one
# drives the real thing -- dsv4_fp4_mi355x_vllm_mtp.sh replaying the Weka
# claude-code trace corpus through aiperf. The two differ in more than
# duration: this run needs the 1.8 GB trace dataset on disk, an isolated
# aiperf venv, a DRAM offload pool, and an hour-scale wall clock, so it gets
# its own launcher rather than a flag on the probe's.
#
# Defaults reproduce the arm under test:
#   DEP8 (TP8 + DP-attention + EP8), round_robin routing, DRAM KV offload
#   via SimpleCPUOffloadConnector, MTP on, CONC=32.
#
# Usage:
#   ./dsv4_agentic_g05.sh
#   CONC=32 VLLM_ROUTER_POLICY=consistent_hash ./dsv4_agentic_g05.sh
#   DP_ATTENTION=false EP_SIZE=1 KV_OFFLOADING=none ./dsv4_agentic_g05.sh   # TP8 baseline

IMAGE="${IMAGE:-jiahcao/vllm-dsv4:optimal-fse-aiter4269}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Pro}"
MODEL_DIR="${MODEL_DIR:-/mnt/models/DeepSeek-V4-Pro}"

TP="${TP:-8}"
EP_SIZE="${EP_SIZE:-8}"
DP_ATTENTION="${DP_ATTENTION:-true}"
CONC="${CONC:-32}"
PORT="${PORT:-8500}"

# DURATION: the CI default (validation.py DEFAULT_AGENTIC_DURATION_SECONDS).
# Do not drop below 900 -- inferencex_agentx_mvp.py sets
# min_benchmark_duration_seconds=900 and aiperf v1.0.1 gates the profile
# metric coverage ratio (0.98) on it, so a shorter run fails validation
# rather than producing a smaller-but-valid number.
DURATION="${DURATION:-3600}"

VLLM_ROUTER_POLICY="${VLLM_ROUTER_POLICY:-round_robin}"

KV_OFFLOADING="${KV_OFFLOADING:-dram}"
KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-vllm-simple}"

# Matches what generate_sweep_configs.py's agentic_dram_offload_gb() would
# compute for this arm, so the pool is the size CI would have used:
#   min(mi355x-amds 3_095_781 MiB, MAX_AGENTIC_AVAILABLE_CPU_DRAM_MIB
#       2_861_022 MiB) * dram-utilization 0.60 * (8/8 GPUs) = 1799 GB.
# The node has ~3.0 TiB total and ~2.8 TiB available, so this leaves ample
# headroom for worker RSS and page cache.
TOTAL_CPU_DRAM_GB="${TOTAL_CPU_DRAM_GB:-1799}"

# GPU_MEM_UTIL: the recipe derives 0.90 for any DRAM arm, but its own comment
# records 0.90 (259.2 GiB) as above the ~256 GiB free-memory hard-fail ceiling
# on these nodes, where every DP worker dies at init. The DRAM arms that chose
# 0.90 are the commented-out ones in amd-master.yaml, i.e. never validated
# here. Hold 0.86 -- the value the active arms actually run at -- and raise it
# only after a green run proves the ceiling has moved.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.86}"

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

# HF cache holding the pre-downloaded trace corpus. $HOME on g05 is an 11 GB
# NFS mount -- far too small for the 1.8 GB dataset plus aiperf's expanded
# artifacts -- so this lives on the 28T local array. /var/tmp is wiped by the
# Slurm epilog, which is fine for a re-downloadable cache but not for results.
HF_CACHE="${HF_CACHE:-/var/tmp/$USER-hf}"

# Results: never under $HOME (11 GB, and aiperf artifacts for an hour-long
# replay are large).
SCRATCH="${SCRATCH:-/var/tmp/$USER-dsv4-agentic}"

if [ "$DP_ATTENTION" = "true" ]; then
    ARM="dep${EP_SIZE}"
else
    ARM="tp${TP}"
fi
RUN_TAG="${RUN_TAG:-${ARM}_c${CONC}_${VLLM_ROUTER_POLICY}_kv${KV_OFFLOADING}}"
HOST_RESULT_DIR="$SCRATCH/$RUN_TAG"

# RESULT_FILENAME is what process_agentic_result writes its aggregate JSON as,
# and what the PROFILE relay path names traces after. Give it the CI shape.
RESULT_FILENAME="${RESULT_FILENAME:-dsv4_vllm_fp4_tp${TP}_ep${EP_SIZE}_conc${CONC}_${VLLM_ROUTER_POLICY}}"

CONTAINER_NAME="${CONTAINER_NAME:-dsv4agentic_${ARM}_$$}"

# ---- preflight ---------------------------------------------------------------

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "Error: model dir not found: $MODEL_DIR" >&2
    exit 1
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Error: image not present locally: $IMAGE (docker pull it first)" >&2
    exit 1
fi

# The trace corpus must already be cached: HF_HUB_OFFLINE is not set here (the
# recipe calls `hf download`, which is idempotent and would just re-verify),
# but a cold download of 1.8 GB mid-run stalls the engine-ready window and can
# trip aiperf's dataset-configuration timeout. Check up front instead.
TRACE_CACHE="$HF_CACHE/hub/datasets--semianalysisai--cc-traces-weka-062126"
if [[ ! -d "$TRACE_CACHE" ]]; then
    echo "Error: trace corpus not cached at $TRACE_CACHE" >&2
    echo "       Fetch it first:" >&2
    echo "         HF_HOME=$HF_CACHE hf download --repo-type dataset \\" >&2
    echo "           semianalysisai/cc-traces-weka-062126" >&2
    exit 1
fi

# Ports. --network host on a shared node means a neighbour on the same port
# kills the run -- but only after the ~9 min weight load, since vLLM binds
# last. The DP-attention path needs three: router on PORT, engine on PORT+1,
# router prometheus on PORT+10000.
PORTS=("$PORT")
if [ "$DP_ATTENTION" = "true" ]; then
    PORTS+=("$((PORT + 1))" "$((PORT + 10000))")
fi
for _p in "${PORTS[@]}"; do
    if ss -ltn 2>/dev/null | grep -q ":$_p "; then
        echo "Error: port $_p is already in use on this node." >&2
        echo "       Pick a free base with PORT=<n> (DP-attention also uses n+1 and n+10000)." >&2
        exit 1
    fi
done
unset _p

mkdir -p "$HOST_RESULT_DIR" "$HF_CACHE"

echo "=============================================="
echo " ARM        : $ARM  (TP=$TP EP=$EP_SIZE DP_ATTENTION=$DP_ATTENTION)"
echo " IMAGE      : $IMAGE"
echo " CONC       : $CONC"
echo " DURATION   : ${DURATION}s"
echo " ROUTER     : $VLLM_ROUTER_POLICY"
echo " KV OFFLOAD : $KV_OFFLOADING / $KV_OFFLOAD_BACKEND (${TOTAL_CPU_DRAM_GB} GB)"
echo " GPU_MEM_UTIL: $GPU_MEM_UTIL"
echo " PORTS      : ${PORTS[*]}"
echo " HF CACHE   : $HF_CACHE"
echo " RESULTS    : $HOST_RESULT_DIR"
echo "=============================================="

cleanup_container() {
    if [[ "${KEEP_CONTAINER:-0}" != "1" ]]; then
        docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fi
}
trap cleanup_container EXIT

# Forward optional knobs that happen to be set. Same rationale as the probe
# launcher: a knob set on the host but not forwarded silently runs the default
# and the run still looks normal.
PASSTHRU=()
for _v in ROCM_ROPE_KVCACHE_FUSION MAX_MODEL_LEN NUM_SPEC_TOKENS EVAL_ONLY \
          WEKA_LOADER_OVERRIDE AIPERF_EXPERIMENTAL_FAST AIPERF_WARMUP_REQUESTS_PER_LANE \
          AIPERF_FAILED_REQUEST_THRESHOLD AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES \
          VLLM_ROUTER_VERSION VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS PROFILE; do
    if [[ -n "${!_v+x}" ]]; then
        PASSTHRU+=(-e "$_v=${!_v}")
        echo " passthru   : $_v=${!_v}"
    fi
done
unset _v

# --entrypoint /bin/bash is mandatory: the image declares
# ENTRYPOINT ["/bin/bash","-c"], so a command passed without overriding it
# expands to `bash -c bash <args>` and is silently dropped.
#
# /mnt/models is root-owned: mount it read-only and never write there.
docker run --rm --name "$CONTAINER_NAME" \
    --entrypoint /bin/bash \
    --device /dev/kfd --device /dev/dri \
    --network host --ipc host \
    --group-add video --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --shm-size 64g \
    -v "$MODEL_DIR":/models/DeepSeek-V4-Pro:ro \
    -v "$REPO_DIR":/workspace \
    -v "$HOST_RESULT_DIR":/results \
    -v "$HF_CACHE":/hfcache \
    -e MODEL="$MODEL" \
    -e MODEL_PATH=/models/DeepSeek-V4-Pro \
    -e MODEL_PREFIX=dsv4 \
    -e TP="$TP" \
    -e EP_SIZE="$EP_SIZE" \
    -e DP_ATTENTION="$DP_ATTENTION" \
    -e CONC="$CONC" \
    -e PORT="$PORT" \
    -e DURATION="$DURATION" \
    -e KV_OFFLOADING="$KV_OFFLOADING" \
    -e KV_OFFLOAD_BACKEND="$KV_OFFLOAD_BACKEND" \
    -e TOTAL_CPU_DRAM_GB="$TOTAL_CPU_DRAM_GB" \
    -e GPU_MEM_UTIL="$GPU_MEM_UTIL" \
    -e VLLM_ROUTER_POLICY="$VLLM_ROUTER_POLICY" \
    -e RESULT_DIR=/results \
    -e RESULT_FILENAME="$RESULT_FILENAME" \
    -e AGENTIC_OUTPUT_DIR=/results \
    -e SCENARIO_TYPE=agentic-coding \
    -e IS_AGENTIC=1 \
    -e FRAMEWORK=vllm \
    -e PRECISION=fp4 \
    -e HF_HOME=/hfcache \
    -e HF_HUB_CACHE=/hfcache/hub \
    -e INFMAX_CONTAINER_WORKSPACE=/workspace \
    -e TMPDIR=/results/tmp \
    "${PASSTHRU[@]}" \
    "$IMAGE" \
    -lc 'set -o pipefail
         mkdir -p "$TMPDIR"
         cd /workspace
         bash benchmarks/single_node/agentic/dsv4_fp4_mi355x_vllm_mtp.sh 2>&1 \
           | tee /results/run.log' \
    || { echo "RUN FAILED (arm=$ARM). Tail of log:" >&2
         tail -60 "$HOST_RESULT_DIR/run.log" 2>/dev/null >&2 || true
         exit 1; }

echo
echo "Done. Results in $HOST_RESULT_DIR"
ls -la "$HOST_RESULT_DIR"
