#!/usr/bin/env bash
set -euo pipefail

# Local-only runner for reproducing the MI355X Kimi-K3 1P1D vLLM-disagg
# (MoRIIO RDMA P/D + Mooncake DRAM cache, DSpark) recipe OUTSIDE GitHub Actions,
# directly on the MIA / amd-aim SLURM cluster.
#
# CI enters through benchmark-multinode-tmpl.yml -> runner ->
# benchmarks/multi_node/agentic/kimik3_fp4_mi355x_vllm-disagg.sh.  This file keeps
# MIA-specific knobs (model cache layout, router image, reuse of an already
# allocated Slurm job, dummy-weight bring-up) out of the CI path, and calls the
# SAME agentic entrypoint so local and CI share one code path.
#
# Usage:
#   # attach to an existing salloc (fastest for iterative debug):
#   SLURM_REUSE_JOBID=<jobid> bash benchmarks/multi_node/local_runner/kimik3_fp4_mi355x_vllm-disagg.sh [CONC] [DURATION]
#   # dummy-weight bring-up (skip 1.5TB main checkpoint; DSpark stays on):
#   LOAD_FORMAT=dummy SLURM_REUSE_JOBID=<jobid> bash .../kimik3_fp4_mi355x_vllm-disagg.sh 1 600
#   # real weights + DSpark:
#   LOAD_FORMAT=auto bash .../kimik3_fp4_mi355x_vllm-disagg.sh 8 3600

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

CONC="${CONC:-${1:-8}}"
DURATION="${DURATION:-${2:-3600}}"
RUN_ROOT="${RUN_ROOT:-/it-share/yichaozhu/kimi-k3-agentx/runs}"
RUN_NAME="${RUN_NAME:-c${CONC}_$(date -u +%Y%m%d_%H%M%S)}"
BENCHMARK_LOGS_DIR="${BENCHMARK_LOGS_DIR:-${RUN_ROOT}/${RUN_NAME}}"
mkdir -p "$BENCHMARK_LOGS_DIR"

export GITHUB_WORKSPACE="${GITHUB_WORKSPACE:-$REPO_ROOT}"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-amd-aim}"
export SLURM_PARTITION="${SLURM_PARTITION:-amd-aim}"
export TIME_LIMIT="${TIME_LIMIT:-08:00:00}"

# --- Model / image (MIA layout) -------------------------------------------------
# MODEL_PATH is the on-disk HF hub root; MODEL_NAME must equal ${MODEL##*/} so
# models_vllm.yaml (key "Kimi-K3") resolves.
export MODEL="${MODEL:-moonshotai/Kimi-K3}"
export MODEL_NAME="${MODEL_NAME:-Kimi-K3}"
export MODEL_PATH="${MODEL_PATH:-/it-share/hf_cache}"
export MODEL_DIR="${MODEL_DIR:-$MODEL_PATH}"
# LOCAL FAST PATH: setup_deps' skip conditions (ibv_devinfo present, `import
# quark`, /opt/vllm-k3-fork/.inferencex_installed_ref) all live inside the
# container, which is recreated per run -- so apt + amd-quark + the fork overlay
# re-run every local iteration. After one successful bring-up, `docker commit`
# the container to e.g. vllm-openai-rocm:kimi-k3-prepared and set
# CONTAINER_IMAGE to it; every step then hits its "already present" branch.
# Cloud CI intentionally keeps the full path from the pristine image.
# Default to the prepared -mc image: it already ships a HIP-linked mooncake +
# mooncake_master + amd-quark, so install_mooncake_rocm and install_amd_quark
# take their fast paths and no apt runs for them. CI can stay on the pristine
# kimi-k3 image; this only changes what the local debug loop pays per iteration.
export CONTAINER_IMAGE="${CONTAINER_IMAGE:-yukiozzz/vllm-openai-rocm:kimi-k3-mc}"
# The -mc image lacks ibverbs-utils/iproute2 and the MIA apt mirrors stall, so
# let the diagnostics-only tools be absent rather than losing the container.
export ALLOW_MISSING_RDMA_TOOLS="${ALLOW_MISSING_RDMA_TOOLS:-1}"
export IMAGE="${IMAGE:-$CONTAINER_IMAGE}"
# Must be a tag that actually resolves. The pinned nightly-<date>-<sha> form is
# not on Docker Hub any more; rank 0 then dies with `docker: manifest unknown`
# (exit 125) BEFORE creating its container, and srun --kill-on-bad-exit=1 reaps
# the decode container on the other node in the same second -- which reads like a
# mystery hang in slurm_job-*.out because the docker error only lands in .err.
export VLLM_ROUTER_IMAGE="${VLLM_ROUTER_IMAGE:-vllm/vllm-router:nightly}"

# --- Recipe identity ------------------------------------------------------------
export RUNNER_NAME="${RUNNER_NAME:-kimi-k3-local-1p1d-c${CONC}}"
export FRAMEWORK="${FRAMEWORK:-vllm-disagg}"
export PRECISION="${PRECISION:-fp4}"
export MODEL_PREFIX="${MODEL_PREFIX:-kimik3}"
export ROUTER_TYPE="${ROUTER_TYPE:-vllm-router}"
export IS_MULTINODE="${IS_MULTINODE:-true}"
export IS_AGENTIC="${IS_AGENTIC:-1}"
export SCENARIO_TYPE="${SCENARIO_TYPE:-agentic-coding}"

# DSpark on by default (via models_vllm.yaml flags). For a model-less plumbing
# smoke set LOAD_FORMAT=dummy (main weights only; draft still loads from HF).
export SPEC_DECODING="${SPEC_DECODING:-draft_model}"
export LOAD_FORMAT="${LOAD_FORMAT:-auto}"
export DISABLE_SPECULATIVE="${DISABLE_SPECULATIVE:-0}"

# --- Workload -------------------------------------------------------------------
export CONC_LIST="${CONC_LIST:-$CONC}"
export DURATION
export ISL="${ISL:-8192}"
export OSL="${OSL:-1024}"
export RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-1}"

# --- 1P1D TP8 + TP8 -------------------------------------------------------------
export PREFILL_NODES="${PREFILL_NODES:-1}"
export PREFILL_NUM_WORKERS="${PREFILL_NUM_WORKERS:-1}"
export PREFILL_TP="${PREFILL_TP:-8}"
export PREFILL_EP="${PREFILL_EP:-1}"
export PREFILL_DP_ATTN="${PREFILL_DP_ATTN:-false}"
export DECODE_NODES="${DECODE_NODES:-1}"
export DECODE_NUM_WORKERS="${DECODE_NUM_WORKERS:-1}"
export DECODE_TP="${DECODE_TP:-8}"
export DECODE_EP="${DECODE_EP:-1}"
export DECODE_DP_ATTN="${DECODE_DP_ATTN:-false}"

# --- KV transfer: MoRIIO RDMA P/D + Mooncake DRAM (MultiConnector) --------------
export KV_OFFLOADING="${KV_OFFLOADING:-dram}"
# Only default the backend when the tier is actually on. benchmark_lib.sh rejects
# KV_OFFLOADING=none with a non-empty KV_OFFLOAD_BACKEND, and "${VAR:-mooncake}"
# substitutes for empty as well as unset -- so an unconditional default made the
# no-offload arm impossible to express through this runner at all, which is
# exactly the baseline a with/without-Mooncake comparison needs.
if [[ "${KV_OFFLOADING}" == "none" ]]; then
    export KV_OFFLOAD_BACKEND=""
else
    export KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-mooncake}"
fi
# Sized the way the cloud path sizes it, instead of a round placeholder. CI never
# sets this directly: generate_sweep_configs.agentic_dram_offload_gb() derives it
# from the recipe's `dram-utilization: 0.80` and the runner's hardware metadata --
#   min(available-cpu-dram-mib, MAX_AGENTIC_AVAILABLE_CPU_DRAM_MIB=2_861_022)
#     * 0.80 * prefill_gpus_per_node / gpus-per-node
# For cluster:mi355x-amds (3_095_781 MiB, 8 GPUs/node) the cap binds, so a TP8
# prefill worker gets 2400 GB. The old hard-coded 256 was ~9x under that, and it
# is what made MooncakeStore fail every put once the corpus got large:
#   client_service.cpp: BatchPut failed for 32 keys due to insufficient space
# which then failed the decode request and aborted the replay in warmup.
#
# 2400 does NOT fit for K3, though, even on a ~2.9 TB node: the pool is pinned for
# RDMA, so it is not reclaimable page cache. Measured on g07 mid-run, each of the
# 8 TP workers held ~231 GiB RSS (~1.85 TiB for the container) and only ~1.1 TiB
# of the node was left available. On the decode node that tipped over: 556 GB of
# the 892 GB swap in use, then
#   multiproc_executor.py: Worker proc VllmWorker-6 died unexpectedly
#     (exit code: None)   -> shm_broadcast RuntimeError("cancelled")
#     -> EngineDeadError  -> aiperf --failed-request-threshold aborts the conc point
# with no Python traceback anywhere, because the worker was signalled, not raised.
# 1200 GB (150 GB/rank) leaves ~1.8 TB of headroom and is the size the earlier
# validated local run used. Still far more DRAM than the GPU KV pool, so the tier
# is not the limiting factor -- ext_cache_hit was already 87-92% at 2400.
export TOTAL_CPU_DRAM_GB="${TOTAL_CPU_DRAM_GB:-1200}"
# GMU / MAX_MODEL_LEN are deliberately NOT set here: they come from
# benchmarks/multi_node/agentic/kimik3_fp4_mi355x_vllm-disagg.sh, which is the
# same file CI goes through (gmu 0.88 per PR #2403's fleet measurements, and the
# native 1M context that benchmark_lib.sh's agentic unset enforces). Overriding
# either from the local runner would make local numbers non-comparable.
#
# KV dtype: bf16 (empty == do not pass --kv-cache-dtype), same as the recipe.
# Backend ladder for K3 TP8 (96/8 = 12 MLA heads/rank, so nhead <= 16 and aiter
# serves decode from mla_gluon, which picks its regime by KV dtype):
#   TRITON_MLA            - head-agnostic but slow
#   aiter gluon bh16bn64  - bf16 KV, batch_size >= 1        <-- this
#   aiter gluon bh16bn128 - fp8  KV, batch_size == 1 ONLY
#   aiter ASM             - batched AND fp8, but needs nhead % 16 == 0
# This file used to default to fp8 KV + VLLM_PATCH_MLA_HEAD_PAD=1 to reach ASM.
# That is a dead end under DSpark: ASM rejects qo_len > 4 outside persistent mode
# and the n=7 verify step presents 8, while plain fp8 without the pad patch lands
# on bh16bn128 and aborts every conc > 1 with "mla_gluon[bh16bn128] requires
# batch_size=1". bf16 KV batches normally and is what the validated real-weight
# run (GSM8K 44/50) served on, so local now matches
# configs/amd-master.yaml's kimik3-fp4-mi355x-vllm-disagg-agentic{,-dummy}.
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
export VLLM_PATCH_MLA_HEAD_PAD="${VLLM_PATCH_MLA_HEAD_PAD:-0}"
# Empty -> keep models_vllm.yaml's ROCM_AITER_MLA (bh16bn64 on bf16 KV), which is
# what the validated real-weight run used. Set TRITON_MLA only to A/B against it.
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
# Draft (DSpark) attention backend. Empty -> models_vllm.yaml's TRITON_MLA.
# ROCM_AITER_MLA is the A/B arm: the draft is qo_len==1 so it never trips the
# target's qo_len>4 fp8 ASM gate, yet it runs 7 of the 8 passes per DSpark step.
export SPEC_ATTN_BACKEND="${SPEC_ATTN_BACKEND:-}"
# Persistent-metadata patch for the target's qo_len>4 fp8 ASM decode: shelved.
export VLLM_PATCH_MLA_PERSISTENT_MTP="${VLLM_PATCH_MLA_PERSISTENT_MTP:-0}"
# The draft/target KV-group split that cost aiter 1.65x KV bytes/token, and the
# hybrid invalid-block crash, are both fixed in the pinned fork now (1755c10c,
# eed3a092) -- no runtime patch knob for them any more. Measured on g05, TP8,
# dummy, bf16 KV, ROCM_AITER_MLA:
#   before -> bucket_sizes=[69,24,5] group_size=24 -> 1,000,400 tokens
#   after  -> bucket_sizes=[69,29]   group_size=29 -> 1,624,554 tokens (== Triton)
# Graphs ON: AiterMLA declares UNIFORM_BATCH cudagraph support and the K3 fork
# adds _uniform_padded_mtp_qo_len for padded MTP capture. Set 1 to fall back.
export ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
export VLLM_PATCH_46240="${VLLM_PATCH_46240:-1}"
# Mooncake is built from source at job start; install_mooncake_rocm caches the
# built artifact at $HF_HUB_CACHE/inferencex/mooncake/<tag-os-abi-rocm-arch>.tar.gz
# and untars it on later runs. job.slurm's fallback is /tmp/mooncake-hf-cache,
# which is per-node and wiped, so every local iteration would rebuild on BOTH
# nodes (~15 min each). Point it at the shared wekafs cache the cloud recipe
# already uses, so the build happens once and both nodes reuse it.
export HF_HUB_CACHE="${HF_HUB_CACHE:-/models/.cache/huggingface/hub}"
export INFERENCEX_MOONCAKE_MAX_TRANSFER_BATCH_KEYS="${INFERENCEX_MOONCAKE_MAX_TRANSFER_BATCH_KEYS:-32}"
export MC_WORKERS_PER_CTX="${MC_WORKERS_PER_CTX:-4}"
export MOONCAKE_DECODE_STORE="${MOONCAKE_DECODE_STORE:-0}"
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"

# --- vLLM K3 DSpark/MoRIIO-KDA fork (online install over the base image) ---------
export VLLM_K3_FORK_REPO="${VLLM_K3_FORK_REPO:-https://github.com/YukioZzz/vllm}"
export VLLM_K3_FORK_REF="${VLLM_K3_FORK_REF:-yichaozhu/k3-tpdcp-hetero}"
# export VLLM_K3_FORK_TOKEN=...   # set if the fork repo is private

# --- MIA node/RDMA env ----------------------------------------------------------
export NODELIST="${NODELIST:-mia1-p01-g05,mia1-p01-g06}"
export IBDEVICES="${IBDEVICES:-rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7}"
# The RDMA QoS/DCQCN pre-flight shells out to `sudo nicctl show qos`, which needs
# passwordless sudo the interactive MIA accounts do not have ("sudo nicctl show
# qos returned nothing" -> FATAL). Skip by default here: this is the local debug
# path, and PFC/DCQCN is a fabric-side setting we cannot assert from the account.
# CI keeps the check (it runs on hosts where nicctl is permitted). Set
# SKIP_RDMA_CHECK=0 to re-enable locally once sudo is available -- required before
# trusting any cross-node MoRIIO *performance* number from this runner.
export SKIP_RDMA_CHECK="${SKIP_RDMA_CHECK:-1}"
export MORI_RDMA_TC="${MORI_RDMA_TC:-104}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export BENCHMARK_LOGS_DIR

cat > "${BENCHMARK_LOGS_DIR}/local_runner_notes.txt" <<EOF
Local-only runner (do NOT source from CI).
MODEL=${MODEL} MODEL_PATH=${MODEL_PATH} IMAGE=${IMAGE}
LOAD_FORMAT=${LOAD_FORMAT} DISABLE_SPECULATIVE=${DISABLE_SPECULATIVE} SPEC_DECODING=${SPEC_DECODING}
KV_OFFLOADING=${KV_OFFLOADING}/${KV_OFFLOAD_BACKEND} KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-<bf16, no flag>} ENFORCE_EAGER=${ENFORCE_EAGER}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-<yaml ROCM_AITER_MLA>} SPEC_ATTN_BACKEND=${SPEC_ATTN_BACKEND:-<yaml TRITON_MLA>}
VLLM_PATCH_MLA_HEAD_PAD=${VLLM_PATCH_MLA_HEAD_PAD} VLLM_PATCH_MLA_PERSISTENT_MTP=${VLLM_PATCH_MLA_PERSISTENT_MTP}
GMU/MAX_MODEL_LEN: set downstream by benchmarks/multi_node/agentic/kimik3_fp4_mi355x_vllm-disagg.sh
  (GMU=${GPU_MEMORY_UTILIZATION:-<agentic default 0.88>} MAX_MODEL_LEN=${MAX_MODEL_LEN:-<agentic native 1048576>})
NODELIST=${NODELIST}
SLURM_REUSE_JOBID=${SLURM_REUSE_JOBID:-<unset>}  (set to attach to an existing salloc)
VLLM_K3_FORK=${VLLM_K3_FORK_REPO}@${VLLM_K3_FORK_REF}
EOF
printenv | sort > "${BENCHMARK_LOGS_DIR}/launch_env.txt"

cd "${REPO_ROOT}/benchmarks/multi_node/amd_utils"

launch_cmd=( bash "${REPO_ROOT}/benchmarks/multi_node/agentic/kimik3_fp4_mi355x_vllm-disagg.sh" )

if [[ "${DETACH:-1}" == "1" ]]; then
    nohup "${launch_cmd[@]}" \
        > "${BENCHMARK_LOGS_DIR}/launch.out" \
        2> "${BENCHMARK_LOGS_DIR}/launch.err" &
    echo "$!" > "${BENCHMARK_LOGS_DIR}/launch.pid"
    echo "BENCHMARK_LOGS_DIR=${BENCHMARK_LOGS_DIR}"
    echo "LAUNCH_PID=$(cat "${BENCHMARK_LOGS_DIR}/launch.pid")"
else
    "${launch_cmd[@]}"
fi
