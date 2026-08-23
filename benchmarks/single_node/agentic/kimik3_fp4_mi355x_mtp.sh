#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Kimi-K3 MXFP4 on MI355X / MI350X (gfx950)
# using vLLM.
#
# The server command is the AMD reference `vllm serve` for this model, i.e. the
# upstream vLLM recipe's amd block (vllm-project/recipes,
# https://recipes.vllm.ai/moonshotai/Kimi-K3) as run in practice:
#
#   --trust-remote-code --moe-backend auto --tensor-parallel-size 8
#   --load-format auto --gpu-memory-utilization 0.95 --mm-encoder-tp-mode data
#   --max-num-seqs 128 --max-num-batched-tokens 4096 --enable-auto-tool-choice
#   --tool-call-parser kimi_k3 --reasoning-parser kimi_k3
#
# with env VLLM_ROCM_USE_AITER=1 SAFETENSORS_FAST_GPU=1 AITER_SITUV2_A8W4=1
# AITER_BF16_FP8_MOE_BOUND=0 VLLM_USE_BREAKABLE_CUDAGRAPH=0.
#
# K3 is a 2.8T-parameter natively-multimodal MoE (896 routed experts, 16/token
# plus shared) on Kimi Delta Attention, gated MLA and Attention Residuals, with
# a 1M-token native context.
#
# TP=8 ONLY. The MXFP4 checkpoint is 1.561 TB decimal (1.420 TiB, 96
# safetensors), ~195 GB/GPU across 8 GPUs of the 288 GB part; TP=4 would need
# ~390 GB/GPU and cannot load. Upstream strategy_min_gpus agrees (single_node_tp
# and multi_node_tep both 8, DEP 16+), which is why there is no DP-attention arm.
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION,
#   EP_SIZE
#
# Perf-search knobs. Each defaults to the reference command's value, so an
# otherwise-unset run reproduces the reference exactly:
#   GPU_MEM_UTIL             0.95   (reference)
#   MAX_NUM_BATCHED_TOKENS   8192   (default)
#   AITER_A8W4               1      (reference; 0 = aiter a16w4 MoE path)
#   LANGUAGE_MODEL_ONLY      true   
#   KV_CACHE_DTYPE           fp8    (default for every arm; =auto for a bf16 A/B)
#   KV_BLOCK_SIZE            unset  (unset -> vLLM sizes the page; 128 under fp8)
#   MAX_MODEL_LEN            1M     
#   SPEC_DECODING            mtp    (set to none for the no-DSpark arm)
#   SPEC_NUM_TOKENS          2      (DSpark draft length; validated by the _mtp config)

source "$(dirname "$0")/../../benchmark_lib.sh"

wait_for_amd_gpu_clean

du -h /dev/shm

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [ "$TP" -ne 8 ]; then
    echo "Error: Kimi-K3 MXFP4 is a 1.56 TB checkpoint and only fits at TP=8 on" >&2
    echo "       288 GB gfx950 parts (~195 GB/GPU). Got TP=$TP." >&2
    exit 1
fi

# ROCR/HIP visibility for vLLM 0.14+
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

# `hf download` creates the target dir if missing and is itself idempotent. The
# 1.56 TB checkpoint is normally pre-staged, so these calls are a no-op there.
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

rocm-smi || true
amd-smi || true

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# ---- In-container patches ----------------------------------------------------
# The pinned image now ships the whole runtime pre-applied (vLLM tree at the DCP
# branch, aiter pybind11 include order, KV block-pool clamp, AITER fp8 Gluon
# overlay) and sets SKIP_KIMI_PATCHES=1 itself, so this is a no-op there. The
# script and its patch files stay in the tree so the image can be rebuilt and
# the diffs reviewed; see /opt/k3-patches/BAKED.md inside the image for which
# patches are baked and which are deliberately skipped.
bash "$(dirname "$0")/apply_k3_container_patches.sh"

# ---- Reference env block ----------------------------------------------------
# Keep ALL of these. Commenting them out does not avoid the AITER FMHA crash:
# that crash is gated on VLLM_ROCM_USE_AITER alone (AiterFlashAttnPrefillBackend
# .is_available() consults only rocm_aiter_ops.is_enabled()), so disabling the
# others just loses the MoE kernels while keeping the failure.
export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm
export VLLM_ROCM_USE_AITER=1
export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1
export AITER_BF16_FP8_MOE_BOUND=0
# REQUIRED on ROCm per the upstream recipe: the build auto-enables this to 1.
export VLLM_USE_BREAKABLE_CUDAGRAPH=0

# The other gfx950 recipes pin this to 1 as an MEC FW <177 RCCL reclaim
# workaround, and the CI parent environment exports 1, so a :- default would
# never take effect here. Assign it: at TP8xDCP8 pinned scratch is what starves
# the run -- capture and warmup leave ~3.6 GB free and the engine then dies
# under load with HSA_STATUS_ERROR_OUT_OF_RESOURCES (0x1008).
export HSA_NO_SCRATCH_RECLAIM=0

# 2.8T of weights off a shared/NFS mount takes far longer than the default.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"

# Long agentic turns against a 1M context: keep the client from timing out
# mid-request while the server is prefill-bound.
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
MOONCAKE_MASTER_LOG="$RESULT_DIR/mooncake_master.log"
mkdir -p "$RESULT_DIR"

SERVER_PID=""
MOONCAKE_MASTER_PID=""
MEMWATCH_PID=""

cleanup_agentic_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    [ -n "$MEMWATCH_PID" ] && kill "$MEMWATCH_PID" 2>/dev/null
    stop_background_process_tree "$SERVER_PID" "vLLM server" 60
    stop_background_process_tree "$MOONCAKE_MASTER_PID" "Mooncake master" 10
    exit "$exit_code"
}
trap cleanup_agentic_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ---- KV offload -------------------------------------------------------------
# TOTAL_CPU_DRAM_GB is the aggregate host-DRAM budget the matrix generator
# derives from dram-utilization and the runner's available-cpu-dram-mib, capped
# at the 3,095,781 MiB (3 TB decimal) agentic limit. Per
# benchmarks/single_node/agentic/README.md it must be consumed as given and
# never replaced with a model-specific constant.
OFFLOAD_ARGS=()

if agentic_kv_offload_enabled; then
case "${KV_OFFLOAD_BACKEND:-}" in
  vllm-simple)
    require_agentic_kv_offload_backend "$KV_OFFLOAD_BACKEND"
    CPU_BYTES_PER_RANK=$(( TOTAL_CPU_DRAM_GB * 1000 * 1000 * 1000 / TP ))
    # Identical prefixes must hash to identical block keys across ranks.
    export PYTHONHASHSEED=42
    SIMPLE_LAZY_OFFLOAD="${SIMPLE_LAZY_OFFLOAD:-false}"
    OFFLOAD_ARGS=(
        --kv-transfer-config
        "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use_per_rank\":$CPU_BYTES_PER_RANK,\"lazy_offload\":$SIMPLE_LAZY_OFFLOAD}}"
    )
    echo "SimpleCPUOffloadConnector: ${CPU_BYTES_PER_RANK} B/rank x ${TP} ranks, lazy_offload=$SIMPLE_LAZY_OFFLOAD"
    ;;
  mooncake)
    require_agentic_kv_offload_backend "$KV_OFFLOAD_BACKEND"

    # Mooncake and its ROCm DMABUF overlay are baked into the pinned image.
    # Keep vLLM independently patched from yichaozhu/k3-dspark-dcp-v3.
    MOONCAKE_RUNTIME_ROOT=/opt/mooncake
    MOONCAKE_DMABUF_ROOT=/opt/mooncake-dmabuf
    MOONCAKE_EXPECTED_SHA=4c6d23c8f77230dd5974cf9bc87344dcc946ee77
    if [ ! -x "$MOONCAKE_RUNTIME_ROOT/bin/mooncake_master" ] ||
       [ ! -d "$MOONCAKE_RUNTIME_ROOT/python/mooncake" ]; then
        echo "Error: baked Mooncake runtime is missing from the image" >&2
        exit 1
    fi
    if ! grep -q "source=$MOONCAKE_EXPECTED_SHA" "$MOONCAKE_DMABUF_ROOT/manifest.txt" ||
       [ ! -f "$MOONCAKE_RUNTIME_ROOT/python/mooncake/engine.cpython-312-x86_64-linux-gnu.so" ] ||
       [ ! -f "$MOONCAKE_RUNTIME_ROOT/python/mooncake/store.cpython-312-x86_64-linux-gnu.so" ]; then
        echo "Error: baked Mooncake DMABUF overlay is missing or unpinned" >&2
        exit 1
    fi

    export PYTHONPATH="$MOONCAKE_RUNTIME_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
    export LD_LIBRARY_PATH="$MOONCAKE_RUNTIME_ROOT/lib:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PATH="$MOONCAKE_RUNTIME_ROOT/bin:$PATH"
    python3 -c "from mooncake.store import MooncakeDistributedStore" >/dev/null

    PER_RANK_GB=$(( TOTAL_CPU_DRAM_GB / TP ))
    MOONCAKE_MASTER_PORT=$((PORT + 12000))
    MOONCAKE_CONFIG_PATH="$RESULT_DIR/mooncake_config.json"

    # Ionic plugins on this fleet are relative symlinks
    # (libionic-rdmav34.so -> ../libionic.so.1.1.54.0-187). Bind-mounting the
    # plugin directory leaves that target outside the mount, so ibverbs fails
    # with "cannot open libionic-rdmav34.so" (CI run 32594695582). The runner
    # copies the resolved .so files into /workspace/rdma-host-libs.
    if [ -d /workspace/rdma-host-libs/plugins ]; then
        mkdir -p /usr/lib/x86_64-linux-gnu/libibverbs
        cp -a /workspace/rdma-host-libs/plugins/. /usr/lib/x86_64-linux-gnu/libibverbs/
        cp -a /workspace/rdma-host-libs/libionic.so* /usr/lib/x86_64-linux-gnu/ 2>/dev/null || true
        export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/libibverbs:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
        echo "installed host ionic plugins from /workspace/rdma-host-libs"
        ls /usr/lib/x86_64-linux-gnu/libibverbs | head
    fi

    # JSON default is rdma0. Each vLLM worker then rewrites MOONCAKE_CONFIG_PATH
    # to rdma${LOCAL_RANK} (see sitecustomize below). Pinning every rank to
    # rdma0 still ENOMEM'd ibv_reg_mr at 8GB x 8 (CI run 32595896920); listing
    # all NICs in one JSON opened 8 devices per rank (32595345336).
    MOONCAKE_RDMA_DEVICE="${MOONCAKE_RDMA_DEVICE:-rdma0}"
    if [ -z "$MOONCAKE_RDMA_DEVICE" ] || [ ! -e "/sys/class/infiniband/${MOONCAKE_RDMA_DEVICE%%,*}" ]; then
        echo "Error: RDMA device $MOONCAKE_RDMA_DEVICE not present" >&2
        ls /sys/class/infiniband 2>/dev/null >&2 || true
        exit 1
    fi

    HUGEPAGE_FREE=$(awk '/HugePages_Free:/ {print $2}' /proc/meminfo)
    HUGEPAGE_TOTAL=$(awk '/HugePages_Total:/ {print $2}' /proc/meminfo)
    echo "hugepages: free=$HUGEPAGE_FREE total=$HUGEPAGE_TOTAL"
    if [ "${HUGEPAGE_FREE:-0}" -ge 512 ]; then
        export MC_STORE_USE_HUGEPAGE=1
        export MC_STORE_HUGEPAGE_SIZE=2MB
        # Size the tier from the pool with headroom. Shrinking it was chased for
        # four CI cycles and never mattered: the reproducer in run 32598493726
        # fails identically at 2 GB and 8 GB.
        HUGEPAGE_GB=$(( HUGEPAGE_FREE * 2 / 1024 * 9 / 10 / TP ))
        if [ "$HUGEPAGE_GB" -lt "$PER_RANK_GB" ]; then
            PER_RANK_GB=$HUGEPAGE_GB
            echo "clamping RDMA segment to ${PER_RANK_GB} GB/rank" >&2
        fi
    else
        # CI cannot grow nr_hugepages (EACCES even from srun). Keep RDMA but
        # shrink the segment so 4K PTEs stay in the RNIC budget.
        export MC_STORE_USE_HUGEPAGE=0
        if [ "$PER_RANK_GB" -gt 4 ]; then
            PER_RANK_GB=4
        fi
        echo "no free hugepages; RDMA with 4K pages at ${PER_RANK_GB} GB/rank" >&2
    fi
    if [ "$PER_RANK_GB" -lt 1 ]; then
        echo "Error: DRAM segment collapsed to ${PER_RANK_GB} GB/rank" >&2
        awk '/HugePages_/ {print}' /proc/meminfo >&2
        exit 1
    fi

    cat > "$MOONCAKE_CONFIG_PATH" <<EOF
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:$MOONCAKE_MASTER_PORT",
  "global_segment_size": "${PER_RANK_GB}GB",
  "local_buffer_size": "128MB",
  "protocol": "rdma",
  "device_name": "$MOONCAKE_RDMA_DEVICE",
  "enable_offload": true
}
EOF
    export MOONCAKE_CONFIG_PATH
    # ibv_reg_mr ENOMEM on ionic is often RLIMIT_MEMLOCK, not hugepage shortage.
    # CI 32596419301: 4GB x 8 ranks on rdma0, mmap ok, register failed [12].
    ulimit -l unlimited 2>/dev/null || true
    echo "memlock ulimit=$(ulimit -l)" >&2

    cat > "$RESULT_DIR/sitecustomize.py" <<'PY'
import json
import os
from pathlib import Path

rank = os.environ.get("LOCAL_RANK", os.environ.get("RANK"))
path = os.environ.get("MOONCAKE_CONFIG_PATH")
if rank is not None and path:
    src = Path(path)
    if src.is_file() and ".rank" not in src.name:
        cfg = json.loads(src.read_text())
        nics = sorted(
            p.name for p in Path("/sys/class/infiniband").iterdir() if p.is_dir()
        )
        if nics:
            cfg["device_name"] = nics[int(rank) % len(nics)]
            dst = src.with_name(f"{src.stem}.rank{rank}{src.suffix}")
            dst.write_text(json.dumps(cfg))
            os.environ["MOONCAKE_CONFIG_PATH"] = str(dst)
PY
    export PYTHONPATH="$RESULT_DIR${PYTHONPATH:+:$PYTHONPATH}"
    PY_STDLIB=$(python3 -c "import sysconfig; print(sysconfig.get_path('stdlib'))")
    if [ -n "$PY_STDLIB" ] && [ -d "$PY_STDLIB" ]; then
        cp "$RESULT_DIR/sitecustomize.py" "$PY_STDLIB/sitecustomize.py"
        echo "installed sitecustomize into $PY_STDLIB" >&2
    fi
    # A forked caller is the only condition under which registration could be
    # made to fail on this node (CI run 32598493726); the bare sweep passed even
    # at 8 ranks x 8 GB, so segment size, page size, NIC binding, max_mr_size and
    # memlock are all cleared. libibverbs' fork-safe path did not help
    # (32598888005), so take fork out of the picture instead. vLLM otherwise
    # forks its workers off the engine core.
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export MC_STORE_MEMCPY=1
    export MC_ENABLE_PARALLEL_REG_MR=0
    export MC_GID_INDEX="${MC_GID_INDEX:-1}"
    # Mooncake enforces max_mr_size client-side, so a value below the largest KV
    # region rejects its registration; every GPU-sourced PUT then fails
    # TRANSFER_FAIL and the external tier silently never hits. Keep it well above
    # this model's ~2.2GB KV regions.
    export MC_MAX_MR_SIZE="${MC_MAX_MR_SIZE:-34359738368}"
    export PYTHONHASHSEED=42

    mooncake_master --port "$MOONCAKE_MASTER_PORT" \
        --eviction_high_watermark_ratio=0.90 \
        --eviction_ratio=0.10 \
        > "$MOONCAKE_MASTER_LOG" 2>&1 &
    MOONCAKE_MASTER_PID=$!
    sleep 2
    if ! kill -0 "$MOONCAKE_MASTER_PID" 2>/dev/null; then
        echo "Error: Mooncake master died during startup" >&2
        cat "$MOONCAKE_MASTER_LOG" >&2
        exit 1
    fi

    # MOONCAKE_PROBE=1 sweeps segment size, page size, NIC binding and process
    # count against this node's RNICs and exits before the engine loads 1.5 TB
    # of weights, so one CI cycle answers the whole matrix instead of one knob.
    # Sample the hugepage pool so a registration failure can be read against the
    # pool state at the moment it happens instead of the state at startup.
    (
        while true; do
            awk '/HugePages_Free|HugePages_Rsvd|MemAvailable/ {printf "%s=%s ", $1, $2}' \
                /proc/meminfo
            echo "MEMWATCH $(date -u +%H:%M:%S)"
            sleep 10
        done
    ) &
    MEMWATCH_PID=$!

    if [ "${MOONCAKE_PROBE:-0}" = "1" ]; then
        python3 "$(dirname "$0")/mooncake_rdma_probe.py" \
            --master "127.0.0.1:$MOONCAKE_MASTER_PORT" --verify || {
            echo "Error: Mooncake cannot register memory in a forked GPU worker" >&2
            exit 1
        }
        echo "probe passed; continuing to the benchmark"
    fi

    OFFLOAD_ARGS=(
        --kv-transfer-config
        '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"load_async":true,"lookup_async":true,"enable_group_semantics":true}}'
    )
    echo "MooncakeStoreConnector: ${PER_RANK_GB} GB/rank x ${TP} ranks, RDMA DRAM"
    ;;
  *)
    echo "Error: unsupported KV_OFFLOAD_BACKEND='$KV_OFFLOAD_BACKEND'" >&2
    exit 1
    ;;
esac
fi

# ---- LLM server  ------------------------------------------------------------


# ---- Parallelism ------------------------------------------------------------
EP_ARGS=()
if [ "$EP_SIZE" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
fi

# ---- Speculative ------------------------------------------------------------
SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-2}"
SYNTHETIC_ACCEPT_LEN=2.51

if [ "${SPEC_DECODING:-mtp}" = "none" ]; then
    SPEC_ARGS=()
elif [ "${EVAL_ONLY:-false}" = "true" ]; then
    SPEC_ARGS=(
        --speculative-config
        "{\"model\":\"Inferact/Kimi-K3-DSpark\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"TRITON_MLA\",\"kv_cache_dtype\":\"auto\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\": \"block\"}"
    )
else
    SPEC_ARGS=(
        --speculative-config
        "{\"model\":\"Inferact/Kimi-K3-DSpark\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"TRITON_MLA\",\"kv_cache_dtype\":\"auto\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}"
    )
fi

# ---- Async scheduling / KV block-pool stability ------------------------------
# DSpark is the ONLY spec method exempted from vLLM's async-scheduling disable
# list (config/vllm.py:1181), so async_scheduling resolves True here. That gives
# max_concurrent_batches = pp_size + 1 = 2 (vllm.py:563-569), and with
# kv_role=kv_both (is_kv_consumer=True) the scheduler sets defer_block_free=True
# (sched/scheduler.py:155-157). Its own comment: "a step may still be writing a
# freed request's KV blocks. A consumer KV Connector can reallocate and fill
# those blocks via a load that isn't ordered against that write."
#
# That limbo state matches our crash signature exactly -- the engine dies with
#   block_pool.py:667  assert block.ref_cnt == 0
# i.e. a block sitting on the FREE list that is still referenced. Crash time
# scales inversely with concurrency: c10 survived 3612 s, c12 died at 487 s,
# c16 at 354 s. Note vLLM already disables async scheduling for ROCm DeepEP DBO
# because "that combination can corrupt" state.
#
# Setting max_concurrent_batches back to 1 makes defer_block_free unreachable.
# Cost: async scheduling exists to fill GPU-utilisation gaps, so expect to give
# some throughput back. Set ASYNC_SCHEDULING=1 to restore the default.
ASYNC_SCHED_ARGS=()
if [ "${ASYNC_SCHEDULING:-0}" != "1" ]; then
    ASYNC_SCHED_ARGS=(--no-async-scheduling)
fi

# ---- MLA prefill backend -----------------------------------------------------
# On ROCm the prefill priority is [ROCM_AITER_FA, FLASH_ATTN]. ROCM_AITER_FA
# JIT-builds module_fmha_fwd_bf16_opus at runtime; that module registers its own
# aiter_tensor_t, distinct from the one in the prebuilt module_aiter_core, so the
# first call dies with:
#   TypeError: fmha_fwd_bf16_opus_fwd(): incompatible function arguments
# during compile_or_warm_up_model -> _dummy_run, before the server binds.
# Pinning FLASH_ATTN keeps every AITER MoE kernel (and its throughput) while
# skipping only the broken FMHA prefill path.
# UPDATE: the AITER packaging issue is now fixed at source by
# apply_kimi_k3_patches.sh (run above), so ROCM_AITER_FA is usable again and
# is the default. Measured on 8x MI355X / Kimi-K3 MXFP4 TP8, cold prefill:
#   ~24k ctx  FLASH_ATTN 12,953 -> AITER 13,524 tok/s  (+4.4%)
#   ~93k ctx  FLASH_ATTN 11,174 -> AITER 13,423 tok/s  (+20.1%)
# This workload averages ~99k input tokens, so the ~93k figure is the relevant
# one. Set MLA_PREFILL_BACKEND=FLASH_ATTN to fall back if AITER regresses.
MLA_PREFILL_BACKEND="${MLA_PREFILL_BACKEND:-ROCM_AITER_FA}"
MLA_PREFILL_ARGS=()
if [ -n "$MLA_PREFILL_BACKEND" ]; then
    MLA_PREFILL_ARGS=(
        --attention-config
        "{\"mla_prefill_backend\":\"$MLA_PREFILL_BACKEND\"}"
    )
fi

# ---- Decode context parallelism ---------------------------------------------
# Compare TP8 and TP8+DCP8 while keeping every non-topology setting identical.
DCP_SIZE="${DCP_SIZE:-8}"
DCP_ARGS=()
if [ "$DCP_SIZE" -eq 8 ]; then
    # Only meaningful when full graphs are actually requested; under PIECEWISE
    # the ROCm platform has nothing to demote.
    if [ "${CUDAGRAPH_MODE:-PIECEWISE}" != "PIECEWISE" ]; then
        export VLLM_ALLOW_DCP_FULL_CUDAGRAPH=1
    fi
    DCP_ARGS=(
        --decode-context-parallel-size "$DCP_SIZE"
        --dcp-comm-backend a2a
        --cp-kv-cache-interleave-size 1
    )
elif [ "$DCP_SIZE" -ne 1 ]; then
    echo "Error: this recipe supports DCP_SIZE=1 or 8, got $DCP_SIZE." >&2
    exit 1
fi

# ---- HIP graph ------------------------------------------------------------
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$(( CONC * 2 ))}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-$MAX_NUM_SEQS}"
CUDAGRAPH_CAPTURE_SIZES="$(seq -s, 1 "$MAX_CUDAGRAPH_CAPTURE_SIZE")"
# Run 32608081933 mounted all eight Mooncake segments, then segfaulted inside
# aiter loadBinary during PIECEWISE warmup. --enforce-eager skips that capture
# path so the KV-offload arm can serve; drop ENFORCE_EAGER to restore graphs.
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-PIECEWISE}"
if [ "$ENFORCE_EAGER" = "1" ]; then
    COMPILATION_CONFIG_ARGS=(--enforce-eager)
else
    COMPILATION_CONFIG_ARGS=(--compilation-config "{\"mode\":3,\"cudagraph_mode\":\"$CUDAGRAPH_MODE\",\"max_cudagraph_capture_size\":$MAX_CUDAGRAPH_CAPTURE_SIZE,\"custom_ops\":[\"+fused_rms_norm_gated\"],\"cudagraph_capture_sizes\":[$CUDAGRAPH_CAPTURE_SIZES]}")
fi

# At 0.88 vLLM preallocates ~48.6 GiB/rank of KV, while the AgentX warmup's
# M=8190 prefill leaves HSA only ~1.3 GiB for runtime scratch. 0.84 returns
# ~11.5 GiB/rank to transient graph/GEMM/MoE allocations.
# 0.84 leaves nothing for warmup: run 32607408391 reached 232.77 GiB allocated
# with 0 bytes free and died asking for 112 MiB. The earlier segfaults in
# trivial elementwise kernels were the same exhaustion hitting an unchecked
# allocation, so give warmup real headroom.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.78}"

echo "Starting vllm server..."
export PYTHONNOUSERSITE=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1200}"

{ set +x; } 2>/dev/null
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --moe-backend auto
    --tensor-parallel-size "$TP"
    "${EP_ARGS[@]}"
    --load-format fastsafetensors
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --language-model-only
    --max-num-seqs "$MAX_NUM_SEQS"
    "${DCP_ARGS[@]}"
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
    --max-model-len 1048576
    --enable-prefix-caching
    --kv-cache-dtype "fp8"
    --max-num-batched-tokens 8192
    "${ASYNC_SCHED_ARGS[@]}"
    "${MLA_PREFILL_ARGS[@]}"
    "${COMPILATION_CONFIG_ARGS[@]}"
    "${SPEC_ARGS[@]}"
    "${OFFLOAD_ARGS[@]}"
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
