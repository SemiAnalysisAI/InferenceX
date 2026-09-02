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
#   SPEC_DECODE              true   (this is the _mtp DSpark recipe; =false for a no-spec A/B)
#   SPEC_NUM_TOKENS          2      (DSpark draft length; validated by the _mtp config)

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE

K3_PERF_VARIANT="${K3_PERF_VARIANT:-baseline}"
K3_ABA_GPU_MEM_UTIL=""
K3_DEFERRED_SESSION_ROOT="${K3_DEFERRED_SESSION_ROOT:-}"
K3_DEFERRED_VALIDATION_DIR="${K3_DEFERRED_VALIDATION_DIR:-}"

# C1 changes cannot be adjudicated reliably across different physical nodes.
# These variants keep one exclusive Slurm allocation while
# running baseline -> candidate -> baseline with fresh server processes. The
# candidate retains the standard results/aiperf_artifacts layout so the normal
# workflow validation remains authoritative; both controls are uploaded below
# results/same_node_aba/.
case "$K3_PERF_VARIANT" in
    m7tunedgemmaba)
        K3_ABA_CANDIDATE=m7tunedgemm
        ;;
    spec3aba)
        K3_ABA_CANDIDATE=spec3
        ;;
    metadatareuseaba)
        K3_ABA_CANDIDATE=metadatareuse
        # Match the clean B300 control without changing the canonical C1
        # recipe or the existing A/B/A variants.
        K3_ABA_GPU_MEM_UTIL=0.85
        ;;
    deferredfinalizeaba)
        K3_ABA_CANDIDATE=deferredfinalize
        K3_ABA_GPU_MEM_UTIL=0.85
        ;;
    *)
        K3_ABA_CANDIDATE=""
        ;;
esac

if [[ -n "$K3_ABA_CANDIDATE" ]]; then
    if [[ "$CONC" != "1" || "${DCP_SIZE:-}" != "1" || "$KV_OFFLOADING" != "none" ]]; then
        echo "Error: same-node A/B/A requires CONC=1, DCP_SIZE=1, and KV_OFFLOADING=none" >&2
        exit 1
    fi
    if [[ "$DURATION" != "1200" || "${AIPERF_EXPERIMENTAL_FAST:-0}" != "0" ]]; then
        echo "Error: same-node A/B/A requires DURATION=1200 and agentx-fast=false" >&2
        exit 1
    fi

    root_result_dir="$RESULT_DIR"
    root_result_filename="${RESULT_FILENAME:?RESULT_FILENAME must be set for same-node A/B/A}"
    aba_dir="$root_result_dir/same_node_aba"
    script_path="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    node_name="${SLURMD_NODENAME:-$(hostname)}"
    if [[ -e "$root_result_dir/aiperf_artifacts" || -e "$aba_dir" ]]; then
        echo "Error: same-node A/B/A requires a fresh result directory" >&2
        exit 1
    fi
    cache_session_root="$(mktemp -d "${TMPDIR:-/tmp}/k3-aba-cache.XXXXXX")"
    mkdir -p "$aba_dir"
    printf 'node\t%s\nvariant\t%s\ncandidate\t%s\ngpu_memory_utilization\t%s\n' \
        "$node_name" "$K3_PERF_VARIANT" "$K3_ABA_CANDIDATE" \
        "${K3_ABA_GPU_MEM_UTIL:-default}" \
        >"$aba_dir/manifest.tsv"

    if [[ "$K3_ABA_CANDIDATE" == "deferredfinalize" ]]; then
        K3_DEFERRED_SESSION_ROOT="$cache_session_root/deferred_finalize"
        deferred_validation_dir="$aba_dir/deferred_finalize_validation"
        K3_DEFERRED_VALIDATION_DIR="$deferred_validation_dir"
        mkdir -p "$K3_DEFERRED_VALIDATION_DIR" "$cache_session_root/prepare"
        K3_ARM_CACHE_ROOT="$cache_session_root/prepare" \
            RESULT_DIR="$deferred_validation_dir" \
            K3_DEFERRED_SESSION_ROOT="$K3_DEFERRED_SESSION_ROOT" \
            K3_DEFERRED_VALIDATION_DIR="$deferred_validation_dir" \
            bash "$(dirname "$script_path")/k3_perf_overlays/prepare_deferred_finalize.sh" \
                prepare
    fi

    capture_kfd_owners() {
        local output="$1"
        if [[ -d /sys/class/kfd/kfd/proc ]]; then
            find /sys/class/kfd/kfd/proc -mindepth 1 -maxdepth 1 \
                -printf '%f\n' 2>/dev/null | LC_ALL=C sort >"$output"
        elif command -v fuser >/dev/null 2>&1; then
            fuser /dev/kfd 2>/dev/null \
                | tr ' ' '\n' | sed '/^$/d' | LC_ALL=C sort -n >"$output" || true
        else
            : >"$output"
        fi
    }

    capture_kfd_owner_details() {
        local owners="$1"
        local output="$2"
        local pid
        : >"$output"
        while IFS= read -r pid; do
            [[ "$pid" =~ ^[1-9][0-9]*$ ]] || continue
            if ! ps -o pid=,ppid=,uid=,stat=,comm=,args= -p "$pid" \
                    >>"$output" 2>/dev/null; then
                printf '%s\tprocess-exited-before-inspection\n' "$pid" >>"$output"
            fi
        done <"$owners"
    }

    capture_shared_memory() {
        local output="$1"
        find /dev/shm -mindepth 1 -maxdepth 1 -printf '%f\n' 2>/dev/null \
            | grep -E '^(nccl|rccl|psm_|sem\.|torch_|vllm)' \
            | LC_ALL=C sort >"$output" || true
    }

    capture_server_processes() {
        local output="$1"
        pgrep -af '[v]llm serve|VLLM::(EngineCore|Worker)|[m]ultiproc_executor' \
            >"$output" || true
    }

    mkdir -p "$aba_dir/cleanup"
    wait_for_amd_gpu_clean
    rocm-smi --showmemuse \
        >"$aba_dir/cleanup/initial_rocm_smi_memuse.txt" 2>&1 || true
    capture_kfd_owners "$aba_dir/cleanup/initial_kfd_owners.txt"
    capture_kfd_owner_details \
        "$aba_dir/cleanup/initial_kfd_owners.txt" \
        "$aba_dir/cleanup/initial_kfd_owner_details.txt"
    capture_shared_memory "$aba_dir/cleanup/initial_shared_memory.txt"
    capture_server_processes "$aba_dir/cleanup/initial_server_processes.txt"
    # Entering an Enroot allocation can itself create a stable KFD owner, and
    # the shared node may retain harmless Python shared-memory names from older
    # jobs. Treat those as the allocation baseline after proving VRAM is clean;
    # every arm must restore exactly this state. A pre-existing vLLM process is
    # still an unconditional failure.
    if [[ -s "$aba_dir/cleanup/initial_server_processes.txt" ]]; then
        echo "Error: same-node A/B/A allocation had a server at startup" >&2
        exit 1
    fi

    verify_aba_arm_cleanup() {
        local label="$1"
        local arm_result_dir="$2"
        local cleanup_dir="$aba_dir/cleanup/$label"
        local server_pid=""
        local cleanup_complete=false
        local cleanup_attempt
        mkdir -p "$cleanup_dir"

        if [[ -s "$arm_result_dir/server_pid.txt" ]]; then
            server_pid="$(<"$arm_result_dir/server_pid.txt")"
            if [[ "$server_pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$server_pid" 2>/dev/null; then
                echo "Error: $label server PID $server_pid survived arm cleanup" >&2
                return 1
            fi
        fi

        wait_for_amd_gpu_clean
        rocm-smi --showmemuse >"$cleanup_dir/rocm_smi_memuse.txt" 2>&1 || true
        for ((cleanup_attempt = 1; cleanup_attempt <= 120; cleanup_attempt++)); do
            capture_kfd_owners "$cleanup_dir/kfd_owners.txt"
            capture_kfd_owner_details \
                "$cleanup_dir/kfd_owners.txt" \
                "$cleanup_dir/kfd_owner_details.txt"
            capture_shared_memory "$cleanup_dir/shared_memory.txt"
            capture_server_processes "$cleanup_dir/server_processes.txt"
            if [[ ! -s "$cleanup_dir/server_processes.txt" ]] && \
                    cmp -s \
                        "$aba_dir/cleanup/initial_kfd_owners.txt" \
                        "$cleanup_dir/kfd_owners.txt" && \
                    cmp -s \
                        "$aba_dir/cleanup/initial_shared_memory.txt" \
                        "$cleanup_dir/shared_memory.txt"; then
                cleanup_complete=true
                break
            fi
            sleep 1
        done
        if [[ "$cleanup_complete" != "true" ]]; then
            echo "Error: $label did not release its GPU, process, or shared-memory state" >&2
            diff -u \
                "$aba_dir/cleanup/initial_kfd_owners.txt" \
                "$cleanup_dir/kfd_owners.txt" >&2 || true
            diff -u \
                "$aba_dir/cleanup/initial_shared_memory.txt" \
                "$cleanup_dir/shared_memory.txt" >&2 || true
            return 1
        fi
        printf 'clean\n' >"$cleanup_dir/status.txt"
    }

    run_aba_process() {
        local label="$1"
        local variant="$2"
        local arm_result_dir="$3"
        local arm_result_filename="$4"
        local arm_output_dir="$5"
        local startup_smoke="$6"
        local starts_file="$7"
        local completions_file="$8"
        local failures_file="$9"
        local arm_cache_root="$cache_session_root/$label"
        local arm_rc=0
        local cleanup_rc=0
        local restore_rc=0

        mkdir -p \
            "$arm_result_dir" \
            "$arm_cache_root/triton" \
            "$arm_cache_root/torchinductor" \
            "$arm_cache_root/torch_extensions" \
            "$arm_cache_root/aiter_jit" \
            "$arm_cache_root/flydsl" \
            "$arm_cache_root/vllm" \
            "$arm_cache_root/pycache"
        printf 'TRITON_CACHE_DIR\t%s\nTORCHINDUCTOR_CACHE_DIR\t%s\nTORCH_EXTENSIONS_DIR\t%s\nAITER_JIT_DIR\t%s\nFLYDSL_RUNTIME_CACHE_DIR\t%s\nVLLM_CACHE_ROOT\t%s\nPYTHONPYCACHEPREFIX\t%s\n' \
            "$arm_cache_root/triton" \
            "$arm_cache_root/torchinductor" \
            "$arm_cache_root/torch_extensions" \
            "$arm_cache_root/aiter_jit" \
            "$arm_cache_root/flydsl" \
            "$arm_cache_root/vllm" \
            "$arm_cache_root/pycache" \
            >"$arm_result_dir/cache_paths.tsv"
        printf '%s\t%s\t%s\t%s\n' \
            "$label" "$variant" "$node_name" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            >>"$starts_file"
        set +e
        K3_PERF_VARIANT="$variant" \
            K3_STARTUP_SMOKE="$startup_smoke" \
            K3_ARM_CACHE_ROOT="$arm_cache_root" \
            K3_GPU_MEM_UTIL_OVERRIDE="$K3_ABA_GPU_MEM_UTIL" \
            K3_DEFERRED_SESSION_ROOT="$K3_DEFERRED_SESSION_ROOT" \
            K3_DEFERRED_VALIDATION_DIR="$K3_DEFERRED_VALIDATION_DIR" \
            TRITON_CACHE_DIR="$arm_cache_root/triton" \
            TORCHINDUCTOR_CACHE_DIR="$arm_cache_root/torchinductor" \
            TORCH_EXTENSIONS_DIR="$arm_cache_root/torch_extensions" \
            AITER_JIT_DIR="$arm_cache_root/aiter_jit" \
            FLYDSL_RUNTIME_CACHE_DIR="$arm_cache_root/flydsl" \
            VLLM_CACHE_ROOT="$arm_cache_root/vllm" \
            PYTHONPYCACHEPREFIX="$arm_cache_root/pycache" \
            RESULT_DIR="$arm_result_dir" \
            RESULT_FILENAME="$arm_result_filename" \
            AGENTIC_OUTPUT_DIR="$arm_output_dir" \
            bash "$script_path"
        arm_rc=$?
        set -e

        # The metadata candidate replaces installed vLLM Python sources. Put
        # the exact baseline files back before any later baseline can start,
        # even when the candidate process itself failed.
        if [[ "$variant" == "metadatareuse" ]]; then
            K3_ARM_CACHE_ROOT="$arm_cache_root" \
                RESULT_DIR="$arm_result_dir" \
                bash "$(dirname "$script_path")/k3_perf_overlays/apply_vllm_metadata_reuse_overlay.sh" \
                    restore || restore_rc=$?
        fi
        if [[ "$variant" == "deferredfinalize" ]]; then
            K3_ARM_CACHE_ROOT="$arm_cache_root" \
                RESULT_DIR="$arm_result_dir" \
                K3_DEFERRED_SESSION_ROOT="$K3_DEFERRED_SESSION_ROOT" \
                K3_DEFERRED_VALIDATION_DIR="$K3_DEFERRED_VALIDATION_DIR" \
                bash "$(dirname "$script_path")/k3_perf_overlays/prepare_deferred_finalize.sh" \
                    restore || restore_rc=$?
        fi

        if ! verify_aba_arm_cleanup "$label" "$arm_result_dir"; then
            cleanup_rc=1
        fi
        if ((restore_rc != 0)); then
            printf '%s\t%s\t%s\t%s\toverlay-restore-rc=%s\n' \
                "$label" "$variant" "$node_name" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
                "$restore_rc" >>"$failures_file"
            return "$restore_rc"
        fi
        if ((cleanup_rc != 0)); then
            printf '%s\t%s\t%s\t%s\tcleanup\n' \
                "$label" "$variant" "$node_name" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
                >>"$failures_file"
            return 1
        fi
        if ((arm_rc != 0)); then
            printf '%s\t%s\t%s\t%s\trc=%s\n' \
                "$label" "$variant" "$node_name" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
                "$arm_rc" >>"$failures_file"
            return "$arm_rc"
        fi
        printf '%s\t%s\t%s\t%s\n' \
            "$label" "$variant" "$node_name" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            >>"$completions_file"
    }

    run_startup_smoke() {
        local label="$1"
        local variant="$2"
        local arm_result_dir="$aba_dir/startup_smoke/$label"
        run_aba_process \
            "$label" "$variant" "$arm_result_dir" \
            "${root_result_filename}_${label}" "$arm_result_dir" 1 \
            "$aba_dir/smoke_starts.tsv" "$aba_dir/smoke_completions.tsv" \
            "$aba_dir/smoke_failures.tsv"
    }

    run_aba_arm() {
        local label="$1"
        local variant="$2"
        local arm_result_dir="$3"
        local arm_result_filename="$4"
        local arm_output_dir="$5"
        run_aba_process \
            "$label" "$variant" "$arm_result_dir" "$arm_result_filename" \
            "$arm_output_dir" 0 "$aba_dir/arm_starts.tsv" \
            "$aba_dir/arm_completions.tsv" "$aba_dir/arm_failures.tsv"
    }

    # Reproduce the exact baseline -> candidate restart boundary that failed in
    # the earlier M=7 run before committing roughly one hour to three full arms.
    run_startup_smoke smoke_baseline baseline
    run_startup_smoke smoke_candidate "$K3_ABA_CANDIDATE"

    run_aba_arm \
        baseline_pre baseline \
        "$aba_dir/baseline_pre" \
        "${root_result_filename}_aba_baseline_pre" \
        "$aba_dir/baseline_pre"
    run_aba_arm \
        candidate "$K3_ABA_CANDIDATE" \
        "$root_result_dir" \
        "$root_result_filename" \
        "$INFMAX_CONTAINER_WORKSPACE"
    run_aba_arm \
        baseline_post baseline \
        "$aba_dir/baseline_post" \
        "${root_result_filename}_aba_baseline_post" \
        "$aba_dir/baseline_post"

    cp "$INFMAX_CONTAINER_WORKSPACE/$root_result_filename.json" \
        "$aba_dir/candidate_aggregate.json"
    find "$root_result_dir" -type f \
        ! -path '*/SHA256SUMS' -print0 \
        | sort -z \
        | xargs -0 sha256sum >"$aba_dir/SHA256SUMS"
    printf 'Kimi-K3 same-node A/B/A completed: baseline -> %s -> baseline on %s\n' \
        "$K3_ABA_CANDIDATE" "$node_name"
    exit 0
fi

wait_for_amd_gpu_clean

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

if [[ -n "${K3_ARM_CACHE_ROOT:-}" ]]; then
    aiter_jit_source="$(python3 - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("aiter")
if spec is None or spec.origin is None:
    raise SystemExit("cannot locate installed AITER package")
print(Path(spec.origin).resolve().parent / "jit")
PY
)"
    if [[ ! -d "$aiter_jit_source" ]]; then
        echo "Error: installed AITER JIT directory is missing: $aiter_jit_source" >&2
        exit 1
    fi
    shopt -s nullglob
    for aiter_module in "$aiter_jit_source"/*.so; do
        ln -s "$aiter_module" "$AITER_JIT_DIR/$(basename "$aiter_module")"
    done
    shopt -u nullglob
    if [[ -d "$aiter_jit_source/flydsl_cache" ]]; then
        if ! cp -a --reflink=auto \
            "$aiter_jit_source/flydsl_cache/." "$FLYDSL_RUNTIME_CACHE_DIR/"; then
            cp -a "$aiter_jit_source/flydsl_cache/." "$FLYDSL_RUNTIME_CACHE_DIR/"
        fi
    fi
    {
        printf 'K3_ARM_CACHE_ROOT\t%s\n' "$K3_ARM_CACHE_ROOT"
        printf 'TRITON_CACHE_DIR\t%s\n' "$TRITON_CACHE_DIR"
        printf 'TORCHINDUCTOR_CACHE_DIR\t%s\n' "$TORCHINDUCTOR_CACHE_DIR"
        printf 'TORCH_EXTENSIONS_DIR\t%s\n' "$TORCH_EXTENSIONS_DIR"
        printf 'AITER_JIT_DIR\t%s\n' "$AITER_JIT_DIR"
        printf 'AITER_JIT_SEED\t%s\n' "$aiter_jit_source"
        printf 'FLYDSL_RUNTIME_CACHE_DIR\t%s\n' "$FLYDSL_RUNTIME_CACHE_DIR"
        printf 'VLLM_CACHE_ROOT\t%s\n' "$VLLM_CACHE_ROOT"
        printf 'PYTHONPYCACHEPREFIX\t%s\n' "$PYTHONPYCACHEPREFIX"
    } >"$RESULT_DIR/cache_provenance.tsv"
fi

# ---- Reference env block ----------------------------------------------------
export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm
export VLLM_ROCM_USE_AITER=1
export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1
export AITER_SITUV2_A8W4=1
export AITER_BF16_FP8_MOE_BOUND=0
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export AITER_QUICK_REDUCE_QUANTIZATION=INT4

case "$K3_PERF_VARIANT" in
    baseline|spec3)
        if [[ -n "$K3_DEFERRED_SESSION_ROOT" ]]; then
            bash "$(dirname "$0")/k3_perf_overlays/prepare_deferred_finalize.sh" \
                activate-base
        fi
        ;;
    mla52494)
        bash "$(dirname "$0")/k3_perf_overlays/apply_vllm_overlay.sh" pr52494
        ;;
    pynccl)
        export K3_DISABLE_CUSTOM_ALL_REDUCE=1
        ;;
    m7tunedgemm)
        bash "$(dirname "$0")/k3_perf_overlays/prepare_m7_bf16_gemm_config.sh" \
            "$RESULT_DIR"
        export AITER_CONFIG_GEMM_BF16="$RESULT_DIR/k3_m7_bf16_runtime_config.csv"
        export AITER_LOG_TUNED_CONFIG=1
        ;;
    metadatareuse)
        bash "$(dirname "$0")/k3_perf_overlays/apply_vllm_metadata_reuse_overlay.sh" \
            apply
        ;;
    deferredfinalize)
        export VLLM_ROCM_USE_AITER_CUSTOM_AR=1
        bash "$(dirname "$0")/k3_perf_overlays/prepare_deferred_finalize.sh" \
            activate-candidate
        ;;
    tritonmla)
        # Preserve AITER for the rest of the ROCm stack while making MLA
        # decode auto-selection choose the Triton backend as an isolated A/B.
        export VLLM_ROCM_USE_AITER_MLA=0
        ;;
    *)
        echo "Error: unsupported Kimi-K3 performance variant '$K3_PERF_VARIANT'" >&2
        exit 1
        ;;
esac
echo "Kimi-K3 C1 performance variant: $K3_PERF_VARIANT"

# Workaround for MEC FW <177 RCCL memory reclaim issue (shared with the other
# gfx950 recipes in this tree).
mec_version=$(rocm-smi --showfw 2>/dev/null | grep MEC | head -n 1 | awk '{print $NF}')
if [[ "$mec_version" == "" || ${mec_version:-0} -lt 177 ]]; then
    export HSA_NO_SCRATCH_RECLAIM=1
fi

# 2.8T of weights off a shared/NFS mount takes far longer than the default.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"

# Long agentic turns against a 1M context: keep the client from timing out
# mid-request while the server is prefill-bound.
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"
printf '%s\n' "${SLURMD_NODENAME:-unknown}" >"$RESULT_DIR/slurm_node.txt"

SERVER_PID=""
LMCACHE_PID=""

cleanup_agentic_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "vLLM server" 60
    stop_background_process_tree "$LMCACHE_PID" "LMCache server"
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
      lmcache)
    require_agentic_kv_offload_backend "$KV_OFFLOAD_BACKEND"

    # Keep the image's tested torch/ROCm stack and install only LMCache's
    # missing runtime dependencies, same as the MiniMax-M3 lmcache arm.
    LMCACHE_VERSION="0.5.5.dev60+rocm7.2"
    LMCACHE_ROCM_INDEX="https://github.com/LMCache/LMCache/releases/expanded_assets/nightly-rocm"
    agentic_pip_install --quiet --no-cache-dir --no-deps \
        "sortedcontainers==2.4.0" \
        "opentelemetry-exporter-prometheus==0.61b0" \
        "cupy-rocm-7-0==14.1.1" \
        "lmcache==${LMCACHE_VERSION}" --find-links "$LMCACHE_ROCM_INDEX"

    # LMCache 0.5.5's transfer-channel layer eagerly imports the Mooncake
    # backend (mooncake_te_impl.py -> `from mooncake.engine import
    # TransferEngine`), whose native .so resolves all of its DT_NEEDED libs at
    # import. The vLLM ROCm image ships none of them, so the import sanity
    # check below (and the LMCache server) would otherwise fail with
    # "ImportError: lib*.so: cannot open shared object file" (first libglog,
    # then libjsoncpp, ...). Provision Mooncake's full runtime lib set from the
    # distro before importing. apt-get install is idempotent, so run it
    # whenever any of the libs is still missing rather than gating on one.
    LMCACHE_NATIVE_LIBS=(libglog.so.0 libjsoncpp.so.25 libibverbs.so.1 librdmacm.so.1 libnuma.so.1)
    for lib in "${LMCACHE_NATIVE_LIBS[@]}"; do
        if ! ldconfig -p | grep -q "$lib"; then
            apt-get update
            apt-get install -y \
                libgoogle-glog0v5 libjsoncpp25 libibverbs1 librdmacm1 libnuma1
            break
        fi
    done
    python3 -c \
        "import cupy; import lmcache.integration.vllm.lmcache_mp_connector; import opentelemetry.exporter.prometheus" \
        >/dev/null

    # One MP server for the node, per the Kimi-K3 recipe
    # (docs.lmcache.ai/recipes/kimi_k3.html), with --chunk-size sized for
    # THIS stack rather than the recipe's CUDA-path 768: the connector
    # requires the chunk to be a multiple of every engine KV group's
    # tokens_per_block, and the hybrid KDA/MLA layout here registers
    # attention groups at 1536 ("Setting attention block size to 1536",
    # run 31644990546) plus a KDA state group at 3072 (run 31645828378),
    # so 3072 is the minimum valid chunk. The multi-group layout also
    # requires one object group per sliding-window size:
    # --separate-object-groups.
    LMCACHE_PORT=6555
    LMCACHE_HTTP_PORT=8090
    LMCACHE_LOG="$RESULT_DIR/lmcache_server.log"

    LMCACHE_L1_SIZE_GB="$TOTAL_CPU_DRAM_GB"

    LMCACHE_CMD=(
        lmcache server
        --host 127.0.0.1
        --port "$LMCACHE_PORT"
        --http-host 127.0.0.1
        --http-port "$LMCACHE_HTTP_PORT"
        --l1-size-gb "$LMCACHE_L1_SIZE_GB"
        --l1-init-size-gb 10
        --chunk-size 3072
        --separate-object-groups
        --enable-extra-logging
        --extra-logging-interval 30
        --max-cpu-workers 8
        --max-gpu-workers 1
        --eviction-policy LRU
        --supported-transfer-mode lmcache_driven
        --shm-name ""
    )
    append_command "$RESULT_DIR/lmcache_command.txt" "${LMCACHE_CMD[@]}"
    "${LMCACHE_CMD[@]}" > "$LMCACHE_LOG" 2>&1 &
    LMCACHE_PID=$!
    wait_for_ready \
        --endpoint "http://127.0.0.1:${LMCACHE_HTTP_PORT}/healthcheck" \
        --log "$LMCACHE_LOG" \
        --pid "$LMCACHE_PID" \
        --sleep-interval 1 \
        --timeout 600

    # 100k-330k-token agentic prefixes make single retrieves large; use the
    # same MQ timeout headroom as the MiniMax-M3 arm.
    OFFLOAD_ARGS=(
        --kv-transfer-config
        "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.port\":$LMCACHE_PORT,\"lmcache.mp.mq_timeout\":6000.0}}"
    )
    ;;
    *)
    echo "Error: unsupported KV_OFFLOAD_BACKEND='$KV_OFFLOAD_BACKEND' (expected vllm-simple or lmcache)" >&2
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

# ---- Speculative / Util------------------------------------------------------
case "$CONC" in
    # No KV offload; the working set fits in HBM.
    1)
        SYNTHETIC_ACCEPT_LEN=3.75
        SPEC_NUM_TOKENS=6
        GPU_MEM_UTIL=0.9
        MAX_NUM_BATCHED_TOKENS=16384
        ;;
    2|4|8|10|12|14)
        SYNTHETIC_ACCEPT_LEN=3.00
        SPEC_NUM_TOKENS=3
        GPU_MEM_UTIL=0.9
        MAX_NUM_BATCHED_TOKENS=8192
        ;;
    *)
        SYNTHETIC_ACCEPT_LEN=0
        SPEC_NUM_TOKENS=0
        GPU_MEM_UTIL=0.85
        MAX_NUM_BATCHED_TOKENS=4096
        ;;
esac

if [[ -n "${K3_GPU_MEM_UTIL_OVERRIDE:-}" ]]; then
    GPU_MEM_UTIL="$K3_GPU_MEM_UTIL_OVERRIDE"
fi

if [[ "$K3_PERF_VARIANT" == "spec3" ]]; then
    if [[ "$CONC" != "1" || "${DCP_SIZE:-}" != "1" || "$KV_OFFLOADING" != "none" ]]; then
        echo "Error: spec3 requires CONC=1, DCP_SIZE=1, and KV_OFFLOADING=none" >&2
        exit 1
    fi
    SPEC_NUM_TOKENS=3
fi
echo "Kimi-K3 speculative config: drafts=$SPEC_NUM_TOKENS synthetic_acceptance_length=$SYNTHETIC_ACCEPT_LEN"

SPEC_ARGS=()
if [ "$SPEC_NUM_TOKENS" -gt 0 ]; then
if [ "${EVAL_ONLY:-false}" = "true" ]; then
    SPEC_ARGS=(
        --speculative-config
        "{\"model\":\"Inferact/Kimi-K3-DSpark\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"TRITON_MLA\",\"kv_cache_dtype\":\"fp8\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\": \"block\"}"
    )
else
    SPEC_ARGS=(
        --speculative-config
        "{\"model\":\"Inferact/Kimi-K3-DSpark\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"TRITON_MLA\",\"kv_cache_dtype\":\"fp8\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}"
    )
    fi
fi

# ---- HIP graph ------------------------------------------------------------
MAX_NUM_SEQS=$((2 * CONC))
MAX_CUDAGRAPH_CAPTURE_SIZE=$((MAX_NUM_SEQS * (1 + SPEC_NUM_TOKENS)))
CUDAGRAPH_CAPTURE_SIZES="$(seq -s, 2 "$MAX_CUDAGRAPH_CAPTURE_SIZE")"
COMPILATION_CONFIG_ARGS=(--compilation-config "{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"max_cudagraph_capture_size\":$MAX_CUDAGRAPH_CAPTURE_SIZE,\"custom_ops\":[\"+fused_rms_norm_gated\"],\"cudagraph_capture_sizes\":[$CUDAGRAPH_CAPTURE_SIZES]}")
printf 'variant\t%s\nspec_num_tokens\t%s\nsynthetic_acceptance_length\t%s\ngpu_memory_utilization\t%s\nmax_cudagraph_capture_size\t%s\ncudagraph_capture_sizes\t%s\n' \
    "$K3_PERF_VARIANT" "$SPEC_NUM_TOKENS" "$SYNTHETIC_ACCEPT_LEN" \
    "$GPU_MEM_UTIL" \
    "$MAX_CUDAGRAPH_CAPTURE_SIZE" "$CUDAGRAPH_CAPTURE_SIZES" \
    >"$RESULT_DIR/spec_decode_provenance.tsv"

echo "Starting vllm server..."
export PYTHONNOUSERSITE=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1200}"


# ---- DCP       ------------------------------------------------------------
# DCP shards decode KV across the TP ranks, so it must divide TP.
DCP_SIZE="${DCP_SIZE:-8}"
if [ $((TP % DCP_SIZE)) -ne 0 ]; then
    echo "Error: TP='$TP' must be divisible by DCP_SIZE='$DCP_SIZE'" >&2
    exit 1
fi
CP_ARGS=()
ATTN_BE_ARGS=()
if [ "$DCP_SIZE" -gt 1 ]; then
    CP_ARGS+=(--decode-context-parallel-size "$DCP_SIZE" --dcp-comm-backend a2a)
    ATTN_BE_ARGS+=(--attention-backend TRITON_MLA)
fi
export VLLM_USE_DIRECT_DCP_A2A=0
export VLLM_USE_DIRECT_DCP_Q_GATHER=0
export VLLM_USE_DIRECT_DCP_KV_GATHER=0

CUSTOM_ALL_REDUCE_ARGS=()
if [[ "${K3_DISABLE_CUSTOM_ALL_REDUCE:-0}" == "1" ]]; then
    CUSTOM_ALL_REDUCE_ARGS+=(--disable-custom-all-reduce)
fi

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
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
    --max-model-len 1048576
    --enable-prefix-caching
    --kv-cache-dtype "fp8"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --attention-config '{"mla_prefill_backend":"ROCM_AITER_FA"}'
    "${ATTN_BE_ARGS[@]}"
    "${CUSTOM_ALL_REDUCE_ARGS[@]}"
    "${COMPILATION_CONFIG_ARGS[@]}"
    "${SPEC_ARGS[@]}"
    "${OFFLOAD_ARGS[@]}"
    "${CP_ARGS[@]}"
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" >"$RESULT_DIR/server_pid.txt"
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "$K3_PERF_VARIANT" == "tritonmla" ]]; then
    if ! grep -F "Using TRITON_MLA backend" "$SERVER_LOG" >/dev/null; then
        echo "Error: tritonmla variant did not select the TRITON_MLA backend" >&2
        exit 1
    fi
    if grep -F "Using ROCM_AITER_MLA backend" "$SERVER_LOG" >/dev/null; then
        echo "Error: tritonmla variant retained the ROCM_AITER_MLA backend" >&2
        exit 1
    fi
fi

if [[ "$K3_PERF_VARIANT" == "deferredfinalize" ]]; then
    if ! grep -F \
        "Kimi-K3 latent-MoE tail: using deferred AITER route reduction with fused all-reduce and RMSNorm." \
        "$SERVER_LOG" >/dev/null; then
        echo "Error: deferredfinalize did not execute the fused deferred route" >&2
        exit 1
    fi
fi

grep -F "all-reduce backends" "$SERVER_LOG" || true
if [[ "${K3_DISABLE_CUSTOM_ALL_REDUCE:-0}" == "1" ]]; then
    if ! grep -F "Using ['PYNCCL'] all-reduce backends" "$SERVER_LOG" >/dev/null; then
        echo "Error: --disable-custom-all-reduce did not expose a PYNCCL backend" >&2
        exit 1
    fi
fi

if ! curl -fsS --max-time 10 "http://localhost:${PORT}/health" >/dev/null; then
    echo "Error: vLLM server failed the post-start health check" >&2
    exit 1
fi

if [[ "${K3_STARTUP_SMOKE:-0}" == "1" ]]; then
    printf 'healthy\n' >"$RESULT_DIR/startup_smoke_status.txt"
    echo "Kimi-K3 startup smoke completed: variant=$K3_PERF_VARIANT"
    exit 0
fi

if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi

if ! curl -fsS --max-time 10 "http://localhost:${PORT}/health" >/dev/null; then
    echo "Error: vLLM server was unhealthy after the benchmark" >&2
    exit 1
fi
printf 'healthy\n' >"$RESULT_DIR/post_run_health.txt"
