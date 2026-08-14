#!/usr/bin/env bash
set -eo pipefail
set -x

# Agentic trace replay benchmark for Kimi-K3 (MXFP4) on B300 using vLLM with
# DSpark speculative decoding at level 2, probabilistic drafting, synthetic
# acceptance pinned to the committed golden AL (real verification for evals).
#
# MEASURED, for anyone reading the throughput numbers: this draft head's REAL
# acceptance collapses on the agentic corpus. Its native window is 32k
# (rope_parameters yarn, factor 32, original_max_position_embeddings 32768,
# stretched to the declared 1M), and at the corpus's 100k-330k ISL the measured
# acceptance length is 1.16-2.01 with 13-41% position-1 acceptance -- versus 4.18
# AL and 0.826 position-1 on short-context work under the identical image, draft
# and spec config. Synthetic acceptance is the AgentX-prescribed way to measure
# system performance at a fixed acceptance target (see the SPEC_CONFIG block), so
# these throughput numbers are NOT evidence that the draft predicts well at
# agentic context lengths. It does not.
#
# MTP sibling of kimik3_fp4_b300_vllm.sh. Everything about the target server is
# identical; the deltas are:
#   - --speculative-config  method "dspark", draft Inferact/Kimi-K3-DSpark,
#                           attention_backend FLASHINFER_MLA,
#                           draft_sample_method probabilistic,
#                           rejection_sample_method block
#   - cudagraph capture sizes are expressed in TOKENS, not sequences
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION
#
# TP8 is the only single-node layout. The MXFP4 checkpoint is ~1.5 TB on disk;
# at TP4 that is ~375 GB of weights per GPU against B300's 288 GB of HBM, so
# only the full 8-GPU shard (~188 GB/GPU) fits. Do not add TP4/TP2 arms.
#
# Chat formatting: AGENTS.md requires MTP scripts to pass --use-chat-template to
# run_benchmark_serving. Agentic recipes never call it -- the replay drives
# AIPerf against /v1/chat/completions, so prompts are already chat-formatted and
# spec-decode acceptance is measured on chat-shaped inputs. Nothing to add here.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION

# The 2.8T MXFP4 checkpoint only fits across all 8 B300s (see header).
if [ "$TP" -ne 8 ]; then
    echo "Error: Kimi-K3 on B300 requires TP=8 (a ~1.5 TB MXFP4 checkpoint does not fit at TP<8), got TP='$TP'" >&2
    exit 1
fi

if [[ -n "${EP_SIZE:-}" && "${EP_SIZE}" -gt 1 ]]; then
    echo "Error: this recipe ships the pure-TP8 profile; EP_SIZE='$EP_SIZE' is not wired yet" >&2
    exit 1
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

DRAFT_MODEL="${DRAFT_MODEL:-Inferact/Kimi-K3-DSpark}"

# `hf download` creates the target dir if missing and is itself idempotent.
# When MODEL_PATH is unset (stand-alone runs), fall back to the HF_HUB_CACHE.
# Either way, MODEL_PATH is what the server is launched with.
#
# The draft must NOT land next to a pre-staged target: Kimi-K3 is in the
# b300-nv launcher's STAGED_MODELS, so dirname(MODEL_PATH) is the read-only
# /scratch/models mount and writing there fails with PermissionError. Use the
# launcher's writable models dir instead, which also caches the draft across
# the jobs in a sweep (same placement as DeepSeek-V4-Pro-DSpark).
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
    DRAFT_MODEL_PATH="${WRITABLE_MODELS_DIR:-/data/models}/${DRAFT_MODEL##*/}"
    if [[ ! -d "$DRAFT_MODEL_PATH" || -z "$(ls -A "$DRAFT_MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$DRAFT_MODEL" --local-dir "$DRAFT_MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
    hf download "$DRAFT_MODEL"
    DRAFT_MODEL_PATH="$DRAFT_MODEL"
fi
nvidia-smi

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# ---- Kimi-K3 production serving environment ---------------------------------
export NCCL_DMABUF_ENABLE=0
export VLLM_ALLREDUCE_USE_FLASHINFER=1
# Required by the upstream recipe on BOTH its blackwell and nvidia paths
# (recipes.vllm.ai/moonshotai/Kimi-K3), and never set here before. It defaults to
# 0, and it is threaded into LatentMoERunner as
# runner_args={"enable_k3_latent_moe_tail_fusion": ...} at
# vllm/models/kimi_k3/nvidia/model.py:549 -- the same runner whose shared-experts
# output buffer asserted (fused_moe/runner/shared_experts.py:165, all 8 TP ranks)
# when the wider flag alignment was tried. With this unset we have been running a
# MoE tail path upstream never exercises, so enable it on its own before
# re-attempting any of the other upstream flags.
export VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1
export VLLM_USE_RUST_FRONTEND=1
# Loading ~1.5 TB of MXFP4 shards off the staged mount takes well past the
# default readiness window even with fastsafetensors.
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export PYTHONNOUSERSITE=1
# AIPerf pins one pooled keep-alive connection per agentic session and reuses it
# across turns, while the Rust frontend's default VLLM_HTTP_TIMEOUT_KEEP_ALIVE is
# 5s. An inter-turn idle gap longer than that lets the client reuse a socket at
# the moment the server closes it -> aiohttp ServerDisconnectedError -> AIPerf
# treats it as a terminal warmup failure and aborts the whole job. Outlast the
# client pool so the race cannot occur.
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900
# Agentic warmup dispatches large prompts at once; allow up to 15 minutes of TCP
# progress before AIPerf declares a connection dead.
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

# The LMCache arm starts a second long-lived process that must not outlive this
# job (its L1 holds the whole host-DRAM budget). The vLLM server's lifecycle is
# left exactly as it was -- the job wrapper still owns it.
LMCACHE_PID=""

cleanup_lmcache_server() {
    local exit_code=$?
    trap - EXIT
    set +e
    stop_background_process_tree "$LMCACHE_PID" "LMCache server"
    exit "$exit_code"
}
trap cleanup_lmcache_server EXIT

# ---- KV offloading ----------------------------------------------------------
# The generated TOTAL_CPU_DRAM_GB budget is the aggregate host-DRAM pool for the
# node; SimpleCPUOffloadConnector is sized per rank. At dram-utilization 0.63 on
# cluster:b300-nv this resolves to ~220 GiB per rank across the 8 TP ranks.
OFFLOAD_ARGS=()
case "${KV_OFFLOAD_BACKEND:-}" in
    "")
        require_agentic_kv_offload_none
        ;;
    vllm-simple)
        require_agentic_kv_offload_backend vllm-simple
        CPU_BYTES_PER_RANK=$(( TOTAL_CPU_DRAM_GB * 1000 * 1000 * 1000 / TP ))
        # Identical prefixes must hash to identical block keys run-to-run.
        export PYTHONHASHSEED=42
        # lazy_offload must be a JSON boolean, not a quoted string: the
        # connector does bool(extra_config.get("lazy_offload", False)), and
        # bool("false") is True in Python.
        OFFLOAD_ARGS=(
            --kv-transfer-config
            "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use_per_rank\":${CPU_BYTES_PER_RANK},\"lazy_offload\":false}}"
        )
        ;;
    lmcache)
        require_agentic_kv_offload_backend lmcache

        # Plain PyPI. The GitHub release `expanded_assets` URLs were never
        # doing anything here -- pip resolved the identical PyPI wheel
        # (lmcache-0.5.4rc2-cp312-cp312-manylinux_2_27_x86_64...whl, 15.6 MB)
        # in every run, whether --find-links pointed at the -cu129 assets or
        # the default ones.
        #
        # --force-reinstall IS required: this image ships lmcache 0.5.3, so a
        # plain install of 0.5.4rc2 upgrades it, but without the flag an
        # already-matching version would be left in place silently.
        #
        # Why any of this matters: LMCache does not fail when its compiled
        # `lmcache.c_ops` extension cannot be loaded. It logs "compiled
        # extension not found; CudaDeviceOps stays on the torch baseline" and
        # silently falls back to lmcache/v1/platform/torch_ops.py, which is
        # broken for this stack's hybrid multi-KV-group KDA/MLA layout: 77-98%
        # of stores die with cudaErrorInvalidValue, so the offload tier stays
        # empty and the arm measures a cache that never stored anything.
        # The guard after server start is what makes that loud.
        # --force-reinstall is REQUIRED, not cosmetic: this image already ships
        # lmcache 0.5.4rc2, so a plain `pip install lmcache==0.5.4rc2` finds the
        # requirement already satisfied and installs nothing -- the --find-links
        # URL is never consulted. Two earlier sweeps proved it: one pinned to
        # the -cu129 assets and one to the default assets both produced the
        # identical build, `LMCache v0.5.4rc2 (gf82f6fd3)`, in ~3 seconds. We
        # were always running the image's copy, whose c_ops does not load here.
        LMCACHE_VERSION="0.5.4rc2"
        agentic_pip_install --no-cache-dir --force-reinstall --no-deps \
            "lmcache==${LMCACHE_VERSION}"
        # Record what actually landed. Silent installs are how the no-op above
        # went unnoticed for two full sweeps; do not add --quiet back.
        python3 -m pip show lmcache 2>/dev/null | grep -E "^(Version|Location):"
        python3 -c "import lmcache, glob, os; print('lmcache:', lmcache.__file__); print('c_ops so:', glob.glob(os.path.join(os.path.dirname(lmcache.__file__), 'c_ops*')))"
        ls -1 /usr/local/cuda*/lib64/libcudart.so* /usr/lib/x86_64-linux-gnu/libcudart.so* 2>/dev/null || true
        # device_ops.ensure_native() swallows this in `except ImportError`, so
        # surface the real dlopen error ourselves -- without it the failure is
        # indistinguishable from a missing file.
        python3 - <<'PYEOF' || true
# Definitive c_ops instrumentation.
#
# ensure_native() swallows the real exception in `except ImportError` and
# latches _native_bound=True on the first attempt, so neither the error nor the
# caller is ever visible. Wrap __import__ to capture the exception, and hook the
# logger to capture the stack at the moment the warning is emitted -- together
# these name both WHAT fails and WHO triggered it.
import builtins
import logging
import sys
import traceback

_real_import = builtins.__import__
_failures = []


def _tracing_import(name, globals=None, locals=None, fromlist=(), level=0):
    try:
        return _real_import(name, globals, locals, fromlist, level)
    except BaseException as exc:  # noqa: BLE001 - diagnostic
        if "c_ops" in name:
            _failures.append((name, exc))
            print(f"[probe] IMPORT FAILED: {name}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
        raise


class _WarnHook(logging.Handler):
    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return
        if "c_ops compiled extension not found" in msg:
            print("[probe] ensure_native() gave up -- caller stack:", flush=True)
            traceback.print_stack()


builtins.__import__ = _tracing_import
logging.getLogger().addHandler(_WarnHook())
logging.getLogger().setLevel(logging.DEBUG)

print("[probe] torch loaded before lmcache?", "torch" in sys.modules, flush=True)

# Import exactly what the server imports, in the server's order.
import lmcache.integration.vllm.lmcache_mp_connector  # noqa: E402,F401

print("[probe] torch in sys.modules now:", "torch" in sys.modules, flush=True)
print("[probe] c_ops in sys.modules:", "lmcache.c_ops" in sys.modules, flush=True)
print(f"[probe] captured c_ops import failures: {len(_failures)}", flush=True)
for name, exc in _failures:
    print(f"[probe]   {name}: {type(exc).__name__}: {exc}", flush=True)

# Now show whether the singleton is latched off despite the extension being loadable.
try:
    from lmcache.v1.platform.cuda.device_ops import CudaDeviceOps

    ops = CudaDeviceOps()
    print("[probe] CudaDeviceOps._native_bound =", getattr(ops, "_native_bound", "?"), flush=True)
except BaseException as exc:  # noqa: BLE001
    print("[probe] could not inspect CudaDeviceOps:", type(exc).__name__, exc, flush=True)

# And whether a direct import works at this point.
try:
    import lmcache.c_ops  # noqa: F401

    print("[probe] direct import lmcache.c_ops AFTER: OK", flush=True)
except BaseException as exc:  # noqa: BLE001
    print("[probe] direct import lmcache.c_ops AFTER FAILED:", type(exc).__name__, exc, flush=True)
PYEOF
        # c_ops.so links libtorch/libc10 but they are not on the default
        # loader path -- `ldd` reports libc10.so, libtorch.so, libtorch_cpu.so,
        # libtorch_python.so, libc10_cuda.so and libtorch_cuda.so as "not
        # found" (libcudart resolves fine). Normally that is harmless because
        # `import torch` loads them RTLD_GLOBAL first, but
        # CudaDeviceOps.ensure_native() sets `self._native_bound = True` BEFORE
        # its `import lmcache.c_ops`, so one early failure -- before torch is
        # in the process -- permanently disables native ops and silently pins
        # the whole server to the broken torch fallback. Putting torch's lib
        # dir on LD_LIBRARY_PATH makes the extension loadable regardless of
        # import order.
        TORCH_LIB_DIR=$(python3 -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
        export LD_LIBRARY_PATH="${TORCH_LIB_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

        # ldd names the missing/unresolved shared object directly.
        LMCACHE_SO=$(python3 -c "import lmcache, glob, os; print((glob.glob(os.path.join(os.path.dirname(lmcache.__file__), 'c_ops*.so')) or [''])[0])")
        if [ -n "$LMCACHE_SO" ]; then
            echo "ldd $LMCACHE_SO"
            ldd "$LMCACHE_SO" 2>&1 | grep -E "not found|libcudart|libtorch|libc10" || true
        fi
        # Which torch library actually EXPORTS the symbol c_ops needs?
        # LD_PRELOAD of libc10/libtorch_cpu/libtorch did not resolve it, so
        # locate the definition rather than guessing which .so to preload.
        echo "=== searching torch libs for materialize_cow_storage ==="
        for _so in "$TORCH_LIB_DIR"/libc10.so "$TORCH_LIB_DIR"/libc10_cuda.so \
                   "$TORCH_LIB_DIR"/libtorch.so "$TORCH_LIB_DIR"/libtorch_cpu.so \
                   "$TORCH_LIB_DIR"/libtorch_cuda.so "$TORCH_LIB_DIR"/libtorch_python.so; do
            [ -f "$_so" ] || continue
            _n=$(nm -D --defined-only "$_so" 2>/dev/null | grep -c "materialize_cow_storage" || true)
            _u=$(nm -D --undefined-only "$_so" 2>/dev/null | grep -c "materialize_cow_storage" || true)
            echo "  $(basename "$_so"): defined=$_n undefined=$_u"
        done
        echo "  (c_ops needs it) $(nm -D --undefined-only "$LMCACHE_SO" 2>/dev/null | grep -c materialize_cow_storage || true)"

        python3 -c \
            "import cupy; import lmcache.integration.vllm.lmcache_mp_connector; import opentelemetry.exporter.prometheus" \
            >/dev/null

        # One MP server for the node, per the Kimi-K3 recipe
        # (docs.lmcache.ai/recipes/kimi_k3.html), but NOT that recipe's
        # --chunk-size 768: the connector requires the chunk to be a multiple
        # of every engine KV group's tokens_per_block, and this stack's blocks
        # are far larger than 768. On this exact image and script, vLLM pins
        # the attention group to 1536 (run 31404943911, interface.py:911,
        # "Setting attention block size to 1536 tokens to ensure that
        # attention page size is >= mamba page size") and pads the KDA/mamba
        # page to match (interface.py:935). That alignment is generic vLLM
        # code, not platform-specific, which is why the MI355X sister arm
        # lands on the same 1536 plus a 3072-token KDA state group and also
        # needs 3072. 768 is smaller than the attention block, so it is not a
        # multiple of it and fails at connector init. The multi-group layout
        # additionally requires one object group per sliding-window size:
        # --separate-object-groups.
        LMCACHE_PORT=6555
        LMCACHE_HTTP_PORT=8090
        LMCACHE_LOG="$RESULT_DIR/lmcache_server.log"

        # Consume the generated aggregate budget verbatim, per
        # benchmarks/single_node/agentic/README.md. --shm-name "" keeps L1 in
        # ordinary process memory so the budget is not silently capped by the
        # size of the node's /dev/shm mount.
        LMCACHE_L1_SIZE_GB="$TOTAL_CPU_DRAM_GB"

        # c_ops.so resolves c10/torch symbols only once libtorch is loaded
        # RTLD_GLOBAL. Captured directly:
        #
        #   ImportError: .../lmcache/c_ops.cpython-312-x86_64-linux-gnu.so:
        #   undefined symbol: _ZN3c104impl3cow23materialize_cow_storageERNS_11StorageImplE
        #     ( c10::impl::cow::materialize_cow_storage(c10::StorageImpl&) )
        #
        # It is NOT an ABI mismatch -- the same import succeeds moments later,
        # once torch is in the process, so the symbol does exist in this
        # image's torch. But CudaDeviceOps.ensure_native() latches
        # _native_bound = True on its FIRST attempt and swallows the error, so
        # one early miss pins the process to the torch fallback for good, and
        # that fallback is broken for this stack's multi-KV-group KDA/MLA
        # layout (77-98% of stores fail).
        #
        # Preloading torch in the parent is not enough: the server forks CPU
        # and GPU workers that import lmcache fresh. LD_PRELOAD applies to the
        # whole process tree, so the symbols are resolvable everywhere. Scoped
        # to this command via `env` rather than exported, to leave the vLLM
        # server's environment untouched.
        LMCACHE_LD_PRELOAD="${TORCH_LIB_DIR}/libc10.so:${TORCH_LIB_DIR}/libtorch_cpu.so:${TORCH_LIB_DIR}/libtorch.so"
        if [ -f "${TORCH_LIB_DIR}/libc10_cuda.so" ]; then
            LMCACHE_LD_PRELOAD="${LMCACHE_LD_PRELOAD}:${TORCH_LIB_DIR}/libc10_cuda.so:${TORCH_LIB_DIR}/libtorch_cuda.so"
        fi
        echo "LMCACHE_LD_PRELOAD=$LMCACHE_LD_PRELOAD"
        LMCACHE_CMD=(
            env "LD_PRELOAD=$LMCACHE_LD_PRELOAD"
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

        # Second guard, on the server process itself: the import check above
        # only proves c_ops loads in this shell's python3. Abort rather than
        # benchmark a silently-degraded offload tier (see the install note).
        if grep -q "c_ops compiled extension not found" "$LMCACHE_LOG"; then
            echo "Error: LMCache fell back to the torch baseline (c_ops did not load)." >&2
            echo "       The installed lmcache's compiled extension is unusable here." >&2
            echo "       Check the pip show / c_ops / libcudart lines logged above:" >&2
            echo "       either the install did not take, or the build's CUDA ABI" >&2
            echo "       does not match this image." >&2
            grep -m1 "c_ops compiled extension not found" "$LMCACHE_LOG" >&2
            exit 1
        fi

        # 100k-330k-token agentic prefixes make single retrieves large; use the
        # same MQ timeout headroom as the MI355X arm.
        OFFLOAD_ARGS=(
            --kv-transfer-config
            "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.port\":$LMCACHE_PORT,\"lmcache.mp.mq_timeout\":6000.0}}"
        )
        ;;
    *)
        echo "Error: unsupported KV_OFFLOAD_BACKEND='$KV_OFFLOAD_BACKEND' (expected empty, vllm-simple, or lmcache)" >&2
        exit 1
        ;;
esac

# ---- DSpark speculative decoding -------------------------------------------
# Probabilistic drafting at DSpark level 2, with synthetic acceptance pinned to
# the committed golden AL, per the AgentX policy in
# golden_al_distribution/README.md: "a submission may choose any supported draft
# length, but it may not substitute a different acceptance target", because
# "InferenceX is evaluating inference-system performance, not the ability to
# fine-tune a benchmark-specific speculative head". AL is workload-dependent by
# the README's own admission, so the prescribed acceptance -- not the acceptance
# this corpus happens to produce -- is what makes submissions comparable.
#
# Level 2 rather than 7. Draft length IS ours to choose, and the verify width is
# a real system cost that synthetic acceptance does not remove: at K=7 the target
# verifies 8 positions per step, which loses to no-speculation once the GPU
# saturates even at the forced golden AL of 3.84. Using cost ~ 1 + m(K-1) + f,
# K=2 clears break-even at both ends of the m range while K=3 is a wash at
# saturation and K=7 is negative:
#     K=2  golden AL 2.51  ->  ~1.7x low conc, ~1.2x saturated
#     K=3  golden AL 3.00  ->  ~1.7x low conc, ~0.95x saturated
#     K=7  golden AL 3.84  ->  ~1.3x low conc, ~0.54x saturated
# The drafter's KV footprint (~25% of GPU KV) is unaffected by K and remains a
# genuine cost of running spec decoding at all on 100k+ prompts.
NUM_SPEC_TOKENS=2
TOKENS_PER_SEQ=$((1 + NUM_SPEC_TOKENS))
# Committed golden AL at K=2 on the probabilistic curve we run
# (golden_al_distribution/kimik3_dspark_probabilistic_sample_method_block_rejection_sample_method.yaml:
# thinking_on 2 -> 2.51). The greedy/standard curve's K=2 value is 2.45 -- do not
# mix curves.
SYNTHETIC_ACCEPT_LEN=2.51

# Throughput runs pin synthetic acceptance; the EVAL_ONLY accuracy run must use
# real target verification instead. Synthetic acceptance commits drafted tokens
# regardless of the target's logits, so the generated text is wrong and the
# SWE-bench eval scores 0.0000 -- the same split dsv4_fp4_b300_vllm_mtp.sh makes
# (follow the DSV4 MTP precedent for this split).
# rejection_sample_method=block does real verification, so it is what EVAL_ONLY
# uses. vLLM rejects synthetic_acceptance_length unless the method is 'synthetic'.
if [ "${EVAL_ONLY:-false}" = "true" ]; then
    SPEC_CONFIG="{\"method\": \"dspark\", \"model\": \"$DRAFT_MODEL_PATH\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"attention_backend\": \"FLASHINFER_MLA\", \"draft_sample_method\": \"probabilistic\", \"rejection_sample_method\": \"block\"}"
else
    SPEC_CONFIG="{\"method\": \"dspark\", \"model\": \"$DRAFT_MODEL_PATH\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"attention_backend\": \"FLASHINFER_MLA\", \"draft_sample_method\": \"probabilistic\", \"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}"
fi

# Agentic fan-out: the scheduler headroom convention shared by the other agentic
# recipes. Upstream pins 32 instead, but the aligned flag set crashed the engine
# core on every concurrency (see the serve-flag note below), so this stays on the
# value that ran green. Capture decode graphs only to this bound — a 93-layer
# 2.8T model makes capturing vLLM's full 2048-wide ladder prohibitively slow.
MAX_NUM_SEQS=$((2 * CONC))

# With spec decoding, cudagraph capture sizes are in TOKENS. vLLM rounds
# configured sizes up to multiples of (1 + num_speculative_tokens) and dedups
# them (adjust_cudagraph_sizes_for_spec_decode), so a plain 1..MAX_NUM_SEQS list
# would collapse to covering only MAX_NUM_SEQS/(1+N) sequences and drop the
# largest decode batches to eager. Enumerate the multiples explicitly so there
# is one graph per decode batch of 1..MAX_NUM_SEQS sequences. The graph mode is
# left at the recipe default that the non-MTP K3 arm already validated.
CUDA_GRAPH_CAPTURE_SIZES=""
for ((num_seqs = 1; num_seqs <= MAX_NUM_SEQS; num_seqs++)); do
    if [ -n "$CUDA_GRAPH_CAPTURE_SIZES" ]; then
        CUDA_GRAPH_CAPTURE_SIZES+=","
    fi
    CUDA_GRAPH_CAPTURE_SIZES+="$((num_seqs * TOKENS_PER_SEQ))"
done
COMPILATION_CONFIG="{\"cudagraph_capture_sizes\":[${CUDA_GRAPH_CAPTURE_SIZES}]}"

echo "Starting vllm server..."

{ set +x; } 2>/dev/null
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --tensor-parallel-size "$TP"
    # Adopting the full upstream flag set (gpu-memory-utilization 0.95,
    # max-num-seqs 32, max-model-len 1000000, max-num-batched-tokens 32768,
    # --no-enable-flashinfer-autotune, VLLM_USE_V2_MODEL_RUNNER=1) killed the
    # engine core on every concurrency with an assertion in the FlashInfer MoE
    # runner's shared-experts output buffer
    # (fused_moe/runner/shared_experts.py:165, all 8 TP ranks at once). Only
    # mla_prefill_backend=TRTLLM_RAGGED is retained from that set; the rest stay
    # on the values that ran 12/12 green in run 30326393603.
    --gpu-memory-utilization 0.90
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-model-len 1048576
    --trust-remote-code
    --load-format fastsafetensors
    --moe-backend auto
    --enable-prefix-caching
    --kv-cache-dtype fp8
    --reasoning-parser kimi_k3
    --tool-call-parser kimi_k3
    --enable-auto-tool-choice
    # FP8 KV cache requires prefill query quantization. Use FlashInfer MLA
    # prefill, matching the current published upstream Kimi-K3 recipe.
    --attention-config '{"mla_prefill_backend":"flashinfer","use_prefill_query_quantization":true}'
    --speculative-config "$SPEC_CONFIG"
    --compilation-config "$COMPILATION_CONFIG"
    --disable-uvicorn-access-log
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
