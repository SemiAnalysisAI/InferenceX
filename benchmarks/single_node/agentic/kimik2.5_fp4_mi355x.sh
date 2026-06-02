#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Kimi-K2.5 FP4 on MI355X using vLLM.
#
# Required env vars:
#   MODEL, TP, CONC, OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR
#
# OFFLOADING values:
#   none    - vLLM GPU KV only.
#   cpu     - vLLM native CPU offload.
#   lmcache - LMCache MP server + vLLM LMCacheMPConnector.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE

# Kimi-K2.5 advertises a 262144-token context window in vLLM 0.21.0.
# Matrix defaults may export MAX_MODEL_LEN=0 to mean "server default"; for this
# script we need the concrete value so AgentX filters prompt+max_tokens against
# the same limit vLLM enforces.
if [[ -z "${MAX_MODEL_LEN:-}" || "$MAX_MODEL_LEN" == "0" ]]; then
    MAX_MODEL_LEN=262144
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# ROCR/HIP visibility for vLLM 0.14+
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
rocm-smi || true
amd-smi || true

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# Install amd-quark for MXFP4 (manual install due to ROCm vLLM bug)
pip install amd-quark

# Disable AITER RMSNorm for TP < 8 due to accuracy issues
if [ "${TP}" -lt 8 ]; then
  export VLLM_ROCM_USE_AITER_RMSNORM=0
fi

write_lmcache_rocm_mp_patch() {
    local patch_dir="$1"
    mkdir -p "$patch_dir"
    cat > "$patch_dir/sitecustomize.py" <<'PY'
"""Runtime compatibility for LMCache MP on ROCm Kimi MLA KV caches."""

import os
import threading

if os.environ.get("LMCACHE_ROCM_DEMAND_PINNED_ALLOCATOR") == "1":
    import builtins
    import sys

    _orig_import = builtins.__import__

    def _patch_lazy_memory_allocator(_lazy_memory_allocator) -> None:
        _LazyMemoryAllocator = _lazy_memory_allocator.LazyMemoryAllocator

        if getattr(_LazyMemoryAllocator, "_agentic_rocm_demand_patch", False):
            return

        _orig_init = _LazyMemoryAllocator.__init__
        _orig_allocate = _LazyMemoryAllocator.allocate
        _orig_batched_allocate = _LazyMemoryAllocator.batched_allocate

        def _expand_to(self, target_size: int) -> None:
            target_size = min(
                self._final_size,
                _lazy_memory_allocator.align_to(target_size, self.PIN_CHUNK_SIZE),
            )
            lock = self._agentic_rocm_demand_expand_lock
            with lock:
                if target_size <= self._curr_size:
                    return

                start_size = self._curr_size
                while self._curr_size < target_size:
                    commit_start = self._curr_size
                    commit_target = min(target_size, self._curr_size + self.COMMIT_SIZE)
                    while self._curr_size < commit_target:
                        self._pin_memory_chunk(self._curr_size, self.PIN_CHUNK_SIZE)
                        self._curr_size += self.PIN_CHUNK_SIZE
                    self._commit_expansion(self._curr_size - commit_start)

                self._log_expansion_progress(self._curr_size - start_size)

        def _retry_with_demand_expansion(self, allocate_once):
            obj = allocate_once()
            step_gb = float(os.environ.get("LMCACHE_ROCM_DEMAND_PINNED_STEP_GB", "64"))
            step_bytes = max(self.COMMIT_SIZE, int(step_gb * (1024**3)))

            while obj is None and self._curr_size < self._final_size:
                _expand_to(self, self._curr_size + step_bytes)
                obj = allocate_once()

            return obj

        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            self._agentic_rocm_demand_expand_lock = threading.Lock()

            # LMCache MP's upstream LazyMemoryAllocator currently expands to
            # the final pinned size in a background thread. On ROCm Kimi TP4,
            # vLLM reaches KV-cache registration only after that 2.5 TB pool
            # is fully pinned, and the server-side IPC open path can stall
            # before acknowledging register_kv_caches. Keep the same final
            # capacity, but pin/commit extra host memory only when L1
            # allocations actually need it.
            self._stop_expand.set()
            self._expand_thread.join()
            _lazy_memory_allocator.logger.info(
                "Agentic ROCm patch: using demand-driven LMCache pinned "
                "memory expansion; final capacity remains %s MB",
                self._final_size >> 20,
            )

        def _patched_allocate(
            self,
            shapes,
            dtypes,
            fmt=_lazy_memory_allocator.MemoryFormat.UNDEFINED,
            allocator_type=None,
        ):
            return _retry_with_demand_expansion(
                self,
                lambda: _orig_allocate(self, shapes, dtypes, fmt, allocator_type),
            )

        def _patched_batched_allocate(
            self,
            shapes,
            dtypes,
            batch_size,
            fmt=_lazy_memory_allocator.MemoryFormat.UNDEFINED,
            allocator_type=None,
        ):
            return _retry_with_demand_expansion(
                self,
                lambda: _orig_batched_allocate(
                    self, shapes, dtypes, batch_size, fmt, allocator_type
                ),
            )

        _LazyMemoryAllocator.__init__ = _patched_init
        _LazyMemoryAllocator.allocate = _patched_allocate
        _LazyMemoryAllocator.batched_allocate = _patched_batched_allocate
        _LazyMemoryAllocator._agentic_rocm_demand_patch = True

    def _patch_l1_memory_manager(_memory_manager) -> None:
        _L1MemoryManager = getattr(_memory_manager, "L1MemoryManager", None)
        _LazyMemoryAllocator = getattr(_memory_manager, "LazyMemoryAllocator", None)
        if _L1MemoryManager is None or _LazyMemoryAllocator is None:
            return
        if getattr(_L1MemoryManager, "_agentic_rocm_final_capacity_patch", False):
            return

        _orig_get_memory_usage = _L1MemoryManager.get_memory_usage

        def _patched_get_memory_usage(self):
            allocator = getattr(self, "_allocator", None)
            if isinstance(allocator, _LazyMemoryAllocator):
                address_manager = allocator.get_address_manager()
                used_size = (
                    address_manager.get_heap_size() - address_manager.get_free_size()
                )
                return used_size, allocator._final_size
            return _orig_get_memory_usage(self)

        _L1MemoryManager.get_memory_usage = _patched_get_memory_usage
        _L1MemoryManager._agentic_rocm_final_capacity_patch = True

    def _maybe_patch_lazy_memory_allocator() -> None:
        module = sys.modules.get("lmcache.v1.lazy_memory_allocator")
        if module is not None and hasattr(module, "LazyMemoryAllocator"):
            _patch_lazy_memory_allocator(module)

    def _maybe_patch_l1_memory_manager() -> None:
        module = sys.modules.get("lmcache.v1.distributed.memory_manager")
        if module is not None and hasattr(module, "L1MemoryManager"):
            _patch_l1_memory_manager(module)

    def _agentic_rocm_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = _orig_import(name, globals, locals, fromlist, level)
        if name == "lmcache.v1.lazy_memory_allocator" or (
            name.startswith("lmcache") and "lmcache.v1.lazy_memory_allocator" in sys.modules
        ):
            _maybe_patch_lazy_memory_allocator()
        if name == "lmcache.v1.distributed.memory_manager" or (
            name.startswith("lmcache")
            and "lmcache.v1.distributed.memory_manager" in sys.modules
        ):
            _maybe_patch_l1_memory_manager()
        return module

    builtins.__import__ = _agentic_rocm_import
    _maybe_patch_lazy_memory_allocator()
    _maybe_patch_l1_memory_manager()

if os.environ.get("LMCACHE_ROCM_MP_BLOCK_FALLBACK") == "1":
    import torch
    import lmcache.non_cuda_equivalents as lmc

    if not hasattr(lmc, "multi_layer_block_kv_transfer"):
        _DTYPE_BY_NAME = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }

        def _dtype_from_env() -> torch.dtype:
            name = os.environ.get("LMCACHE_ROCM_MP_BLOCK_FALLBACK_DTYPE", "bfloat16")
            try:
                return _DTYPE_BY_NAME[name]
            except KeyError as exc:
                raise ValueError(f"Unsupported LMCache ROCm fallback dtype: {name}") from exc

        def _paged_view(ptr: int, shape_desc, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
            block_stride = shape_desc.block_stride_elems or (
                shape_desc.bs * shape_desc.nh * shape_desc.hs
            )
            base = lmc._tensor_from_ptr(
                ptr,
                (shape_desc.nb * block_stride,),
                dtype,
                device,
            )
            return torch.as_strided(
                base,
                (shape_desc.nb, shape_desc.bs, shape_desc.nh * shape_desc.hs),
                (block_stride, shape_desc.nh * shape_desc.hs, 1),
            )

        def _tmp_view(ptr: int, shape_desc, num_layers: int, chunk_slots: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
            return lmc._tensor_from_ptr(
                ptr,
                (shape_desc.kv_size, num_layers, chunk_slots, shape_desc.nh * shape_desc.hs),
                dtype,
                device,
            )

        def multi_layer_block_kv_transfer(
            group_kv_pointers,
            tmp_buffer_ptrs,
            block_ids,
            paged_memory_device,
            direction,
            shape_desc,
            lmcache_chunk_size,
            gpu_kv_format,
            skip_blocks=0,
        ) -> None:
            # Kimi K2.5 uses vLLM MLA: one KV tensor per layer with
            # shape [num_blocks, block_size, hidden_size]. LMCache's Python
            # fallback has no block-transfer entrypoint yet, so implement the
            # same gather/scatter contract with torch indexing on ROCm.
            if shape_desc.kv_size != 1:
                raise NotImplementedError(
                    "ROCm LMCache MP block fallback currently supports MLA KV caches only"
                )

            dtype = _dtype_from_env()
            device = (
                paged_memory_device
                if isinstance(paged_memory_device, torch.device)
                else torch.device(paged_memory_device)
            )
            num_layers = int(group_kv_pointers.numel())
            blocks_per_chunk = lmcache_chunk_size // shape_desc.bs
            direction_name = getattr(direction, "name", str(direction))

            for chunk_idx, tmp_ptr in enumerate(tmp_buffer_ptrs):
                start = chunk_idx * blocks_per_chunk
                end = start + blocks_per_chunk
                chunk_blocks = block_ids[start:end].to(device=device, dtype=torch.long)

                dest_slot_offset = 0
                if skip_blocks and chunk_idx == 0:
                    chunk_blocks = chunk_blocks[int(skip_blocks):]
                    dest_slot_offset = int(skip_blocks) * shape_desc.bs
                if chunk_blocks.numel() == 0:
                    continue

                num_slots = int(chunk_blocks.numel()) * shape_desc.bs
                tmp = _tmp_view(
                    int(tmp_ptr),
                    shape_desc,
                    num_layers,
                    lmcache_chunk_size,
                    dtype,
                    device,
                )

                for layer_idx in range(num_layers):
                    paged = _paged_view(
                        int(group_kv_pointers[layer_idx].item()),
                        shape_desc,
                        dtype,
                        device,
                    )
                    tmp_slice = tmp[
                        0,
                        layer_idx,
                        dest_slot_offset : dest_slot_offset + num_slots,
                        :,
                    ]
                    if direction_name == "D2H":
                        gathered = paged.index_select(0, chunk_blocks).reshape(
                            num_slots, shape_desc.nh * shape_desc.hs
                        )
                        tmp_slice.copy_(gathered)
                    elif direction_name == "H2D":
                        src = tmp_slice.reshape(
                            int(chunk_blocks.numel()),
                            shape_desc.bs,
                            shape_desc.nh * shape_desc.hs,
                        )
                        paged.index_copy_(0, chunk_blocks, src)
                    else:
                        raise ValueError(f"Unsupported transfer direction: {direction}")

        lmc.multi_layer_block_kv_transfer = multi_layer_block_kv_transfer

# ---- Chunked KV loading (prevents GPU block exhaustion at high concurrency) ----
if os.environ.get("CHUNKED_LMCACHE_MAX_TOKENS_PER_LOAD", "0") != "0":
    import chunked_connector_patch  # noqa: F401

# ---- vLLM scheduler assertion fix (stale KV transfer notifications) ----
import scheduler_assertion_patch  # noqa: F401
PY
}

write_chunked_connector_patch() {
    local patch_dir="$1"
    mkdir -p "$patch_dir"
    cat > "$patch_dir/chunked_connector_patch.py" <<'PY'
"""
Monkey-patch for LMCacheMPConnector to add chunked KV loading.

Fixes GPU block exhaustion deadlock at high concurrency by capping
the number of external tokens reported AND retrieved per scheduling step.

Usage: set CHUNKED_LMCACHE_MAX_TOKENS_PER_LOAD=<tokens> and import this
module from sitecustomize.py before LMCache is loaded.
"""

import logging
import os
import sys
import builtins

logger = logging.getLogger("chunked_lmcache_patch")

_MAX_TOKENS = int(os.environ.get("CHUNKED_LMCACHE_MAX_TOKENS_PER_LOAD", "32768"))

# Per-request chunk tracking (module-level, survives across calls)
_chunk_state: dict[str, dict] = {}


def _apply_patch():
    """Patch LMCacheMPConnector in-place."""
    mod = sys.modules.get("lmcache.integration.vllm.lmcache_mp_connector")
    if mod is None:
        return
    cls = getattr(mod, "LMCacheMPConnector", None)
    if cls is None or getattr(cls, "_chunked_patch_applied", False):
        return

    LMCacheMPRequestState = getattr(mod, "LMCacheMPRequestState", None)
    _orig_get_matched = cls.get_num_new_matched_tokens
    _orig_get_finished = cls.get_finished

    def _get_blocks_per_chunk(self):
        block_size = getattr(self, "block_size", 1)
        return max(1, _MAX_TOKENS // block_size)

    def _patched_get_num_new_matched_tokens(self, request, num_computed_tokens):
        full_match = _orig_get_matched(self, request, num_computed_tokens)
        if full_match <= 0 or _MAX_TOKENS <= 0:
            return full_match

        req_id = request.request_id
        block_size = getattr(self, "block_size", 1)
        blocks_per_chunk = _get_blocks_per_chunk(self)
        full_match_blocks = full_match // block_size

        state = _chunk_state.get(req_id)
        if state is None or state.get("num_computed_at_start") != num_computed_tokens:
            state = {
                "full_match_blocks": full_match_blocks,
                "chunk_end_blocks": 0,
                "num_computed_at_start": num_computed_tokens,
                "lookup_done": False,
            }
            _chunk_state[req_id] = state

        if state["lookup_done"]:
            return 0

        remaining = state["full_match_blocks"] - state["chunk_end_blocks"]
        if remaining <= 0:
            state["lookup_done"] = True
            return 0

        this_chunk = min(remaining, blocks_per_chunk)
        state["chunk_end_blocks"] += this_chunk
        if state["chunk_end_blocks"] >= state["full_match_blocks"]:
            state["lookup_done"] = True

        capped = this_chunk * block_size
        if capped < full_match:
            logger.debug(
                "Chunked LMCache: req %s capped %d -> %d tokens "
                "(chunk %d/%d blocks)",
                req_id, full_match, capped, this_chunk, full_match_blocks,
            )

        # Cap the tracker's hit blocks to match what we report
        tracker = getattr(request, "kv_transfer_params", None)
        if tracker is not None:
            orig_hits = getattr(tracker, "num_lmcache_hit_blocks", 0)
            if orig_hits > this_chunk:
                tracker.num_lmcache_hit_blocks = this_chunk

        return capped

    def _patched_get_finished(self, scheduler_output):
        result = _orig_get_finished(self, scheduler_output)
        # Clean up chunk state for finished requests.
        # vLLM passes scheduler_output as a set of request-ID strings
        # (not a SchedulerOutput object), so iterate directly when it
        # is a set/frozenset; fall back to the attribute path for
        # forward compatibility.
        if isinstance(scheduler_output, (set, frozenset)):
            finished = scheduler_output
        else:
            finished = getattr(scheduler_output, "finished_req_ids", [])
        for req in finished:
            _chunk_state.pop(req, None)
        return result

    cls.get_num_new_matched_tokens = _patched_get_num_new_matched_tokens
    cls.get_finished = _patched_get_finished
    cls._chunked_patch_applied = True
    logger.info(
        "Chunked LMCache connector patch applied "
        "(max_tokens_per_load=%d)", _MAX_TOKENS,
    )


_orig_import = builtins.__import__


def _patching_import(name, *args, **kwargs):
    module = _orig_import(name, *args, **kwargs)
    if (
        name == "lmcache.integration.vllm.lmcache_mp_connector"
        or (
            name.startswith("lmcache")
            and "lmcache.integration.vllm.lmcache_mp_connector" in sys.modules
        )
    ):
        _apply_patch()
    return module


builtins.__import__ = _patching_import
_apply_patch()
PY
}

write_scheduler_assertion_patch() {
    local patch_dir="$1"
    mkdir -p "$patch_dir"
    cat > "$patch_dir/scheduler_assertion_patch.py" <<'PY'
"""
Patch vLLM scheduler to handle stale finished_recving gracefully.

The assertion at scheduler.py crashes when a KV transfer reports
"finished recving" but the request is already in RUNNING state.
This happens when transfers complete asynchronously and the scheduler
has already moved the request forward.

Fix: Instead of asserting, log a warning and skip.
"""

import logging
import sys
import builtins

logger = logging.getLogger("scheduler_assertion_patch")


def _apply_patch():
    """Patch vLLM scheduler's _update_from_kv_xfer_finished."""
    sched_mod = sys.modules.get("vllm.v1.core.sched.scheduler")
    if sched_mod is None:
        return
    req_mod = sys.modules.get("vllm.v1.request")
    if req_mod is None:
        return
    Scheduler = getattr(sched_mod, "Scheduler", None)
    RequestStatus = getattr(req_mod, "RequestStatus", None)
    if Scheduler is None or RequestStatus is None:
        return
    if getattr(Scheduler, "_kv_xfer_patch_applied", False):
        return

    _orig_update = Scheduler._update_from_kv_xfer_finished

    def _patched_update(self, kv_connector_output):
        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)
        for req_id in kv_connector_output.finished_recving or ():
            if req_id not in self.requests:
                continue
            req = self.requests[req_id]
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
            elif RequestStatus.is_finished(req.status):
                self._free_blocks(self.requests[req_id])
            else:
                logger.warning(
                    "Stale finished_recving for req %s in status %s; skipping.",
                    req_id, req.status.name,
                )
        for req_id in kv_connector_output.finished_sending or ():
            if req_id not in self.requests:
                continue
            self._free_blocks(self.requests[req_id])

    Scheduler._update_from_kv_xfer_finished = _patched_update
    Scheduler._kv_xfer_patch_applied = True
    logger.info("Scheduler KV transfer assertion patch applied")


_orig_import = builtins.__import__


def _patching_import(name, *args, **kwargs):
    module = _orig_import(name, *args, **kwargs)
    if (
        name == "vllm.v1.core.sched.scheduler"
        or (
            name.startswith("vllm")
            and "vllm.v1.core.sched.scheduler" in sys.modules
        )
    ):
        _apply_patch()
    return module


builtins.__import__ = _patching_import
_apply_patch()
PY
}

# Workaround for MEC FW <177 RCCL memory reclaim issue
version=$(rocm-smi --showfw 2>/dev/null | grep MEC | head -n 1 | awk '{print $NF}')
if [[ "$version" == "" || ${version:-0} -lt 177 ]]; then
    export HSA_NO_SCRATCH_RECLAIM=1
fi

export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
LMCACHE_LOG="$RESULT_DIR/lmcache_server.log"
mkdir -p "$RESULT_DIR"

OFFLOAD_ARGS=()
PREFIX_CACHE_ARGS=()
LMCACHE_PID=""

cleanup_lmcache_server() {
    if [[ -n "$LMCACHE_PID" ]] && kill -0 "$LMCACHE_PID" 2>/dev/null; then
        kill "$LMCACHE_PID" 2>/dev/null || true
        wait "$LMCACHE_PID" 2>/dev/null || true
    fi
}

trap cleanup_lmcache_server EXIT

wait_for_lmcache_ready() {
    { set +x; } 2>/dev/null
    local attempts="${LMCACHE_READY_ATTEMPTS:-120}"
    local tail_pid=""

    while [ ! -f "$LMCACHE_LOG" ]; do
        if [[ -n "$LMCACHE_PID" ]] && ! kill -0 "$LMCACHE_PID" 2>/dev/null; then
            echo "LMCache server died before creating log file. Exiting." >&2
            exit 1
        fi
        sleep 1
    done

    tail -f -n +1 "$LMCACHE_LOG" &
    tail_pid=$!

    for ((i = 1; i <= attempts; i++)); do
        if curl --output /dev/null --silent --fail "http://127.0.0.1:${LMCACHE_HTTP_PORT}/healthcheck"; then
            kill "$tail_pid" 2>/dev/null || true
            wait "$tail_pid" 2>/dev/null || true
            return 0
        fi
        if [[ -n "$LMCACHE_PID" ]] && ! kill -0 "$LMCACHE_PID" 2>/dev/null; then
            echo "LMCache server died before becoming healthy. Log follows:" >&2
            kill "$tail_pid" 2>/dev/null || true
            wait "$tail_pid" 2>/dev/null || true
            cat "$LMCACHE_LOG" >&2 || true
            exit 1
        fi
        sleep 1
    done

    echo "Timed out waiting for LMCache server healthcheck. Log follows:" >&2
    kill "$tail_pid" 2>/dev/null || true
    wait "$tail_pid" 2>/dev/null || true
    cat "$LMCACHE_LOG" >&2 || true
    exit 1
}

case "$OFFLOADING" in
    none) ;;
    cpu)
        unset VLLM_USE_SIMPLE_KV_OFFLOAD
        # MI355X nodes have ~2.7 TiB of host DRAM available for offload;
        # reserve 2.5 TB for the offload pool (leaves ~200 GB headroom for
        # worker RSS / page cache / slurm cgroup).
        TOTAL_CPU_DRAM_GB=2500
        # Use vLLM's regular native KV-offload path (OffloadingConnector),
        # NOT the SimpleCPUOffloadConnector. The "native" backend resolves to
        # OffloadingConnector by default; setting VLLM_USE_SIMPLE_KV_OFFLOAD=1
        # would switch it to SimpleCPUOffloadConnector. We intentionally leave
        # that env var UNSET here so the regular OffloadingConnector path is
        # used. The shortcut --kv_offloading_backend native + --kv_offloading_size
        # form constructs the KVTransferConfig at engine startup
        # (vllm/config/vllm.py:662).
        OFFLOAD_ARGS=(
            --kv_offloading_backend native
            --kv_offloading_size "$TOTAL_CPU_DRAM_GB"
            --disable-hybrid-kv-cache-manager
        )
        ;;
    lmcache)
        { set +x; } 2>/dev/null
        unset VLLM_USE_SIMPLE_KV_OFFLOAD

        agentic_pip_install --quiet --no-cache-dir lmcache
        # LMCache's current dependency chain can install NVIDIA/CUDA NIXL and
        # CuPy packages on ROCm. vLLM 0.21.0 treats ROCm as "cuda-like", and
        # during Kimi fused-MoE model inspection it imports nixl_ep whenever
        # that module is importable, even when this run is not using EP/NIXL
        # kernels. The CUDA extension then fails immediately on AMD nodes with
        # "ImportError: libcuda.so.1".
        #
        # LMCache MP also uses CuPy stream APIs while registering vLLM's KV
        # caches. The CUDA CuPy wheel imports on ROCm, but it fails at runtime
        # with cudaErrorInsufficientDriver when LMCache touches the stream. Use
        # the ROCm 7 CuPy wheel so the same API dispatches through HIP.
        python3 -m pip uninstall -y \
            nixl nixl-cu12 nixl-cu13 nixl_ep \
            >/dev/null 2>&1 || true
        python3 -m pip uninstall -y \
            cupy cupy-cuda11x cupy-cuda12x cupy-cuda13x \
            >/dev/null 2>&1 || true
        agentic_pip_install --quiet --no-cache-dir cupy-rocm-7-0
        python3 - <<'PY'
import importlib.util
import sys

spec = importlib.util.find_spec("nixl_ep")
if spec is not None:
    locations = ", ".join(spec.submodule_search_locations or [spec.origin or "unknown"])
    print(
        "Error: nixl_ep is still importable after LMCache install; "
        "this ROCm Kimi run would import a CUDA-only nixl_ep module. "
        f"location={locations}",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from cupy_backends.cuda.api import runtime as cupy_runtime
except Exception as exc:
    print(f"Error: failed to import CuPy runtime after ROCm CuPy install: {exc}", file=sys.stderr)
    sys.exit(1)

if not getattr(cupy_runtime, "is_hip", False):
    print(
        "Error: CuPy is still using the CUDA backend after installing "
        "cupy-rocm-7-0; LMCache MP would fail during KV-cache registration.",
        file=sys.stderr,
    )
    sys.exit(1)
PY
        LMCACHE_ROCM_PATCH_DIR="$RESULT_DIR/lmcache_rocm_patch"
        write_lmcache_rocm_mp_patch "$LMCACHE_ROCM_PATCH_DIR"
        write_chunked_connector_patch "$LMCACHE_ROCM_PATCH_DIR"
        write_scheduler_assertion_patch "$LMCACHE_ROCM_PATCH_DIR"
        export LMCACHE_ROCM_MP_BLOCK_FALLBACK=1
        export LMCACHE_ROCM_MP_BLOCK_FALLBACK_DTYPE=bfloat16
        export LMCACHE_ROCM_DEMAND_PINNED_ALLOCATOR=1
        # Cap external KV tokens loaded per scheduling step to prevent GPU
        # block exhaustion deadlock at high concurrency (c>=32).  Default
        # 32768 keeps peak block demand within the GPU KV pool.  Set to 0 to
        # disable chunking (only safe at low concurrency).
        export CHUNKED_LMCACHE_MAX_TOKENS_PER_LOAD="${CHUNKED_LMCACHE_MAX_TOKENS_PER_LOAD:-32768}"
        export PYTHONPATH="$LMCACHE_ROCM_PATCH_DIR${PYTHONPATH:+:$PYTHONPATH}"
        python3 -c "import lmcache.integration.vllm.lmcache_mp_connector" >/dev/null

        # Match the B200 Kimi LMCache setup: keep a 2.5 TB semantic CPU KV
        # pool, but let the external MP server own that pool so vLLM does not
        # split --kv-offloading-size across TP ranks through the integrated
        # LMCache backend.
        TOTAL_CPU_DRAM_GB=2500
        LMCACHE_HOST="${LMCACHE_HOST:-127.0.0.1}"
        LMCACHE_PORT="${LMCACHE_PORT:-5555}"
        LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
        # LMCacheMPConnector concatenates lmcache.mp.host and port into the
        # ZMQ endpoint. Bind the server to a raw host, but pass the connector a
        # ZMQ-style host string.
        LMCACHE_CONNECT_HOST="${LMCACHE_CONNECT_HOST:-tcp://$LMCACHE_HOST}"
        LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-$TOTAL_CPU_DRAM_GB}"
        LMCACHE_L1_INIT_SIZE_GB="${LMCACHE_L1_INIT_SIZE_GB:-20}"
        # LMCache read locks are leases on chunks that lookup has promised
        # vLLM can retrieve. The default 300s TTL is too short for this
        # long-context agentic queue: TP8/conc32 can spend >300s between
        # lookup and retrieve while GPU KV is saturated, which leaves the
        # object present in L1 but no longer readable. Keep the 2.5 TB pool
        # size unchanged and only extend the lookup-to-retrieve lease.
        LMCACHE_L1_READ_TTL_SECONDS="${LMCACHE_L1_READ_TTL_SECONDS:-3600}"
        LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
        LMCACHE_MAX_WORKERS="${LMCACHE_MAX_WORKERS:-$TP}"
        export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

        echo "Starting LMCache MP server..."
        LMCACHE_CMD=(
            lmcache server
            --host "$LMCACHE_HOST"
            --port "$LMCACHE_PORT"
            --http-host "$LMCACHE_HOST"
            --http-port "$LMCACHE_HTTP_PORT"
            --l1-size-gb "$LMCACHE_L1_SIZE_GB"
            --l1-init-size-gb "$LMCACHE_L1_INIT_SIZE_GB"
            --l1-read-ttl-seconds "$LMCACHE_L1_READ_TTL_SECONDS"
            --chunk-size "$LMCACHE_CHUNK_SIZE"
            --max-workers "$LMCACHE_MAX_WORKERS"
            --eviction-policy LRU
        )
        printf '%q ' "${LMCACHE_CMD[@]}" > "$RESULT_DIR/lmcache_command.txt"
        printf '\n' >> "$RESULT_DIR/lmcache_command.txt"
        "${LMCACHE_CMD[@]}" > "$LMCACHE_LOG" 2>&1 &
        LMCACHE_PID=$!
        echo "LMCache server PID: $LMCACHE_PID"
        wait_for_lmcache_ready

        PREFIX_CACHE_ARGS=(--enable-prefix-caching)
        OFFLOAD_ARGS=(
            --kv-transfer-config
            "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"$LMCACHE_CONNECT_HOST\",\"lmcache.mp.port\":$LMCACHE_PORT}}"
            --disable-hybrid-kv-cache-manager
        )
        ;;
    *) echo "Error: unsupported OFFLOADING value '$OFFLOADING'" >&2; exit 1 ;;
esac

EP_ARGS=()
if [ "$EP_SIZE" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
fi

echo "Starting vllm server..."
export PYTHONNOUSERSITE=1

{ set +x; } 2>/dev/null
VLLM_CMD=(
    vllm serve "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --tensor-parallel-size="$TP"
    "${EP_ARGS[@]}"
    --gpu-memory-utilization 0.90
    --block-size=1
    --trust-remote-code
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$CONC"
    --mm-encoder-tp-mode data
    "${PREFIX_CACHE_ARGS[@]}"
    "${OFFLOAD_ARGS[@]}"
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Run benchmark ----------------------------------------------------------
build_replay_cmd "$RESULT_DIR"

run_agentic_replay_and_write_outputs "$RESULT_DIR"
