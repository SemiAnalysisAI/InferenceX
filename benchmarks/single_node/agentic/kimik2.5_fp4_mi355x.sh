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

check_env_vars MODEL TP CONC OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR

PORT=${PORT:-8888}
DURATION=${DURATION:-1800}
MAX_DELAY=${MAX_DELAY:-60}
ADVANCE_MIN=${ADVANCE_MIN:-0.0}
ADVANCE_MAX=${ADVANCE_MAX:-0.7}
EP_SIZE=${EP_SIZE:-1}

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
        export LMCACHE_ROCM_MP_BLOCK_FALLBACK=1
        export LMCACHE_ROCM_MP_BLOCK_FALLBACK_DTYPE=bfloat16
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
