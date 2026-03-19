#!/bin/bash
# =============================================================================
# setup_deps.sh — Install missing vLLM disagg dependencies at container start.
#
# Base image: vllm/vllm-openai-rocm:v0.17.1
# Sourced by server.sh so PATH / LD_LIBRARY_PATH exports persist.
# Idempotent: each component is skipped if already present.
#
# Build steps run in subshells to avoid CWD pollution between installers.
# =============================================================================

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
UCX_HOME="${UCX_HOME:-/usr/local/ucx}"
RIXL_HOME="${RIXL_HOME:-/usr/local/rixl}"

_SETUP_START=$(date +%s)
_SETUP_INSTALLED=()

git_clone_retry() {
    local url="$1" dest="$2" max_tries=3 try=1
    while (( try <= max_tries )); do
        if git clone --quiet "$url" "$dest" 2>/dev/null; then return 0; fi
        echo "[SETUP] git clone attempt $try/$max_tries failed for $url, retrying in 10s..."
        rm -rf "$dest"
        sleep 10
        (( try++ ))
    done
    echo "[SETUP] git clone failed after $max_tries attempts: $url"
    return 1
}

# ---------------------------------------------------------------------------
# 1. UCX (ROCm fork — required for GPU-direct RDMA via Nixl)
# ---------------------------------------------------------------------------
install_ucx() {
    if [[ -x "${UCX_HOME}/bin/ucx_info" ]]; then
        echo "[SETUP] UCX already present at ${UCX_HOME}"
        return 0
    fi

    echo "[SETUP] Installing UCX build dependencies..."
    apt-get update -q -y && apt-get install -q -y \
        autoconf automake libtool pkg-config \
        librdmacm-dev rdmacm-utils libibverbs-dev ibverbs-utils ibverbs-providers \
        infiniband-diags perftest ethtool rdma-core strace \
        && rm -rf /var/lib/apt/lists/*

    echo "[SETUP] Building UCX from source (ROCm/ucx @ da3fac2a)..."
    (
        set -e
        mkdir -p /usr/local/src && cd /usr/local/src
        git_clone_retry https://github.com/ROCm/ucx.git ucx && cd ucx
        git checkout da3fac2a
        ./autogen.sh && mkdir -p build && cd build
        ../configure \
            --prefix="${UCX_HOME}" \
            --enable-shared --disable-static \
            --disable-doxygen-doc --enable-optimizations \
            --enable-devel-headers --enable-mt \
            --with-rocm="${ROCM_PATH}" --with-verbs --with-dm
        make -j"$(nproc)" && make install
    )
    rm -rf /usr/local/src/ucx

    if [[ ! -x "${UCX_HOME}/bin/ucx_info" ]]; then
        echo "[SETUP] ERROR: UCX build failed"; exit 1
    fi
    _SETUP_INSTALLED+=("UCX")
}

# ---------------------------------------------------------------------------
# 2. RIXL (ROCm fork of NIXL — KV cache transfer for disaggregated vLLM)
# ---------------------------------------------------------------------------
install_rixl() {
    if python3 -c "import rixl" 2>/dev/null; then
        echo "[SETUP] RIXL Python bindings already present"
        return 0
    fi

    echo "[SETUP] Installing RIXL build dependencies..."
    apt-get update -q -y && apt-get install -q -y \
        libgrpc-dev libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc \
        libcpprest-dev libaio-dev \
        && rm -rf /var/lib/apt/lists/*
    pip3 install --quiet meson "pybind11[global]"

    echo "[SETUP] Building RIXL from source (ROCm/RIXL @ f33a5599)..."
    (
        set -e
        git_clone_retry https://github.com/ROCm/RIXL.git /opt/rixl && cd /opt/rixl
        git checkout f33a5599
        meson setup build --prefix="${RIXL_HOME}" \
            -Ducx_path="${UCX_HOME}" \
            -Drocm_path="${ROCM_PATH}"
        cd build && ninja && ninja install
        cd /opt/rixl
        pip install --quiet \
            --config-settings=setup-args="-Drocm_path=${ROCM_PATH}" \
            --config-settings=setup-args="-Ducx_path=${UCX_HOME}" .
    )
    rm -rf /opt/rixl

    if ! python3 -c "import rixl" 2>/dev/null; then
        echo "[SETUP] ERROR: RIXL build failed"; exit 1
    fi
    _SETUP_INSTALLED+=("RIXL")
}

# ---------------------------------------------------------------------------
# 3. etcd (distributed KV store for vLLM disagg service discovery)
# ---------------------------------------------------------------------------
install_etcd() {
    if [[ -x /usr/local/bin/etcd/etcd ]]; then
        echo "[SETUP] etcd already present"
        return 0
    fi

    local version="v3.6.0-rc.5"
    echo "[SETUP] Downloading etcd ${version}..."
    wget -q "https://github.com/etcd-io/etcd/releases/download/${version}/etcd-${version}-linux-amd64.tar.gz" \
        -O /tmp/etcd.tar.gz
    mkdir -p /usr/local/bin/etcd
    tar -xf /tmp/etcd.tar.gz -C /usr/local/bin/etcd --strip-components=1
    rm /tmp/etcd.tar.gz
    _SETUP_INSTALLED+=("etcd")
}

# ---------------------------------------------------------------------------
# 4. libionic1 (Pensando ionic RDMA verbs provider for RoCEv2 KV transfer)
#    Harmless on non-Pensando nodes (shared lib is simply unused).
# ---------------------------------------------------------------------------
install_libionic() {
    if dpkg -l libionic1 2>/dev/null | grep -q '^ii'; then
        echo "[SETUP] libionic1 already installed"
        return 0
    fi

    echo "[SETUP] Downloading and installing libionic1..."
    wget -q "https://repo.radeon.com/amdainic/pensando/ubuntu/1.117.5/pool/main/r/rdma-core/libionic1_54.0-149.g3304be71_amd64.deb" \
        -O /tmp/libionic1.deb
    dpkg -i /tmp/libionic1.deb || true
    rm -f /tmp/libionic1.deb
    _SETUP_INSTALLED+=("libionic1")
}

# ---------------------------------------------------------------------------
# 5. MoRI-IO proxy deps (Python packages for the MoRI-IO-aware proxy server)
#    The proxy replaces vllm-router: it handles both HTTP routing AND the
#    MoRI-IO ZMQ registration/request-enrichment protocol.
#    Only needed on NODE_RANK=0 (proxy node).
# ---------------------------------------------------------------------------
install_mori_proxy_deps() {
    if python3 -c "import quart, aiohttp, msgpack, zmq" 2>/dev/null; then
        echo "[SETUP] MoRI-IO proxy Python deps already present"
        return 0
    fi

    echo "[SETUP] Installing MoRI-IO proxy Python deps..."
    pip install --quiet --ignore-installed blinker
    pip install --quiet quart aiohttp msgpack pyzmq

    if ! python3 -c "import quart, aiohttp, msgpack, zmq" 2>/dev/null; then
        echo "[SETUP] ERROR: MoRI-IO proxy deps install failed"; exit 1
    fi
    _SETUP_INSTALLED+=("mori-proxy-deps")
}

# ---------------------------------------------------------------------------
# 6. MoRI (Modular RDMA Interface — EP dispatch/combine kernels for MoE)
#    Required for --all2all-backend mori (Expert Parallelism via RDMA).
#    GPU kernels are JIT-compiled on first use; no hipcc needed at install.
# ---------------------------------------------------------------------------
install_mori() {
    if python3 -c "import mori" 2>/dev/null; then
        echo "[SETUP] MoRI Python bindings already present"
        return 0
    fi

    echo "[SETUP] Installing MoRI build dependencies..."
    apt-get update -q -y && apt-get install -q -y \
        libopenmpi-dev openmpi-bin libpci-dev \
        && rm -rf /var/lib/apt/lists/*

    echo "[SETUP] Building MoRI from source (ROCm/mori @ b645fc8)..."
    (
        set -e
        git_clone_retry https://github.com/ROCm/mori.git /opt/mori && cd /opt/mori
        git checkout b645fc8
        pip install --quiet .
    )
    rm -rf /opt/mori

    if ! python3 -c "import mori" 2>/dev/null; then
        echo "[SETUP] ERROR: MoRI build failed"; exit 1
    fi
    _SETUP_INSTALLED+=("MoRI")
}

# ---------------------------------------------------------------------------
# 7. Patch vLLM v0.17.1 MoRI-EP + FP8 incompatibility
#    v0.17.1 asserts MoRI requires AITER fused_moe, but AITER's FP8 kernel
#    uses defer_input_quant=True which MoRI's prepare/finalize rejects.
#    Patch: remove both the AITER requirement assertion and the
#    defer_input_quant NotImplementedError so non-AITER kernels work.
# ---------------------------------------------------------------------------
patch_mori_fp8_compat() {
    python3 -c '
import re, os, sys
patched = []

# 1. Patch layer.py: remove multi-line AITER assertion for MoRI
try:
    import vllm.model_executor.layers.fused_moe.layer as lm
    f = lm.__file__
    src = open(f).read()
    if "Mori needs to be used with aiter" in src:
        new = re.sub(
            r"assert self\.rocm_aiter_fmoe_enabled,\s*\([^)]*Mori needs[^)]*\)",
            "pass  # [PATCHED] AITER requirement removed for MoRI-EP + FP8",
            src, flags=re.DOTALL)
        if new != src:
            open(f, "w").write(new)
            patched.append("layer.py")
except Exception as e:
    print(f"[SETUP] WARN patch layer.py: {e}", file=sys.stderr)

# 2. Patch mori_prepare_finalize.py: remove defer_input_quant restriction
try:
    import vllm.model_executor.layers.fused_moe.mori_prepare_finalize as mm
    f = mm.__file__
    src = open(f).read()
    if "defer_input_quant" in src:
        new = re.sub(
            r"raise NotImplementedError\([^)]*defer_input_quant[^)]*\)",
            "pass  # [PATCHED] defer_input_quant check removed for MoRI-EP + FP8",
            src)
        if new != src:
            open(f, "w").write(new)
            patched.append("mori_prepare_finalize.py")
except Exception as e:
    print(f"[SETUP] WARN patch mori_pf: {e}", file=sys.stderr)

if patched:
    print(f"[SETUP] Patched: {chr(44).join(patched)}")
else:
    print("[SETUP] No MoRI-FP8 patches needed")
'
    _SETUP_INSTALLED+=("MoRI-FP8-patch")
}

# ---------------------------------------------------------------------------
# 8. Patch vLLM MoRI-IO save_kv_layer busy-spin (C128 tail-batch deadlock)
#    In WRITE mode, save_kv_layer spins forever waiting for the handshake
#    callback to set write_ready_flags. This blocks the model worker thread,
#    preventing it from responding to EngineCore shm_broadcast, causing a
#    TimeoutError cascade and crash.
#    Patch: add time.sleep(0.001) and a 30s timeout to yield CPU and prevent
#    the model worker from deadlocking.
# ---------------------------------------------------------------------------
patch_moriio_save_kv_timeout() {
    python3 -c '
import os, sys

try:
    import vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector as mc
    f = mc.__file__
    src = open(f).read()

    # Already patched?
    if "[PATCHED] save_kv_layer timeout" in src:
        print("[SETUP] save_kv_layer timeout patch already applied")
        sys.exit(0)

    old = """        while True:
            if (
                self._ready_requests.empty()
                and remote_engine_id not in self.write_ready_flags
            ):
                continue"""

    if old not in src:
        print("[SETUP] WARN: save_kv_layer busy-spin pattern not found, skipping patch")
        sys.exit(0)

    new = """        # [PATCHED] save_kv_layer — null guard + timeout + sleep
        if remote_engine_id is None:
            return
        import time as _time, os as _os
        _wait_start = _time.monotonic()
        _SAVE_KV_TIMEOUT = float(_os.environ.get("VLLM_MORIIO_HANDSHAKE_TIMEOUT", "30"))
        while True:
            if (
                self._ready_requests.empty()
                and remote_engine_id not in self.write_ready_flags
            ):
                _elapsed = _time.monotonic() - _wait_start
                if _elapsed > _SAVE_KV_TIMEOUT:
                    import logging as _logging
                    _logging.getLogger("vllm.moriio").warning(
                        "[HANGFIX] save_kv_layer: timeout (%.1fs) waiting for "
                        "write_ready_flags[%s], breaking to unblock model "
                        "worker", _elapsed, remote_engine_id)
                    break
                _time.sleep(0.001)
                continue"""

    new_src = src.replace(old, new)
    if new_src == src:
        print("[SETUP] WARN: replacement had no effect")
        sys.exit(0)

    open(f, "w").write(new_src)
    print("[SETUP] Patched save_kv_layer: null guard + timeout + sleep")
except Exception as e:
    print(f"[SETUP] WARN patch save_kv_layer: {e}", file=sys.stderr)
'
    _SETUP_INSTALLED+=("MoRIIO-save-kv-timeout-patch")
}

# ---------------------------------------------------------------------------
# 9. Patch MoRIIO waiting_for_transfer_complete with bounded timeout
#    The original status.Wait() blocks forever if an RDMA completion never
#    arrives (e.g., NIC queue saturation at C256). This replaces the unbounded
#    wait with a polling loop using status.Succeeded() + configurable timeout.
#    Also adds error handling to the write worker loop so a single failed
#    transfer doesn't kill the background thread.
# ---------------------------------------------------------------------------
patch_moriio_transfer_timeout() {
    python3 -c '
import os, sys, textwrap

try:
    import vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine as me
    f = me.__file__
    src = open(f).read()

    if "[PATCHED] transfer completion timeout" in src:
        print("[SETUP] transfer completion timeout patch already applied")
        sys.exit(0)

    # --- Patch 1: Replace waiting_for_transfer_complete with polling + timeout ---
    old_wait = """    def waiting_for_transfer_complete(self):
        if not self.transfer_status:
            return

        transfers_to_wait = []
        with self.lock:
            transfers_to_wait = self.transfer_status[:]
            self.transfer_status.clear()

        for status in transfers_to_wait:
            try:
                status.Wait()
                if not status.Succeeded():
                    logger.error(
                        "Transfer failed: %s, Code: %s", status.Message(), status.Code()
                    )
                    raise TransferError("MoRIIO transfer failed!")
            except Exception as e:
                logger.error("Transfer %s failed: %s", status, e)
                raise"""

    new_wait = """    def waiting_for_transfer_complete(self):
        # [PATCHED] transfer completion timeout — bounded polling loop
        import time as _time, os as _os
        if not self.transfer_status:
            return

        _timeout = float(_os.environ.get("VLLM_MORIIO_TRANSFER_TIMEOUT", "120"))

        transfers_to_wait = []
        with self.lock:
            transfers_to_wait = self.transfer_status[:]
            self.transfer_status.clear()

        _start = _time.monotonic()
        remaining = list(transfers_to_wait)
        _polls = 0
        _completed = 0

        while remaining:
            _elapsed = _time.monotonic() - _start
            if _elapsed > _timeout:
                logger.error(
                    "[HANGFIX] transfer_timeout elapsed=%.1fs "
                    "pending=%d/%d completed=%d polls=%d "
                    "action=raise_transfer_error",
                    _elapsed, len(remaining), len(transfers_to_wait),
                    _completed, _polls,
                )
                raise TransferError(
                    f"RDMA transfer timeout after {_elapsed:.1f}s, "
                    f"{len(remaining)}/{len(transfers_to_wait)} pending"
                )

            still_waiting = []
            for status in remaining:
                try:
                    if status.Succeeded():
                        _completed += 1
                        continue
                    still_waiting.append(status)
                except Exception as e:
                    logger.error(
                        "[HANGFIX] transfer_poll_error error=%s", e)
                    raise TransferError(
                        f"Transfer failed during poll: {e}"
                    ) from e

            remaining = still_waiting
            if remaining:
                _time.sleep(0.005)
                _polls += 1
                if _polls % 2000 == 0:
                    logger.warning(
                        "[HANGFIX] transfer_wait pending=%d "
                        "completed=%d elapsed=%.1fs timeout=%.0fs",
                        len(remaining), _completed,
                        _time.monotonic() - _start, _timeout,
                    )"""

    if old_wait not in src:
        print("[SETUP] WARN: waiting_for_transfer_complete pattern not found")
        sys.exit(0)

    new_src = src.replace(old_wait, new_wait)

    # --- Patch 2: Add error handling + cleanup to _write_worker_loop ---
    old_loop = """            self._execute_write_task(task)"""

    new_loop = """            try:
                self._execute_write_task(task)
            except Exception as _e:
                logger.error(
                    "[HANGFIX] req=%s write_task_failed error=%s "
                    "action=cleanup_and_mark_done",
                    task.request_id, _e,
                )
                try:
                    _wr = self.worker.moriio_wrapper
                    with _wr.lock:
                        _wr.done_req_ids.append(task.request_id)
                    _wr.done_remote_allocate_req_dict.pop(
                        task.request_id, None
                    )
                except Exception:
                    pass"""

    if old_loop in new_src:
        new_src = new_src.replace(old_loop, new_loop, 1)
    else:
        print("[SETUP] WARN: _write_worker_loop pattern not found for error handling")

    # --- Patch 3: Add deferred task timeout to _process_deferred_tasks ---
    old_deferred = """    def _process_deferred_tasks(self) -> None:
        \"\"\"Process tasks that were previously deferred.\"\"\"
        if not self._deferred_tasks:
            return

        still_deferred: list[WriteTask] = []
        for task in self._deferred_tasks:
            if self._is_remote_ready(task):
                self._execute_write_task(task)
            else:
                still_deferred.append(task)

        self._deferred_tasks = still_deferred"""

    new_deferred = """    def _process_deferred_tasks(self) -> None:
        \"\"\"Process tasks that were previously deferred.\"\"\"
        # [PATCHED] deferred task timeout — prune stale tasks
        import time as _time, os as _os
        if not self._deferred_tasks:
            return

        _DEFER_TIMEOUT = float(
            _os.environ.get("VLLM_MORIIO_DEFER_TIMEOUT", "60"))

        still_deferred: list[WriteTask] = []
        for task in self._deferred_tasks:
            _age = _time.monotonic() - getattr(task, "_defer_ts", _time.monotonic())
            if _age > _DEFER_TIMEOUT:
                logger.error(
                    "[HANGFIX] req=%s deferred_task_expired age=%.1fs "
                    "action=drop_and_mark_done",
                    task.request_id, _age,
                )
                try:
                    _wr = self.worker.moriio_wrapper
                    with _wr.lock:
                        _wr.done_req_ids.append(task.request_id)
                    _wr.done_remote_allocate_req_dict.pop(
                        task.request_id, None)
                except Exception:
                    pass
                continue
            if self._is_remote_ready(task):
                try:
                    self._execute_write_task(task)
                except Exception as _e:
                    logger.error(
                        "[HANGFIX] req=%s deferred_write_failed error=%s",
                        task.request_id, _e,
                    )
                    try:
                        _wr = self.worker.moriio_wrapper
                        with _wr.lock:
                            _wr.done_req_ids.append(task.request_id)
                        _wr.done_remote_allocate_req_dict.pop(
                            task.request_id, None)
                    except Exception:
                        pass
            else:
                still_deferred.append(task)

        self._deferred_tasks = still_deferred"""

    if old_deferred in new_src:
        new_src = new_src.replace(old_deferred, new_deferred, 1)
    else:
        print("[SETUP] WARN: _process_deferred_tasks pattern not found")

    # --- Patch 4: Stamp defer time when task is deferred ---
    old_defer_add = """                self._deferred_tasks.append(task)"""
    new_defer_add = """                import time as _time2
                if not hasattr(task, "_defer_ts"):
                    task._defer_ts = _time2.monotonic()
                self._deferred_tasks.append(task)"""
    if old_defer_add in new_src:
        new_src = new_src.replace(old_defer_add, new_defer_add, 1)
    else:
        print("[SETUP] WARN: deferred task timestamp patch target not found")

    open(f, "w").write(new_src)
    print("[SETUP] Patched: transfer timeout + writer error handling")

except Exception as e:
    print(f"[SETUP] WARN patch transfer_timeout: {e}", file=sys.stderr)
'
    _SETUP_INSTALLED+=("MoRIIO-transfer-timeout-patch")
}

# ---------------------------------------------------------------------------
# 10. Patch MoRIIO start_load_kv busy-spin (same pattern as save_kv_layer)
#     The READ-mode spin loop in start_load_kv has the same unbounded-spin
#     issue as save_kv_layer. Add timeout + sleep + null guard.
# ---------------------------------------------------------------------------
patch_moriio_load_kv_timeout() {
    python3 -c '
import os, sys

try:
    import vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector as mc
    f = mc.__file__
    src = open(f).read()

    if "[PATCHED] start_load_kv timeout" in src:
        print("[SETUP] start_load_kv timeout patch already applied")
        sys.exit(0)

    old = """        while True:
            if (
                self._ready_requests.empty()
                and remote_engine_id not in self.load_ready_flag
                and wait_handshake_readd_req
            ):
                continue"""

    if old not in src:
        print("[SETUP] WARN: start_load_kv busy-spin pattern not found, skipping")
        sys.exit(0)

    new = """        # [PATCHED] start_load_kv timeout — prevent model worker deadlock
        if remote_engine_id is None and not wait_handshake_readd_req:
            self._reqs_to_send.update(metadata.reqs_to_send)
            return
        import time as _time, os as _os
        _wait_start = _time.monotonic()
        _LOAD_KV_TIMEOUT = float(_os.environ.get("VLLM_MORIIO_HANDSHAKE_TIMEOUT", "30"))
        while True:
            if (
                self._ready_requests.empty()
                and remote_engine_id not in self.load_ready_flag
                and wait_handshake_readd_req
            ):
                if _time.monotonic() - _wait_start > _LOAD_KV_TIMEOUT:
                    import logging as _logging
                    _logging.getLogger("vllm.moriio").warning(
                        "[HANGFIX] start_load_kv: timeout (%.1fs) waiting for "
                        "load_ready_flag[%s]", _time.monotonic() - _wait_start,
                        remote_engine_id)
                    break
                _time.sleep(0.001)
                continue"""

    new_src = src.replace(old, new)
    if new_src == src:
        print("[SETUP] WARN: start_load_kv replacement had no effect")
        sys.exit(0)

    open(f, "w").write(new_src)
    print("[SETUP] Patched start_load_kv busy-spin with timeout + sleep")
except Exception as e:
    print(f"[SETUP] WARN patch start_load_kv: {e}", file=sys.stderr)
'
    _SETUP_INSTALLED+=("MoRIIO-load-kv-timeout-patch")
}

# ---------------------------------------------------------------------------
# 11. Fix READ-mode scheduler assertion in _update_from_kv_xfer_finished
#     vLLM v0.17.1 asserts that a request in finished_recving must be either
#     WAITING_FOR_REMOTE_KVS or finished.  In READ mode the request can
#     transition to RUNNING before the aggregated recv notification arrives,
#     crashing the engine with AssertionError.
# ---------------------------------------------------------------------------
patch_scheduler_read_mode_fix() {
    python3 -c '
import os, sys

try:
    import vllm.v1.core.sched.scheduler as smod
    f = smod.__file__
    src = open(f).read()

    if "[PATCHED] read-mode recv assertion" in src:
        print("[SETUP] scheduler read-mode assertion fix already applied")
        sys.exit(0)

    old_recv = """        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            assert req_id in self.requests
            req = self.requests[req_id]
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
            else:
                assert RequestStatus.is_finished(req.status)
                self._free_blocks(self.requests[req_id])"""

    new_recv = """        # [PATCHED] read-mode recv assertion — handle intermediate states
        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            if req_id not in self.requests:
                logger.debug("Request %s already removed, skipping recv", req_id)
                continue
            req = self.requests[req_id]
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
            elif RequestStatus.is_finished(req.status):
                self._free_blocks(self.requests[req_id])
            else:
                logger.debug(
                    "Request %s recv finished but status=%s (not "
                    "WAITING_FOR_REMOTE_KVS or finished), skipping "
                    "block free — will be freed on request completion",
                    req_id, req.status.name)"""

    if old_recv not in src:
        print("[SETUP] WARN: scheduler finished_recving pattern not found, skipping")
        sys.exit(0)

    new_src = src.replace(old_recv, new_recv, 1)

    old_send = """        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests
            self._free_blocks(self.requests[req_id])"""

    new_send = """        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            if req_id not in self.requests:
                logger.debug("Request %s already removed, skipping send", req_id)
                continue
            req = self.requests[req_id]
            if RequestStatus.is_finished(req.status):
                self._free_blocks(req)
            else:
                logger.debug(
                    "Request %s send finished but status=%s, "
                    "deferring block free to request completion",
                    req_id, req.status.name)"""

    if old_send in new_src:
        new_src = new_src.replace(old_send, new_send, 1)
    else:
        print("[SETUP] WARN: scheduler finished_sending pattern not found")

    open(f, "w").write(new_src)
    print("[SETUP] Patched: scheduler _update_from_kv_xfer_finished read-mode fix")

except Exception as e:
    print(f"[SETUP] WARN patch scheduler read-mode: {e}", file=sys.stderr)
'
    _SETUP_INSTALLED+=("scheduler-read-mode-fix")
}

# =============================================================================
# Run installers
# =============================================================================

install_ucx
install_rixl
install_etcd
install_libionic
install_mori
patch_mori_fp8_compat
patch_moriio_save_kv_timeout
patch_moriio_transfer_timeout
patch_moriio_load_kv_timeout
patch_scheduler_read_mode_fix

if [[ "${NODE_RANK:-0}" -eq 0 ]]; then
    install_mori_proxy_deps
fi

# =============================================================================
# Export paths (persists for server.sh since this file is sourced)
# =============================================================================

export ROCM_PATH="${ROCM_PATH}"
export UCX_HOME="${UCX_HOME}"
export RIXL_HOME="${RIXL_HOME}"
export PATH="${UCX_HOME}/bin:/usr/local/bin/etcd:/root/.cargo/bin:${PATH}"
export LD_LIBRARY_PATH="${UCX_HOME}/lib:${RIXL_HOME}/lib:${RIXL_HOME}/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

_SETUP_END=$(date +%s)
if [[ ${#_SETUP_INSTALLED[@]} -eq 0 ]]; then
    echo "[SETUP] All dependencies already present (${_SETUP_END}s wallclock)"
else
    echo "[SETUP] Installed: ${_SETUP_INSTALLED[*]} in $(( _SETUP_END - _SETUP_START ))s"
fi
