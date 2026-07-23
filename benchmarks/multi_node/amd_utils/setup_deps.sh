#!/bin/bash
# =============================================================================
# setup_deps.sh — Install missing disagg dependencies at container start.
#
# Dispatched by $ENGINE (set by server.sh dispatcher):
#   vllm-disagg   -> vLLM/MoRI-IO patches + UCX/RIXL path exports
#                    (base image: vllm/vllm-openai-rocm:v0.18.0)
#   sglang-disagg -> SGLang aiter gluon patch + per-model installs
#                    (base image: lmsysorg/sglang-rocm:v0.5.12-rocm720-mi35x-*)
#
# Sourced by server_vllm.sh and server_sglang.sh so PATH / LD_LIBRARY_PATH
# exports persist. Each patch is idempotent: skipped if already applied.
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
# 5. Container RDMA/net tools
#    - ibv_devinfo comes from ibverbs-utils
#    - iproute2 provides the `ip` command
#    Used for in-container NIC/RDMA validation and routing checks.
# ---------------------------------------------------------------------------
install_recipe_deps() {
    if command -v ibv_devinfo >/dev/null 2>&1 && command -v ip >/dev/null 2>&1; then
        echo "[SETUP] Container RDMA/net tools already present"
        return 0
    fi

    echo "[SETUP] Installing ibv_devinfo + iproute2 in container..."
    apt-get update -q -y && apt-get install -q -y \
        ibverbs-utils iproute2 \
        && rm -rf /var/lib/apt/lists/*

    if ! command -v ibv_devinfo >/dev/null 2>&1 || ! command -v ip >/dev/null 2>&1; then
        echo "[SETUP] ERROR: Failed to install ibv_devinfo/iproute2"; exit 1
    fi
    _SETUP_INSTALLED+=("ibverbs-utils+iproute2")
}

# ---------------------------------------------------------------------------
# 6b. amd-quark (MXFP4 quantization support for Kimi-K2.5-MXFP4 and similar)
#     Required due to ROCm vLLM missing the quark dependency:
#     https://github.com/vllm-project/vllm/issues/35633
# ---------------------------------------------------------------------------
install_amd_quark() {
    if python3 -c "import quark" 2>/dev/null; then
        echo "[SETUP] amd-quark already present"
        return 0
    fi

    echo "[SETUP] Installing amd-quark for MXFP4 quantization support..."
    pip install --quiet amd-quark

    if ! python3 -c "import quark" 2>/dev/null; then
        echo "[SETUP] WARN: amd-quark install failed (non-fatal for non-MXFP4 models)"
        return 0
    fi
    _SETUP_INSTALLED+=("amd-quark")
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
# 10b. Patch MoRIIO WRITE-mode release path for latest vLLM.
#
# vLLM 0.25 introduced a WRITE-mode prefill-block release path in
# MoRIIOConnector.request_finished.  For Kimi PD-disagg the decode-side request
# can finish with do_remote_prefill still set and a plain request_id, so the new
# release helper cannot parse the router-embedded notify address and the request
# hangs behind:
#
#   Could not find ... in transfer_id_to_request_id lookup table
#   Cannot release WRITE prefill blocks ... missing remote notify address
#
# The older working bf610c2f image did not run this release path.  Restore that
# behavior behind an env-enabled patch while we validate the latest image.
# ---------------------------------------------------------------------------
patch_moriio_write_release_kimi_hang() {
    python3 -c '
import os, sys

try:
    import vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector as mc
    f = mc.__file__
    src = open(f).read()

    if "[PATCHED] skip WRITE release on unfinished remote prefill" in src:
        print("[SETUP] MoRIIO WRITE release hang patch already applied")
        sys.exit(0)

    old = """                self._release_write_prefill_blocks(request.request_id, params)"""
    new = """                logger.warning(
                    "[HANGFIX] skipping WRITE prefill-block release for "
                    "request %s because do_remote_prefill remained set at "
                    "request_finished; preserving old MoRIIO behavior",
                    request.request_id,
                )"""

    if old not in src:
        print("[SETUP] WARN: MoRIIO WRITE release pattern not found, skipping")
        sys.exit(0)

    open(f, "w").write(src.replace(old, new, 1))
    print("[SETUP] Patched MoRIIO WRITE release path for Kimi hang")
except Exception as e:
    print(f"[SETUP] WARN patch MoRIIO WRITE release hang: {e}", file=sys.stderr)
'
    _SETUP_INSTALLED+=("MoRIIO-write-release-kimi-hang-patch")
}

# ---------------------------------------------------------------------------
# 10c. Patch MoRIIO producer DP-rank handoff.
#
# Correct long-term shape for DEP/DTP prefill is not "all decode ranks read
# from prefill dp0".  The prefill engine that actually produced the KV must
# export its producer DP rank, and decode must use that concrete remote DP
# engine for handshake/session lookup before applying heterogeneous-TP mapping.
# ---------------------------------------------------------------------------
patch_moriio_producer_dp_rank() {
    python3 -c '
import os, sys

def replace_once(src, old, new, label):
    if old not in src:
        print(f"[SETUP] WARN: {label} pattern not found")
        return src
    return src.replace(old, new, 1)

try:
    import vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common as common
    f = common.__file__
    src = open(f).read()

    if "[PATCHED] MoRIIO producer remote_dp_rank metadata" in src:
        print("[SETUP] MoRIIO producer DP rank common patch already applied")
    else:
        src = replace_once(
            src,
            """    remote_dp_size: int\n""",
            """    remote_dp_size: int\n    # [PATCHED] MoRIIO producer remote_dp_rank metadata.\n    remote_dp_rank: int = 0\n""",
            "ReqMeta remote_dp_rank field",
        )
        src = replace_once(
            src,
            """            remote_dp_size=kv_transfer_params.get("remote_dp_size", 1),\n""",
            """            remote_dp_size=kv_transfer_params.get("remote_dp_size", 1),\n            remote_dp_rank=kv_transfer_params.get("remote_dp_rank", 0),\n""",
            "ReqMeta remote_dp_rank constructor",
        )
        open(f, "w").write(src)
        print("[SETUP] Patched MoRIIO producer DP rank metadata")
except Exception as e:
    print(f"[SETUP] WARN patch MoRIIO producer DP rank common: {e}", file=sys.stderr)

try:
    import vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector as mc
    f = mc.__file__
    src = open(f).read()

    if "[PATCHED] export producer remote_dp_rank" in src:
        print("[SETUP] MoRIIO producer DP rank connector patch already applied")
        sys.exit(0)

    src = replace_once(
        src,
        """            remote_dp_size=self.vllm_config.parallel_config.data_parallel_size,\n            tp_size=self.vllm_config.parallel_config.tensor_parallel_size,\n""",
        """            remote_dp_size=self.vllm_config.parallel_config.data_parallel_size,\n            # [PATCHED] export producer remote_dp_rank so decode reads the\n            # concrete prefill DP replica that produced this request.\n            remote_dp_rank=self.dp_rank,\n            tp_size=self.vllm_config.parallel_config.tensor_parallel_size,\n""",
        "request_finished remote_dp_rank handoff",
    )
    open(f, "w").write(src)
    print("[SETUP] Patched MoRIIO producer DP rank handoff")
except Exception as e:
    print(f"[SETUP] WARN patch MoRIIO producer DP rank connector: {e}", file=sys.stderr)
'
    _SETUP_INSTALLED+=("MoRIIO-producer-dp-rank-patch")
}

# ---------------------------------------------------------------------------
# 10d. Patch MoRIIO heterogeneous-TP readiness.
#
# Kimi MPND needs prefill TP4 feeding decode TP8/TP16.  vLLM MoRIIO has
# heterogeneous TP helpers, but the READ-mode scheduler path currently assumes
# one remote-engine readiness flag is enough for all local TP ranks and then
# continues down the normal request path after a handshake timeout.  In
# prefill-TP4 -> decode-TP8 tests this produces:
#
#   Timed out waiting for load_ready_flag[host:6301]
#   ValueError: Cannot parse peer zmq_address from request_id: '<plain id>'
#
# Decode ranks that do not finish the heterogeneous handshake in the first
# synchronous wait then see a plain internal/rejection request ID and crash the
# EngineCore while trying to parse router-embedded ZMQ addresses.
#
# Fixes:
#   1. Scope load/write readiness to the remote mapped TP rank, so local TP
#      ranks that share one remote TP rank do not wait on a nonexistent peer.
#   2. On handshake timeout, return and let the scheduler retry instead of
#      falling through into a request path with incomplete remote metadata.
#   3. Treat plain request IDs without remote host/ports as non-PD cleanup
#      requests and skip MoRIIO metadata construction instead of crashing.
# ---------------------------------------------------------------------------
patch_moriio_heterogeneous_tp_readiness() {
    python3 -c '
import os, sys

try:
    import vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common as common
    f = common.__file__
    src = open(f).read()

    if "[PATCHED] tolerate plain request_id for heterogeneous TP" in src:
        print("[SETUP] MoRIIO heterogeneous-TP common patch already applied")
    else:
        import re
        old = re.compile(
            r"(?m)^ {12}peer_zmq = get_peer_zmq_from_request_id"
            r"\(request_id, is_producer=write_mode\)\n"
            r"^ {12}remote_host, remote_handshake_port, remote_notify_port = \(\n"
            r"^ {16}parse_moriio_zmq_address\(peer_zmq\)\n"
            r"^ {12}\)"
        )
        new = """            # Parse host/ports from the request_id. The router embeds both
            # zmq_addresses in PD request IDs, but WRITE decode requests may carry
            # a plain request ID and get the remote address via kv_transfer_params.
            # [PATCHED] tolerate plain request_id for heterogeneous TP cleanup.
            try:
                peer_zmq = get_peer_zmq_from_request_id(
                    request_id, is_producer=write_mode)
                remote_host, remote_handshake_port, remote_notify_port = (
                    parse_moriio_zmq_address(peer_zmq)
                )
            except ValueError:
                remote_engine_id = kv_transfer_params.get("remote_engine_id")
                if (not write_mode) and remote_engine_id:
                    # Decode READ-mode cleanup / retry paths can carry a plain
                    # request_id while still containing enough kv_transfer_params
                    # to construct metadata.  Do not drop these requests: doing
                    # so loses transfer_id_to_request_id and can hang the proxy.
                    remote_host, _, _remote_port_s = str(remote_engine_id).partition(":")
                    remote_handshake_port = int(
                        kv_transfer_params.get(
                            "remote_handshake_port",
                            _remote_port_s or MoRIIOConstants.DEFAULT_HANDSHAKE_PORT,
                        )
                    )
                    remote_notify_port = int(
                        kv_transfer_params.get(
                            "remote_notify_port",
                            MoRIIOConstants.DEFAULT_NOTIFY_PORT,
                        )
                    )
                    logger.warning(
                        "[HANGFIX] recovered MoRIIO metadata from plain "
                        "request_id=%s remote_engine_id=%s remote_host=%s "
                        "handshake=%s notify=%s kv_transfer_keys=%s",
                        request_id, remote_engine_id, remote_host,
                        remote_handshake_port, remote_notify_port,
                        sorted(kv_transfer_params.keys()),
                    )
                else:
                    logger.warning(
                        "[HANGFIX] skipping MoRIIO metadata for non-PD/plain "
                        "request_id=%s write_mode=%s kv_transfer_keys=%s",
                        request_id, write_mode, sorted(kv_transfer_params.keys()),
                    )
                    return"""
        new_src, count = old.subn(new, src, count=1)
        if count == 0:
            print("[SETUP] WARN: MoRIIO plain request_id pattern not found")
        else:
            open(f, "w").write(new_src)
            print("[SETUP] Patched MoRIIO plain request_id handling")
except Exception as e:
    print(f"[SETUP] WARN patch MoRIIO plain request_id: {e}", file=sys.stderr)

try:
    import vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector as mc
    f = mc.__file__
    src = open(f).read()

    if "[PATCHED] heterogeneous TP mapped ready key" in src:
        print("[SETUP] MoRIIO heterogeneous-TP connector patch already applied")
        sys.exit(0)

    old_ready_head = """        # Do MoRIIO handshake in background and add to _ready_requests when done.
        fut = None
        if remote_engine_id is not None:
            fut = self._handshake_futures.get(remote_engine_id)"""
    new_ready_head = """        # Do MoRIIO handshake in background and add to _ready_requests when done.
        # [PATCHED] heterogeneous TP mapped ready key.  Multiple local TP ranks
        # can map to one remote TP rank; readiness is scoped to that remote rank.
        remote_tp_size = max(1, int(
            getattr(meta, "tp_size", 0)
            or getattr(meta, "remote_tp_size", 0)
            or getattr(meta, "attn_tp_size", 0)
            or 1
        ))
        remote_dp_size = max(1, int(getattr(meta, "remote_dp_size", 1) or 1))
        producer_dp_rank = max(0, int(getattr(meta, "remote_dp_rank", 0) or 0))
        remote_dp_rank = producer_dp_rank % remote_dp_size
        mapped_tp_rank = int(self.tp_rank) % remote_tp_size
        concrete_remote_engine_id = self.get_engine_name_with_dp(
            remote_engine_id, remote_dp_rank)
        ready_key = f"{concrete_remote_engine_id}|mapped_tp{mapped_tp_rank}"
        future_key = (
            f"{concrete_remote_engine_id}|mapped_tp{mapped_tp_rank}"
        )
        fut = None
        if remote_engine_id is not None:
            fut = self._handshake_futures.get(future_key)"""
    if old_ready_head in src:
        src = src.replace(old_ready_head, new_ready_head, 1)
    else:
        print("[SETUP] WARN: MoRIIO handshake ready-key head pattern not found")

    old_request_ready = """        def request_ready(_f: Future[Any], entry=(req_id, meta)):
            logger.info("MoRIIO handshake done for request %s", req_id)
            self._ready_requests.put(entry)
            self.load_ready_flag[remote_engine_id] = True
            self.write_ready_flags[remote_engine_id] = True"""
    new_request_ready = """        def request_ready(_f: Future[Any], entry=(req_id, meta), key=ready_key):
            _handshake_exc = _f.exception()
            if _handshake_exc is not None:
                logger.warning(
                    "[HANGFIX] MoRIIO handshake failed for request %s "
                    "local_tp=%s ready_key=%s exc=%r",
                    req_id, self.tp_rank, key, _handshake_exc,
                )
                return
            logger.info(
                "MoRIIO handshake done for request %s local_tp=%s "
                "producer_dp_rank=%s remote_dp_rank=%s remote_dp_size=%s "
                "remote_tp_size=%s mapped_tp=%s ready_key=%s",
                req_id, self.tp_rank, producer_dp_rank, remote_dp_rank,
                remote_dp_size,
                remote_tp_size, mapped_tp_rank, key)
            self._ready_requests.put(entry)
            self.load_ready_flag[key] = True
            self.write_ready_flags[key] = True"""
    if old_request_ready in src:
        src = src.replace(old_request_ready, new_request_ready, 1)
    else:
        print("[SETUP] WARN: MoRIIO request_ready local key pattern not found")

    old_existing_future = """        fut_list = []"""
    new_existing_future = """        if fut is not None:
            fut.add_done_callback(request_ready)
            return

        fut_list = []"""
    if old_existing_future in src:
        src = src.replace(old_existing_future, new_existing_future, 1)
    else:
        print("[SETUP] WARN: MoRIIO existing handshake future pattern not found")

    old_future = """            future.add_done_callback(done_callback)
            self._handshake_futures[dp_engine_id] = future"""
    new_future = """            future.add_done_callback(done_callback)
            _remote_tp_size = max(1, int(
                getattr(meta, "tp_size", 0)
                or getattr(meta, "remote_tp_size", 0)
                or getattr(meta, "attn_tp_size", 0)
                or 1
            ))
            _mapped_tp_rank = int(self.tp_rank) % _remote_tp_size
            self._handshake_futures[
                f"{dp_engine_id}|mapped_tp{_mapped_tp_rank}"
            ] = future"""
    if old_future in src:
        src = src.replace(old_future, new_future, 1)
    else:
        print("[SETUP] WARN: MoRIIO handshake future key pattern not found")

    old_remote_dp_init = """            remote_dp_size = int(meta.remote_dp_size)"""
    new_remote_dp_init = """            remote_dp_size = int(meta.remote_dp_size)
            producer_dp_rank = max(0, int(getattr(meta, "remote_dp_rank", 0) or 0))
            if remote_dp_size <= 0:
                remote_dp_size = 1
            if producer_dp_rank >= remote_dp_size:
                logger.warning(
                    "[HANGFIX] remote_dp_rank=%s out of range for "
                    "remote_dp_size=%s req=%s; using modulo mapping",
                    producer_dp_rank, remote_dp_size, req_id,
                )
                producer_dp_rank = producer_dp_rank % remote_dp_size
                meta.remote_dp_rank = producer_dp_rank
            remote_dp_rank = producer_dp_rank
            """
    if old_remote_dp_init in src:
        src = src.replace(old_remote_dp_init, new_remote_dp_init, 1)
    else:
        print("[SETUP] WARN: MoRIIO remote_dp_rank init pattern not found")

    old_port_offset = """        port_offset = get_port_offset(
            remote_dp_rank, self._remote_tp_rank(remote_tp_size)
        )"""
    new_port_offset = """        local_tp_size = max(1, int(
            self.vllm_config.parallel_config.tensor_parallel_size or 1))
        if remote_tp_size == local_tp_size:
            remote_tp_rank = self.tp_rank
        elif remote_tp_size > local_tp_size:
            remote_tp_rank = self.tp_rank * (remote_tp_size // local_tp_size)
        else:
            remote_tp_rank = self.tp_rank // (local_tp_size // remote_tp_size)
        # [PATCHED] Port offsets are in the remote engine namespace:
        # offset = remote_dp_rank * remote_tp_size + remote_tp_rank.
        # Do not call get_port_offset here; older vLLM variants bind its
        # optional size argument to the local TP namespace, which makes
        # local TP ranks 1..N hit nonexistent remote TP ports.
        port_offset = remote_dp_rank * remote_tp_size + remote_tp_rank"""
    if old_port_offset in src:
        src = src.replace(old_port_offset, new_port_offset, 1)
    else:
        old_port_offset_v024 = """        port_offset = get_port_offset(remote_dp_rank, self.tp_rank)"""
        if old_port_offset_v024 in src:
            src = src.replace(old_port_offset_v024, new_port_offset, 1)
        else:
            print("[SETUP] WARN: MoRIIO remote port offset pattern not found")

    old_handshake_path = """        path = make_zmq_path("tcp", host, port + port_offset)
        logger.debug("handshake Querying metadata on path: %s", path)"""
    new_handshake_path = """        path = make_zmq_path("tcp", host, port + port_offset)
        # [PATCHED] stagger heterogeneous TP handshakes.  Several decode TP
        # processes may query one producer DP/TP metadata endpoint at once;
        # MoRIIO/ZMQ metadata serving is fragile under that burst.
        _local_tp_size_for_stagger = max(1, int(
            self.vllm_config.parallel_config.tensor_parallel_size or 1))
        if remote_tp_size == 1 and _local_tp_size_for_stagger > 1:
            time.sleep(1.0 * int(self.tp_rank))
        logger.info(
            "[HANGFIX] MoRIIO handshake begin local_tp=%s "
            "remote_dp_rank=%s remote_tp_size=%s path=%s expected_engine_id=%s",
            self.tp_rank, remote_dp_rank, remote_tp_size,
            path, expected_engine_id,
        )"""
    if old_handshake_path in src:
        src = src.replace(old_handshake_path, new_handshake_path, 1)
    else:
        print("[SETUP] WARN: MoRIIO handshake stagger pattern not found")

    old_sock_send = """            logger.debug("prepare send msg INSTAZNCE: %s", path)
            sock.send(MoRIIOConstants.GET_META_MSG)
            received_frame = sock.recv_multipart()"""
    new_sock_send = """            logger.debug("prepare send msg INSTAZNCE: %s", path)
            sock.setsockopt(zmq.RCVTIMEO, int(self.moriio_config.transfer_timeout * 1000))
            sock.setsockopt(zmq.SNDTIMEO, int(self.moriio_config.transfer_timeout * 1000))
            try:
                sock.send(MoRIIOConstants.GET_META_MSG)
                received_frame = sock.recv_multipart()
            except Exception:
                logger.exception(
                    "[HANGFIX] MoRIIO handshake metadata recv failed "
                    "local_tp=%s remote_dp_rank=%s path=%s",
                    self.tp_rank, remote_dp_rank, path,
                )
                raise"""
    if old_sock_send in src:
        src = src.replace(old_sock_send, new_sock_send, 1)
    else:
        print("[SETUP] WARN: MoRIIO handshake socket timeout pattern not found")

    old_second_recv = """            received_frame = sock.recv_multipart()
            if len(received_frame) != 2 or received_frame[0] != b"":"""
    new_second_recv = """            try:
                received_frame = sock.recv_multipart()
            except Exception:
                logger.exception(
                    "[HANGFIX] MoRIIO handshake tensor-meta recv failed "
                    "local_tp=%s remote_dp_rank=%s path=%s",
                    self.tp_rank, remote_dp_rank, path,
                )
                raise
            if len(received_frame) != 2 or received_frame[0] != b"":"""
    if old_second_recv in src:
        src = src.replace(old_second_recv, new_second_recv, 1)
    else:
        print("[SETUP] WARN: MoRIIO handshake second recv timeout pattern not found")

    old_dp_loop = """        for cur_dp_rank in range(remote_dp_size):"""
    new_dp_loop = """        for cur_dp_rank in (remote_dp_rank,):"""
    if old_dp_loop in src:
        src = src.replace(old_dp_loop, new_dp_loop, 1)
    else:
        print("[SETUP] WARN: MoRIIO single remote DP handshake pattern not found")

    old_start = """        for req_id, meta in metadata.reqs_to_recv.items():
            remote_engine_id = (
                str(meta.remote_host) + ":" + str(meta.remote_handshake_port)
            )
            meta.remote_engine_id = remote_engine_id
            dp0_remote_engine_id = self.get_engine_name_with_dp(remote_engine_id, 0)
            if dp0_remote_engine_id not in self._remote_agents:
                # Initiate handshake with remote engine to exchange metadata.
                with self._handshake_lock:
                    if remote_engine_id not in self._remote_agents:
                        self._background_moriio_handshake(
                            req_id, remote_engine_id, meta
                        )
                        wait_handshake_readd_req = True

                        continue

            # Handshake already completed, start async read xfer.
            self._read_blocks_for_req(req_id, meta)
        # Start transfers for requests whose handshakes have now finished.

        if remote_engine_id is None and not wait_handshake_readd_req:
            return
        _deadline = time.monotonic() + self.moriio_config.transfer_timeout
        while True:
            if (
                self._ready_requests.empty()
                and remote_engine_id not in self.load_ready_flag
                and wait_handshake_readd_req
            ):
                if time.monotonic() > _deadline:
                    logger.warning(
                        "Timed out waiting for load_ready_flag[%s]; "
                        "adjust with kv_connector_extra_config.transfer_timeout",
                        remote_engine_id,
                    )
                    break
                time.sleep(0.001)
                continue
            elif (
                not self._ready_requests.empty()
                and remote_engine_id in self.load_ready_flag
            ):
                self._read_blocks_for_req(*self._ready_requests.get_nowait())
                break
            else:
                break

        self._reqs_to_send.update(metadata.reqs_to_send)"""

    new_start = """        ready_key = None
        for req_id, meta in metadata.reqs_to_recv.items():
            remote_engine_id = (
                str(meta.remote_host) + ":" + str(meta.remote_handshake_port)
            )
            meta.remote_engine_id = remote_engine_id
            remote_tp_size = max(1, int(
                getattr(meta, "tp_size", 0)
                or getattr(meta, "remote_tp_size", 0)
                or getattr(meta, "attn_tp_size", 0)
                or 1
            ))
            remote_dp_size = max(1, int(getattr(meta, "remote_dp_size", 1) or 1))
            producer_dp_rank = max(0, int(getattr(meta, "remote_dp_rank", 0) or 0))
            remote_dp_rank = producer_dp_rank % remote_dp_size
            mapped_tp_rank = int(self.tp_rank) % remote_tp_size
            concrete_remote_engine_id = self.get_engine_name_with_dp(
                remote_engine_id, remote_dp_rank)
            ready_key = f"{concrete_remote_engine_id}|mapped_tp{mapped_tp_rank}"
            mapped_remote_key = f"{concrete_remote_engine_id}|mapped_tp{mapped_tp_rank}"
            logger.warning(
                "[HANGFIX] start_load_kv recv req=%s local_tp=%s "
                "local_blocks=%s remote_blocks=%s remote_engine_id=%s "
                "concrete_remote_engine_id=%s remote_dp_rank=%s "
                "remote_dp_size=%s remote_tp_size=%s mapped_tp=%s "
                "ready_key=%s has_agent=%s ready_queue=%s",
                req_id, self.tp_rank, meta.local_block_ids,
                meta.remote_block_ids, remote_engine_id,
                concrete_remote_engine_id, remote_dp_rank, remote_dp_size,
                remote_tp_size, mapped_tp_rank, ready_key,
                mapped_remote_key in self._remote_agents,
                self._ready_requests.qsize(),
            )
            if mapped_remote_key not in self._remote_agents:
                # Initiate handshake with remote engine to exchange metadata.
                with self._handshake_lock:
                    if mapped_remote_key not in self._remote_agents:
                        self._background_moriio_handshake(
                            req_id, remote_engine_id, meta
                        )
                        wait_handshake_readd_req = True

                        continue

            # Handshake already completed, start async read xfer.
            self._read_blocks_for_req(req_id, meta)
        # Start transfers for requests whose handshakes have now finished.

        if remote_engine_id is None and not wait_handshake_readd_req:
            return
        _deadline = time.monotonic() + self.moriio_config.transfer_timeout
        while True:
            if (
                self._ready_requests.empty()
                and ready_key not in self.load_ready_flag
                and wait_handshake_readd_req
            ):
                if time.monotonic() > _deadline:
                    logger.warning(
                        "[HANGFIX] timed out waiting for heterogeneous-TP "
                        "load_ready_flag[%s] remote_engine_id=%s producer_dp_rank=%s "
                        "remote_dp_rank=%s remote_dp_size=%s local_tp=%s "
                        "remote_tp_size=%s mapped_tp=%s; deferring request "
                        "so the scheduler can retry",
                        ready_key, remote_engine_id, producer_dp_rank,
                        remote_dp_rank, remote_dp_size, self.tp_rank,
                        remote_tp_size, mapped_tp_rank,
                    )
                    return
                time.sleep(0.001)
                continue
            elif (
                not self._ready_requests.empty()
                and ready_key in self.load_ready_flag
            ):
                while not self._ready_requests.empty():
                    self._read_blocks_for_req(*self._ready_requests.get_nowait())
                break
            else:
                break

        self._reqs_to_send.update(metadata.reqs_to_send)"""

    if old_start not in src:
        print("[SETUP] WARN: MoRIIO start_load_kv heterogeneous TP pattern not found")
    else:
        src = src.replace(old_start, new_start, 1)

    # Store remote agents with the original key plus local/mapped TP scoped keys
    # so existing paths keep working while heterogeneous TP readiness tests the
    # mapped key.
    old_agents = """                        self._remote_agents[eid] = f.result()"""
    new_agents = """                        _agents = f.result()
                        self._remote_agents[eid] = _agents
                        self._remote_agents[f"{eid}|local_tp{self.tp_rank}"] = _agents
                        _remote_tp_size = max(1, int(
                            getattr(meta, "tp_size", 0)
                            or getattr(meta, "remote_tp_size", 0)
                            or getattr(meta, "attn_tp_size", 0)
                            or 1
                        ))
                        _mapped_tp_rank = int(self.tp_rank) % _remote_tp_size
                        self._remote_agents[
                            f"{eid}|mapped_tp{_mapped_tp_rank}"
                        ] = _agents
                        if eid in self.layer_name_to_remote_kv_cache_metadata:
                            _base_eid = eid.rsplit("_dp", 1)[0]
                            self.layer_name_to_remote_kv_cache_metadata.setdefault(
                                _base_eid,
                                self.layer_name_to_remote_kv_cache_metadata[eid],
                            )
                            self.remote_moriio_metadata.setdefault(
                                _base_eid,
                                self.remote_moriio_metadata[eid],
                            )
                        logger.info(
                            "[HANGFIX] MoRIIO registered remote agent "
                            "eid=%s local_tp=%s has_meta=%s aliases=%s",
                            eid, self.tp_rank,
                            eid in self.layer_name_to_remote_kv_cache_metadata,
                            sorted(k for k in self._remote_agents if str(eid) in k),
                        )"""
    if old_agents in src:
        src = src.replace(old_agents, new_agents, 1)
    else:
        print("[SETUP] WARN: MoRIIO remote_agents local key pattern not found")

    old_read_for_req = """        self._read_blocks(
            request_id=req_id,
            transfer_id=meta.transfer_id,
            dst_engine_id=meta.remote_engine_id,
            local_block_ids=meta.local_block_ids,
            remote_block_ids=meta.remote_block_ids,
            remote_host=meta.remote_host,
            remote_notify_port=meta.remote_notify_port,
        )"""
    new_read_for_req = """        remote_dp_size = max(1, int(getattr(meta, "remote_dp_size", 1) or 1))
        remote_tp_size = max(1, int(
            getattr(meta, "tp_size", 0)
            or getattr(meta, "remote_tp_size", 0)
            or getattr(meta, "attn_tp_size", 0)
            or 1
        ))
        if not meta.local_block_ids:
            logger.warning(
                "[HANGFIX] skipping MoRIIO read for empty local_block_ids "
                "req=%s transfer_id=%s remote_blocks=%s remote_engine_id=%s",
                req_id, meta.transfer_id, meta.remote_block_ids,
                meta.remote_engine_id,
            )
            return
        producer_dp_rank = max(0, int(getattr(meta, "remote_dp_rank", 0) or 0))
        remote_dp_rank = producer_dp_rank % remote_dp_size
        local_tp_size = max(1, int(
            self.vllm_config.parallel_config.tensor_parallel_size or 1))
        if remote_tp_size == local_tp_size:
            mapped_remote_tp_rank = self.tp_rank
        elif remote_tp_size > local_tp_size:
            mapped_remote_tp_rank = self.tp_rank * (remote_tp_size // local_tp_size)
        else:
            mapped_remote_tp_rank = self.tp_rank // (local_tp_size // remote_tp_size)
        remote_notify_port = (
            meta.remote_notify_port
            + remote_dp_rank * remote_tp_size
            + mapped_remote_tp_rank
        )
        concrete_remote_engine_id = self.get_engine_name_with_dp(
            meta.remote_engine_id, remote_dp_rank)
        logger.warning(
            "[HANGFIX] _read_blocks_for_req req=%s local_tp=%s transfer_id=%s "
            "local_blocks=%s remote_blocks=%s concrete_remote_engine_id=%s "
            "remote_notify_port=%s remote_dp_rank=%s remote_tp_size=%s "
            "mapped_remote_tp=%s",
            req_id, self.tp_rank, meta.transfer_id, meta.local_block_ids,
            meta.remote_block_ids, concrete_remote_engine_id,
            remote_notify_port, remote_dp_rank, remote_tp_size,
            mapped_remote_tp_rank,
        )
        self._read_blocks(
            request_id=req_id,
            transfer_id=meta.transfer_id,
            dst_engine_id=concrete_remote_engine_id,
            local_block_ids=meta.local_block_ids,
            remote_block_ids=meta.remote_block_ids,
            remote_host=meta.remote_host,
            remote_notify_port=remote_notify_port,
        )
        # [PATCHED] READ mode schedules async RDMA reads in worker processes,
        # but get_finished() is polled from the scheduler-side connector and
        # cannot see worker-local _recving_transfers.  Wait here so forward
        # only starts after KV is resident, and release the producer blocks.
        _deadline = time.monotonic() + self.moriio_config.transfer_timeout
        _last_log = 0.0
        while True:
            with self.moriio_wrapper.lock:
                _statuses = list(self._recving_transfers.get(req_id, []))
                _callback = self._recving_transfers_callback_addr.get(req_id)
            _num_status = len(_statuses)
            _all_done = _num_status > 0 and all(s.Succeeded() for s in _statuses)
            _any_failed = any(s.Failed() for s in _statuses)
            _now = time.monotonic()
            if _all_done:
                if _callback is not None:
                    _host, _port, _xfer_id = _callback
                    logger.warning(
                        "[HANGFIX] synchronous READ complete req=%s local_tp=%s "
                        "num_status=%s notify=%s:%s xfer_id=%s",
                        req_id, self.tp_rank, _num_status, _host, _port, _xfer_id,
                    )
                    self.moriio_wrapper.send_notify(_xfer_id, _host, _port)
                with self.moriio_wrapper.lock:
                    self._recving_transfers.pop(req_id, None)
                    self._recving_transfers_callback_addr.pop(req_id, None)
                break
            if _any_failed:
                logger.error(
                    "[HANGFIX] synchronous READ failed req=%s local_tp=%s "
                    "num_status=%s",
                    req_id, self.tp_rank, _num_status,
                )
                break
            if _now > _deadline:
                logger.warning(
                    "[HANGFIX] synchronous READ timed out req=%s local_tp=%s "
                    "num_status=%s callback=%s",
                    req_id, self.tp_rank, _num_status, _callback,
                )
                break
            if _now - _last_log > 10.0:
                logger.warning(
                    "[HANGFIX] synchronous READ waiting req=%s local_tp=%s "
                    "num_status=%s callback=%s",
                    req_id, self.tp_rank, _num_status, _callback,
                )
                _last_log = _now
            time.sleep(0.001)"""
    if old_read_for_req in src:
        src = src.replace(old_read_for_req, new_read_for_req, 1)
    else:
        print("[SETUP] WARN: MoRIIO read concrete DP engine pattern not found")

    old_read_session = """        dp0_engine_id = self.get_engine_name_with_dp(dst_engine_id, 0)
        sessions, remote_moriio_meta = self._get_built_session(dp0_engine_id)"""
    new_read_session = """        # [PATCHED] dst_engine_id is already the concrete remote DP engine
        # selected by the producer handoff; do not force reads through dp0.
        sessions, remote_moriio_meta = self._get_built_session(dst_engine_id)"""
    if old_read_session in src:
        src = src.replace(old_read_session, new_read_session, 1)
    else:
        print("[SETUP] WARN: MoRIIO read session DP0 pattern not found")

    old_read_status = """            transfer_status = self.moriio_wrapper.read_remote_data(
                offs[2], offs[0], offs[1], sessions[sess_idx]
            )
            with self.moriio_wrapper.lock:
                self._recving_transfers[request_id].append(transfer_status)"""
    new_read_status = """            _num_layers = len(self.layer_name_to_local_kv_cache_metadata)
            _log_layer = sess_idx == 0 or sess_idx == (_num_layers - 1)
            if _log_layer:
                logger.warning(
                    "[HANGFIX] _read_blocks layer=%s req=%s local_tp=%s "
                    "dst_engine_id=%s sess_idx=%s/%s n_ranges=%s total_bytes=%s "
                    "local_offsets_head=%s remote_offsets_head=%s sizes_head=%s",
                    layer_name, request_id, self.tp_rank, dst_engine_id,
                    sess_idx, _num_layers, len(offs[2]),
                    sum(offs[2]) if offs[2] else 0,
                    offs[0][:3], offs[1][:3], offs[2][:3],
                )
            transfer_status = self.moriio_wrapper.read_remote_data(
                offs[2], offs[0], offs[1], sessions[sess_idx]
            )
            if _log_layer:
                logger.warning(
                    "[HANGFIX] _read_blocks scheduled layer=%s req=%s local_tp=%s "
                    "transfer_status=%r",
                    layer_name, request_id, self.tp_rank, transfer_status,
                )
            with self.moriio_wrapper.lock:
                self._recving_transfers[request_id].append(transfer_status)
                if _log_layer:
                    logger.warning(
                        "[HANGFIX] _read_blocks tracking req=%s local_tp=%s "
                        "num_status=%s callback_before=%s",
                        request_id, self.tp_rank,
                        len(self._recving_transfers[request_id]),
                        self._recving_transfers_callback_addr.get(request_id),
                    )"""
    if old_read_status in src:
        src = src.replace(old_read_status, new_read_status, 1)
    else:
        print("[SETUP] WARN: MoRIIO read status instrumentation pattern not found")

    old_recv_callback = """                self._recving_transfers_callback_addr[request_id] = (
                    remote_host,
                    str(remote_notify_port + self.tp_rank),
                    transfer_id,
                )"""
    new_recv_callback = """                # [PATCHED] remote_notify_port is already the exact producer
                # DP/TP notify port computed by _read_blocks_for_req.
                _had_callback = request_id in self._recving_transfers_callback_addr
                self._recving_transfers_callback_addr[request_id] = (
                    remote_host,
                    str(remote_notify_port),
                    transfer_id,
                )
                if not _had_callback:
                    logger.warning(
                        "[HANGFIX] _read_blocks callback req=%s local_tp=%s "
                        "callback=%s",
                        request_id, self.tp_rank,
                        self._recving_transfers_callback_addr[request_id],
                    )"""
    if old_recv_callback in src:
        src = src.replace(old_recv_callback, new_recv_callback, 1)
    else:
        print("[SETUP] WARN: MoRIIO recv callback notify port pattern not found")

    old_pop_done = """            for req_id, status_list in self._recving_transfers.items():
                last = status_list[-1]
                if last.Succeeded():
                    host, port, xfer_id = self._recving_transfers_callback_addr[req_id]
                    done_req_ids.add(xfer_id)
                    self.moriio_wrapper.send_notify(xfer_id, host, port)
                    to_remove.append(req_id)
                elif last.Failed():
                    logger.error(
                        "RDMA transfer failed for request %s: %s (code=%s). "
                        "Notifying prefill to free blocks; request will be "
                        "aborted by timeout.",
                        req_id,
                        last.Message(),
                        last.Code(),
                    )"""
    new_pop_done = """            for req_id, status_list in self._recving_transfers.items():
                last = status_list[-1]
                try:
                    _succeeded = last.Succeeded()
                    _failed = last.Failed()
                    _msg = last.Message() if _failed else ""
                    _code = last.Code() if _failed else ""
                except Exception:
                    logger.exception(
                        "[HANGFIX] _pop_done_transfers status probe failed "
                        "req=%s num_status=%s last=%r",
                        req_id, len(status_list), last,
                    )
                    _succeeded = False
                    _failed = False
                    _msg = ""
                    _code = ""
                logger.warning(
                    "[HANGFIX] _pop_done_transfers req=%s num_status=%s "
                    "last=%r succeeded=%s failed=%s callback=%s",
                    req_id, len(status_list), last, _succeeded, _failed,
                    self._recving_transfers_callback_addr.get(req_id),
                )
                if _succeeded:
                    host, port, xfer_id = self._recving_transfers_callback_addr[req_id]
                    done_req_ids.add(xfer_id)
                    logger.warning(
                        "[HANGFIX] _pop_done_transfers send_notify req=%s "
                        "xfer_id=%s host=%s port=%s",
                        req_id, xfer_id, host, port,
                    )
                    self.moriio_wrapper.send_notify(xfer_id, host, port)
                    to_remove.append(req_id)
                elif _failed:
                    logger.error(
                        "RDMA transfer failed for request %s: %s (code=%s). "
                        "Notifying prefill to free blocks; request will be "
                        "aborted by timeout.",
                        req_id,
                        _msg,
                        _code,
                    )"""
    if old_pop_done in src:
        src = src.replace(old_pop_done, new_pop_done, 1)
    else:
        print("[SETUP] WARN: MoRIIO _pop_done_transfers instrumentation pattern not found")

    old_all_done = """        all_done_future = self._handshake_initiation_executor.submit(wait_all_dp)
        all_done_future.add_done_callback(request_ready)"""
    new_all_done = """        all_done_future = self._handshake_initiation_executor.submit(wait_all_dp)
        self._handshake_futures[future_key] = all_done_future
        all_done_future.add_done_callback(request_ready)"""
    if old_all_done in src:
        src = src.replace(old_all_done, new_all_done, 1)
    else:
        print("[SETUP] WARN: MoRIIO all-dp handshake future pattern not found")

    open(f, "w").write(src)
    print("[SETUP] Patched MoRIIO heterogeneous TP readiness")
except Exception as e:
    print(f"[SETUP] WARN patch MoRIIO heterogeneous TP readiness: {e}", file=sys.stderr)
'
    _SETUP_INSTALLED+=("MoRIIO-heterogeneous-tp-readiness-patch")
}

# ---------------------------------------------------------------------------
# 11. Fix READ-mode scheduler assertion in _update_from_kv_xfer_finished
#     vLLM asserts that a request in finished_recving must be either
#     WAITING_FOR_REMOTE_KVS or finished.  In READ mode the request can
#     transition to RUNNING before the aggregated recv notification arrives,
#     crashing the engine with AssertionError.
#     (present in v0.17.1 & v0.18.0)
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
        if kv_connector_output.finished_recving or kv_connector_output.finished_sending:
            logger.warning(
                "[HANGFIX] scheduler kv_xfer output finished_recving=%s "
                "finished_sending=%s requests=%s waiting=%s running=%s",
                kv_connector_output.finished_recving,
                kv_connector_output.finished_sending,
                list(self.requests.keys())[:8],
                getattr(self, "waiting", None),
                getattr(self, "running", None),
            )
        for req_id in kv_connector_output.finished_recving or ():
            logger.warning("Finished recving KV transfer for request %s", req_id)
            if req_id not in self.requests:
                logger.warning("Request %s already removed, skipping recv", req_id)
                continue
            req = self.requests[req_id]
            logger.warning(
                "[HANGFIX] scheduler finished_recving req=%s status=%s "
                "num_computed_tokens=%s",
                req_id, req.status.name,
                getattr(req, "num_computed_tokens", None),
            )
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
            elif RequestStatus.is_finished(req.status):
                self._free_blocks(self.requests[req_id])
            else:
                logger.warning(
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
            logger.warning("Finished sending KV transfer for request %s", req_id)
            if req_id not in self.requests:
                logger.warning("Request %s already removed, skipping send", req_id)
                continue
            self._free_blocks(self.requests[req_id])"""

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

# ---------------------------------------------------------------------------
# 12. Idle KV block reaper for disaggregated prefill (READ mode)
#     The RIXL notification path can lose `finished_sending` signals under
#     high concurrency with ibv_post_send failures. This leaves KV blocks
#     permanently allocated on the prefill engine even after the decode has
#     finished reading. Over multiple benchmark rounds, leaked blocks
#     accumulate and eventually saturate the prefill KV cache.
#
#     Fix: instrument the scheduler's `schedule()` method to detect idle
#     periods (0 running, 0 waiting for >5s) and force-free blocks for
#     any remaining requests whose status is finished.
# ---------------------------------------------------------------------------
patch_prefill_idle_kv_reaper() {
    python3 -c '
import os, sys

try:
    import vllm.v1.core.sched.scheduler as smod
    f = smod.__file__
    src = open(f).read()

    if "[PATCHED] idle-kv-reaper" in src:
        print("[SETUP] idle KV block reaper already applied")
        sys.exit(0)

    # Find the _update_from_kv_xfer_finished method end and add reaper logic
    # We inject into the method that processes KV transfer completions.
    marker = "[PATCHED] read-mode recv assertion"
    if marker not in src:
        print("[SETUP] WARN: scheduler read-mode patch not found, skipping reaper")
        sys.exit(0)

    # Add reaper state initialization to __init__
    old_init_marker = "self.finished_recving_kv_req_ids"
    if old_init_marker not in src:
        print("[SETUP] WARN: finished_recving_kv_req_ids not found in scheduler")
        sys.exit(0)

    # Find the first occurrence to insert reaper state
    init_pos = src.find(old_init_marker)
    # Find the line containing it
    line_end = src.find("\n", init_pos)
    init_line = src[init_pos:line_end]

    # Add reaper state after this line
    reaper_init = init_line + """
        # [PATCHED] idle-kv-reaper state
        self._idle_kv_reaper_ts = 0.0
        self._idle_kv_reaper_active = False"""

    src = src.replace(init_line, reaper_init, 1)

    # Now add the reaper logic at the end of _update_from_kv_xfer_finished
    # Find the finished_sending handler we patched
    send_handler = """        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            if req_id not in self.requests:
                logger.debug("Request %s already removed, skipping send", req_id)
                continue
            self._free_blocks(self.requests[req_id])"""

    reaper_logic = send_handler + """

        # [PATCHED] idle-kv-reaper — force-free leaked prefill KV blocks
        import time as _time
        _REAPER_IDLE_SECS = 5.0
        _num_running = sum(1 for r in self.requests.values()
                          if r.status == RequestStatus.RUNNING)
        _should_reap = (_num_running == 0)

        if _should_reap:
            if not self._idle_kv_reaper_active:
                self._idle_kv_reaper_active = True
                self._idle_kv_reaper_ts = _time.monotonic()
            elif _time.monotonic() - self._idle_kv_reaper_ts > _REAPER_IDLE_SECS:
                _reaped = 0
                _reap_ids = []
                for _rid, _req in list(self.requests.items()):
                    if RequestStatus.is_finished(_req.status):
                        _reap_ids.append(_rid)
                for _rid in _reap_ids:
                    try:
                        _req = self.requests[_rid]
                        self._free_blocks(_req)
                        _reaped += 1
                    except Exception as _e:
                        logger.debug("[KV-REAPER] free_blocks failed for %s: %s", _rid, _e)
                if _reaped > 0:
                    logger.warning(
                        "[KV-REAPER] Force-freed blocks for %d finished "
                        "requests after %.1fs idle",
                        _reaped, _time.monotonic() - self._idle_kv_reaper_ts)
                self._idle_kv_reaper_ts = _time.monotonic()
        else:
            self._idle_kv_reaper_active = False"""

    if send_handler in src:
        src = src.replace(send_handler, reaper_logic, 1)
    else:
        print("[SETUP] WARN: send handler not found for reaper injection")
        sys.exit(0)

    open(f, "w").write(src)
    print("[SETUP] Patched: idle KV block reaper for prefill")

except Exception as e:
    print(f"[SETUP] WARN patch idle-kv-reaper: {e}", file=sys.stderr)
'
    _SETUP_INSTALLED+=("idle-kv-reaper")
}

# ---------------------------------------------------------------------------
# SGLang: Patch aiter gluon pa_mqa_logits — fix 2D → 3D instr_shape for
# Triton ≥ 3.5.
#
# Bug: _gluon_deepgemm_fp8_paged_mqa_logits (the non-preshuffle variant)
# hardcodes AMDMFMALayout(instr_shape=[16, 16]) which fails on Triton
# builds where AMDMFMALayout requires 3D (M, N, K) format.
#
# The two preshuffle variants already conditionally select 2D vs 3D via
# the module-level _Use_2d_instr_shape_mfma_layout flag, but the base
# variant was missed. This patch brings it in line.
#
# Affects: GLM-5 (NSA attention) and any future model that uses
# deepgemm_fp8_paged_mqa_logits with Preshuffle=False.
# ---------------------------------------------------------------------------
patch_gluon_pa_mqa_logits_instr_shape() {
    python3 -c '
import os, sys

target = "/sgl-workspace/aiter/aiter/ops/triton/gluon/pa_mqa_logits.py"
if not os.path.isfile(target):
    print("[SETUP] gluon pa_mqa_logits.py not found, skipping")
    sys.exit(0)

src = open(target).read()

if "[PATCHED] 3D instr_shape for base gluon variant" in src:
    print("[SETUP] gluon pa_mqa_logits 3D instr_shape patch already applied")
    sys.exit(0)

# The buggy code: the base _gluon_deepgemm_fp8_paged_mqa_logits uses 2D
# instr_shape unconditionally.  We replace it with a conditional that
# mirrors the preshuffle variants.
old = """\
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=CDNA_VERSION,
        instr_shape=[16, 16],
        transposed=False,
        warps_per_cta=[1, NumWarps],
    )
    mfma_layout_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    mfma_layout_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )"""

new = """\
    # [PATCHED] 3D instr_shape for base gluon variant
    if _Use_2d_instr_shape_mfma_layout:
        mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
            version=CDNA_VERSION,
            instr_shape=[16, 16],
            transposed=False,
            warps_per_cta=[1, NumWarps],
        )
    else:
        mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
            version=CDNA_VERSION,
            instr_shape=[16, 16, 32],
            transposed=False,
            warps_per_cta=[1, NumWarps],
        )
    mfma_layout_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    mfma_layout_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )"""

if old not in src:
    print("[SETUP] WARN: gluon pa_mqa_logits pattern not found — aiter version may have changed")
    sys.exit(0)

# Only replace the FIRST occurrence (the base variant, not preshuffle ones)
new_src = src.replace(old, new, 1)

open(target, "w").write(new_src)
print("[SETUP] Patched: gluon pa_mqa_logits 3D instr_shape for base variant")
'
    _SETUP_INSTALLED+=("gluon-instr-shape-fix")
}

# ---------------------------------------------------------------------------
# SGLang: Install latest transformers for GLM-5 model type support.
#
# GLM-5 (zai-org/GLM-5-FP8) requires a transformers build that includes
# the glm_moe_dsa model type. The mori images do not ship it.
# Only install if GLM-5 is the active model (avoid overhead otherwise).
# ---------------------------------------------------------------------------
install_transformers_glm5() {
    if [[ "$MODEL_NAME" != "GLM-5-FP8" ]]; then
        return 0
    fi

    if python3 -c "from transformers import AutoConfig; AutoConfig.from_pretrained('zai-org/GLM-5-FP8', trust_remote_code=True)" 2>/dev/null; then
        echo "[SETUP] transformers already supports GLM-5 model type"
        return 0
    fi

    echo "[SETUP] Installing transformers with GLM-5 (glm_moe_dsa) support..."
    pip install --quiet -U --no-cache-dir \
        "git+https://github.com/huggingface/transformers.git@6ed9ee36f608fd145168377345bfc4a5de12e1e2"
    _SETUP_INSTALLED+=("transformers-glm5")
}

# =============================================================================
# Run installers (engine-gated)
# =============================================================================

if [[ "$ENGINE" == "vllm-disagg" ]]; then
    install_recipe_deps
    install_amd_quark
    patch_moriio_save_kv_timeout
    patch_moriio_transfer_timeout
    patch_moriio_load_kv_timeout
    patch_moriio_write_release_kimi_hang
    patch_moriio_producer_dp_rank
    patch_moriio_heterogeneous_tp_readiness
    patch_scheduler_read_mode_fix
    patch_prefill_idle_kv_reaper

    # =========================================================================
    # vLLM: Export UCX/RIXL paths (persists since this file is sourced)
    # =========================================================================
    export ROCM_PATH="${ROCM_PATH}"
    export UCX_HOME="${UCX_HOME}"
    export RIXL_HOME="${RIXL_HOME}"
    export PATH="${UCX_HOME}/bin:/usr/local/bin/etcd:/root/.cargo/bin:${PATH}"
    export LD_LIBRARY_PATH="${UCX_HOME}/lib:${RIXL_HOME}/lib:${RIXL_HOME}/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
else
    patch_gluon_pa_mqa_logits_instr_shape
    install_transformers_glm5
fi

_SETUP_END=$(date +%s)
if [[ ${#_SETUP_INSTALLED[@]} -eq 0 ]]; then
    echo "[SETUP] All dependencies already present ($(( _SETUP_END - _SETUP_START ))s wallclock)"
else
    echo "[SETUP] Installed: ${_SETUP_INSTALLED[*]} in $(( _SETUP_END - _SETUP_START ))s"
fi
