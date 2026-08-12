#!/usr/bin/env bash
# =============================================================================
# apply_k3_moriio_patches.sh
#
# Kimi-K3 needs three things from the MoRIIO connector that the pinned image's
# vLLM does not have. They are carried here as one patch against
# vllm/vllm-openai-rocm:nightly-cb8104839c141609d99f1254459ef3a4f1bd4263 rather
# than as a fork checkout, so a disagg run installs exactly the same way the
# single-node recipe installs apply_k3_container_patches.sh:
#
#   * hybrid mamba/KDA (conv+ssm) KV transfer. K3 is MLA + Kimi Delta Attention,
#     so a request's state is a (conv, ssm) tuple in one KV-cache group and paged
#     blocks in the others. Stock MoRIIO only moves paged blocks.
#   * per-layer KV block length. K3's KDA group gets a larger page than its MLA
#     groups (3072 vs 1536 tokens at fp8), and MoRIIO used to assert every
#     attention layer matched the first one, aborting all 8 ranks with
#     "MoRIIO KV cache block size mismatch for layer model.layers.93.self_attn".
#   * draft-only layer skip. Under DSpark the decoder loads a draft model whose
#     layers do not exist on the prefill side; MoRIIO must not try to pull them.
#
# Also carried: a peer KV-layout guard (fail the handshake instead of
# transferring into the wrong layers), the DSpark rejection-sampler bounds
# checks, drafter-invariant hybrid KV grouping, and a KV-group-size log line.
#
# This patch set is DISJOINT from apply_k3_container_patches.sh -- verified file
# by file -- so the two can be applied in either order without clobbering each
# other. Keep it that way: before adding anything here, check the file list
# against that script's.
#
# Run INSIDE the engine container, before `vllm serve`. Idempotent.
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${K3_MORIIO_PATCH:-$HERE/patches/k3_moriio_dspark_trim.patch}"

ROOT="$(python3 -c 'import importlib.util as u, os; print(os.path.dirname(os.path.dirname(u.find_spec("vllm").origin)))')"
if [ -z "$ROOT" ] || [ ! -d "$ROOT/vllm" ]; then
  echo "[k3-moriio] ERROR: could not resolve dist-packages (ROOT='$ROOT')"; exit 1
fi
echo "[k3-moriio] ROOT=$ROOT"
echo "[k3-moriio] patch=$PATCH_FILE"
[ -f "$PATCH_FILE" ] || { echo "[k3-moriio] ERROR: patch file missing"; exit 1; }

MORIIO_CONNECTOR="$ROOT/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
WS_RACE="${WS_RACE:-/tmp/k3_moriio_race}"; mkdir -p "$WS_RACE"

# Post-state marker: the draft-only layer set only exists after this patch.
if grep -qF "_draft_only_layers" "$MORIIO_CONNECTOR" 2>/dev/null; then
  echo "[k3-moriio] already applied (skip)"
else
  # The patch is generated against the vLLM source tree, so paths are
  # vllm/... and examples/...; dist-packages has vllm/ at its root and no
  # examples/, hence -p1 plus an explicit exclude for the example proxy.
  if ( cd "$ROOT" && git apply -p1 --exclude='examples/*' --whitespace=nowarn "$PATCH_FILE" ) 2>/dev/null; then
    echo "[k3-moriio] APPLIED (git apply)"
  elif patch -p1 -d "$ROOT" --fuzz=3 --forward --no-backup-if-mismatch \
         -x 'examples/*' < "$PATCH_FILE" >/dev/null 2>&1; then
    echo "[k3-moriio] APPLIED (patch)"
  else
    echo "[k3-moriio] ERROR: patch did not apply; refusing to serve a half-patched tree"
    ( cd "$ROOT" && git apply -p1 --exclude='examples/*' --check "$PATCH_FILE" ) 2>&1 | head -20
    exit 1
  fi
  find "$ROOT" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null
fi

# --- MoRIIO transfer-completion correctness -----------------------------------
# Three fixes, all narrowing the window in which the decoder can act on KV that
# has not landed. Applied after the trim patch above because they build on it.
#
#   1. start_load_kv waits for the reads it just posted. READ mode reports
#      load_kv_async=False, which promises the scheduler that the KV is in place
#      by the time the forward runs -- but the reads were posted non-blocking and
#      wait_for_layer_load is a no-op, so nothing honoured the promise and the
#      forward could read blocks the NIC was still filling. Since
#      delay_cache_blocks tracks load_kv_async, those same blocks were also
#      published into the prefix cache in that step.
#   2. A request's completion now considers every layer's status instead of only
#      the newest. _post_read_with_backoff deliberately returns a failed status
#      when the send queue never drains, so a failure on any layer but the last
#      was silently reported as success and the decoder ran on partial KV.
#   3. Failed reads reach the scheduler through get_block_ids_with_load_errors,
#      as NIXL does, instead of being logged and left to expire on a timeout.
#
# The blocking wait uses mori's batched WaitAll where available (GIL released,
# condvar instead of a 1 ms spin) and falls back to polling otherwise; the
# non-blocking verdict stays in Python on purpose, since mori's zero-timeout
# wait would drive PollProgress from the engine thread.
cat > "$WS_RACE/MORIIO_RACE.diff" <<'DIFF_MORIIO_RACE'
diff --git a/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py b/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py
--- a/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py
+++ b/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py
@@ -191,12 +191,26 @@ def get_role() -> ROLE:
 
 class MoRIIOMode(Enum):
     READ = "read"
     WRITE = "write"
 
 
+class TransferBatchState(Enum):
+    """Verdict for a group of transfer statuses that belong to one request.
+
+    A request's KV transfer is spread over one status per layer (two for a KDA
+    layer), so a single status never decides the request: PENDING means at
+    least one is still in flight and none has failed, FAILED means at least
+    one failed, DONE means all succeeded.
+    """
+
+    DONE = "done"
+    FAILED = "failed"
+    PENDING = "pending"
+
+
 class MoRIIOError(Exception):
     """Base exception for MoRIIO operations."""
 
     pass
 
 
diff --git a/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py b/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py
--- a/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py
+++ b/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py
@@ -34,12 +34,14 @@ from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
     MoRIIOConstants,
     MoRIIOError,
     MoRIIOMode,
     MoRIIOTransferAck,
     ReqId,
     ReqMeta,
+    TransferBatchState,
+    TransferError,
     TransferId,
     WriteTask,
     as_attn_mamba,
     fold_local_rank,
     get_moriio_mode,
     get_peer_zmq_from_request_id,
@@ -329,12 +331,18 @@ class MoRIIOConnector(KVConnectorBase_V1, SupportsHMA):
 
     def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
         """Get the finished recving and sending requests."""
         assert self.connector_worker is not None
         return self.connector_worker.get_finished()
 
+    def get_block_ids_with_load_errors(self) -> set[int]:
+        """Blocks whose MoRIIO read failed, for the scheduler to recompute."""
+        if self.connector_worker is None:
+            return set()
+        return self.connector_worker.get_block_ids_with_load_errors()
+
     def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
         assert self.connector_worker is not None
         if self.mode == MoRIIOMode.WRITE and get_role() == ROLE.CONSUMER:
             self.connector_worker.moriio_wrapper.async_wait_reqid()
 
         assert isinstance(self._connector_metadata, MoRIIOConnectorMetadata)
@@ -587,12 +595,26 @@ class MoRIIOConnectorScheduler:
                 # Hybrid: the decoder recomputes the final token locally from
                 # the pushed KDA state, so only N-1 tokens come from remote.
                 num_tokens -= 1
             return num_tokens - num_computed_tokens, True
 
         # READ mode always recomputes the last token locally on the decoder.
+        #
+        # The second element declares WHO waits for the KV, not whether it has
+        # already arrived. False keeps the wait on this side: the scheduler
+        # runs the forward in the same step, and the connector is expected to
+        # block in start_load_kv (or wait_for_layer_load) until the KV is
+        # really there. READ=False / WRITE=True is MoRIIO's paradigm across
+        # every model, so it stays -- flipping READ to the NIXL-style async
+        # park would change behaviour for every MoRIIO deployment, not just
+        # this arm.
+        #
+        # What was actually broken is that nobody honoured the False: the reads
+        # were posted non-blocking and wait_for_layer_load was a no-op, so the
+        # forward ran against blocks the NIC was still filling. start_load_kv
+        # now waits for this step's reads before returning.
         return len(token_ids) - 1 - num_computed_tokens, False
 
     def _truncate_mamba_request_for_prefill(self, request: "Request") -> None:
         """P-side only: drop the last prompt token so the prefiller computes
         h(N-1) instead of h(N).
 
@@ -1439,12 +1461,20 @@ class MoRIIOConnectorWorker:
         # Values are (remote_host, remote_notify_port, transfer_id).
         self._recving_transfers_callback_addr: dict[ReqId, tuple[str, str, str]] = {}
         # Monotonic-clock start times for each in-flight recv transfer.
         # Used by _pop_done_transfers to abort transfers whose RDMA
         # completion is lost, instead of hanging forever.
         self._recving_transfers_start: dict[str, float] = {}
+        # Local attention block ids per in-flight recv, kept so a failed read
+        # can hand them to the scheduler as load errors (see NIXL's
+        # _handle_failed_transfer).
+        self._recving_local_blocks: dict[ReqId, list[int]] = {}
+        # Block ids drained by get_block_ids_with_load_errors.
+        self._invalid_block_ids: set[int] = set()
+        # Statuses posted by this step's reads, awaited before the forward.
+        self._reads_issued_this_step: list = []
 
         # Track the expiration time of requests that are waiting to be sent.
         self._reqs_to_send: dict[ReqId, float] = {}
 
         # Background thread for handling new handshake requests.
         self._moriio_handshake_listener_t: threading.Thread | None = None
@@ -2278,19 +2308,18 @@ class MoRIIOConnectorWorker:
                 fresh = self.moriio_wrapper.pop_finished_write_req_ids()
                 # Accumulate with any completions that arrived before their
                 # transfer_id was registered in transfer_id_to_request_id.
                 self._unmatched_write_completions |= fresh
                 done_recving = self._unmatched_write_completions
             else:
-                # READ mode: the scheduler treats KV loads as synchronous
-                # (load_kv_async=False), so requests go directly to RUNNING
-                # instead of WAITING_FOR_REMOTE_KVS. We still call
-                # _pop_done_transfers() to send the notify to the prefill
-                # side and clean up internal state, but we must NOT report
-                # these as done_recving because the scheduler doesn't
-                # expect a finished_recving signal for RUNNING requests.
+                # READ mode is synchronous (load_kv_async=False), so requests
+                # are already RUNNING and the scheduler does not expect a
+                # finished_recving signal for them. _pop_done_transfers still
+                # runs to notify the producer and clean up state; failed reads
+                # reach the scheduler through get_block_ids_with_load_errors
+                # instead.
                 self._pop_done_transfers()
 
         done_recving = {
             self.transfer_id_to_request_id[id]
             for id in filter(
                 lambda id: id in self.transfer_id_to_request_id, done_recving
@@ -2314,33 +2343,40 @@ class MoRIIOConnectorWorker:
     def _pop_done_transfers(self) -> set[str]:
         done_req_ids: set[str] = set()
         _xfer_timeout = int(os.environ.get("VLLM_MORIIO_TRANSFER_TIMEOUT_S", "120"))
         with self.moriio_wrapper.lock:
             to_remove = []
             for req_id, status_list in self._recving_transfers.items():
-                last = status_list[-1]
-                if last.Succeeded():
+                # Every layer's read must land before the request is complete,
+                # so the whole batch is polled rather than just the newest
+                # status.
+                state = self.moriio_wrapper.poll_transfer_batch(status_list)
+                if state is TransferBatchState.DONE:
                     host, port, xfer_id = self._recving_transfers_callback_addr[req_id]
                     done_req_ids.add(xfer_id)
                     self.moriio_wrapper.send_notify(
                         xfer_id,
                         host,
                         port,
                         message_type="release",
                         message_fields={"consumer_tp_size": self.world_size},
                     )
                     to_remove.append(req_id)
-                elif last.Failed():
+                elif state is TransferBatchState.FAILED:
+                    failed = next(
+                        (status for status in status_list if status.Failed()), None
+                    )
                     logger.error(
                         "RDMA transfer failed for request %s: %s (code=%s). "
-                        "Notifying prefill to free blocks; request will be "
-                        "aborted by timeout.",
+                        "Notifying prefill to free blocks and reporting the "
+                        "load error so the scheduler can recover the request.",
                         req_id,
-                        last.Message(),
-                        last.Code(),
+                        failed.Message() if failed is not None else "unknown",
+                        failed.Code() if failed is not None else "unknown",
                     )
+                    self._record_failed_recv(req_id)
                     host, port, xfer_id = self._recving_transfers_callback_addr[req_id]
                     try:
                         self.moriio_wrapper.send_notify(
                             xfer_id,
                             host,
                             port,
@@ -2350,14 +2386,16 @@ class MoRIIOConnectorWorker:
                     except Exception:
                         logger.exception(
                             "Failed to send error notification for request %s",
                             req_id,
                         )
                     to_remove.append(req_id)
-                    # Do NOT add to done_req_ids: decode KV cache is incomplete.
-                    # The request will expire via the normal request timeout.
+                    # Deliberately not in done_req_ids: the decode KV is
+                    # incomplete, so this is a load error rather than a
+                    # completion. _record_failed_recv above routes it to
+                    # get_finished/get_block_ids_with_load_errors instead.
                 elif req_id in self._recving_transfers_start:
                     # Abort still-in-flight transfers that exceed the
                     # configured deadline. Otherwise a lost RDMA
                     # completion would leave the decode worker hung
                     # indefinitely on this request.
                     _age = time.monotonic() - self._recving_transfers_start[req_id]
@@ -2366,20 +2404,74 @@ class MoRIIOConnectorWorker:
                             "RDMA read TIMED OUT for req %s after %.1fs "
                             "(VLLM_MORIIO_TRANSFER_TIMEOUT_S=%d)",
                             req_id,
                             _age,
                             _xfer_timeout,
                         )
+                        self._record_failed_recv(req_id)
                         to_remove.append(req_id)
             for req_id in to_remove:
                 del self._recving_transfers[req_id]
                 del self._recving_transfers_callback_addr[req_id]
                 self._recving_transfers_start.pop(req_id, None)
+                self._recving_local_blocks.pop(req_id, None)
 
             return done_req_ids
 
+    def _await_reads_issued_this_step(self) -> None:
+        """Block until this step's reads land, honouring load_kv_async=False.
+
+        The sync contract promises the KV is in place by the time the forward
+        runs, and this is the only place that can keep it: the reads are posted
+        non-blocking and wait_for_layer_load is a no-op, so without this the
+        forward read blocks the NIC was still filling, and delay_cache_blocks
+        (== load_kv_async) published those same blocks into the prefix cache in
+        the same step.
+
+        Waiting for the whole step rather than per layer is the conservative
+        shape: wait_for_layer_load could overlap the forward's early layers with
+        later layers' transfers, but it needs a layer-indexed status map and a
+        per-layer deadline. Correctness first.
+
+        A read that fails or times out is reported through
+        get_block_ids_with_load_errors rather than raised, so one bad transfer
+        costs its own request instead of the engine.
+        """
+        statuses = self._reads_issued_this_step
+        self._reads_issued_this_step = []
+        if not statuses:
+            return
+        try:
+            self.moriio_wrapper.waiting_for_transfer_complete(statuses)
+        except TransferError:
+            logger.exception(
+                "MoRIIO reads did not complete before the forward; the "
+                "affected requests are reported as load errors."
+            )
+
+    def _record_failed_recv(self, req_id: ReqId) -> None:
+        """Hand a failed or timed-out read to the scheduler's recovery path.
+
+        Mirrors the block-marking half of NIXL's _handle_failed_transfer: the
+        destination blocks go to get_block_ids_with_load_errors, so the
+        scheduler recomputes the affected prefix instead of running on whatever
+        the NIC left behind. Previously a failed read was only logged and the
+        request was left to expire on a timeout.
+
+        Skipped for hybrid models, as NIXL skips it when HMA is required: the
+        block-id channel carries [attn, mamba] halves and the recurrent-state
+        slots do not map onto the scheduler's token-prefix recovery.
+        """
+        if not self._has_mamba:
+            self._invalid_block_ids.update(self._recving_local_blocks.get(req_id, ()))
+
+    def get_block_ids_with_load_errors(self) -> set[int]:
+        """Drain the blocks whose read failed, for the scheduler to recompute."""
+        invalid, self._invalid_block_ids = self._invalid_block_ids, set()
+        return invalid
+
     def save_kv_layer(
         self,
         metadata: MoRIIOConnectorMetadata,
         layer_name: str,
         kv_layer: torch.Tensor,
         attn_metadata: "AttentionMetadata | None",
@@ -2635,17 +2727,18 @@ class MoRIIOConnectorWorker:
                             req_id, remote_engine_id, meta
                         )
                         wait_handshake_readd_req = True
 
                         continue
 
-            # Handshake already completed, start async read xfer.
+            # Handshake already completed, start the read xfer.
             self._read_blocks_for_req(req_id, meta)
         # Start transfers for requests whose handshakes have now finished.
 
         if remote_engine_id is None and not wait_handshake_readd_req:
+            self._await_reads_issued_this_step()
             return
         _deadline = time.monotonic() + self.moriio_config.transfer_timeout
         while True:
             if (
                 self._ready_requests.empty()
                 and remote_engine_id not in self.load_ready_flag
@@ -2666,12 +2759,13 @@ class MoRIIOConnectorWorker:
             ):
                 self._read_blocks_for_req(*self._ready_requests.get_nowait())
                 break
             else:
                 break
 
+        self._await_reads_issued_this_step()
         self._reqs_to_send.update(metadata.reqs_to_send)
 
     def wait_for_save(self, metadata: MoRIIOConnectorMetadata):
         if self.mode == MoRIIOMode.WRITE and self.is_producer:
             for layer_name, kv_layer in self.kv_caches.items():
                 self.save_kv_layer(metadata, layer_name, kv_layer, None)
@@ -3160,12 +3254,16 @@ class MoRIIOConnectorWorker:
                         _sq_deadline,
                     )
                 )
             with self.moriio_wrapper.lock:
                 self._recving_transfers[request_id].extend(statuses)
                 self._recving_transfers_start.setdefault(request_id, time.monotonic())
+                # Destination blocks, kept for _record_failed_recv.
+                self._recving_local_blocks.setdefault(request_id, list(local_attn))
+                # Awaited at the end of start_load_kv, before the forward.
+                self._reads_issued_this_step.extend(statuses)
                 self._recving_transfers_callback_addr[request_id] = (
                     remote_host,
                     str(
                         remote_notify_port
                         + get_port_offset(
                             int(remote_dp_rank),
diff --git a/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py b/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py
--- a/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py
+++ b/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py
@@ -27,12 +27,13 @@ from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
     LayerTransferPlan,
     MoRIIOAgentMetadata,
     MoRIIOConstants,
     MoRIIOError,
     MoRIIOTransferAck,
     RemoteAllocInfo,
+    TransferBatchState,
     TransferError,
     TransferId,
     WriteTask,
     get_port_offset,
     get_role,
     zmq_ctx,
@@ -56,12 +57,20 @@ try:
     )
 
     logger.info("MoRIIO is available")
 except ImportError:
     logger.error("MoRIIO is not available")
 
+try:
+    from mori.io import StatusCode
+except ImportError:
+    # Builds older than ROCm/mori#341 expose neither StatusCode nor
+    # IOEngine.WaitAll. sglang's MI35x CI broke on exactly that, so the Python
+    # polling path has to stay reachable.
+    StatusCode = None
+
 
 """Write task execution logic for MoRIIO connector."""
 
 
 _MAX_TERMINAL_TRANSFER_IDS = 4096
 
@@ -613,12 +622,13 @@ class MoRIIOWrapper:
         self.done_write_cache_req_ids: list[str] = []
         self._terminal_transfer_ids: OrderedDict[TransferId, None] = OrderedDict()
         self._transfer_timeout = transfer_timeout
         self.notify_thread: threading.Thread | None = None
         self.sessions: list[IOEngine.Session] = []
         self.paths: dict[str, zmq.Socket] = {}
+        self._wait_all_supported: bool | None = None
 
     def set_moriio_engine(self, moriio_engine):
         assert moriio_engine is not None, (
             "You Cannot pass None engine to MoRIIOWrapper!"
         )
         self.moriio_engine = moriio_engine
@@ -730,23 +740,116 @@ class MoRIIOWrapper:
             remote_offset,
             transfer_size_byte,
             self.moriio_engine.allocate_transfer_uid(),
         )
         return transfer_status
 
+    def _batch_wait_available(self) -> bool:
+        """True if this mori build exposes the batched wait (ROCm/mori#341).
+
+        Probed once and cached. An image pinning an older mori raises
+        AttributeError on the first transfer instead -- which is how this same
+        API broke sglang's MI35x CI -- so the polling path stays reachable.
+        """
+        if self._wait_all_supported is None:
+            self._wait_all_supported = StatusCode is not None and hasattr(
+                self.moriio_engine, "wait_all"
+            )
+            logger.info(
+                "MoRIIO batched transfer wait (wait_all) available: %s",
+                self._wait_all_supported,
+            )
+        return self._wait_all_supported
+
+    def poll_transfer_batch(self, transfer_statuses: list[Any]) -> TransferBatchState:
+        """Non-blocking verdict over every status of one request.
+
+        Checking only the newest status calls a request done while an earlier
+        layer's read is unfinished or already failed -- and
+        _post_read_with_backoff deliberately hands back a failed status when the
+        send queue never drains, so failures do land on layers other than the
+        last.
+
+        Kept as a Python scan on purpose even where wait_all exists: mori's
+        zero-timeout wait runs PollProgress on the calling thread, which would
+        drive the backend's progress callback from the engine thread alongside
+        its own CQ poller. The semantics here are identical and it introduces no
+        new concurrency, which matters while we are chasing a race.
+        """
+        state = TransferBatchState.DONE
+        for status in transfer_statuses:
+            if status.Failed():
+                return TransferBatchState.FAILED
+            if not status.Succeeded():
+                state = TransferBatchState.PENDING
+        return state
+
     def waiting_for_transfer_complete(self, transfer_statuses: list[Any] | None = None):
         if transfer_statuses is None:
             with self.lock:
                 transfers_to_wait = self.transfer_status[:]
                 self.transfer_status.clear()
         else:
             transfers_to_wait = list(transfer_statuses)
 
         if not transfers_to_wait:
             return
 
+        if self._batch_wait_available():
+            self._wait_all_transfers(transfers_to_wait)
+        else:
+            self._poll_transfers_until_done(transfers_to_wait)
+
+    def _wait_all_transfers(self, transfers_to_wait: list[Any]) -> None:
+        """Block on the whole batch inside mori, with the GIL released.
+
+        This is the blocking path only, where the benefit is concrete and the
+        risk is not: start_load_kv now waits here on the forward thread, so the
+        Python fallback would spin at 1 ms holding the GIL for the full transfer
+        and starve the very threads it waits on. mori blocks on a condition
+        variable instead, shares one deadline across the batch, and gives
+        failure precedence over still-in-flight.
+        """
+        timeout = self._transfer_timeout
+        # timeout_ms == 0 means "poll once" to mori, so never round down to it.
+        timeout_ms = max(int(timeout * 1000), 1)
+        rc = self.moriio_engine.wait_all(transfers_to_wait, timeout_ms=timeout_ms)
+        if rc == StatusCode.SUCCESS:
+            return
+        raise TransferError(self._describe_transfer_failure(transfers_to_wait, rc))
+
+    def _describe_transfer_failure(self, transfers_to_wait: list[Any], rc) -> str:
+        """Recover the per-status detail a batch return code loses.
+
+        Runs only on the failure path, so it costs nothing in steady state.
+        """
+        timeout = self._transfer_timeout
+        errors: list[str] = []
+        for status in transfers_to_wait:
+            if status.Succeeded():
+                continue
+            if status.Failed():
+                errors.append(
+                    f"RDMA transfer failed: {status.Message()} "
+                    f"(code={status.Code()})"
+                )
+            else:
+                errors.append(
+                    f"RDMA transfer timed out after {timeout:.0f}s. "
+                    f"Check NIC queue depth or reduce concurrency; "
+                    f"adjust with kv_connector_extra_config.transfer_timeout."
+                )
+        if not errors:
+            errors.append(f"batch wait returned {rc} with every status terminal")
+        return (
+            f"{len(errors)}/{len(transfers_to_wait)} transfers failed "
+            f"(batch wait rc={rc}):\n" + "\n".join(errors)
+        )
+
+    def _poll_transfers_until_done(self, transfers_to_wait: list[Any]) -> None:
+        """Fallback for mori builds without the batched wait."""
         timeout = self._transfer_timeout
         deadline = time.monotonic() + timeout
         remaining = list(transfers_to_wait)
         errors: list[str] = []
 
         while remaining:
DIFF_MORIIO_RACE
if grep -qF "_await_reads_issued_this_step" "$MORIIO_CONNECTOR" 2>/dev/null; then
  echo "[k3-moriio] race patch: already present (skip)"
elif ( cd "$ROOT" && git apply -p1 --whitespace=nowarn "$WS_RACE/MORIIO_RACE.diff" ) 2>/dev/null; then
  echo "[k3-moriio] race patch: APPLIED (git apply)"
elif patch -p1 -d "$ROOT" --fuzz=3 --forward --no-backup-if-mismatch \
       < "$WS_RACE/MORIIO_RACE.diff" >/dev/null 2>&1; then
  echo "[k3-moriio] race patch: APPLIED (patch)"
else
  echo "[k3-moriio] ERROR: race patch did not apply; refusing to serve a half-patched tree"
  ( cd "$ROOT" && git apply -p1 --check "$WS_RACE/MORIIO_RACE.diff" ) 2>&1 | head -20
  exit 1
fi
find "$ROOT" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null

echo "[k3-moriio] verify"
for pair in \
  "moriio_connector.py:_draft_only_layers" \
  "moriio_connector.py:_await_reads_issued_this_step" \
  "moriio_connector.py:get_block_ids_with_load_errors" \
  "moriio_engine.py:def poll_transfer_batch" \
  "moriio_connector.py:attention layers carry their own" \
  "moriio_layout.py:def build_dcp_block_pairing|def compute_block_transfer_offsets" \
  "kv_cache_utils.py:KV cache grouping: buckets" \
  "rejection_sampler_utils.py:num_sampling_positions|clamp"
do
  f="${pair%%:*}"; pat="${pair#*:}"
  path=$(find "$ROOT/vllm" -name "$f" | head -1)
  n=$(grep -cE "$pat" "$path" 2>/dev/null || echo 0)
  echo "  $f  /$pat/ = $n"
done

python3 -m py_compile \
  "$ROOT/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py" \
  "$ROOT/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py" \
  "$ROOT/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py" \
  "$ROOT/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_layout.py" \
  "$ROOT/vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py" \
  "$ROOT/vllm/v1/core/kv_cache_utils.py" \
  "$ROOT/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py" \
  && echo "[k3-moriio] PY_COMPILE_OK" || { echo "[k3-moriio] PY_COMPILE_FAIL"; exit 1; }

python3 - <<'PYEOF'
try:
    from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import MoRIIOConnector  # noqa: F401
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector  # noqa: F401
    print("[k3-moriio] IMPORT_OK")
except Exception as e:
    print("[k3-moriio] IMPORT_SKIPPED (needs GPU?):", type(e).__name__, e)
PYEOF
