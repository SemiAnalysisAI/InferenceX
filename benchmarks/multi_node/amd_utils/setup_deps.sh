#!/bin/bash
# =============================================================================
# setup_deps.sh — Install missing disagg dependencies at container start.
#
# Dispatched by $ENGINE (set by server.sh dispatcher):
#   vllm-disagg   -> recipe deps + amd-quark + UCX/RIXL path exports
#                    (base image: vllm/vllm-openai-rocm:nightly)
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
# SGLang: prevent TP-rank collective desync deadlock in disaggregation prefill.
#
# resolve_waiting_queue_bootstrap() runs poll_and_all_reduce_attn_cp_tp_group()
# over `candidates`. The upstream candidate set (all non-aborted waiting reqs)
# can differ across TP ranks, so some ranks enter the all_reduce while others
# skip it -> hang. Narrow candidates to optimistic (pending_bootstrap) requests,
# which is consistent across ranks and is the only set finalize_bootstrap acts on.
# ---------------------------------------------------------------------------
patch_disagg_prefill_bootstrap_desync() {
    python3 -c '
import os, sys

target = "/sgl-workspace/sglang/python/sglang/srt/disaggregation/prefill.py"
if not os.path.isfile(target):
    print("[SETUP] disaggregation/prefill.py not found, skipping")
    sys.exit(0)

src = open(target).read()

old = "        candidates = [req for req in self.waiting_queue if not is_aborted(req)]"
new = (
    "        candidates = [\n"
    "            req\n"
    "            for req in self.waiting_queue\n"
    "            if req.pending_bootstrap and not is_aborted(req)\n"
    "        ]"
)

if new in src:
    print("[SETUP] prefill bootstrap-desync patch already applied")
    sys.exit(0)

if old not in src:
    print("[SETUP] WARN: resolve_waiting_queue_bootstrap pattern not found — sglang version may have changed")
    sys.exit(0)

open(target, "w").write(src.replace(old, new))
print("[SETUP] Patched: disaggregation/prefill.py resolve_waiting_queue_bootstrap candidates")
'
    _SETUP_INSTALLED+=("prefill-bootstrap-desync-fix")
}

# ---------------------------------------------------------------------------
# SGLang: tp-group agreement for disagg decode queue (decode_tp_queue_agree).
#
# The metadata gate (poll_and_all_reduce) issues a tp-group collective whose
# shape/order == self.queue. Per-rank prealloc/enqueue timing can leave ranks
# with different queue membership/order, desyncing the collective -> decode
# hang (JID-17417 / JID-17445). This inserts _agree_and_order_queue() so every
# rank polls an identical, identically-ordered subset, and replaces the
# rank-divergent retracted-queue early return in process_decode_queue with a
# rank-symmetric MAX all_reduce. Mirrors patches/decode_tp_queue_agree.patch.
# ---------------------------------------------------------------------------
patch_decode_tp_queue_agree() {
    python3 - <<'PYEOF'
import os, sys

target = "/sgl-workspace/sglang/python/sglang/srt/disaggregation/decode.py"
if not os.path.isfile(target):
    print("[SETUP] disaggregation/decode.py not found, skipping")
    sys.exit(0)

src = open(target).read()

if "_agree_and_order_queue" in src:
    print("[SETUP] decode tp-queue-agree patch already applied")
    sys.exit(0)

# --- Hunk 1+2: insert agreement method + gate pop_transferred -------------
old1 = (
    "    def pop_transferred(self, rids_to_check: Optional[List[str]] = None) -> List[Req]:\n"
    "        if not self.queue:\n"
    "            return []\n"
)
new1 = '''    def _agree_and_order_queue(self) -> Tuple[List["DecodeRequest"], List["DecodeRequest"]]:
        """Split self.queue into (gated, deferred) using a tp-group agreement.

        The metadata gate (utils.poll_and_all_reduce) issues a tp-group all_reduce
        whose tensor shape == len(self.queue) and whose per-index meaning is the
        i-th queued request. That is only correct if every tp rank polls an
        IDENTICAL queue (same requests, same order). Per-rank prealloc/enqueue
        timing can leave queues with different membership or order, which desyncs
        the collective -> deadlock (the JID-17417 decode hang).

        We all_gather the local request ids, keep only requests present on EVERY
        rank, and order them deterministically (request ids are rank-invariant), so
        the subsequent gate runs a matching collective on all ranks. Requests not
        yet on every rank are returned as `deferred` and retried on a later call.

        NOTE: every rank that reaches pop_transferred must call this (it contains a
        collective). That holds for the same reason the existing gate is safe:
        pop_transferred is entered rank-symmetrically over self.gloo_group.
        """
        tp_size = torch.distributed.get_world_size(self.gloo_group)
        if tp_size <= 1:
            return self.queue, []

        local_rids = [dr.req.rid for dr in self.queue]
        gathered: List[Optional[List[str]]] = [None] * tp_size
        torch.distributed.all_gather_object(
            gathered, local_rids, group=self.gloo_group
        )

        common = set(gathered[0] or [])
        for rids in gathered[1:]:
            common &= set(rids or [])

        if not common:
            return [], list(self.queue)

        by_rid = {dr.req.rid: dr for dr in self.queue}
        # sorted() yields an identical order on every rank (rids are rank-invariant).
        gated = [by_rid[rid] for rid in sorted(common)]
        deferred = [dr for dr in self.queue if dr.req.rid not in common]
        return gated, deferred

    def pop_transferred(self, rids_to_check: Optional[List[str]] = None) -> List[Req]:
        # Agree on a tp-rank-identical, identically-ordered subset BEFORE issuing any
        # metadata-gate collective. Do NOT add a local `if not self.queue` early
        # return here: an empty-queue rank must still join the agreement collective,
        # otherwise it skips it and desyncs the tp group (root cause of the hang).
        self.queue, _deferred = self._agree_and_order_queue()
        if not self.queue:
            # Empty agreement is rank-symmetric: all ranks skip the gate together.
            self.queue = _deferred
            return []
'''

# --- Hunk 3: re-attach deferred reqs after removal ------------------------
old3 = (
    "            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove\n"
    "        ]\n"
    "\n"
    "        return transferred_reqs\n"
)
new3 = (
    "            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove\n"
    "        ]\n"
    "        # Re-attach requests that were not yet present on every tp rank this\n"
    "        # iteration; they are gated again on a later call once all ranks have them.\n"
    "        if _deferred:\n"
    "            self.queue.extend(_deferred)\n"
    "\n"
    "        return transferred_reqs\n"
)

# --- Hunk 4: rank-symmetric retracted-queue gate --------------------------
old4 = (
    "        if len(self.disagg_decode_prealloc_queue.retracted_queue) > 0:\n"
    "            # if there are still retracted requests, we do not allocate new requests\n"
    "            return\n"
)
new4 = '''        # PATCH(call-site tp-agreement): the retracted-queue early return below is
        # rank-divergent — retracted_queue length is per-rank (per-rank KV-cache
        # pressure). A divergent return skips the polling_count increment and the
        # pop_transferred() collectives on some ranks, permanently desyncing the tp
        # group: one rank ends up a full collective ahead, so pop_transferred's
        # _agree_and_order_queue all_gather_object on the lagging ranks lines up
        # against the metadata-gate all_reduce on the leading rank -> mismatched-
        # collective deadlock (GPU 0%, detokenizer heartbeat freeze; JID-17445).
        # process_decode_queue is called unconditionally every event-loop iteration
        # on every rank, so an all_reduce here (before any divergent return) is
        # rank-symmetric. Hold off new allocation iff ANY rank still has retracted
        # reqs, so all ranks branch identically every iteration and pop_transferred
        # is entered symmetrically.
        _agree_gg = self.disagg_decode_transfer_queue.gloo_group
        _local_retracted = (
            1 if len(self.disagg_decode_prealloc_queue.retracted_queue) > 0 else 0
        )
        if torch.distributed.get_world_size(_agree_gg) > 1:
            _retracted_t = torch.tensor([_local_retracted], dtype=torch.int32)
            torch.distributed.all_reduce(
                _retracted_t, op=torch.distributed.ReduceOp.MAX, group=_agree_gg
            )
            _any_retracted = bool(_retracted_t.item())
        else:
            _any_retracted = bool(_local_retracted)
        if _any_retracted:
            # if any rank still has retracted requests, no rank allocates new ones
            return
'''

for label, old, new in (
    ("agreement-method+pop_transferred", old1, new1),
    ("deferred-reattach", old3, new3),
    ("retracted-gate", old4, new4),
):
    if old not in src:
        print(f"[SETUP] WARN: decode.py anchor for '{label}' not found — sglang version may have changed")
        sys.exit(0)
    src = src.replace(old, new, 1)

open(target, "w").write(src)
print("[SETUP] Patched: disaggregation/decode.py tp-group decode queue agreement")
PYEOF
    _SETUP_INSTALLED+=("decode-tp-queue-agree-fix")
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

    # =========================================================================
    # vLLM: Export UCX/RIXL paths (persists since this file is sourced)
    # =========================================================================
    export ROCM_PATH="${ROCM_PATH}"
    export UCX_HOME="${UCX_HOME}"
    export RIXL_HOME="${RIXL_HOME}"
    export PATH="${UCX_HOME}/bin:/usr/local/bin/etcd:/root/.cargo/bin:${PATH}"
    export LD_LIBRARY_PATH="${UCX_HOME}/lib:${RIXL_HOME}/lib:${RIXL_HOME}/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
else
    patch_disagg_prefill_bootstrap_desync
    # patch_decode_tp_queue_agree

    install_transformers_glm5
fi

_SETUP_END=$(date +%s)
if [[ ${#_SETUP_INSTALLED[@]} -eq 0 ]]; then
    echo "[SETUP] All dependencies already present ($(( _SETUP_END - _SETUP_START ))s wallclock)"
else
    echo "[SETUP] Installed: ${_SETUP_INSTALLED[*]} in $(( _SETUP_END - _SETUP_START ))s"
fi
