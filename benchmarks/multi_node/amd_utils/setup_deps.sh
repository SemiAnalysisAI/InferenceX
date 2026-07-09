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
# SGLang: fix stale SWA ring buffer on radix-prefix reuse for DeepSeek-V4
# unified_kv backend. Mirrors upstream sgl-project/sglang#30339.
#
# The unified_kv layout keeps SWA in a per-request ring (addressed by
# req_pool_idx * window + pos % window) that is NOT content-addressed and is
# never written into the radix tree. Radix prefix reuse is safe for the
# index-addressed compressed KV, but the SWA ring slots covering a reused
# prefix still hold whatever a previous occupant of that req_pool slot wrote.
# If the decode sliding window reaches back into the reused-prefix region,
# it reads stale SWA from an unrelated request.
#
# Adds a generic BasePrefixCache.swa_reprefill_tail_tokens() hook (default 0,
# no-op) that the scheduler prefix-match paths (schedule_batch.py,
# schedule_policy.py) use to cap the radix match by the trailing sliding
# window, so that tail is re-prefilled into this request's own ring. Overrides
# the hook on SWARadixCache to return sliding_window_size only when the
# unified_kv_triton backend is active on HIP; every other layout/backend stays
# a complete no-op via the base class's 0.
#
# The HiCache/UnifiedRadixCache counterpart (unified_kv + hierarchical cache
# ON) is owned by patch_dsv4_unified_kv_hicache() below, mirroring upstream
# sgl-project/sglang#29417 — that override builds on the base hook this patch
# installs, so this patch must run first.
# ---------------------------------------------------------------------------
patch_swa_reprefill_tail_unified_kv() {
    python3 - <<'PYEOF'
import os


def patch_file(path, marker, hunks):
    if not os.path.isfile(path):
        print(f"[SETUP] {path} not found, skipping")
        return
    src = open(path).read()
    if marker in src:
        print(f"[SETUP] {os.path.basename(path)} swa-reprefill-tail patch already applied")
        return
    for label, old, new in hunks:
        if old not in src:
            print(
                f"[SETUP] WARN: {os.path.basename(path)} anchor for '{label}' not "
                "found — sglang version may have changed"
            )
            return
        src = src.replace(old, new, 1)
    open(path, "w").write(src)
    print(f"[SETUP] Patched: {os.path.basename(path)} swa-reprefill-tail")


# --- base_prefix_cache.py: generic no-op hook, shared with patch_dsv4_unified_kv_hicache() ---
patch_file(
    "/sgl-workspace/sglang/python/sglang/srt/mem_cache/base_prefix_cache.py",
    marker="def swa_reprefill_tail_tokens",
    hunks=[(
        "swa_reprefill_tail_tokens base hook",
        "    def supports_swa(self) -> bool:\n"
        "        return False\n"
        "\n"
        "    def supports_mamba(self) -> bool:\n"
        "        return False\n",
        "    def supports_swa(self) -> bool:\n"
        "        return False\n"
        "\n"
        "    def swa_reprefill_tail_tokens(self) -> int:\n"
        "        # Only cache layouts with a non-content-stable SWA ring (unified_kv)\n"
        "        # need to hold back a trailing sliding window for re-prefill; every\n"
        "        # other cache keeps SWA content-stable and overrides this where needed.\n"
        "        return 0\n"
        "\n"
        "    def supports_mamba(self) -> bool:\n"
        "        return False\n",
    )],
)

# --- schedule_batch.py: cap key_limit by the trailing sliding window ------
patch_file(
    "/sgl-workspace/sglang/python/sglang/srt/managers/schedule_batch.py",
    marker="reprefill_tail = tree_cache.swa_reprefill_tail_tokens()",
    hunks=[(
        "init_next_round_input reprefill cap",
        "        token_ids_to_match = self.full_untruncated_fill_ids\n"
        "        key_limit: Optional[int] = self._compute_max_prefix_len(input_len)\n"
        "\n"
        "        # Disable prefix caching when embed overrides are present: same token IDs\n"
        "        # with different override vectors must not share cached KV values.\n"
        "        if self.positional_embed_overrides is not None:",
        "        token_ids_to_match = self.full_untruncated_fill_ids\n"
        "        key_limit: Optional[int] = self._compute_max_prefix_len(input_len)\n"
        "\n"
        "        # unified_kv SWA lives in a per-request ring that is not content-stable\n"
        "        # and never cached in the radix tree, so a reused prefix carries stale\n"
        "        # SWA. Cap the match by the trailing sliding window so it is re-prefilled\n"
        "        # into this request's ring. No-op for other layouts (returns 0).\n"
        "        if tree_cache is not None:\n"
        "            reprefill_tail = tree_cache.swa_reprefill_tail_tokens()\n"
        "            if reprefill_tail:\n"
        "                capped = max(0, input_len - reprefill_tail)\n"
        "                key_limit = capped if key_limit is None else min(key_limit, capped)\n"
        "\n"
        "        # Disable prefix caching when embed overrides are present: same token IDs\n"
        "        # with different override vectors must not share cached KV values.\n"
        "        if self.positional_embed_overrides is not None:",
    )],
)

# --- schedule_policy.py: same cap for the non-batch match_prefix_for_req path ---
patch_file(
    "/sgl-workspace/sglang/python/sglang/srt/managers/schedule_policy.py",
    marker="reprefill_tail = tree_cache.swa_reprefill_tail_tokens()",
    hunks=[(
        "match_prefix_for_req reprefill cap",
        "    if token_ids is None:\n"
        "        token_ids = req.origin_input_ids + req.output_ids\n"
        "\n"
        "    match_result = tree_cache.match_prefix(\n"
        "        MatchPrefixParams(\n"
        "            key=RadixKey(token_ids=token_ids, extra_key=req.extra_key),\n"
        "            cow_mamba=cow_mamba,\n"
        "            req=req if include_req else None,\n"
        "        )",
        "    if token_ids is None:\n"
        "        token_ids = req.origin_input_ids + req.output_ids\n"
        "\n"
        "    # unified_kv SWA lives in a per-request ring (not content-stable, never cached\n"
        "    # in the radix tree), so a reused prefix carries stale SWA. Cap the match by the\n"
        "    # trailing sliding window so it is re-prefilled. No-op for other layouts.\n"
        "    reprefill_tail = tree_cache.swa_reprefill_tail_tokens()\n"
        "    key_limit = max(0, len(token_ids) - reprefill_tail) if reprefill_tail else None\n"
        "\n"
        "    match_result = tree_cache.match_prefix(\n"
        "        MatchPrefixParams(\n"
        "            key=RadixKey(token_ids=token_ids, extra_key=req.extra_key, limit=key_limit),\n"
        "            cow_mamba=cow_mamba,\n"
        "            req=req if include_req else None,\n"
        "        )",
    )],
)

# --- swa_radix_cache.py: SWARadixCache override (plain radix, HiCache off) ---
patch_file(
    "/sgl-workspace/sglang/python/sglang/srt/mem_cache/swa_radix_cache.py",
    marker="def swa_reprefill_tail_tokens",
    hunks=[(
        "SWARadixCache swa_reprefill_tail_tokens override",
        "        ), \"sliding_window_size must be set for SWARadixCache\"\n"
        "        return True\n"
        "\n"
        "    def reset(self) -> None:\n"
        "        self.root_node = TreeNode()\n"
        "        self.root_node.key = []",
        "        ), \"sliding_window_size must be set for SWARadixCache\"\n"
        "        return True\n"
        "\n"
        "    def swa_reprefill_tail_tokens(self) -> int:\n"
        "        \"\"\"Tokens at the tail of a matched prefix that must NOT be reused.\n"
        "\n"
        "        The DeepSeek-V4 unified_kv layout keeps SWA in a per-request ring\n"
        "        (addressed by ``req_pool_idx * window + pos % window``), which is NOT\n"
        "        content-stable and is never stored in the radix tree. A reused prefix\n"
        "        therefore carries another request's stale SWA in the ring. Hold back the\n"
        "        trailing sliding window from the match so it gets re-prefilled into THIS\n"
        "        request's ring, making the decode window read freshly-written data.\n"
        "\n"
        "        No-op (0) for the index-addressed SWA pool, whose slots are\n"
        "        content-stable and safe to reuse.\n"
        "        \"\"\"\n"
        "        from sglang.srt.layers.attention.dsv4.unified_kv_kernels.env_gate import (\n"
        "            is_unified_kv_triton,\n"
        "        )\n"
        "\n"
        "        if self.sliding_window_size and is_unified_kv_triton():\n"
        "            return self.sliding_window_size\n"
        "        return 0\n"
        "\n"
        "    def reset(self) -> None:\n"
        "        self.root_node = TreeNode()\n"
        "        self.root_node.key = []",
    )],
)
PYEOF
    _SETUP_INSTALLED+=("swa-reprefill-tail-unified-kv-fix")
}

# ---------------------------------------------------------------------------
# SGLang: enable unified-KV HiCache on DeepSeek-V4. Mirrors upstream
# sgl-project/sglang#29417.
#
# Builds on patch_swa_reprefill_tail_unified_kv()'s generic
# swa_reprefill_tail_tokens() hook (must run first) and adds the
# HiCache/UnifiedRadixCache counterpart: when unified_kv is paired with
# --enable-hierarchical-cache (compress-only HiCache, no separate SWA host
# pool), the SWA ring is still per-request/non-content-stable, so
# UnifiedRadixCache gets its own override. This is mutually exclusive with
# SWARadixCache's override (plain radix, HiCache off) — no functional overlap.
#
# Also updates DeepSeekV4UnifiedKVPool/hybrid_pool_assembler so the
# compressed (C4/C128) region of the unified buffer can be paged into a
# HiCache host pool (previously unified_kv forced HiCache off entirely via
# _resolve_unified_kv_hicache_compatibility(), which this patch removes), and
# gates the SWA match validator / hicache transfer builder to skip the (now
# nonexistent) separate SWA host pool for unified_kv.
#
# Only relevant when unified_kv + HiCache are both active; gated by
# KV_OFFLOADING=dram + KV_OFFLOAD_BACKEND=hicache + unified_kv_triton so the
# container's sglang install is left untouched otherwise.
# ---------------------------------------------------------------------------
patch_dsv4_unified_kv_hicache() {
    if [[ "${KV_OFFLOADING:-none}" == "none" || "${KV_OFFLOAD_BACKEND:-}" != "hicache" || "${SGLANG_HACK_FLASHMLA_BACKEND:-}" != "unified_kv_triton" ]]; then
        return 0
    fi
    echo "[SETUP] Patching sglang for unified_kv + HiCache (DeepSeek-V4) ..."
    python3 - <<'PYEOF'
import os


def patch_file(path, marker, hunks, optional=False):
    if not os.path.isfile(path):
        print(f"[SETUP] {path} not found, skipping")
        return
    src = open(path).read()
    if marker is not None and marker in src:
        print(f"[SETUP] {os.path.basename(path)} unified_kv-hicache patch already applied")
        return
    changed = False
    for label, old, new in hunks:
        if old not in src:
            level = "INFO" if optional else "WARN"
            print(
                f"[SETUP] {level}: {os.path.basename(path)} anchor for '{label}' not "
                "found — sglang version may have changed"
            )
            if optional:
                continue
            return
        src = src.replace(old, new, 1)
        changed = True
    if not changed:
        print(f"[SETUP] {os.path.basename(path)} unified_kv-hicache patch already applied")
        return
    open(path, "w").write(src)
    print(f"[SETUP] Patched: {os.path.basename(path)} unified_kv-hicache")


# --- unified_radix_cache.py: UnifiedRadixCache override (unified_kv + HiCache) ---
patch_file(
    "/sgl-workspace/sglang/python/sglang/srt/mem_cache/unified_radix_cache.py",
    marker="def swa_reprefill_tail_tokens",
    hunks=[(
        "UnifiedRadixCache swa_reprefill_tail_tokens override",
        "        swa = self.components.get(ComponentType.SWA)\n"
        "        return swa.sliding_window_size if swa else None\n"
        "\n"
        "    def supports_swa(self) -> bool:\n"
        "        return ComponentType.SWA in self.components\n",
        "        swa = self.components.get(ComponentType.SWA)\n"
        "        return swa.sliding_window_size if swa else None\n"
        "\n"
        "    def swa_reprefill_tail_tokens(self) -> int:\n"
        "        \"\"\"\n"
        "        Only unified_kv + HiCache needs this: SWA lives in a per-request ring\n"
        "        (state_slot/pos), not content-stable and never offloaded to host, so a\n"
        "        reused prefix's trailing sliding window would read another request's\n"
        "        stale ring slots. Re-prefilling that window rewrites this request's ring\n"
        "        (what plain radix reuse does via its SWA match gate). 0 for every other\n"
        "        layout.\n"
        "        \"\"\"\n"
        "        swa = self.components.get(ComponentType.SWA)\n"
        "        unified_compress_only_hicache = (\n"
        "            self.cache_controller is not None\n"
        "            and swa is not None\n"
        "            and swa._swa_kv_pool_host is None\n"
        "        )\n"
        "        return swa.sliding_window_size if unified_compress_only_hicache else 0\n"
        "\n"
        "    def supports_swa(self) -> bool:\n"
        "        return ComponentType.SWA in self.components\n",
    )],
)

# --- server_args.py: remove the unified_kv+HiCache incompatibility fallback ---
patch_file(
    "/sgl-workspace/sglang/python/sglang/srt/server_args.py",
    marker=None,
    optional=True,
    hunks=[(
        "drop _resolve_unified_kv_hicache_compatibility call+method",
        "        # Step 2: Storage-layout normalization without changing io backend.\n"
        "        self._resolve_storage_layout_compatibility()\n"
        "\n"
        "        # Step 3: HiCache is not yet supported with the DeepSeek-V4 hip unified_kv\n"
        "        # layout, so fall back to the default tilelang FlashMLA backend.\n"
        "        self._resolve_unified_kv_hicache_compatibility()\n"
        "\n"
        "    def _resolve_unified_kv_hicache_compatibility(self):\n"
        "        # The DeepSeek-V4 unified_kv layout (SGLANG_HACK_FLASHMLA_BACKEND=\n"
        "        # unified_kv_triton) keeps swa/c4/c128 in a single per-layer buffer and\n"
        "        # has no HiCache host-pool support yet, so reset the backend to the\n"
        "        # default (tilelang) so the server still starts.\n"
        "        if not self.enable_hierarchical_cache:\n"
        "            return\n"
        "\n"
        "        if envs.SGLANG_HACK_FLASHMLA_BACKEND.get() == \"unified_kv_triton\":\n"
        "            envs.SGLANG_HACK_FLASHMLA_BACKEND.set(\"tilelang\")\n"
        "            logger.warning(\n"
        "                \"SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton is not yet \"\n"
        "                \"compatible with --enable-hierarchical-cache; falling back to \"\n"
        "                \"SGLANG_HACK_FLASHMLA_BACKEND=tilelang.\"\n"
        "            )\n"
        "\n"
        "    def _resolve_layout_io_compatibility(self):",
        "        # Step 2: Storage-layout normalization without changing io backend.\n"
        "        self._resolve_storage_layout_compatibility()\n"
        "\n"
        "    def _resolve_layout_io_compatibility(self):",
    )],
)

# --- deepseek_v4_memory_pool.py: pad compressed region + per-layer H->D wait ---
mp_path = "/sgl-workspace/sglang/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py"
if os.path.isfile(mp_path):
    src = open(mp_path).read()
    if "def unified_region_buffers" in src:
        print("[SETUP] deepseek_v4_memory_pool.py unified_kv-hicache patch already applied")
    else:
        hunks = [
            (
                "class docstring padded_compress_rows",
                "class DeepSeekV4UnifiedKVPool:\n"
                "    \"\"\"\n"
                "    Layout:\n"
                "    unified_kv[L]: ``[swa_pages + compress_pages, head_dim]`` bf16\n"
                "    - rows ``[0, swa_pages)``   = SWA ring (``req_pool_indices * swa_window + pos % swa_window``)\n"
                "    - rows ``[swa_pages, ...)`` = compressed (``swa_pages + page_index``)\n"
                "    \"\"\"",
                "class DeepSeekV4UnifiedKVPool:\n"
                "    \"\"\"\n"
                "    Layout:\n"
                "    unified_kv[L]: ``[swa_pages + padded_compress_rows, head_dim]`` bf16\n"
                "    - rows ``[0, swa_pages)``   = SWA ring (``req_pool_indices * swa_window + pos % swa_window``)\n"
                "    - rows ``[swa_pages, ...)`` = compressed (``swa_pages + page_index``)\n"
                "    \"\"\"",
            ),
            (
                "__init__ signature page_size param",
                "        stage_ratios: List[int],\n"
                "        num_slots: int,\n"
                "        num_blocks: int,\n"
                "        qk_nope_head_dim: int,\n"
                "        qk_rope_head_dim: int,\n"
                "        device: str,",
                "        stage_ratios: List[int],\n"
                "        num_slots: int,\n"
                "        num_blocks: int,\n"
                "        page_size: int,\n"
                "        qk_nope_head_dim: int,\n"
                "        qk_rope_head_dim: int,\n"
                "        device: str,",
            ),
            (
                "__init__ body self.page_size",
                "        self.num_slots = num_slots\n"
                "        self.swa_pages = num_slots * self.swa_ring_size\n"
                "        self.num_blocks = num_blocks\n"
                "        self.k_per_block = dict(self.K_PER_BLOCK)",
                "        self.num_slots = num_slots\n"
                "        self.swa_pages = num_slots * self.swa_ring_size\n"
                "        self.num_blocks = num_blocks\n"
                "        self.page_size = page_size\n"
                "        self.k_per_block = dict(self.K_PER_BLOCK)",
            ),
            (
                "buffer alloc padded_compress_rows",
                "                for ratio in stage_ratios:\n"
                "                    compress_pages = self.num_blocks * self.k_per_block[ratio]\n"
                "                    bufs.append(\n"
                "                        torch.zeros(\n"
                "                            self.swa_pages + compress_pages,\n"
                "                            self.head_dim,\n"
                "                            dtype=torch.bfloat16,\n"
                "                            device=device,",
                "                for ratio in stage_ratios:\n"
                "                    # Pad by one extra page. The KV pool reserves a null slot\n"
                "                    # (token indices run 1..size).\n"
                "                    compress_rows = self.num_blocks * self.k_per_block[ratio]\n"
                "                    rows_per_page = self.page_size // ratio if ratio else 0\n"
                "                    padded_compress_rows = compress_rows + rows_per_page\n"
                "                    bufs.append(\n"
                "                        torch.zeros(\n"
                "                            self.swa_pages + padded_compress_rows,\n"
                "                            self.head_dim,\n"
                "                            dtype=torch.bfloat16,\n"
                "                            device=device,",
            ),
            (
                "pass page_size to DeepSeekV4UnifiedKVPool(...)",
                "                stage_ratios=stage_ratios,\n"
                "                num_slots=self.num_req_slots,\n"
                "                num_blocks=self.c128_size,\n"
                "                qk_nope_head_dim=qk_nope_head_dim,\n"
                "                qk_rope_head_dim=qk_rope_head_dim,\n"
                "                device=device,",
                "                stage_ratios=stage_ratios,\n"
                "                num_slots=self.num_req_slots,\n"
                "                num_blocks=self.c128_size,\n"
                "                page_size=page_size,\n"
                "                qk_nope_head_dim=qk_nope_head_dim,\n"
                "                qk_rope_head_dim=qk_rope_head_dim,\n"
                "                device=device,",
            ),
            (
                "get_unified_kv wait_layer_transfer",
                "    def get_unified_kv(self, layer_id: int) -> torch.Tensor:\n"
                "        return self.unified_kv_pool.get_unified_kv(layer_id - self._stage_start)",
                "    def get_unified_kv(self, layer_id: int) -> torch.Tensor:\n"
                "        # Under HiCache the compressed region is loaded H->D per layer; wait for this\n"
                "        # layer's transfer before attention reads it. No-op when HiCache is off.\n"
                "        self.wait_layer_transfer(layer_id)\n"
                "        return self.unified_kv_pool.get_unified_kv(layer_id - self._stage_start)",
            ),
            (
                "get_contiguous_buf_infos comment update (cosmetic)",
                "        if self._unified_kv:\n"
                "            # Unified buffer per layer: [swa_pages + compress_pages, head_dim].\n"
                "            # Compressed region [swa_pages:] is page-contiguous (row swa_pages +",
                "        if self._unified_kv:\n"
                "            # Unified buffer per layer: [swa_pages + padded_compress_rows, head_dim].\n"
                "            # Compressed region [swa_pages:] is page-contiguous (row swa_pages +",
            ),
            (
                "insert unified_region_buffers before get_state_buf_infos",
                "            item_lens.append(row_bytes)\n"
                "        return data_ptrs, data_lens, item_lens\n"
                "\n"
                "    def get_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:",
                "            item_lens.append(row_bytes)\n"
                "        return data_ptrs, data_lens, item_lens\n"
                "\n"
                "    def unified_region_buffers(self, ratio: int) -> Tuple[List[torch.Tensor], int]:\n"
                "        \"\"\"\n"
                "        In unified_kv, swa/c4/c128 share one buffer with one slot per row. But the\n"
                "        HiCache host pool transfers a whole page per indexed row, so we reshape the\n"
                "        compressed region into the layout it expects: skip the SWA segment, reshape to\n"
                "        one row per page, then cast to uint8.\n"
                "        \"\"\"\n"
                "        assert self._unified_kv, \"unified_region_buffers requires unified_kv layout\"\n"
                "        assert ratio in (4, 128), f\"unsupported compression ratio: {ratio}\"\n"
                "\n"
                "        swa_pages = self.unified_kv_pool.swa_pages\n"
                "        head_dim = self.unified_kv_pool.head_dim\n"
                "        rows_per_page = self.page_size // ratio\n"
                "        stage_ratios = self.compression_ratios[self._stage_start : self._stage_end]\n"
                "        local_layer_ids = [i for i, r in enumerate(stage_ratios) if r == ratio]\n"
                "\n"
                "        views: List[torch.Tensor] = []\n"
                "        for local_layer_id in local_layer_ids:\n"
                "            buf = self.unified_kv_pool.kv_buffer[local_layer_id]\n"
                "            compress_rows = buf.shape[0] - swa_pages\n"
                "            assert compress_rows % rows_per_page == 0, (\n"
                "                f\"compressed rows {compress_rows} not a multiple of \"\n"
                "                f\"rows_per_page {rows_per_page} for ratio {ratio}\"\n"
                "            )\n"
                "            num_pages = compress_rows // rows_per_page\n"
                "            page_view = (\n"
                "                buf.narrow(0, swa_pages, compress_rows)\n"
                "                .reshape(num_pages, rows_per_page * head_dim)\n"
                "                .view(torch.uint8)\n"
                "            )\n"
                "            views.append(page_view)\n"
                "\n"
                "        item_bytes = (\n"
                "            rows_per_page * head_dim * self.unified_kv_pool.kv_buffer[0].element_size()\n"
                "        )\n"
                "        return views, item_bytes\n"
                "\n"
                "    def get_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:",
            ),
        ]
        for label, old, new in hunks:
            if old not in src:
                print(
                    f"[SETUP] WARN: deepseek_v4_memory_pool.py anchor for '{label}' not "
                    "found — sglang version may have changed"
                )
                break
            src = src.replace(old, new, 1)
        else:
            open(mp_path, "w").write(src)
            print("[SETUP] Patched: deepseek_v4_memory_pool.py unified_kv-hicache")
else:
    print(f"[SETUP] {mp_path} not found, skipping")

# --- hybrid_pool_assembler.py: assemble the HiCache pool stack for unified_kv ---
# NOTE: this file's hunks are the largest/riskiest reconstruction in this patch
# (upstream restructures ~150 lines, moving blocks into new conditionals). Each
# hunk below is applied independently and will WARN-and-skip (not corrupt the
# file) if its anchor text doesn't match byte-for-byte — verify
# [SETUP] Patched: hybrid_pool_assembler.py lines appear in the container log
# before relying on unified_kv + HiCache actually working.
hpa_path = "/sgl-workspace/sglang/python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py"
if os.path.isfile(hpa_path):
    src = open(hpa_path).read()
    if "_dsv4_compressed_region_buffers" in src:
        print("[SETUP] hybrid_pool_assembler.py unified_kv-hicache patch already applied")
    else:
        hunks = [
            (
                "insert _dsv4_compressed_region_buffers helper",
                "    return full_host_pages, swa_host_pages\n"
                "\n"
                "\n"
                "def build_deepseek_v4_hicache_stack(",
                "    return full_host_pages, swa_host_pages\n"
                "\n"
                "\n"
                "def _dsv4_compressed_region_buffers(kvcache: Any, ratio: int) -> tuple[list, int]:\n"
                "    \"\"\"\n"
                "    Resolve ``(device_buffers, item_bytes)`` for a DeepSeek V4 C4/C128 main-KV\n"
                "    HiCache pool, hiding the device KV layout from the stack builder.\n"
                "    \"\"\"\n"
                "    if getattr(kvcache, \"_unified_kv\", False):\n"
                "        return kvcache.unified_region_buffers(ratio)\n"
                "    pool = kvcache.c4_kv_pool if ratio == 4 else kvcache.c128_kv_pool\n"
                "    return pool.kv_buffer, pool.bytes_per_page_padded\n"
                "\n"
                "\n"
                "def build_deepseek_v4_hicache_stack(",
            ),
            (
                "gate swa_layer_mapping / swa_kv_pool assert by is_unified_kv",
                "    transfer_layer_num = kvcache.end_layer - kvcache.start_layer\n"
                "    full_layer_mapping = {layer_id: layer_id for layer_id in range(transfer_layer_num)}\n"
                "    if len(kvcache.swa_kv_pool.kv_buffer) != transfer_layer_num:\n"
                "        raise ValueError(\n"
                "            \"DeepSeek V4 SWA KV pool must be PP-stage-local: \"\n"
                "            f\"got {len(kvcache.swa_kv_pool.kv_buffer)} buffers for \"\n"
                "            f\"{transfer_layer_num} local layers\"\n"
                "        )\n"
                "    swa_layer_mapping = {layer_id: layer_id for layer_id in range(transfer_layer_num)}",
                "    transfer_layer_num = kvcache.end_layer - kvcache.start_layer\n"
                "    full_layer_mapping = {layer_id: layer_id for layer_id in range(transfer_layer_num)}\n"
                "\n"
                "    is_unified_kv = getattr(kvcache, \"_unified_kv\", False)\n"
                "    if is_unified_kv:\n"
                "        # unified_kv keeps the SWA ring inside the unified pool and never offloads it,\n"
                "        # so there is no separate SWA host pool to map.\n"
                "        swa_layer_mapping = {}\n"
                "    else:\n"
                "        if len(kvcache.swa_kv_pool.kv_buffer) != transfer_layer_num:\n"
                "            raise ValueError(\n"
                "                \"DeepSeek V4 SWA KV pool must be PP-stage-local: \"\n"
                "                f\"got {len(kvcache.swa_kv_pool.kv_buffer)} buffers for \"\n"
                "                f\"{transfer_layer_num} local layers\"\n"
                "            )\n"
                "        swa_layer_mapping = {\n"
                "            layer_id: layer_id for layer_id in range(transfer_layer_num)\n"
                "        }",
            ),
            (
                "move swa_host_pool construction out of the top-level block",
                "    logical_host_pool = LogicalHostPool(\n"
                "        num_host_pages * page_size, page_size, layout=server_args.hicache_mem_layout\n"
                "    )\n"
                "    swa_host_pool = DeepSeekV4PagedHostPool(\n"
                "        pool_name=str(PoolName.SWA),\n"
                "        device_buffers=kvcache.swa_kv_pool.kv_buffer,\n"
                "        item_bytes=kvcache.swa_kv_pool.bytes_per_page_padded,\n"
                "        num_host_pages=swa_num_host_pages,\n"
                "        slot_page_size=kvcache.swa_page_size,\n"
                "        layout=server_args.hicache_mem_layout,\n"
                "        allocator_type=server_args.hicache_storage_backend,\n"
                "    )\n"
                "    swa_attn_allocator = params.token_to_kv_pool_allocator.swa_attn_allocator\n"
                "    entries = [\n"
                "        build_pool_entry(\n"
                "            name=PoolName.KV,",
                "    logical_host_pool = LogicalHostPool(\n"
                "        num_host_pages * page_size, page_size, layout=server_args.hicache_mem_layout\n"
                "    )\n"
                "    entries = [\n"
                "        build_pool_entry(\n"
                "            name=PoolName.KV,",
            ),
            (
                "move swa build_pool_entry into `if not is_unified_kv:`, add C4 helper call",
                "            transfer_layer_num=transfer_layer_num,\n"
                "            is_anchor=True,\n"
                "        ),\n"
                "        build_pool_entry(\n"
                "            name=PoolName.SWA,\n"
                "            host_pool=swa_host_pool,\n"
                "            device_pool=kvcache.swa_kv_pool,\n"
                "            layer_mapping=swa_layer_mapping,\n"
                "            transfer_layer_num=transfer_layer_num,\n"
                "            host_evict_fn=host_swa_evict_fn,\n"
                "            device_evict_fn=device_swa_evict_fn,\n"
                "            device_alloc_fn=swa_attn_allocator.alloc,\n"
                "            device_free_fn=swa_attn_allocator.free,\n"
                "        ),\n"
                "    ]\n"
                "\n"
                "    if c4_layer_mapping:\n"
                "        c4_host_pool = DeepSeekV4PagedHostPool(\n"
                "            pool_name=str(PoolName.DEEPSEEK_V4_C4),\n"
                "            device_buffers=kvcache.c4_kv_pool.kv_buffer,\n"
                "            item_bytes=kvcache.c4_kv_pool.bytes_per_page_padded,\n"
                "            num_host_pages=num_host_pages,\n"
                "            slot_page_size=page_size,\n"
                "            layout=server_args.hicache_mem_layout,",
                "            transfer_layer_num=transfer_layer_num,\n"
                "            is_anchor=True,\n"
                "        ),\n"
                "    ]\n"
                "\n"
                "    if not is_unified_kv:\n"
                "        swa_host_pool = DeepSeekV4PagedHostPool(\n"
                "            pool_name=str(PoolName.SWA),\n"
                "            device_buffers=kvcache.swa_kv_pool.kv_buffer,\n"
                "            item_bytes=kvcache.swa_kv_pool.bytes_per_page_padded,\n"
                "            num_host_pages=swa_num_host_pages,\n"
                "            slot_page_size=kvcache.swa_page_size,\n"
                "            layout=server_args.hicache_mem_layout,\n"
                "            allocator_type=server_args.hicache_storage_backend,\n"
                "        )\n"
                "        swa_attn_allocator = params.token_to_kv_pool_allocator.swa_attn_allocator\n"
                "        entries.append(\n"
                "            build_pool_entry(\n"
                "                name=PoolName.SWA,\n"
                "                host_pool=swa_host_pool,\n"
                "                device_pool=kvcache.swa_kv_pool,\n"
                "                layer_mapping=swa_layer_mapping,\n"
                "                transfer_layer_num=transfer_layer_num,\n"
                "                host_evict_fn=host_swa_evict_fn,\n"
                "                device_evict_fn=device_swa_evict_fn,\n"
                "                device_alloc_fn=swa_attn_allocator.alloc,\n"
                "                device_free_fn=swa_attn_allocator.free,\n"
                "            )\n"
                "        )\n"
                "\n"
                "    if c4_layer_mapping:\n"
                "        c4_device_buffers, c4_item_bytes = _dsv4_compressed_region_buffers(kvcache, 4)\n"
                "        c4_host_pool = DeepSeekV4PagedHostPool(\n"
                "            pool_name=str(PoolName.DEEPSEEK_V4_C4),\n"
                "            device_buffers=c4_device_buffers,\n"
                "            item_bytes=c4_item_bytes,\n"
                "            num_host_pages=num_host_pages,\n"
                "            slot_page_size=page_size,\n"
                "            layout=server_args.hicache_mem_layout,",
            ),
            (
                "gate c4 state pools by is_unified_kv + c128 helper call",
                "        c4_indexer_host_pool = DeepSeekV4PagedHostPool(\n"
                "            pool_name=str(PoolName.DEEPSEEK_V4_C4_INDEXER),\n"
                "            device_buffers=kvcache.c4_indexer_kv_pool.index_k_with_scale_buffer,\n"
                "            item_bytes=(\n"
                "                kvcache.c4_indexer_kv_pool.index_k_with_scale_buffer[0].shape[1]\n"
                "                * kvcache.c4_indexer_kv_pool.index_k_with_scale_buffer[0].element_size()\n"
                "            ),\n"
                "            num_host_pages=num_host_pages,\n"
                "            slot_page_size=page_size,\n"
                "            layout=server_args.hicache_mem_layout,\n"
                "            allocator_type=server_args.hicache_storage_backend,\n"
                "        )\n"
                "        c4_state_host_pool = DeepSeekV4StateHostPool(\n"
                "            pool_name=str(PoolName.DEEPSEEK_V4_C4_STATE),\n"
                "            state_pools=[\n"
                "                kvcache.compress_state_pools[layer_id]\n"
                "                for layer_id in c4_state_global_layers\n"
                "            ],\n"
                "            num_host_pages=swa_num_host_pages,\n"
                "            swa_page_size=kvcache.swa_page_size,\n"
                "            layout=server_args.hicache_mem_layout,\n"
                "            allocator_type=server_args.hicache_storage_backend,\n"
                "        )\n"
                "        c4_indexer_state_host_pool = DeepSeekV4StateHostPool(\n"
                "            pool_name=str(PoolName.DEEPSEEK_V4_C4_INDEXER_STATE),\n"
                "            state_pools=[\n"
                "                kvcache.indexer_compress_state_pools[layer_id]\n"
                "                for layer_id in c4_state_global_layers\n"
                "            ],\n"
                "            num_host_pages=swa_num_host_pages,\n"
                "            swa_page_size=kvcache.swa_page_size,\n"
                "            layout=server_args.hicache_mem_layout,\n"
                "            allocator_type=server_args.hicache_storage_backend,\n"
                "        )\n"
                "        entries.extend(\n"
                "            [\n"
                "                build_pool_entry(\n"
                "                    name=PoolName.DEEPSEEK_V4_C4,\n"
                "                    host_pool=c4_host_pool,\n"
                "                    device_pool=kvcache.c4_kv_pool,\n"
                "                    layer_mapping=c4_layer_mapping,\n"
                "                    transfer_layer_num=transfer_layer_num,\n"
                "                ),\n"
                "                build_pool_entry(\n"
                "                    name=PoolName.DEEPSEEK_V4_C4_INDEXER,\n"
                "                    host_pool=c4_indexer_host_pool,\n"
                "                    device_pool=kvcache.c4_indexer_kv_pool,\n"
                "                    layer_mapping=c4_layer_mapping,\n"
                "                    transfer_layer_num=transfer_layer_num,\n"
                "                ),\n"
                "                build_pool_entry(\n"
                "                    name=PoolName.DEEPSEEK_V4_C4_STATE,\n"
                "                    host_pool=c4_state_host_pool,\n"
                "                    device_pool=None,\n"
                "                    layer_mapping=c4_state_mapping,\n"
                "                    transfer_layer_num=transfer_layer_num,\n"
                "                ),\n"
                "                build_pool_entry(\n"
                "                    name=PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,\n"
                "                    host_pool=c4_indexer_state_host_pool,\n"
                "                    device_pool=None,\n"
                "                    layer_mapping=c4_state_mapping,\n"
                "                    transfer_layer_num=transfer_layer_num,\n"
                "                ),\n"
                "            ]\n"
                "        )\n"
                "\n"
                "    if c128_layer_mapping:\n"
                "        c128_host_pool = DeepSeekV4PagedHostPool(\n"
                "            pool_name=str(PoolName.DEEPSEEK_V4_C128),\n"
                "            device_buffers=kvcache.c128_kv_pool.kv_buffer,\n"
                "            item_bytes=kvcache.c128_kv_pool.bytes_per_page_padded,",
                "        c4_indexer_host_pool = DeepSeekV4PagedHostPool(\n"
                "            pool_name=str(PoolName.DEEPSEEK_V4_C4_INDEXER),\n"
                "            device_buffers=kvcache.c4_indexer_kv_pool.index_k_with_scale_buffer,\n"
                "            item_bytes=(\n"
                "                kvcache.c4_indexer_kv_pool.index_k_with_scale_buffer[0].shape[1]\n"
                "                * kvcache.c4_indexer_kv_pool.index_k_with_scale_buffer[0].element_size()\n"
                "            ),\n"
                "            num_host_pages=num_host_pages,\n"
                "            slot_page_size=page_size,\n"
                "            layout=server_args.hicache_mem_layout,\n"
                "            allocator_type=server_args.hicache_storage_backend,\n"
                "        )\n"
                "        entries.extend(\n"
                "            [\n"
                "                build_pool_entry(\n"
                "                    name=PoolName.DEEPSEEK_V4_C4,\n"
                "                    host_pool=c4_host_pool,\n"
                "                    device_pool=kvcache.c4_kv_pool,\n"
                "                    layer_mapping=c4_layer_mapping,\n"
                "                    transfer_layer_num=transfer_layer_num,\n"
                "                ),\n"
                "                build_pool_entry(\n"
                "                    name=PoolName.DEEPSEEK_V4_C4_INDEXER,\n"
                "                    host_pool=c4_indexer_host_pool,\n"
                "                    device_pool=kvcache.c4_indexer_kv_pool,\n"
                "                    layer_mapping=c4_layer_mapping,\n"
                "                    transfer_layer_num=transfer_layer_num,\n"
                "                ),\n"
                "            ]\n"
                "        )\n"
                "\n"
                "        if not is_unified_kv:\n"
                "            c4_state_host_pool = DeepSeekV4StateHostPool(\n"
                "                pool_name=str(PoolName.DEEPSEEK_V4_C4_STATE),\n"
                "                state_pools=[\n"
                "                    kvcache.compress_state_pools[layer_id]\n"
                "                    for layer_id in c4_state_global_layers\n"
                "                ],\n"
                "                num_host_pages=swa_num_host_pages,\n"
                "                swa_page_size=kvcache.swa_page_size,\n"
                "                layout=server_args.hicache_mem_layout,\n"
                "                allocator_type=server_args.hicache_storage_backend,\n"
                "            )\n"
                "            c4_indexer_state_host_pool = DeepSeekV4StateHostPool(\n"
                "                pool_name=str(PoolName.DEEPSEEK_V4_C4_INDEXER_STATE),\n"
                "                state_pools=[\n"
                "                    kvcache.indexer_compress_state_pools[layer_id]\n"
                "                    for layer_id in c4_state_global_layers\n"
                "                ],\n"
                "                num_host_pages=swa_num_host_pages,\n"
                "                swa_page_size=kvcache.swa_page_size,\n"
                "                layout=server_args.hicache_mem_layout,\n"
                "                allocator_type=server_args.hicache_storage_backend,\n"
                "            )\n"
                "            entries.extend(\n"
                "                [\n"
                "                    build_pool_entry(\n"
                "                        name=PoolName.DEEPSEEK_V4_C4_STATE,\n"
                "                        host_pool=c4_state_host_pool,\n"
                "                        device_pool=None,\n"
                "                        layer_mapping=c4_state_mapping,\n"
                "                        transfer_layer_num=transfer_layer_num,\n"
                "                    ),\n"
                "                    build_pool_entry(\n"
                "                        name=PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,\n"
                "                        host_pool=c4_indexer_state_host_pool,\n"
                "                        device_pool=None,\n"
                "                        layer_mapping=c4_state_mapping,\n"
                "                        transfer_layer_num=transfer_layer_num,\n"
                "                    ),\n"
                "                ]\n"
                "            )\n"
                "\n"
                "    if c128_layer_mapping:\n"
                "        c128_device_buffers, c128_item_bytes = _dsv4_compressed_region_buffers(\n"
                "            kvcache, 128\n"
                "        )\n"
                "        c128_host_pool = DeepSeekV4PagedHostPool(\n"
                "            pool_name=str(PoolName.DEEPSEEK_V4_C128),\n"
                "            device_buffers=c128_device_buffers,\n"
                "            item_bytes=c128_item_bytes,",
            ),
            (
                "make component_host_pools[SWA] optional (unified_kv has no SWA host pool)",
                "        return StackBuildResult(\n"
                "            host_pool_group=host_pool_group,\n"
                "            cache_controller=cache_controller,\n"
                "            component_host_pools={\n"
                "                ComponentType.FULL: host_pool_group.get_pool(PoolName.KV),\n"
                "                ComponentType.SWA: host_pool_group.get_pool(PoolName.SWA),\n"
                "            },\n"
                "            sidecars=sidecars,\n"
                "            transfer_layer_num=kvcache.end_layer - kvcache.start_layer,\n"
                "            pools_desc=\"KV + SWA + DeepSeekV4 sidecars\",",
                "        component_host_pools = {\n"
                "            ComponentType.FULL: host_pool_group.get_pool(PoolName.KV),\n"
                "        }\n"
                "        if PoolName.SWA in host_pool_group.entry_map:\n"
                "            component_host_pools[ComponentType.SWA] = host_pool_group.get_pool(\n"
                "                PoolName.SWA\n"
                "            )\n"
                "\n"
                "        return StackBuildResult(\n"
                "            host_pool_group=host_pool_group,\n"
                "            cache_controller=cache_controller,\n"
                "            component_host_pools=component_host_pools,\n"
                "            sidecars=sidecars,\n"
                "            transfer_layer_num=kvcache.end_layer - kvcache.start_layer,\n"
                "            pools_desc=\"KV + SWA + DeepSeekV4 sidecars\",",
            ),
        ]
        for label, old, new in hunks:
            if old not in src:
                print(
                    f"[SETUP] WARN: hybrid_pool_assembler.py anchor for '{label}' not "
                    "found — sglang version may have changed"
                )
                break
            src = src.replace(old, new, 1)
        else:
            open(hpa_path, "w").write(src)
            print("[SETUP] Patched: hybrid_pool_assembler.py unified_kv-hicache")
else:
    print(f"[SETUP] {hpa_path} not found, skipping")

# --- swa_component.py: skip SWA host-pool bookkeeping for unified_kv -------
patch_file(
    "/sgl-workspace/sglang/python/sglang/srt/mem_cache/unified_cache_components/swa_component.py",
    marker="swa_device_only_hicache",
    hunks=[
        (
            "create_match_validator device-only-hicache short-circuit",
            "        ct = self.component_type\n"
            "        state = {\"len\": float(\"inf\")}\n"
            "\n"
            "        def validator(node: UnifiedTreeNode) -> bool:\n"
            "            cd = node.component_data[ct]\n"
            "            # HiCache: a host-only tombstone is a valid match boundary too\n"
            "            # — load_back will restore SWA from host before use.\n"
            "            if cd.value is None and (match_device_only or cd.host_value is None):\n"
            "                state[\"len\"] = 0\n"
            "                return False",
            "        ct = self.component_type\n"
            "        state = {\"len\": float(\"inf\")}\n"
            "\n"
            "        # unified_kv never caches the SWA ring (per-request, not content-stable),\n"
            "        # so SWA bookkeeping must not gate the match here.\n"
            "        swa_device_only_hicache = (\n"
            "            self._swa_kv_pool_host is None and self.cache.cache_controller is not None\n"
            "        )\n"
            "\n"
            "        def validator(node: UnifiedTreeNode) -> bool:\n"
            "            cd = node.component_data[ct]\n"
            "            # HiCache: a host-only tombstone is a valid match boundary too\n"
            "            # — load_back will restore SWA from host before use.\n"
            "            if cd.value is None and (match_device_only or cd.host_value is None):\n"
            "                state[\"len\"] = 0\n"
            "                if swa_device_only_hicache and (node.backuped or not node.evicted):\n"
            "                    return True\n"
            "                return False",
        ),
        (
            "build_hicache_transfers unified_kv short-circuit",
            "    ) -> Optional[list[PoolTransfer]]:\n"
            "        ct = self.component_type\n"
            "\n"
            "        if phase == CacheTransferPhase.BACKUP_HOST:",
            "    ) -> Optional[list[PoolTransfer]]:\n"
            "        ct = self.component_type\n"
            "\n"
            "        # unified_kv keeps SWA as a device-only ring.\n"
            "        if self._swa_kv_pool_host is None and self.cache.cache_controller is not None:\n"
            "            return None\n"
            "\n"
            "        if phase == CacheTransferPhase.BACKUP_HOST:",
        ),
    ],
)
PYEOF
    _SETUP_INSTALLED+=("dsv4-unified-kv-hicache-fix")
}

# ---------------------------------------------------------------------------
# SGLang: DSA indexer paged-MQA-logits backend abstraction (AITER on ROCm).
# Mirrors upstream sgl-project/sglang#30374.
#
# Adds the sglang.jit_kernel.dsa package (paged_mqa_logits.py +
# paged_mqa_logits_backend.py's DSAPagedMQALogitsBackend enum) and rewires
# dsa_indexer.py's paged-MQA-logits dispatch (previously an inline
# `if _is_hip: ... elif use_dg_native: ... else: ...` chain) through it, plus
# a new `--dsa-paged-mqa-logits-backend` server arg (default "auto").
#
# On ROCm this is a refactor, not a behavior change: DSAPagedMQALogitsBackend
# .resolve() always resolves to AITER on HIP (raises for anything else), and
# the new `is_aiter()` branch calls the same
# aiter.ops.triton.pa_mqa_logits.deepgemm_fp8_paged_mqa_logits kernel the old
# `if _is_hip:` branch called directly. The CUTE DSL backend
# (jit_kernel/dsa/cutedsl_paged_mqa_logits.py, SM100/Blackwell-only) is
# intentionally NOT installed here: jit_kernel/dsa/__init__.py and
# paged_mqa_logits.py's cutedsl_paged_mqa_logits() both gate its import
# behind `is_hip()`/local-import, so it is never imported/executed on ROCm.
#
# Depends on nothing else in this file; safe to run unconditionally.
# ---------------------------------------------------------------------------
patch_dsa_paged_mqa_logits_backend() {
    python3 - <<'PYEOF'
import os

SGLANG_ROOT = "/sgl-workspace/sglang/python/sglang"


def write_new_file(rel_path, marker, content):
    path = os.path.join(SGLANG_ROOT, rel_path)
    if os.path.isfile(path) and marker in open(path).read():
        print(f"[SETUP] {rel_path} already present")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print(f"[SETUP] Wrote: {rel_path}")


def patch_file(path, marker, hunks):
    if not os.path.isfile(path):
        print(f"[SETUP] {path} not found, skipping")
        return
    src = open(path).read()
    if marker in src:
        print(f"[SETUP] {os.path.basename(path)} dsa-paged-mqa-logits-backend patch already applied")
        return
    for label, old, new in hunks:
        if old not in src:
            print(
                f"[SETUP] WARN: {os.path.basename(path)} anchor for '{label}' not "
                "found — sglang version may have changed"
            )
            return
        src = src.replace(old, new, 1)
    open(path, "w").write(src)
    print(f"[SETUP] Patched: {os.path.basename(path)} dsa-paged-mqa-logits-backend")


# --- new file: srt/layers/attention/dsa/paged_mqa_logits_backend.py -------
write_new_file(
    "srt/layers/attention/dsa/paged_mqa_logits_backend.py",
    marker="class DSAPagedMQALogitsBackend",
    content='''from __future__ import annotations

from enum import Enum

from sglang.srt.utils import is_hip, is_sm100_supported


class DSAPagedMQALogitsBackend(Enum):
    DEEPGEMM = "deepgemm"
    CUTEDSL = "cutedsl"
    AITER = "aiter"

    def is_deepgemm(self) -> bool:
        return self == DSAPagedMQALogitsBackend.DEEPGEMM

    def is_cutedsl(self) -> bool:
        return self == DSAPagedMQALogitsBackend.CUTEDSL

    def is_aiter(self) -> bool:
        return self == DSAPagedMQALogitsBackend.AITER

    @staticmethod
    def resolve(value: str) -> DSAPagedMQALogitsBackend:
        if is_hip():
            if value not in ("auto", "aiter"):
                raise ValueError(
                    f"dsa_paged_mqa_logits_backend={value!r} is not supported on "
                    "ROCm; only 'aiter' is implemented."
                )
            return DSAPagedMQALogitsBackend.AITER

        if value == "auto" or value == "deepgemm":
            return DSAPagedMQALogitsBackend.DEEPGEMM
        if value == "aiter":
            raise ValueError("dsa_paged_mqa_logits_backend='aiter' requires ROCm.")
        if value == "cutedsl":
            if not is_sm100_supported():
                raise ValueError(
                    "dsa_paged_mqa_logits_backend='cutedsl' requires SM100 (Blackwell)."
                )
            return DSAPagedMQALogitsBackend.CUTEDSL
        raise ValueError(f"Unknown dsa_paged_mqa_logits_backend: {value!r}")
''',
)

# --- new file: jit_kernel/dsa/__init__.py ---------------------------------
# NOTE: omits the `from .cutedsl_paged_mqa_logits import ...` branch's target
# module (cutedsl_paged_mqa_logits.py, SM100/Blackwell-only) since that
# import is gated by `if not is_hip()` and is therefore dead code on ROCm.
write_new_file(
    "jit_kernel/dsa/__init__.py",
    marker="aiter_paged_mqa_logits",
    content='''from sglang.srt.utils import is_hip

from .paged_mqa_logits import (
    aiter_paged_mqa_logits,
    cutedsl_paged_mqa_logits,
    deepgemm_paged_mqa_logits_native,
    deepgemm_paged_mqa_logits_split,
)

if not is_hip():
    # Preserve the original eager import behavior on non-ROCm platforms.
    from .cutedsl_paged_mqa_logits import CuteDSLPagedMQALogitsRunner, pick_dsl_expand

__all__ = [
    "CuteDSLPagedMQALogitsRunner",
    "pick_dsl_expand",
    "aiter_paged_mqa_logits",
    "cutedsl_paged_mqa_logits",
    "deepgemm_paged_mqa_logits_native",
    "deepgemm_paged_mqa_logits_split",
]
''',
)

# --- new file: jit_kernel/dsa/paged_mqa_logits.py -------------------------
write_new_file(
    "jit_kernel/dsa/paged_mqa_logits.py",
    marker="def aiter_paged_mqa_logits",
    content='''# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Callable

import torch


def deepgemm_paged_mqa_logits_native(
    fp8_paged_mqa_logits_fn: Callable[..., torch.Tensor],
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    ctx_lens_2d: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_seq_len: int,
    *,
    q_offset: int,
    B: int,
    next_n: int,
) -> torch.Tensor:
    # block_tables[::next_n] de-expands the caller's repeat_interleave without a
    # copy (DeepGEMM only checks `stride(1) == 1`).
    return fp8_paged_mqa_logits_fn(
        q_fp8[:q_offset].view(B, next_n, q_fp8.shape[1], q_fp8.shape[2]),
        kv_cache_fp8,
        weights[:q_offset],
        ctx_lens_2d,
        block_tables[::next_n],
        schedule_metadata,
        max_seq_len,
        clean_logits=False,
    )


def deepgemm_paged_mqa_logits_split(
    fp8_paged_mqa_logits_fn: Callable[..., torch.Tensor],
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    ctx_lens_2d: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_seq_len: int,
    *,
    q_offset: int,
) -> torch.Tensor:
    q_fp8 = q_fp8.unsqueeze(1)
    return fp8_paged_mqa_logits_fn(
        q_fp8[:q_offset],
        kv_cache_fp8,
        weights[:q_offset],
        ctx_lens_2d,
        block_tables,
        schedule_metadata,
        max_seq_len,
        clean_logits=False,
    )


def aiter_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_seq_len: int,
    *,
    preshuffle: bool,
    kv_block_size: int,
) -> torch.Tensor:
    from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits

    q_fp8 = q_fp8.unsqueeze(1)
    batch_size, next_n, _, _ = q_fp8.shape
    logits = torch.empty(
        (batch_size * next_n, max_seq_len),
        device=q_fp8.device,
        dtype=torch.float32,
    )
    deepgemm_fp8_paged_mqa_logits(
        q_fp8,
        kv_cache_fp8,
        weights,
        logits,
        seq_lens,
        block_tables,
        max_seq_len,
        Preshuffle=preshuffle,
        KVBlockSize=kv_block_size,
    )
    return logits


def cutedsl_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    ctx_lens_1d: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor | None,
    max_seq_len: int,
    *,
    q_offset: int,
    B: int,
    next_n: int,
    is_target_verify: bool,
    dsl_expand_factor: int,
    dsl_atom: int,
    blocksize: int,
    sm_count: int,
    get_paged_mqa_logits_metadata_fn: Callable[..., torch.Tensor],
) -> torch.Tensor:
    from sglang.jit_kernel.dsa.cutedsl_paged_mqa_logits import (
        CuteDSLPagedMQALogitsRunner,
    )

    dsl_atom_split = dsl_expand_factor > 1 and next_n == dsl_expand_factor * dsl_atom
    if is_target_verify and dsl_atom_split:
        exp_B = B * dsl_expand_factor
        q_dsl = q_fp8[:q_offset].view(exp_B, dsl_atom, q_fp8.shape[1], q_fp8.shape[2])
        ctx_lens_1d = ctx_lens_1d.repeat_interleave(dsl_expand_factor)
        block_tables_dsl = block_tables[::next_n].repeat_interleave(
            dsl_expand_factor, dim=0
        )
        schedule_metadata = get_paged_mqa_logits_metadata_fn(
            ctx_lens_1d.unsqueeze(-1), blocksize, sm_count
        )
    elif is_target_verify and next_n >= 2:
        # Native single-launch: one task per batch entry (the kernel iterates
        # next_n internally), so the schedule must be built from B-length
        # context lens, not the caller's [B, next_n] or per-token layout.
        q_dsl = q_fp8[:q_offset].view(B, next_n, q_fp8.shape[1], q_fp8.shape[2])
        block_tables_dsl = block_tables[::next_n]
        schedule_metadata = get_paged_mqa_logits_metadata_fn(
            ctx_lens_1d.unsqueeze(-1), blocksize, sm_count
        )
    else:
        q_dsl = q_fp8[:q_offset].unsqueeze(1)
        block_tables_dsl = block_tables[:B]

    return CuteDSLPagedMQALogitsRunner.forward(
        q_dsl,
        kv_cache_fp8.view(torch.uint8),
        weights[:q_offset],
        ctx_lens_1d,
        block_tables_dsl,
        schedule_metadata,
        max_seq_len,
    )
''',
)

# --- dsa_indexer.py: rewire the paged-MQA-logits dispatch through the ------
# --- new backend abstraction -----------------------------------------------
patch_file(
    "/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py",
    marker="self.paged_mqa_logits_backend = DSAPagedMQALogitsBackend.resolve",
    hunks=[
        (
            "imports: jit_kernel.dsa + paged_mqa_logits_backend",
            "from sglang.jit_kernel.fused_store_index_cache import (\n"
            "    can_use_dsa_fused_store,\n"
            "    fused_store_index_k_cache,\n"
            ")\n"
            "from sglang.srt.compilation.compilation_config import register_split_op\n"
            "from sglang.srt.environ import envs\n"
            "from sglang.srt.layers.attention.dsa.utils import (\n",
            "from sglang.jit_kernel.dsa import (\n"
            "    aiter_paged_mqa_logits,\n"
            "    cutedsl_paged_mqa_logits,\n"
            "    deepgemm_paged_mqa_logits_native,\n"
            "    deepgemm_paged_mqa_logits_split,\n"
            ")\n"
            "from sglang.jit_kernel.fused_store_index_cache import (\n"
            "    can_use_dsa_fused_store,\n"
            "    fused_store_index_k_cache,\n"
            ")\n"
            "from sglang.srt.compilation.compilation_config import register_split_op\n"
            "from sglang.srt.environ import envs\n"
            "from sglang.srt.layers.attention.dsa.paged_mqa_logits_backend import (\n"
            "    DSAPagedMQALogitsBackend,\n"
            ")\n"
            "from sglang.srt.layers.attention.dsa.utils import (\n",
        ),
        (
            "imports: get_server_args",
            "from sglang.srt.runtime_context import get_parallel\n",
            "from sglang.srt.runtime_context import get_parallel, get_server_args\n",
        ),
        (
            "module-level: conditional pick_dsl_expand import",
            "_is_cuda = is_cuda()\n"
            "_is_hip = is_hip()\n"
            "_is_npu = is_npu()\n"
            '_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip\n',
            "_is_cuda = is_cuda()\n"
            "_is_hip = is_hip()\n"
            "_is_npu = is_npu()\n"
            "if not _is_hip:\n"
            "    # Preserve the original eager import behavior on non-ROCm platforms.\n"
            "    from sglang.jit_kernel.dsa import pick_dsl_expand\n"
            '_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip\n',
        ),
        (
            "__init__: resolve paged_mqa_logits_backend",
            "        self.block_size = block_size\n"
            "        self.scale_fmt = scale_fmt\n"
            "        self.softmax_scale = self.head_dim**-0.5\n"
            "\n"
            "    @contextlib.contextmanager\n"
            "    def _with_real_sm_count(self):\n",
            "        self.block_size = block_size\n"
            "        self.scale_fmt = scale_fmt\n"
            "        self.softmax_scale = self.head_dim**-0.5\n"
            "\n"
            "        self.paged_mqa_logits_backend = DSAPagedMQALogitsBackend.resolve(\n"
            "            get_server_args().dsa_paged_mqa_logits_backend\n"
            "        )\n"
            "\n"
            "    @contextlib.contextmanager\n"
            "    def _with_real_sm_count(self):\n",
        ),
        (
            "forward: compute use_cute_dsl / dsl_expand_factor before use_dg_native",
            "        # Reuse pre-computed schedule metadata if available (from init_forward_metadata),\n"
            "        # otherwise fall back to computing it here.\n"
            '        schedule_metadata = getattr(metadata, "paged_mqa_schedule_metadata", None)\n'
            "\n"
            "        assert len(q_fp8.shape) == 3\n"
            "        # attn_tp_size > 1 or MAX_LEN padding mode can leave padding in the\n"
            "        # hidden states; q_offset is the real (unpadded) q length.\n"
            "        q_offset = sum(metadata.get_dsa_extend_len_cpu())\n"
            "\n"
            "        # DG-native q=[B,next_n,H,D] is faster than expanded q=[B*next_n,1,H,D]\n"
            "        # for target_verify with next_n>=2 (bigger MMA tile, fewer atoms). The\n"
            "        # precomputed ctx_lens_2d's shape is the single source of truth — if\n"
            "        # dsa_backend chose the per-token layout (e.g. non-SM100), fall through\n"
            "        # to the expanded path.\n"
            "        B = metadata.get_seqlens_int32().shape[0]\n"
            "        next_n = q_offset // B if B > 0 else 0\n"
            '        ctx_2d = getattr(metadata, "paged_mqa_ctx_lens_2d", None)\n'
            "        use_dg_native = (\n"
            "            _is_cuda\n"
            "            and forward_batch.forward_mode.is_target_verify()\n"
            "            and next_n >= 2\n"
            "            and ctx_2d is not None\n",
            "        # Reuse pre-computed schedule metadata if available (from init_forward_metadata),\n"
            "        # otherwise fall back to computing it here.\n"
            '        schedule_metadata = getattr(metadata, "paged_mqa_schedule_metadata", None)\n'
            "        assert len(q_fp8.shape) == 3\n"
            "        # attn_tp_size > 1 or MAX_LEN padding mode can leave padding in the\n"
            "        # hidden states; q_offset is the real (unpadded) q length.\n"
            "        q_offset = sum(metadata.get_dsa_extend_len_cpu())\n"
            "\n"
            "        B = metadata.get_seqlens_int32().shape[0]\n"
            "        next_n = q_offset // B if B > 0 else 0\n"
            "        use_cute_dsl = (\n"
            "            self.paged_mqa_logits_backend.is_cutedsl()\n"
            "            and not forward_batch.forward_mode.is_draft_extend_v2()\n"
            "        )\n"
            "        dsl_expand_factor, dsl_atom = 1, 1\n"
            "        if (\n"
            "            use_cute_dsl\n"
            "            and forward_batch.forward_mode.is_target_verify()\n"
            "            and next_n >= 2\n"
            "        ):\n"
            "            dsl_expand_factor, dsl_atom = pick_dsl_expand(\n"
            "                next_n,\n"
            "                batch_size=B,\n"
            "                max_ctx=max_seq_len,\n"
            "                num_sms=self.sm_count,\n"
            "                kernel_atoms=(1, 2, 3, 4),\n"
            "                num_heads=self.n_heads,\n"
            "            )\n"
            '        ctx_2d = getattr(metadata, "paged_mqa_ctx_lens_2d", None)\n'
            "        use_dg_native = (\n"
            "            not use_cute_dsl\n"
            "            and _is_cuda\n"
            "            and forward_batch.forward_mode.is_target_verify()\n"
            "            and next_n >= 2\n"
            "            and ctx_2d is not None\n",
        ),
        (
            "forward: dispatch chain -> backend abstraction",
            "        assert len(weights.shape) == 3\n"
            "        weights = weights.squeeze(2)\n"
            "\n"
            "        if _is_hip:\n"
            "            from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits\n"
            "\n"
            "            q_fp8 = q_fp8.unsqueeze(1)\n"
            "            batch_size, next_n, heads, _ = q_fp8.shape\n"
            "            logits = torch.empty(\n"
            "                (batch_size * next_n, max_seq_len),\n"
            "                device=q_fp8.device,\n"
            "                dtype=torch.float32,\n"
            "            )\n"
            "            deepgemm_fp8_paged_mqa_logits(\n"
            "                q_fp8,\n"
            "                kv_cache_fp8,\n"
            "                weights,\n"
            "                logits,\n"
            "                seqlens_32,\n"
            "                block_tables,\n"
            "                max_seq_len,\n"
            "                Preshuffle=_use_aiter_preshuffle,\n"
            "                KVBlockSize=block_kv,\n"
            "            )\n"
            "        elif use_dg_native:\n"
            "            # block_tables[::next_n] de-expands dsa_backend's repeat_interleave\n"
            "            # without a copy (DG only checks `stride(1) == 1`).\n"
            "            logits = deep_gemm.fp8_paged_mqa_logits(\n"
            "                q_fp8[:q_offset].view(B, next_n, q_fp8.shape[1], q_fp8.shape[2]),\n"
            "                kv_cache_fp8,\n"
            "                weights[:q_offset],\n"
            "                seqlens_32_2d,\n"
            "                block_tables[::next_n],\n"
            "                schedule_metadata,\n"
            "                max_seq_len,\n"
            "                clean_logits=False,\n"
            "            )\n"
            "        else:\n"
            "            q_fp8 = q_fp8.unsqueeze(1)\n"
            "            logits = deep_gemm.fp8_paged_mqa_logits(\n"
            "                q_fp8[:q_offset],\n"
            "                kv_cache_fp8,\n"
            "                weights[:q_offset],\n"
            "                seqlens_32_2d,\n"
            "                block_tables,\n"
            "                schedule_metadata,\n"
            "                max_seq_len,\n"
            "                clean_logits=False,\n"
            "            )\n",
            "        assert len(weights.shape) == 3\n"
            "        weights = weights.squeeze(2)\n"
            "\n"
            "        if self.paged_mqa_logits_backend.is_aiter():\n"
            "            logits = aiter_paged_mqa_logits(\n"
            "                q_fp8,\n"
            "                kv_cache_fp8,\n"
            "                weights,\n"
            "                seqlens_32,\n"
            "                block_tables,\n"
            "                max_seq_len,\n"
            "                preshuffle=_use_aiter_preshuffle,\n"
            "                kv_block_size=block_kv,\n"
            "            )\n"
            "        elif use_cute_dsl:\n"
            "            logits = cutedsl_paged_mqa_logits(\n"
            "                q_fp8,\n"
            "                kv_cache_fp8,\n"
            "                weights,\n"
            "                metadata.get_seqlens_int32(),\n"
            "                block_tables,\n"
            "                schedule_metadata,\n"
            "                max_seq_len,\n"
            "                q_offset=q_offset,\n"
            "                B=B,\n"
            "                next_n=next_n,\n"
            "                is_target_verify=forward_batch.forward_mode.is_target_verify(),\n"
            "                dsl_expand_factor=dsl_expand_factor,\n"
            "                dsl_atom=dsl_atom,\n"
            "                blocksize=blocksize,\n"
            "                sm_count=self.sm_count,\n"
            "                get_paged_mqa_logits_metadata_fn=deep_gemm.get_paged_mqa_logits_metadata,\n"
            "            )\n"
            "        elif use_dg_native:\n"
            "            logits = deepgemm_paged_mqa_logits_native(\n"
            "                deep_gemm.fp8_paged_mqa_logits,\n"
            "                q_fp8,\n"
            "                kv_cache_fp8,\n"
            "                weights,\n"
            "                seqlens_32_2d,\n"
            "                block_tables,\n"
            "                schedule_metadata,\n"
            "                max_seq_len,\n"
            "                q_offset=q_offset,\n"
            "                B=B,\n"
            "                next_n=next_n,\n"
            "            )\n"
            "        else:\n"
            "            logits = deepgemm_paged_mqa_logits_split(\n"
            "                deep_gemm.fp8_paged_mqa_logits,\n"
            "                q_fp8,\n"
            "                kv_cache_fp8,\n"
            "                weights,\n"
            "                seqlens_32_2d,\n"
            "                block_tables,\n"
            "                schedule_metadata,\n"
            "                max_seq_len,\n"
            "                q_offset=q_offset,\n"
            "            )\n",
        ),
    ],
)

# --- server_args.py: add --dsa-paged-mqa-logits-backend --------------------
patch_file(
    "/sgl-workspace/sglang/python/sglang/srt/server_args.py",
    marker="DSA_PAGED_MQA_LOGITS_BACKEND_CHOICES",
    hunks=[
        (
            "add DSA_PAGED_MQA_LOGITS_BACKEND_CHOICES constant",
            "DSA_TOPK_BACKEND_CHOICES = [\"sgl-kernel\", \"torch\", \"flashinfer\"]\n",
            "DSA_TOPK_BACKEND_CHOICES = [\"sgl-kernel\", \"torch\", \"flashinfer\"]\n"
            "\n"
            'DSA_PAGED_MQA_LOGITS_BACKEND_CHOICES = ["auto", "deepgemm", "cutedsl", "aiter"]\n',
        ),
        (
            "add dsa_paged_mqa_logits_backend field",
            "    dsa_topk_backend: A[\n"
            "        str,\n"
            "        Arg(\n"
            '            help="DSA indexer top-k backend. Options: \'sgl-kernel\', \'torch\', \'flashinfer\'. The \'torch\' backend currently requires SGLANG_DSA_FUSE_TOPK=false.",\n'
            "            choices=DSA_TOPK_BACKEND_CHOICES,\n"
            "        ),\n"
            '    ] = "sgl-kernel"\n',
            "    dsa_paged_mqa_logits_backend: A[\n"
            "        str,\n"
            "        Arg(\n"
            '            help="DSA indexer paged MQA logits kernel backend. Options: \'auto\' (default; DeepGEMM on CUDA, aiter on ROCm), \'deepgemm\', \'cutedsl\' (CuTe DSL kernel, SM 100 (Blackwell) only; wins at low batch size and long context), \'aiter\' (ROCm only).",\n'
            "            choices=DSA_PAGED_MQA_LOGITS_BACKEND_CHOICES,\n"
            "        ),\n"
            '    ] = "auto"\n'
            "    dsa_topk_backend: A[\n"
            "        str,\n"
            "        Arg(\n"
            '            help="DSA indexer top-k backend. Options: \'sgl-kernel\', \'torch\', \'flashinfer\'. The \'torch\' backend currently requires SGLANG_DSA_FUSE_TOPK=false.",\n'
            "            choices=DSA_TOPK_BACKEND_CHOICES,\n"
            "        ),\n"
            '    ] = "sgl-kernel"\n',
        ),
    ],
)
PYEOF
    _SETUP_INSTALLED+=("dsa-paged-mqa-logits-backend-fix")
}

# ---------------------------------------------------------------------------
# SGLang: fix cold-start uninitialized C128 compress-state read on HIP.
# Mirrors upstream sgl-project/sglang#30333.
#
# CompressStatePool.clear_all_state() only clears the LAST row of
# kv_score_buffer (the historical "empty state" sentinel row). That is
# correct for the index-addressed C4 pool, but the ROCm/HIP C128 layout
# addresses request-scoped state by req_pool_idx (a per-request ring, not
# content-addressed) — with the pool allocated via torch.empty(), a cold
# server can read uninitialized garbage as a request's "previous" compress
# state before that request's slot has ever been written. Initialize every
# C128 row to the sentinel on HIP; C4 (and non-HIP) keep the original
# last-row-only behavior.
# ---------------------------------------------------------------------------
patch_deepseek_v4_compress_state_coldstart() {
    python3 -c '
import os, sys

target = "/sgl-workspace/sglang/python/sglang/srt/mem_cache/deepseek_v4_compress_state.py"
if not os.path.isfile(target):
    print("[SETUP] deepseek_v4_compress_state.py not found, skipping")
    sys.exit(0)

src = open(target).read()

if "Request-scoped C128 state is addressed by req_pool_idx" in src:
    print("[SETUP] deepseek_v4_compress_state.py cold-start patch already applied")
    sys.exit(0)

old = "        if not online:\n            self.kv_score_buffer[-1].clear()\n"
new = (
    "        if not online:\n"
    "            if _is_hip and ratio == 128:\n"
    "                # Request-scoped C128 state is addressed by req_pool_idx (or a\n"
    "                # per-request ring).  The pool is allocated with torch.empty(),\n"
    "                # so a cold server can otherwise read uninitialized partial\n"
    "                # states before a request slot has been written for the first\n"
    "                # time.  Initialize all C128 rows to the empty-state sentinel;\n"
    "                # C4 keeps the historical last-row sentinel behavior.\n"
    "                self.kv_score_buffer.clear()\n"
    "            else:\n"
    "                self.kv_score_buffer[-1].clear()\n"
)

if old not in src:
    print("[SETUP] WARN: deepseek_v4_compress_state.py anchor not found — sglang version may have changed")
    sys.exit(0)

open(target, "w").write(src.replace(old, new, 1))
print("[SETUP] Patched: deepseek_v4_compress_state.py cold-start C128 init")
'
    _SETUP_INSTALLED+=("dsv4-compress-state-coldstart-fix")
}

# ---------------------------------------------------------------------------
# SGLang: soften host>device KV pool assert (Mooncake / page_first_direct).
#
# With page_first_direct + a storage backend, SGLang asserts the host KV pool
# must be strictly larger than the device pool, which kills startup when the
# host pool is sized smaller. Turn the hard assert into a warning so the server
# starts (lower L2 hit rate is acceptable).
# Gated by MC_PATCH_HOSTPOOL=1 and KV_OFFLOADING=dram/KV_OFFLOAD_BACKEND=hicache.
# ---------------------------------------------------------------------------
patch_memory_pool_host_assert() {
    if [[ "${MC_PATCH_HOSTPOOL:-0}" != "1" || "${KV_OFFLOADING:-none}" == "none" || "${KV_OFFLOAD_BACKEND:-}" != "hicache" ]]; then
        return 0
    fi
    echo "[SETUP] Patching memory_pool_host.py host>device assert -> warning ..."
    python3 -c "
import pathlib
f = pathlib.Path('/sgl-workspace/sglang/python/sglang/srt/mem_cache/memory_pool_host.py')
src = f.read_text()
old = '''        assert (
            self.size > device_pool.size
        ), \"The host memory should be larger than the device memory with the current protocol\"'''
new = '''        if self.size <= device_pool.size:
            import logging as _lg
            _lg.getLogger(__name__).warning(
                \"Host KV pool (%d tokens) <= device pool (%d tokens). L2 hit rate may be low.\",
                self.size, device_pool.size)'''
if new in src:
    print('[SETUP] memory_pool_host.py assert patch already applied')
elif old in src:
    f.write_text(src.replace(old, new))
    print('[SETUP] Patched: memory_pool_host.py assert -> warning')
else:
    print('[SETUP] WARN: memory_pool_host.py assert pattern not found — sglang version may have changed')
" || true
    _SETUP_INSTALLED+=("memory-pool-host-assert-fix")
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
    patch_gluon_pa_mqa_logits_instr_shape
    patch_disagg_prefill_bootstrap_desync
    # patch_decode_tp_queue_agree
    patch_dsa_paged_mqa_logits_backend
    patch_deepseek_v4_compress_state_coldstart
    patch_swa_reprefill_tail_unified_kv
    patch_dsv4_unified_kv_hicache
    patch_memory_pool_host_assert
    
    install_transformers_glm5
fi

_SETUP_END=$(date +%s)
if [[ ${#_SETUP_INSTALLED[@]} -eq 0 ]]; then
    echo "[SETUP] All dependencies already present ($(( _SETUP_END - _SETUP_START ))s wallclock)"
else
    echo "[SETUP] Installed: ${_SETUP_INSTALLED[*]} in $(( _SETUP_END - _SETUP_START ))s"
fi
