#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""PATCH 01 -- dsa_indexer HIP/aiter DP-padded rows.

WHAT: the aiter (HIP) paged-MQA branch sizes its `logits` output from the
DP-PADDED row count while `lengths` is sized to the REAL count, so top-k dies
with "Expected lengths.size(0) == B to be true, but got false". Reconcile the
two counts to `min(real, padded)`: trim q/weights when q_fp8 is the longer side
(the contract every CUDA backend already honours) and restore the padding after
`topk_transform`, or clip the lengths when q_fp8 is the SHORTER side. See
"THE ROW COUNT DIVERGES BOTH WAYS" below.

WHY: without it, DP-attention + EAGLE MTP for GLM-5.2 DSA crashes as soon as
more than one request is in flight -- on gfx950 at the very first batch, and on
gfx942 at conc > 1. CUDA was never affected: those backends take `q_offset=` and
slice internally. ROCm-specific.

THE ROW COUNT DIVERGES BOTH WAYS (`GLM52_P1V3`)
  An earlier revision of this fix guarded one direction only, `real < padded`,
  i.e. it assumed DP padding always makes `q_fp8` LONGER. On a DP-attention IDLE
  rank under MTP draft-extend the inequality inverts. Captured live with this
  patch's own SGLANG_DEBUG_DSA_ROWS=1:

      mode=IDLE q_fp8=(1,32,128) q_offset=2 ntnp=0 agree=False lengths=(2,)
                -> mqa_q=(1,32,128)

  `q_offset` (= sum of dsa_extend_len_cpu) is 2 while only 1 row is materialized
  in q_fp8, so `2 < 1` is false, no trim runs, aiter sizes its logits from
  q_fp8.shape[0] = 1, and fast_topk_v2 gets 1 score row against 2 lengths
  entries -- killing the scheduler rank and dropping the router to
  `active_workers: 1`. A trim cannot fix this direction: there are FEWER query
  rows than lengths entries, so there is nothing to cut. The LENGTHS side is
  clipped instead, through `topk_transform`'s existing `ke_offset` parameter.

  The two directions, and what each does:

    real < padded  the #32762 case: q_fp8 carries DP padding, so TRIM q/weights
                   down to the real count, then re-pad after topk_transform.
                   Trim at _p1v2_real == 0 too -- a DP-IDLE rank has no requests,
                   so the real count is 0 while q_fp8 still carries padding.
                   q_fp8[:0] is a legal empty slice and is what an idle rank
                   should pass; the CUDA path likewise slices unconditionally.
                   DO NOT reintroduce a `0 < q_offset` lower bound here: an
                   earlier revision had one and it made fast_topk_v2 assert on
                   exactly those idle ranks.
    real > padded  the P1V3 case above: nothing to trim, so CLIP the lengths.

  BOUND BEFORE THE BRANCH. `_p1v2_clip` and `_p1v2_rows` are bound before the
  `is_aiter()` test, not inside it. `_p1v2_clip` is read unconditionally at the
  topk_transform call, so a branch-local binding would be a NameError on any
  non-aiter backend; `_p1v2_rows` defaults to `_p1v2_padded`, making the restore
  a no-op if it is ever reached without the branch having run. aiter is
  unconditional on ROCm so neither has fired in practice, but a build-time patch
  should not depend on that.

  This half is what makes the patch survive an AGENTIC workload. The bug needs
  MTP AND DP-attention AND an idle rank at once: an 8-round fixed-shape sweep
  ran MTP + DPA for 660 requests without hitting it, because
  `--random-range-ratio 1.0` keeps batch shapes homogeneous and ranks rarely go
  idle mid-flight. A breathing session population produces ragged batches
  constantly, and the one-directional revision crashed the decode leg roughly
  13 minutes into an agentic benchmark.

  Reproduced twice under that workload (125 s and 766 s into two independent
  runs). With the fix: 0 occurrences of `Expected lengths.size` across two full
  ~4,000 s windows on two different clusters, 0 scheduler exceptions,
  `active_workers: 2` throughout. Second-hand for the reader: it was validated
  as a runtime script before being folded in here, so this is a re-shaping of an
  already-measured fix rather than a fresh one. Evidence in the internal
  reproduction kit -- ask the patch author.

WHY A SCRIPT AND NOT A DIFF. Patches 02 and 04 are `--fuzz=0` context diffs
pinned to one base. This fix is needed on every engine base we ship, and
`dsa_indexer.py` drifts across all of them: v0.5.16/gfx942, v0.5.18/glm53 and
the v0.5.19 gfx950 nightly. Anchoring on source text instead of line numbers
covers all three from one source of truth.

  ANCHORS 1 AND 3 ARE DELIBERATELY LOOSE, so do not "tidy" them back onto the
  adjacent code. Anchor 1 sits on the `contextlib`/`logging` import head rather
  than the typing line, because v0.5.19 dropped `List` from that line. Anchor 3
  starts at `if self.paged_mqa_logits_backend.is_aiter():` rather than the
  `weights.squeeze(2)` above it, because v0.5.19 inserted a nested helper
  between the two. Both forms verified unique on v0.5.16, v0.5.18 and v0.5.19.

  NOTE on the P1V3 anchor. It is the bare `topk_transform(logits,
  self.index_topk)` call. The complete patch was applied and bytecode-verified
  against both release tags on 2026-09-02. If either source drifts, the
  uniqueness check below writes NOTHING and exits 1, which is the intended
  failure: a half-applied fix crashes in the same place the unpatched tree does.

UPSTREAM STATUS (queried with `gh` on 2026-08-01; re-queried and each PR state
re-read 2026-08-03)
  issue          NONE, for either half. Searched sgl-project/sglang for
                 "Expected lengths.size", "DSA padding DP attention" and the
                 GLM-5.2 MTP crash text (P1V2); and for "DSA idle rank MTP draft
                 extend", "q_offset dsa_extend_len", "fast_topk_v2 lengths"
                 (P1V3): no match either time. Weak evidence -- `gh search`
                 matches titles and bodies, not diff content, so an upstream
                 change to this same function that never names the symptom would
                 not surface. The anchor-collision PRs below were found only by
                 searching for the FUNCTION name, not the symptom.
  third-party PR #32762 "[NPU] Fix DSA eager padding mismatch in PD MTP warm-up"
                 (stellaxcpeng, OPEN). SAME BUG CLASS on a different platform,
                 and this patch is written in its shape: one boolean gates both
                 trim and restore, and the returned row count is asserted before
                 the padding is re-attached. It does not touch the aiter branch,
                 so it does not fix HIP.
                 #31683 "[ROCm][MI35X] Enable GLM-5.2-MXFP4 MTP" (long10024070,
                 OPEN). ADJACENT, NOT THE SAME FIX -- same function, but it
                 widens forward_cuda's guard from `seq_lens.numel() == 0` to
                 `is_idle() or ...`. Different symptom; verified against its diff.
                 #30378 / #30427 (both MERGED, in both bases) clamp padded-row
                 seq_lens VALUES in triton_ops/pad.py. This fixes the HIP-side
                 row COUNT -- a different defect in the same area.
  anchor risk    Two OPEN PRs rewrite the code this patch anchors on. Neither
                 fixes this defect; both would make the anchors drift, which
                 fails the build loudly rather than mis-applying.
                 #32738 "[Fix] DSA Indexer: pad heads for DeepGEMM paged MQA
                 logits on decode/target-verify" -- edits _get_topk_paged at the
                 SAME two aiter call sites, replacing `q_fp8` with a
                 `q_fp8_padded` from a new `_pad_heads_for_deep_gemm`. Head-dim
                 padding, orthogonal to our row-count fix, but a direct textual
                 collision with the P1V2 anchor.
                 #31480 "[DSA] Add an arch-independent torch paged-MQA-logits
                 backend" (updated 2026-08-03) -- extracts the backend into
                 paged_mqa_logits_backend.py, i.e. restructures the
                 `is_aiter()` dispatch this whole patch hangs off.
  own PR         #33059 "Fix DSA indexer aiter (HIP) padding mismatch under
                 DP-attention" (dorado269, OPEN, REVIEW_REQUIRED, against main,
                 1 file +16/-4). CI is red only at `pr-gate`, which blocks on a
                 missing `run-ci` label, not on code.

THIS PATCH vs OUR OWN PR #33059. Same defect, overlapping edits. They differ
deliberately:
  base       here: v0.5.16/gfx942 and v0.5.18/gfx950, with every edit verified
             against both release tags. #33059: sglang `main`, whose
             `_get_topk_paged` has drifted further (cutedsl / dg_native branches,
             `_mask_init_and_local_tokens`).
  scope      #33059 carries the `real < padded` half ONLY. The P1V3 clip is NOT
             in it and has not been filed upstream at all -- it should be, and
             this repo already ships the instrumentation that proves the bug.
  form       here: explicit `_p1v2_rows` / `_p1v2_trim` / `_p1v2_clip` locals
             with a post-kernel row-count ASSERT, following #32762's shape.
             #33059: the minimal `q_fp8[:q_offset]` / `weights[:q_offset]` slice
             and a plain `if q_offset < q_fp8.shape[0]` restore.
  local-only Stripped for upstream: the `GLM52_P1V2` / `GLM52_P1V3` markers (the
             build script greps `_p1v2_trim` and `_p1v2_rows` out of the
             BYTECODE to prove the patch landed), the `_p1v2_*` naming, and the
             `SGLANG_DEBUG_DSA_ROWS` logging block. All three are scaffolding
             for this repo's verification loop and carry no upstream value.
  trade-off  The upstream form is smaller and easier to review. Ours fails
             LOUDER: on a shape drift the assert crashes instead of silently
             returning a short tensor -- and it handles a direction #33059 does
             not. Do NOT delete this script when #33059 merges; the clip half
             would go with it.
  validation Ours: 4/4 probe, conc=64 1k/1k 256/256, 2540/2540 in the PD set,
             plus two full ~4,000 s agentic windows on two gfx950 clusters for
             the P1V3 half. #33059's port to `main` was NOT re-run on hardware.

RELATION TO THE INDEXSHARE WORKAROUND. NOT substituted.
`--json-model-override-args '{"index_share_for_mtp_iteration":false}'` removes
the rank divergence that draft_cuda_graph_dp_vote and dsa_page_table_rows
address, but this is an independent bug -- present regardless of IndexShare, and
the arm that validated the override had THIS patch applied.

Self-locating and idempotent. All edits or none: an anchor that is missing or no
longer unique writes NOTHING and fails (exit 1), because a half-applied fix
crashes in the same place the unpatched tree does.
"""

import importlib.util
import sys
from pathlib import Path

_TAG = "[dsa-indexer-rows]"

_REL = "layers/attention/dsa/dsa_indexer.py"

# (anchor, anchor + our edit). Each anchor must occur exactly once; the
# replacement doubles as the already-applied marker.
_EDITS: list[tuple[str, str]] = [
    # `os` for the debug-logging env switch below.
    (
        """from __future__ import annotations

import contextlib
import logging
""",
        """from __future__ import annotations

import contextlib
import logging
import os
""",
    ),
    # Anchored on the two platform flags rather than the whole block: v0.5.16
    # adds `_is_xpu` on the line after `_is_npu`.
    (
        """_is_cuda = is_cuda()
_is_hip = is_hip()
""",
        """_is_cuda = is_cuda()
_is_hip = is_hip()
# Set SGLANG_DEBUG_DSA_ROWS=1 to log the indexer row bookkeeping (padded vs real).
_DSA_DEBUG_ROWS = os.environ.get("SGLANG_DEBUG_DSA_ROWS", "0") == "1"

""",
    ),
    # Trim q/weights to the real row count on the aiter branch only.
    (
        """        if self.paged_mqa_logits_backend.is_aiter():
            logits = aiter_paged_mqa_logits(
                q_fp8,
                kv_cache_fp8,
                weights,
""",
        """        # GLM52_P1V2/P1V3: bound BEFORE the branch, not inside it -- _p1v2_clip is
        # read unconditionally below, so a branch-local binding is a NameError on
        # any non-aiter backend. See "BOUND BEFORE THE BRANCH" in the header.
        _p1v2_trim = False
        _p1v2_clip = False
        _p1v2_real = q_offset
        _p1v2_padded = q_fp8.shape[0]
        _p1v2_rows = _p1v2_padded
        if self.paged_mqa_logits_backend.is_aiter():
            # WHY aiter sizes logits from q_fp8.shape[0], which disagrees with
            # `lengths` BOTH ways. HOW reconcile to min(): trim when longer, clip
            # when shorter. Cases: "DIVERGES BOTH WAYS" in the header.
            _p1v2_rows = min(_p1v2_real, _p1v2_padded)
            _p1v2_trim = _p1v2_rows < _p1v2_padded
            _p1v2_clip = _p1v2_rows < _p1v2_real
            _q_mqa, _w_mqa = q_fp8, weights
            if _p1v2_trim:
                _q_mqa = q_fp8[:_p1v2_rows]
                _w_mqa = weights[:_p1v2_rows]
            if _DSA_DEBUG_ROWS:
                # Cross-check against the padding bookkeeping #32762 keys off.
                # NOT an assert: it does NOT agree with q_offset on the MTP
                # draft-extend path -- that disagreement is the P1V3 case above,
                # and this log is how it was caught.
                _p1v2_ntnp = getattr(forward_batch, "_original_num_tokens", None)
                if _p1v2_ntnp is None:
                    _p1v2_ntnp = forward_batch.num_token_non_padded_cpu
                logger.info(
                    "[dsa-rows] mode=%s q_fp8=%s q_offset=%s ntnp=%s agree=%s "
                    "lengths=%s -> mqa_q=%s",
                    forward_batch.forward_mode,
                    tuple(q_fp8.shape),
                    _p1v2_real,
                    _p1v2_ntnp,
                    _p1v2_ntnp == _p1v2_real,
                    tuple(metadata.get_seqlens_expanded().shape),
                    tuple(_q_mqa.shape),
                )
            logits = aiter_paged_mqa_logits(
                _q_mqa,
                kv_cache_fp8,
                _w_mqa,
""",
    ),
    # GLM52_P1V3: clip the lengths when the rows were clipped rather than trimmed.
    # Anchored on the bare call, which occurs once; the `if` wrapper is the
    # already-applied marker.
    (
        """        topk_result = metadata.topk_transform(logits, self.index_topk)
""",
        """        # GLM52_P1V3: rows were CLIPPED (real > padded), so the lengths this
        # indexes must be clipped to match. Via topk_transform's existing
        # `ke_offset` -- an extension point that only overrides seq_lens_topk --
        # NOT by mutating shared, possibly graph-captured metadata.
        if _p1v2_clip:
            topk_result = metadata.topk_transform(
                logits,
                self.index_topk,
                ke_offset=metadata.get_seqlens_expanded()[:_p1v2_rows],
            )
        else:
            topk_result = metadata.topk_transform(logits, self.index_topk)
""",
    ),
    # Restore the trimmed padding, gated on the SAME boolean.
    (
        """        # Restore possible padding exist in the hidden states.
        if not _is_hip and q_offset < q_fp8.shape[0]:
            pad_len = q_fp8.shape[0] - q_offset
            padding = torch.full(
                (pad_len, topk_result.shape[1]),
""",
        """        # GLM52_P1V2: restore the trimmed padding -- gate on the SAME boolean and
        # assert the row count. BEFORE, a re-derived condition skipped the restore
        # on a shape drift, returning a short tensor instead of crashing.
        #
        # Keyed off _p1v2_rows, not _p1v2_real: under P1V3's clip case the two
        # differ, and it is _p1v2_rows that the kernel actually ran over.
        if _p1v2_trim:
            assert topk_result.shape[0] == _p1v2_rows, (
                "GLM52_P1V3: paged-MQA top-k returned "
                f"{topk_result.shape[0]} rows for {_p1v2_rows} trimmed query "
                f"rows (real={_p1v2_real}, padded={_p1v2_padded}); refusing to "
                "pad a mis-shaped result."
            )
            padding = torch.full(
                (_p1v2_padded - _p1v2_rows, topk_result.shape[1]),
""",
    ),
]


def _srt_dir():
    spec = importlib.util.find_spec("sglang")
    if not spec or not spec.origin:
        return None
    d = Path(spec.origin).parent / "srt"
    return d if d.is_dir() else None


def main():
    srt = _srt_dir()
    if srt is None:
        print(f"{_TAG} sglang not importable — skipping")
        return 0

    f = srt / _REL
    if not f.is_file():
        print(f"{_TAG} {f} is missing — sglang layout changed, re-anchor the patch")
        return 1

    src = out = f.read_text()
    for old, new in _EDITS:
        if new in out:
            continue  # this edit is already in the tree
        found = out.count(old)
        if found != 1:
            where = "absent" if found == 0 else f"{found}x ambiguous"
            print(f"{_TAG} anchor {where} in {_REL}: {old.splitlines()[0]!r}")
            print(f"{_TAG} sglang drifted — re-cut the patch, nothing written")
            return 1
        out = out.replace(old, new, 1)

    if out == src:
        print(f"{_TAG} already present — skipping")
        return 0
    f.write_text(out)
    print(f"{_TAG} patched {f}")
    print(f"{_TAG} aiter paged-MQA now sees the real row count under DP-attention")
    return 0


if __name__ == "__main__":
    sys.exit(main())
