#!/usr/bin/env python3
"""Backport of sgl-project/sglang#31478 (issue #31071): broadcast the EAGLE
greedy-verify accept decision across TP ranks.

SGLang's ``eagle_sample()`` broadcasts the verify result across TP ranks only in
the *sampling* (non-greedy) branch. On AMD/HIP the *greedy* branch is ALWAYS
taken (its condition is ``sampling_info.is_all_greedy or _is_npu or _is_hip``),
and that branch does NOT broadcast. With ``tp>1`` and no dp-attention (pure-TP /
EP1 decode) the per-rank ``torch.argmax`` diverges on near-tied logits: ranks
then accept a different number of draft tokens, committed seq_lens / batch shapes
diverge across ranks, and the next tensor-parallel collective either deadlocks
(server hang -> /health dies) or commits divergent tokens (greedy gsm8k drops
from ~0.96 to ~0.88/0.85). Upstream fixes this by broadcasting the finalized
decision for both branches; this backport adds the identical broadcast to the
greedy branch of the installed source.

Design notes:
  * Inert on dp-attention (DEP) topologies: there ``get_attention_tp_group()``
    has world_size 1, so the ``world_size > 1`` guard skips the broadcast --
    which is exactly why DEP8 arms were already correct without this fix.
  * Idempotent: no-op if our marker OR the upstream marker is already present
    (so a future image that ships #31478 is left untouched).
  * Pure insertion at a unique anchor (the ``verify_tree_greedy_func(...)`` call
    site) -- no deletion, so it is robust across the v0.5.14 / v0.5.15 source
    shapes that share that call site.

Exit codes: 0 = applied or already-present (no action needed); 2 = the installed
source shape was not recognized (nothing changed) -- the caller decides whether
that is fatal for the scenario at hand.
"""
import sys

OUR_MARKER = "# Sync the greedy verify decision across TP ranks"
UPSTREAM_MARKER = "# Sync the finalized verify decision across TP ranks"

# Unique anchor: the tail of the greedy-branch verify_tree_greedy_func(...) call.
# `target_predict=` and `topk=verify_input.tree_topk` appear only in the greedy
# branch (the sampling branch uses retrive_*/uniform_samples/target_probs).
ANCHOR = (
    "            target_predict=target_predict,\n"
    "            topk=verify_input.tree_topk,\n"
    "        )\n"
)

INSERT = (
    "\n"
    "        # Sync the greedy verify decision across TP ranks. Backport of\n"
    "        # sgl-project/sglang#31478 (issue #31071). On AMD/HIP the greedy\n"
    "        # branch is always taken; per-rank torch.argmax can diverge on\n"
    "        # near-tied logits, so without this broadcast tp>1 without\n"
    "        # dp-attention deadlocks or commits divergent tokens. Inert on\n"
    "        # dp-attention (attn_tp world_size 1 -> guard skips the broadcast).\n"
    "        tp_group = (\n"
    "            get_attention_tp_group() if is_dp_attention_enabled() else get_tp_group()\n"
    "        )\n"
    "        if tp_group.world_size > 1:\n"
    "            tp_group.broadcast(predict, src=0)\n"
    "            tp_group.broadcast(accept_index, src=0)\n"
    "            tp_group.broadcast(num_correct_drafts, src=0)\n"
)


def main() -> int:
    try:
        import sglang.srt.speculative.eagle_utils as m
    except Exception as e:  # noqa: BLE001
        print(f"[PATCH] eagle_utils: import failed ({e}); NOT patched", flush=True)
        return 2

    path = m.__file__
    with open(path, encoding="utf-8") as f:
        src = f.read()

    if OUR_MARKER in src or UPSTREAM_MARKER in src:
        print(
            f"[PATCH] eagle_utils: TP-broadcast greedy-verify fix already present "
            f"in {path}; skipping",
            flush=True,
        )
        return 0

    n = src.count(ANCHOR)
    if n != 1:
        print(
            f"[PATCH] eagle_utils: greedy verify_tree_greedy_func anchor found {n}x "
            f"(expected 1) in {path}; source shape unrecognized -- NOT patched",
            flush=True,
        )
        return 2

    new = src.replace(ANCHOR, ANCHOR + INSERT, 1)
    if OUR_MARKER not in new or new == src:
        print(
            f"[PATCH] eagle_utils: post-transform verification failed for {path}; "
            f"NOT patched",
            flush=True,
        )
        return 2

    with open(path, "w", encoding="utf-8") as f:
        f.write(new)
    print(
        f"[PATCH] eagle_utils: applied greedy-verify TP-broadcast fix "
        f"(sglang#31478 / #31071 backport) to {path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
