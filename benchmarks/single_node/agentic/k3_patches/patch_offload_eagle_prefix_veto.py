#!/usr/bin/env python3
# PATCH: fix the full-attention EAGLE/MTP "prefix veto" in the native CPU KV-offload
# scheduler that kills the offload READ path for Kimi-K3 DSpark.
#
# FILE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py  (_lookup)
#
# BUG (traced end-to-end against the pinned image, 0.26.1rc1.dev306+gcb8104839):
#   _lookup() iterates full-attention groups first with a PREFIX scan
#   (_maximal_prefix_lookup, from the START), then sliding-window groups with a
#   SUFFIX scan (_sliding_window_lookup, from the END).
#
#   For SLIDING-WINDOW eagle groups the code deliberately OVER-queries one extra
#   chunk (line ~696:  if is_eagle_unverified and sliding_window_size_in_chunks
#   is not None: query_max += tokens_per_chunk) and then trims it back with
#   num_hit_chunks -= 1. Net: it drops exactly the volatile draft tail. Correct.
#
#   For the FULL-ATTENTION eagle group (K3 group 3, is_eagle_group=True,
#   sliding_window_size_in_chunks is None) the extra-chunk over-query is SKIPPED
#   (the gate above is False), yet the SAME num_hit_chunks -= 1 still fires
#   (line ~733). A prefix scan's volatile tail is at the END of the sequence, not
#   at the prefix boundary, so this decrement drops a *legitimately stored,
#   verified* prompt chunk. When the stored prefix is <= 1 chunk it collapses
#   num_hit_chunks to 0 -> the `if num_hit_chunks == 0: return 0` gate vetoes the
#   ENTIRE lookup (all groups) -> offload reads are dead / TTFT never drops.
#
# WHY THE FIX IS CORRECTNESS-SAFE:
#   * offload_prompt_only defaults True (v1/kv_offload/base.py:598) and our serve
#     does not override it. With it True, _calc_num_offloadable_tokens caps the
#     offloadable range at num_prompt_tokens, so storable_chunks() always sees
#     is_decoding=False and NEVER excludes a trailing chunk: the full-attention
#     group only ever persists stable PROMPT chunks. There is no volatile chunk in
#     the store to protect against at load time.
#   * Even with offload_prompt_only=False, storable_chunks() drops the volatile
#     trailing chunk during decode at STORE time (is_eagle_group and is_decoding:
#     num_chunks -= 1), so the store never persists a volatile full-attention
#     chunk regardless. The load-side decrement is redundant either way for a
#     PREFIX scan.
#
# FIX: gate the load-side decrement on the SAME condition as the over-query, i.e.
# only decrement for sliding-window eagle groups. The full-attention eagle group
# keeps its full verified prefix hit.
#
# A/B: default = FIX ON. Set OFFLOAD_EAGLE_PREFIX_VETO=1 in the serve env to
# restore the original upstream (buggy) behavior for a controlled baseline.
#
# Idempotent: guards on a marker; one-time .bak; verifies with py_compile. No GPU.

import os
import py_compile
import vllm

F = os.path.join(os.path.dirname(vllm.__file__),
                 "distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py")

src = open(F).read()

MARKER = "OFFLOAD_EAGLE_PREFIX_VETO"
if MARKER in src:
    print("PATCH(eagle-prefix-veto): already applied; skipping")
    raise SystemExit(0)

# scheduler.py does not import os; add it next to `import time` (idempotent).
if "import os" not in src:
    assert src.count("import time\n") == 1, "expected exactly 1 'import time' anchor"
    src = src.replace("import time\n", "import os\nimport time\n", 1)

OLD = (
    "                    if is_eagle_unverified:\n"
    "                        num_hit_chunks -= 1\n"
    "                        eagle_verified.add(group_idx)\n"
)
assert src.count(OLD) == 1, (
    f"expected exactly 1 eagle decrement block, found {src.count(OLD)}")

NEW = (
    "                    if is_eagle_unverified:\n"
    "                        # FIX(full-attn eagle prefix veto): the SWA eagle path\n"
    "                        # over-queries one extra chunk (query_max += tpc) above\n"
    "                        # and trims it here. The FULL-ATTENTION eagle group runs\n"
    "                        # a PREFIX scan, never over-queries, and (offload_prompt\n"
    "                        # _only / store-side decode drop) never stores a volatile\n"
    "                        # chunk -- so decrementing drops a verified prompt chunk\n"
    "                        # and vetoes <=1-chunk prefixes. Only decrement for SWA.\n"
    "                        # OFFLOAD_EAGLE_PREFIX_VETO=1 restores upstream behavior.\n"
    "                        if (sliding_window_size_in_chunks is not None\n"
    "                                or os.environ.get(\n"
    "                                    \"OFFLOAD_EAGLE_PREFIX_VETO\", \"0\") == \"1\"):\n"
    "                            num_hit_chunks -= 1\n"
    "                        eagle_verified.add(group_idx)\n"
)

import shutil, os
bak = F + ".eagleveto_bak"
if not os.path.exists(bak):
    shutil.copy2(F, bak)
    print(f"PATCH(eagle-prefix-veto): backed up -> {bak}")

src = src.replace(OLD, NEW)
open(F, "w").write(src)
py_compile.compile(F, doraise=True)
print("PATCH(eagle-prefix-veto): applied + py_compile OK")
print("  default = FIX ON (full-attn eagle group keeps full verified prefix)")
print("  set OFFLOAD_EAGLE_PREFIX_VETO=1 to A/B the original upstream behavior")
