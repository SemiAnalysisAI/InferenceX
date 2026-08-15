#!/usr/bin/env python3
"""Reproduce unified-v2's hybrid EAGLE/SimpleCPU-offload scheduler fixes."""

import os
import py_compile
from pathlib import Path


dist = Path(os.environ.get("DIST", "/usr/local/lib/python3.12/dist-packages"))
scheduler_path = (
    dist / "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
)
kv_utils_path = dist / "vllm/v1/core/kv_cache_utils.py"

scheduler = scheduler_path.read_text()
scheduler_old = '''                    if is_eagle_unverified:
                        num_hit_chunks -= 1
                        eagle_verified.add(group_idx)
'''
scheduler_new = '''                    if is_eagle_unverified:
                        # FIX(full-attn eagle prefix veto): the SWA eagle path
                        # over-queries one extra chunk (query_max += tpc) above
                        # and trims it here. The FULL-ATTENTION eagle group runs
                        # a PREFIX scan, never over-queries, and (offload_prompt
                        # _only / store-side decode drop) never stores a volatile
                        # chunk -- so decrementing drops a verified prompt chunk
                        # and vetoes <=1-chunk prefixes. Only decrement for SWA.
                        # OFFLOAD_EAGLE_PREFIX_VETO=1 restores upstream behavior.
                        if (sliding_window_size_in_chunks is not None
                                or os.environ.get(
                                    "OFFLOAD_EAGLE_PREFIX_VETO", "0") == "1"):
                            num_hit_chunks -= 1
                        eagle_verified.add(group_idx)
'''

if "FIX(full-attn eagle prefix veto)" not in scheduler:
    if scheduler.count("import time\n") != 1 or scheduler.count(scheduler_old) != 1:
        raise SystemExit("ERROR: scheduler patch anchors missing or duplicated")
    scheduler = scheduler.replace("import time\n", "import os\nimport time\n", 1)
    scheduler = scheduler.replace(scheduler_old, scheduler_new, 1)
    scheduler_path.write_text(scheduler)

kv_utils = kv_utils_path.read_text()
call_old = '''            groups.append(KVCacheGroupSpec([name], aligned))

    return groups


def generate_scheduler_kv_cache_config(
'''
call_new = '''            groups.append(KVCacheGroupSpec([name], aligned))

    # PATCH(#52047): annotate the draft KV group on the hybrid path so the
    # KV-offload scheduler does not flag Mamba groups as draft groups.
    _annotate_eagle_groups_from_draft_spec(vllm_config, groups)
    _warn_if_unannotated_eagle_mamba(vllm_config, groups)
    return groups


def _annotate_eagle_groups_from_draft_spec(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> None:
    """PATCH(#52047): flag the draft (EAGLE/MTP) KV group on the hybrid path.

    The general multi-group path does not know which group belongs to the draft
    model. The draft attention layer marks its spec with
    ``non_causal_multi_token_decode=True`` (propagated through
    MLAAttentionSpec.merge), so use that marker to annotate only the real draft
    group -- avoiding the offload scheduler's flag-all fallback that would
    otherwise treat Mamba groups as draft groups.
    """
    spec_config = vllm_config.speculative_config
    if spec_config is None or not spec_config.use_eagle():
        return
    for group in kv_cache_groups:
        if getattr(group.kv_cache_spec, "non_causal_multi_token_decode", False):
            group.is_eagle_group = True


def _warn_if_unannotated_eagle_mamba(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> None:
    """PATCH(#52047): warn when spec is on but no group was identified as draft.

    This is exactly the condition that triggers the offload scheduler's
    flag-all fallback, which would wrongly treat Mamba groups as draft groups.
    """
    spec_config = vllm_config.speculative_config
    if spec_config is None or not spec_config.use_eagle():
        return
    if any(getattr(g, "is_eagle_group", False) for g in kv_cache_groups):
        return
    mamba_groups = [
        idx
        for idx, group in enumerate(kv_cache_groups)
        if isinstance(group.kv_cache_spec, MambaSpec)
    ]
    if not mamba_groups:
        return
    logger.warning(
        "Speculative decoding (method=%s) is enabled but no KV cache group "
        "could be identified as the draft model's, so every group -- including "
        "Mamba groups %s -- may be treated as a draft group by the KV-offload "
        "scheduler. External prefix-cache reads may be suppressed.",
        spec_config.method,
        mamba_groups,
    )


def generate_scheduler_kv_cache_config(
'''

if "PATCH(#52047): annotate the draft KV group" not in kv_utils:
    if kv_utils.count(call_old) != 1:
        raise SystemExit("ERROR: hybrid KV-group patch anchor missing or duplicated")
    kv_utils = kv_utils.replace(call_old, call_new, 1)
    kv_utils_path.write_text(kv_utils)

checks = {
    scheduler_path: ("FIX(full-attn eagle prefix veto)", "OFFLOAD_EAGLE_PREFIX_VETO"),
    kv_utils_path: (
        "PATCH(#52047): annotate the draft KV group",
        "_annotate_eagle_groups_from_draft_spec",
        "_warn_if_unannotated_eagle_mamba",
    ),
}
for path, markers in checks.items():
    updated = path.read_text()
    missing = [marker for marker in markers if marker not in updated]
    if missing:
        raise SystemExit(f"ERROR: missing markers in {path}: {missing}")
    py_compile.compile(str(path), doraise=True)
print("hybrid EAGLE/SimpleCPU-offload patches OK")
