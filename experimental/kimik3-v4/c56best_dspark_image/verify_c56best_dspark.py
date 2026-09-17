#!/usr/bin/env python3
"""Fail if the PR 53901/52921/56621 or KV propagation backports are incomplete."""

from __future__ import annotations

from pathlib import Path
import py_compile


SITE = Path("/usr/local/lib/python3.12/dist-packages")
REQUIRED = {
    "vllm/config/speculative.py": (
        "pipeline_parallel_size=1",
        "Drafter runs on the last PP stage only",
    ),
    "vllm/models/kimi_k3/amd/linear.py": (
        "supports_aux_hidden_states_over_pp",
    ),
    "vllm/v1/worker/gpu/model_runner.py": (
        "[50514] isolate draft cudagraph capture across PP",
    ),
    "vllm/v1/worker/gpu/spec_decode/dspark/utils.py": (
        "[50514] draft loads only on the last PP stage",
    ),
    "vllm/v1/core/kv_cache_utils.py": (
        "K3 PP projection preserves global DSpark draft group",
        "K3 PP scheduler merges global DSpark draft groups",
    ),
    "vllm/v1/core/kv_cache_coordinator.py": (
        "K3 DSpark replay boundary",
    ),
    "vllm/v1/core/sched/scheduler.py": (
        "use_eagle_block_drop",
    ),
    "vllm/v1/simple_kv_offload/sizing.py": (
        "sync_num_offload_blocks_across_workers",
        "view_complete_kv_cache_regions",
    ),
    "vllm/v1/simple_kv_offload/worker.py": (
        "local_num_offload_blocks",
        "sync_num_offload_blocks_across_workers",
        "_store_submitted",
        "No-forward steps skip the normal wait_for_save hook.",
    ),
    "vllm/v1/simple_kv_offload/manager.py": (
        "aligned_num_cpu_blocks",
    ),
    "vllm/v1/worker/gpu_worker.py": (
        "get_simple_cpu_offload_num_cpu_blocks",
    ),
    "vllm/v1/engine/core.py": (
        "get_simple_cpu_offload_num_blocks",
    ),
}

for relative, markers in REQUIRED.items():
    path = SITE / relative
    source = path.read_text()
    for marker in markers:
        if marker not in source:
            raise RuntimeError(f"{relative}: missing required marker {marker!r}")
    py_compile.compile(str(path), doraise=True)

kv_source = (SITE / "vllm/v1/core/kv_cache_utils.py").read_text()
projection_start = kv_source.index("def _project_kv_cache_groups_to_worker(")
projection_end = kv_source.index("\ndef ", projection_start + 5)
projection = kv_source[projection_start:projection_end]
if "is_eagle_group=group.is_eagle_group and bool(worker_layer_names)" in projection:
    raise RuntimeError("PP projection still clears the global draft-group flag")

annotate_start = kv_source.index("def _annotate_eagle_groups(")
annotate_end = kv_source.index("\ndef ", annotate_start + 5)
annotate = kv_source[annotate_start:annotate_end]
if "not spec_config.use_eagle_block_drop()" in annotate:
    raise RuntimeError("draft-group annotation is still coupled to block drop")

print("c56best PR 53901 + PR 52921 + PR 56621 verification passed")
