#!/usr/bin/env bash

set -euo pipefail

# Backport fixes merged after the 0618 MiniMax M3 image was built.
python3 - <<'PYEOF'
import importlib.util
import pathlib


def apply_exact_patches(
    target: pathlib.Path,
    patches: list[tuple[str, str]],
    label: str,
) -> None:
    src = target.read_text()
    changed = False

    for old, new in patches:
        if new in src:
            continue
        if src.count(old) != 1:
            raise RuntimeError(f"Expected exactly one {label} patch anchor in {target}")
        src = src.replace(old, new, 1)
        changed = True

    if changed:
        target.write_text(src)
        print(f"[{label}] patched: {target}")
    else:
        print(f"[{label}] already applied: {target}")


spec = importlib.util.find_spec("vllm")
if spec is None or not spec.submodule_search_locations:
    raise RuntimeError("Could not locate the installed vllm package")

vllm_root = pathlib.Path(next(iter(spec.submodule_search_locations)))

# TP1 data-parallel-attention workers retain a non-contiguous head stride in
# the persistent CUDA-graph top-k buffer. Materialize the slice before the
# MiniMax M3 MSA CSR builder consumes it.
msa_target = (
    vllm_root
    / "models"
    / "minimax_m3"
    / "nvidia"
    / "sparse_attention_msa.py"
)
apply_exact_patches(
    msa_target,
    [
        (
            "            prefill_topk = topk[:, nd:num_tokens, :]\n",
            "            prefill_topk = topk[:, nd:num_tokens, :].contiguous()\n",
        )
    ],
    "minimax-m3-msa-contiguity",
)

# vllm-project/vllm#45879: heterogeneous TP must validate SPLIT regions using
# per-rank KV heads. MiniMax M3 has four KV heads, so TP8 replicates one head
# per rank instead of scaling block length by the raw TP ratio of eight.
nixl_target = (
    vllm_root
    / "distributed"
    / "kv_transfer"
    / "kv_connector"
    / "v1"
    / "nixl"
    / "base_worker.py"
)
apply_exact_patches(
    nixl_target,
    [
        (
            """        # only allow the number of blocks to differ; SPLIT regions scale with
        # tp_ratio. Mamba uses the ssm_sizes counterpart, so skip block_len here.
""",
            """        # only allow the number of blocks to differ; SPLIT regions scale with
        # the per-rank KV head ratio rather than the raw tp_ratio, because GQA
        # replication caps per-rank heads at 1 when tp > total_kv_heads
        # (issue #45330). Mamba uses the ssm_sizes counterpart, so skip here.
""",
        ),
        (
            """            model_replicated = self.use_mla or self.transfer_topo.is_kv_replicated(
                remote_engine_id
            )
            for i, local_len in enumerate(self.block_len_per_layer):
""",
            """            model_replicated = self.use_mla or self.transfer_topo.is_kv_replicated(
                remote_engine_id
            )
            total_kv_heads = self.transfer_topo.total_num_kv_heads
            local_heads = self.transfer_topo.local_physical_heads
            remote_heads = max(1, total_kv_heads // remote_tp_size)
            for i, local_len in enumerate(self.block_len_per_layer):
""",
        ),
        (
            """                elif tp_ratio > 0:
                    assert remote_len == (local_len * tp_ratio) // block_size_ratio, (
                        f"SPLIT region {i}: remote P KV block_len {remote_len} "
                        f"must equal local {local_len} * tp_ratio {tp_ratio} "
                        f"// block_size_ratio {block_size_ratio}."
                    )
""",
            """                elif tp_ratio > 0:
                    assert (
                        remote_len
                        == (local_len * remote_heads // local_heads) // block_size_ratio
                    ), (
                        f"SPLIT region {i}: remote P KV block_len {remote_len} "
                        f"must equal local {local_len} * remote_heads "
                        f"{remote_heads} // local_heads {local_heads} "
                        f"// block_size_ratio {block_size_ratio}."
                    )
""",
        ),
        (
            """                    assert remote_len == local_len // (-tp_ratio), (
                        f"SPLIT region {i}: remote P KV block_len "
                        f"{remote_len} must equal local {local_len} "
                        f"// |tp_ratio| {-tp_ratio}."
                    )
""",
            """                    assert remote_len == local_len * remote_heads // local_heads, (
                        f"SPLIT region {i}: remote P KV block_len {remote_len} "
                        f"must equal local {local_len} * remote_heads "
                        f"{remote_heads} // local_heads {local_heads}."
                    )
""",
        ),
    ],
    "minimax-m3-nixl-gqa",
)
PYEOF
