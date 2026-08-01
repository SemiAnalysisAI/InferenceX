"""Backport vLLM Kimi DS prefix-cache fixes #49291 and #50153."""

from __future__ import annotations

import importlib.util
from pathlib import Path

OLD_IMPORT = """from vllm.model_executor.layers.mamba.mamba_utils import (
    get_conv_copy_spec,
    is_conv_state_dim_first,
)
"""

OLD_GUARD = """            # The fused copy kernels shift conv windows assuming the SD layout;
            # the DS layout cannot express a >0 spec-decode shift as a single
            # contiguous copy (mirrors get_conv_copy_spec's NotImplementedError).
            if get_conv_copy_spec in copy_funcs and is_conv_state_dim_first():
                assert self.vllm_config.speculative_config is None, (
                    "DS conv state layout does not support mamba align state "
                    "copies with speculative decoding"
                )
"""

NEW_COMMENT = """            # Both SD and DS conv layouts support a >0 spec-decode shift: the
            # fused pre-copy kernel applies the accepted-token window shift per
            # layout. Backported from vLLM #49291 for the V2 model runner.
"""


BLOCKS_PER_SW = """        self.blocks_per_sw = [
            cdiv(n_tokens, block_size) + 1 if n_tokens else 0
            for n_tokens, block_size in sw_sizes_tokens
        ]
"""

BLOCKS_PER_SW_WITH_SSM = (
    BLOCKS_PER_SW
    + """
        # Trailing scratch slots that mamba managers co-allocate per request
        # for speculative decoding; None for non-SSM groups.
        self._ssm_spec_blocks = [
            g.kv_cache_spec.num_speculative_blocks
            if isinstance(g.kv_cache_spec, MambaSpec)
            else None
            for g in kv_cache_config.kv_cache_groups
        ]
        # Only "all" mode keeps a state per block position; the other modes
        # keep a single running state in the last non-speculative slot.
        self._ssm_state_slots_are_positional = (
            vllm_config.cache_config.mamba_cache_mode == "all"
        )
"""
)

NEW_CLIP_METHOD = '''    def get_exchange_clipped_blocks(
        self, block_ids: BlockIds, clip_ssm: bool = True
    ) -> BlockIds:
        """Clip a request's block lists down to the transferable blocks.

        Sliding-window groups keep only the in-window tail: the KV cache
        manager allocates blocks for the entire sequence length and cleans up
        out-of-window blocks only prior to the `request_finished_all_groups`
        hook.

        SSM groups keep only their state-bearing slots: the trailing
        speculative scratch slots always go, and in single-state cache modes
        so does everything before the running state (null placeholders and
        the previous step's superseded state). "all" mode keeps its remaining
        slots, which the worker pairs position-wise.

        Use this at every block-id exchange point. Pass ``clip_ssm=False``
        for per-step partial lists (host-buffer save), where the SSM strip
        does not apply.
        """
        if len(block_ids) == 0 or not self._is_hma_required:
            # No blocks to clip eg Full prefix cache hit or not a hybrid model.
            return block_ids
        # NOTE (NickLucche) This logic is currently handled at the connector level
        # because offloading connectors might want to receive the whole sequence even
        # for SWA groups. We will abstract this logic once the interface is more stable
        assert len(block_ids) == len(self.blocks_per_sw), (
            "Number of KV cache groups must match"
        )
        clipped = []
        for i, blocks in enumerate(block_ids):
            if n_sw := self.blocks_per_sw[i]:
                blocks = blocks[-n_sw:]
            elif (
                clip_ssm
                and blocks
                and (n_spec_blocks := self._ssm_spec_blocks[i]) is not None
            ):
                if n_spec := min(n_spec_blocks, len(blocks) - 1):
                    blocks = blocks[:-n_spec]
                if not self._ssm_state_slots_are_positional:
                    # Never empty: downstream reads that as a full prefix hit.
                    blocks = blocks[-1:]
            clipped.append(blocks)
        return tuple(clipped)
'''

OLD_BASE_SAVE = """            clipped_block_id_groups = self.get_sw_clipped_blocks(new_block_id_groups)
"""

NEW_BASE_SAVE = """            clipped_block_id_groups = self.get_exchange_clipped_blocks(
                new_block_id_groups, clip_ssm=False
            )
"""

OLD_WORKER_PAIRING = """                if (
                    _is_ssm_spec(self._group_spec_types[i])
                    and num_local_blocks < num_remote_blocks
                ):
                    # NOTE (NickLucche): With prefix caching on SSM, (remote) blocks
                    # prior to the last one are placeholders (null blocks). Mind that
                    # this doesn't really impact transfer, as we only still care about
                    # the last "block", the full in-place state.
                    assert num_local_blocks == 1, "SSM can only have one local block"
                    remote_block_ids[i] = remote_group[-num_local_blocks:]
"""

NEW_WORKER_PAIRING = """                if _is_ssm_spec(self._group_spec_types[i]):
                    if num_local_blocks == num_remote_blocks:
                        continue
                    # Only state-bearing slots reach here, single-state modes
                    # just one (see get_exchange_clipped_blocks), so differing
                    # counts mean position-indexed "all"-mode lists. A longer
                    # remote list carries earlier positions the local side
                    # already has (prefix hit) -> read its tail; a longer local
                    # list holds the position D recomputes itself, which gets
                    # no remote state.
                    assert num_local_blocks - num_remote_blocks <= 1, (
                        f"Group {i}: unpairable SSM state slots, "
                        f"local={num_local_blocks} remote={num_remote_blocks}"
                    )
                    num_blocks = min(num_local_blocks, num_remote_blocks)
                    if num_local_blocks < num_remote_blocks:
                        remote_block_ids[i] = remote_group[-num_blocks:]
                    else:
                        local_block_ids[i] = local_block_ids[i][:num_blocks]
"""

OLD_REMOTE_PREFILL_MATCH = """        if params is not None and params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            token_ids = request.prompt_token_ids or []
            actual = self._get_remote_prefill_token_count(len(token_ids))
"""

NEW_REMOTE_PREFILL_MATCH = """        if (
            params is not None
            and params.get("do_remote_prefill")
            and params.get("remote_block_ids")
            and any(params["remote_block_ids"])
        ):
            # Remote prefill: get all prompt blocks from remote. A producer can
            # export empty NIXL block groups after satisfying the complete prefix
            # from another MultiConnector child (for example Mooncake). Do not
            # claim that request through NIXL: the next connector must load it.
            token_ids = request.prompt_token_ids or []
            actual = self._get_remote_prefill_token_count(len(token_ids))
"""


def replace_once(
    target: Path,
    source: str,
    old: str,
    new: str,
    *,
    already_patched_marker: str,
) -> str:
    """Apply one exact source backport, while remaining safely idempotent."""
    if already_patched_marker in source:
        return source
    if old in source:
        return source.replace(old, new, 1)
    raise RuntimeError(f"Unexpected vLLM source layout in {target}")


def replace_clip_method(target: Path, source: str) -> str:
    """Replace the clipping helper without depending on docstring wrapping."""
    if "def get_exchange_clipped_blocks(" in source:
        return source
    start_marker = "    def get_sw_clipped_blocks("
    end_marker = "    def set_xfer_handshake_metadata("
    start = source.find(start_marker)
    end = source.find(end_marker, start)
    if start < 0 or end < 0:
        raise RuntimeError(f"Unexpected vLLM source layout in {target}")
    return source[:start] + NEW_CLIP_METHOD + "\n" + source[end:]


def patch_mamba_runner(package_dir: Path) -> None:
    target = package_dir / "v1/worker/gpu/model_states/mamba_hybrid.py"
    source = target.read_text()
    source = replace_once(
        target,
        source,
        OLD_IMPORT,
        "",
        already_patched_marker="Both SD and DS conv layouts support",
    )
    source = replace_once(
        target,
        source,
        OLD_GUARD,
        NEW_COMMENT,
        already_patched_marker="Both SD and DS conv layouts support",
    )
    compile(source, str(target), "exec")
    target.write_text(source)
    print(f"[kimi-k3-ds-prefix-cache] Patched {target}")


def patch_nixl_connector(package_dir: Path) -> None:
    nixl_dir = package_dir / "distributed/kv_transfer/kv_connector/v1/nixl"
    base_scheduler = nixl_dir / "base_scheduler.py"
    base_worker = nixl_dir / "base_worker.py"
    pull_scheduler = nixl_dir / "pull_scheduler.py"
    push_scheduler = nixl_dir / "push_scheduler.py"

    source = base_scheduler.read_text()
    source = replace_once(
        base_scheduler,
        source,
        BLOCKS_PER_SW,
        BLOCKS_PER_SW_WITH_SSM,
        already_patched_marker="self._ssm_spec_blocks = [",
    )
    source = replace_clip_method(base_scheduler, source)
    source = replace_once(
        base_scheduler,
        source,
        OLD_BASE_SAVE,
        NEW_BASE_SAVE,
        already_patched_marker="new_block_id_groups, clip_ssm=False",
    )
    compile(source, str(base_scheduler), "exec")
    base_scheduler.write_text(source)

    source = base_worker.read_text()
    source = replace_once(
        base_worker,
        source,
        OLD_WORKER_PAIRING,
        NEW_WORKER_PAIRING,
        already_patched_marker="unpairable SSM state slots",
    )
    compile(source, str(base_worker), "exec")
    base_worker.write_text(source)

    for target in (pull_scheduler, push_scheduler):
        source = target.read_text()
        old_count = source.count("self.get_sw_clipped_blocks(")
        new_count = source.count("self.get_exchange_clipped_blocks(")
        if old_count == 2:
            source = source.replace(
                "self.get_sw_clipped_blocks(",
                "self.get_exchange_clipped_blocks(",
            )
        elif old_count != 0 or new_count != 2:
            raise RuntimeError(f"Unexpected vLLM source layout in {target}")
        compile(source, str(target), "exec")
        target.write_text(source)

    source = pull_scheduler.read_text()
    source = replace_once(
        pull_scheduler,
        source,
        OLD_REMOTE_PREFILL_MATCH,
        NEW_REMOTE_PREFILL_MATCH,
        already_patched_marker="A producer can\n            # export empty NIXL block groups",
    )
    compile(source, str(pull_scheduler), "exec")
    pull_scheduler.write_text(source)

    print(f"[kimi-k3-ds-prefix-cache] Patched NIXL connector in {nixl_dir}")


def patch_package(package_dir: Path) -> None:
    patch_mamba_runner(package_dir)
    patch_nixl_connector(package_dir)


def main() -> None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Cannot locate the installed vllm package")

    patch_package(Path(next(iter(spec.submodule_search_locations))))


if __name__ == "__main__":
    main()
