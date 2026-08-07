#!/usr/bin/env python3
"""Allow DeepSeek-V4 shared-expert fusion (FSE) under expert parallelism.

Upstream `_shared_expert_fusion_possible` rejects EP outright and
`_should_fuse_shared_expert` additionally requires TP=8. Neither is a kernel
limitation:

* EP: aiter's `init_aiter_topK_meta_data` has a dedicated `is_EP` branch that
  hands the shared-expert slot to rank ``i % ep_size`` for global-batch token
  ``i``, so each token's shared expert runs exactly once. That is only sound if
  every EP rank sees the same token array in the same order. With the
  allgather_reducescatter all2all backend it does: `MoERunner._forward_impl`
  calls `_maybe_dispatch` (all-gather across DP ranks) *before*
  `_apply_quant_method` does the routing, so the indices the modulo runs on are
  global-batch indices, identical on every rank. Real dispatch/combine backends
  (mori, deepep) give each rank a different token subset, where the same modulo
  would silently drop the shared contribution for most tokens -- so this only
  lifts the gate for allgather_reducescatter.

* TP=8: the reason string says "not the *validated* TP=8 layout", i.e. scope of
  testing, not capability. FSE pads the FP8 shared expert to
  intermediate_size_per_partition = moe_intermediate_size // tp_size and needs
  that to be a multiple of the 128x128 FP8 block. 3072 // tp is a multiple of
  128 for tp in {1,2,4,8,12,16,24}, so the check is dropped in favour of the
  divisibility assertion `_pad_and_expand_native_fp8_shared_expert` already
  performs.

The `moe_backend != "aiter"` check is deliberately left alone: it *is* a kernel
limitation. FSE runs an FP8 shared expert as slot 385 of an otherwise MXFP4
grouped GEMM, which only aiter's heterogeneous kernel accepts. MegaMoE's
`MegaMoEV2(quant="a8w4", ...)` takes a single weight pair with one quant mode
and has no heterogeneous slot.

The failure mode of getting the EP reasoning wrong is silently wrong numerics,
not a crash, so this also injects a runtime assertion that fires if FSE is
active under EP with anything other than allgather_reducescatter.

Idempotent: running twice is a no-op. Usage:
    python3 patch_fse_ep.py <path-to-model.py>
"""

import sys
from pathlib import Path

OLD_GATE = """    if parallel_config is None:
        parallel_config = get_current_vllm_config().parallel_config
    return not parallel_config.enable_expert_parallel
"""

NEW_GATE = '''    if parallel_config is None:
        parallel_config = get_current_vllm_config().parallel_config
    if not parallel_config.enable_expert_parallel:
        return True
    # Under EP the fused shared expert is claimed by rank i % ep_size for
    # global-batch token i (aiter's init_aiter_topK_meta_data, is_EP branch).
    # That is only correct when every EP rank routes over the same token array
    # in the same order. allgather_reducescatter satisfies this: MoERunner
    # all-gathers across DP ranks before routing, so the modulo runs on global
    # batch indices. A real dispatch/combine backend hands each rank a distinct
    # token subset, where the same modulo would silently drop the shared-expert
    # contribution for most tokens -- reject those rather than return wrong
    # numbers.
    return (
        getattr(parallel_config, "all2all_backend", None)
        == "allgather_reducescatter"
    )
'''

OLD_TP = """    if getattr(parallel_config, "tensor_parallel_size", None) != 8:
        reasons.append("tensor parallelism is not the validated TP=8 layout")
"""

NEW_TP = """    # No TP check: FSE pads the FP8 shared expert to
    # moe_intermediate_size // tp_size, which
    # _pad_and_expand_native_fp8_shared_expert already asserts is a multiple of
    # the 128x128 FP8 block. 3072 // tp satisfies that for every TP size this
    # model runs at, so the former "validated TP=8 only" gate was scope of
    # testing rather than a constraint.
"""

OLD_REASON = '''            "shared-expert fusion is off (needs ROCm, a shared expert, "
            "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1, and no expert parallelism)"
'''

NEW_REASON = '''            "shared-expert fusion is off (needs ROCm, a shared expert, "
            "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1, and either no expert "
            "parallelism or the allgather_reducescatter all2all backend)"
'''

# Anchor for the runtime assertion: the point where the decision is consumed.
OLD_USE = """        self.fuse_heterogeneous_shared_expert = _should_fuse_shared_expert(vllm_config)
"""

NEW_USE = '''        self.fuse_heterogeneous_shared_expert = _should_fuse_shared_expert(vllm_config)
        if self.fuse_heterogeneous_shared_expert:
            _pc = vllm_config.parallel_config
            if _pc.enable_expert_parallel and (
                getattr(_pc, "all2all_backend", None) != "allgather_reducescatter"
            ):
                # Belt and braces: the gate above should already have caught
                # this. Wrong numerics here would be silent, so refuse to run.
                raise RuntimeError(
                    "Shared-expert fusion under expert parallelism requires the "
                    "allgather_reducescatter all2all backend (the per-rank "
                    "modulo claim assumes a globally consistent token order); "
                    f"got {getattr(_pc, 'all2all_backend', None)!r}."
                )
            logger.info_once(
                "DeepSeek V4 shared-expert fusion active with "
                "expert_parallel=%s, tp_size=%s, all2all_backend=%s.",
                _pc.enable_expert_parallel,
                getattr(_pc, "tensor_parallel_size", None),
                getattr(_pc, "all2all_backend", None),
            )
'''

EDITS = [
    ("EP gate", OLD_GATE, NEW_GATE),
    ("TP=8 check", OLD_TP, NEW_TP),
    ("reason string", OLD_REASON, NEW_REASON),
    ("runtime assertion", OLD_USE, NEW_USE),
]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    src = path.read_text()

    applied, skipped = [], []
    for name, old, new in EDITS:
        if new in src:
            skipped.append(name)
            continue
        count = src.count(old)
        if count != 1:
            print(
                f"error: anchor for {name!r} found {count} times, expected 1. "
                "The image's model.py does not match what this patch was "
                "written against; refusing to guess.",
                file=sys.stderr,
            )
            return 1
        src = src.replace(old, new)
        applied.append(name)

    if applied:
        path.write_text(src)
    print(f"applied: {applied or 'none'}")
    print(f"already present: {skipped or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
