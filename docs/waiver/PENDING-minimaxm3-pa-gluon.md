# Inference-engine patch waiver — PR #PENDING

> **Rename this file to `<PR_NUMBER>.md` in the PR that introduces the patch.** The checklist requires the
> waiver to be named after that PR; the number is not known while this work sits on a branch with no PR.

Filed per [`docs/PR_REVIEW_CHECKLIST.md`](../PR_REVIEW_CHECKLIST.md) (image-provenance / patch item): a
benchmark script in this PR patches the pinned serving-stack image before serving, which the checklist
prohibits unless covered by a filled-out waiver.

## Config covered

- **Master config entry:** `minimaxm3-fp4-mi355x-vllm-agentic-mtp` in [`configs/amd-master.yaml`](../../configs/amd-master.yaml)
- **Pinned image:** `vllm/vllm-openai-rocm:v0.27.1`
- **Model:** `amd/MiniMax-M3-MXFP4`
- **Patch entrypoint:** `bash "$(dirname "$0")/apply_minimaxm3_container_patches.sh"` invoked from
  [`benchmarks/single_node/agentic/minimaxm3_fp4_mi355x_mtp.sh`](../../benchmarks/single_node/agentic/minimaxm3_fp4_mi355x_mtp.sh)
- **Patch script:** [`benchmarks/single_node/agentic/apply_minimaxm3_container_patches.sh`](../../benchmarks/single_node/agentic/apply_minimaxm3_container_patches.sh)

## What is patched

Two Python files under `dist-packages`, as one idempotent, marker-gated, offline pristine→container diff.
No network, no pip, no rebuilt wheels, no binary artifacts.

| Upstream PR | What it changes | Notes |
|---|---|---|
| vllm #52849 | Enable AITER PA gluon decode for MiniMax-M3 MTP and dense layers: routes uniform multi-token decodes to `pa_decode_gluon` instead of asserting, moves the shuffled KV cache to a K/V-separated layout (+ a matching `get_kv_cache_stride_order`), advertises the 128-token page only when gluon can serve it, and sizes the fp8 KV scales from a layer the metadata builder owns (`vllm/v1/attention/backends/rocm_aiter_fa.py`, `vllm/v1/worker/utils.py`) | 11 of 13 hunks apply to v0.27.1 unchanged; 2 hand-ported (see below) |
| upstream main, pre-#52849 | Gate `fused_qk_norm_rope_kvcache_supported()` on the shuffle layout, so the shuffled write goes through the dedicated `reshape_and_cache_shuffle_triton` path | **prerequisite, not part of #52849.** MiniMax-M3 uses per-head QK norm so the QK-norm+RoPE+KVCache fusion pass is live; #52849 was validated with this gate in place and v0.27.1 predates it |

Two hunks are hand-ported because v0.27.1 differs structurally from main:

- `_split_kv_cache` does not exist in v0.27.1, which inlines the same transpose/split at three call
  sites. The helper is hoisted exactly as main has it and those sites now call it.
- The KV block zeroer in `vllm/v1/worker/utils.py` has different surrounding code, so the outer-dim
  classification fix is re-expressed against v0.27.1's context.

The patched result is byte-identical to #52849's own post-image in every function the PR touches,
verified by diffing against the PR head branch.

## Why the unmodified upstream image cannot run this benchmark

The TP4 arm of this config runs `--attention-backend ROCM_AITER_FA` with
`VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` (MiniMax-M3's sparse-attention layers need the shuffled layout at
one KV head per rank) and EAGLE3 speculative decoding with 3 speculative tokens. MTP makes every target
decode a 4-token query, and `rocm_aiter_fa.py` in the stock image hard-asserts on exactly that pair:

```
assert not rocm_aiter_ops.is_shuffle_kv_cache_enabled(), (
    "Shuffle KV cache layout is not supported with sliding "
    "window, sinks, or speculative decoding (multi-token decode)."
)
```

The server therefore dies on the first multi-token decode; the config cannot produce a result on the
pinned image as shipped. The decode kernel that handles multi-token queries
(`aiter/ops/triton/gluon/pa_decode_gluon.py`) is already present in the image — only vLLM's dispatch to
it is missing, which is what #52849 adds. The patch script fails loudly if that kernel is ever absent
rather than silently serving a different path.

## Upstream PR / issue links

- vLLM: https://github.com/vllm-project/vllm/pull/52849 (open at time of filing; approved by Rohan138,
  under review by tjtanaa for the `ROCM_AITER_FA` changes and jhu960213 for the KV cache layout)

## Removal plan

Retire `apply_minimaxm3_container_patches.sh` and its invocation from `minimaxm3_fp4_mi355x_mtp.sh` once a
ROCm vLLM image ships #52849 (plus the `fused_qk_norm_rope_kvcache_supported` shuffle gate, which is
already on main). At that point bump the pinned `image:` for `minimaxm3-fp4-mi355x-vllm-agentic-mtp` to
that image, drop the patch script and its `bash …/apply_minimaxm3_container_patches.sh` call, and delete
this waiver in the same PR. The patch script's marker gate means an image that already contains the
change is detected and skipped, so the bump can land before the script is removed.
