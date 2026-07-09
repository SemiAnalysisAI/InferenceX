# In-tree patches for the MoRI / MoRIIO PD-disagg path

This directory carries small overlays that fix up the engine source inside
the docker container at runtime. They are needed because some published
images ship known bugs in the (MoRI / MoRIIO) disaggregation backend that
block our benchmark + accuracy configs — so we can keep reusing the
**stock image** instead of rebuilding a patched one.

- `mori_conn.py` — single-file overlay (bind-mounted) for the **sglang**
  MoRI backend.
- `decode_tp_queue_agree.patch`, `swa_reprefill_tail_unified_kv.patch`,
  `dsv4_unified_kv_hicache.patch`, `dsa_paged_mqa_logits_backend.patch`,
  `deepseek_v4_compress_state_coldstart.patch` — reference-only unified
  diffs. Each mirrors a `patch_*()` function in `../setup_deps.sh`, which is
  what actually gets applied (idempotently, in-container, at startup); the
  `.patch` file here is kept as a human-readable copy of the same edit, not
  something that gets `git apply`'d.

> Note: the vLLM MoRIIO `minimax-m3` overlay (`moriio/`) was retired once the
> upstream fixes (vLLM #46039 / #46290 / #46332) shipped in the ROCm nightly
> image; `minimaxm3-fp8-mi355x-vllm-disagg` now runs the stock nightly directly.

The `mori_conn.py` overlay is wired through the `EXTRA_DOCKER_MOUNTS` env
var that `job.slurm` consumes (an opt-in `${EXTRA_DOCKER_MOUNTS:-}` after
the existing `-v` block). The local-test driver scripts under
`scripts/sglang_disagg/` pre-set this env var to the path of the relevant
overlay; CI runners that need the patch can do the same.

## `mori_conn.py`

Overlays
`/sgl-workspace/sglang/python/sglang/srt/disaggregation/mori/conn.py`.

Source: forked from the file shipped in
`lmsysorg/sglang-rocm:v0.5.12.post1-rocm720-mi35x-20260523`
(sglang [v0.5.12.post1](https://github.com/sgl-project/sglang/tree/v0.5.12.post1)).
Four logical edits, all confined to `MoriKVReceiver.send_state`,
`MoriKVReceiver._register_kv_args`, and
`MoriKVReceiver._send_swa_dsa_state`:

1. **Sender flatten** — handle the framework's nested
   `state_item_lens: List[List[int]]` instead of crashing in the
   naked `struct.pack("I", item_len)` (the legacy `List[int]`
   assumption). Idempotent for legacy flat callers.
2. **`state_type` legacy fallback** — when the legacy singular
   `kv_args.state_type` is `'none'` but `state_mem_descs` is non-empty,
   read `kv_args.state_types[0]` (the modern plural API that Mooncake
   and NIXL already use). Routes `MAMBA → _send_mamba_state` and
   `DSA/SWA → _send_swa_dsa_state` correctly.
3. **Consumer normalization** — flatten `state_item_lens` and
   `state_dim_per_tensor` to flat `List[int]` once at the entry of
   `send_state`, so the existing per-tensor index arithmetic
   (`state_item_lens[i]`) and length checks
   (`len(state_item_lens) == len(state_mem_descs)`) keep working.
4. **DSA index rank+length normalization** — inside
   `_send_swa_dsa_state`, before the `group_concurrent_contiguous`
   call, ravel both `src_state_indices` and `dst_state_indices` to 1-D
   and re-truncate to common length. Upstream's existing truncation
   only slices the outer axis, leaving 2-D `(1, N)` arrays unchanged
   and triggering an `np.diff` broadcasting error
   (`shapes (1,12) (0,)`) for GLM-5 (single-DSA-component) prefill
   traffic. See
   `scripts/sglang_disagg/docs_glm5/01-bug-analysis.md` for the full
   write-up.

Verified passing GSM8K = 0.978 ± 0.004 on Qwen3.5-397B-A17B-FP8 1P+1D
TP=8 dp-attn=false (matches and slightly exceeds upstream
[PR #22665](https://github.com/sgl-project/sglang/pull/22665)'s
reported 0.970 GSM8K on the bf16 baseline). GLM-5 (DSA) verification
in progress under
`scripts/sglang_disagg/docs_glm5/02-fix-and-verification.md`.

This is a stop-gap. The proper upstream fix is to migrate MoRI to the
plural `state_types: List[StateType]` API (full design + diff in
`scripts/sglang_disagg/docs/03-upstream-pr-proposal.md`).

## `swa_reprefill_tail_unified_kv.patch`

Reference mirror of upstream
[sgl-project/sglang#30339](https://github.com/sgl-project/sglang/pull/30339)
(open, not yet merged). Applied via `setup_deps.sh`'s
`patch_swa_reprefill_tail_unified_kv()`, unconditionally (the fix is a no-op
unless `SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton` on HIP with a
sliding-window model and HiCache off).

Fixes a stale SWA ring buffer read on DeepSeek-V4 with the unified_kv
backend: unified_kv keeps SWA in a per-request ring
(`req_pool_idx * window + pos % window`) that is not content-addressed and
never stored in the radix tree, so a request that reuses a cached prefix can
read another request's stale SWA if its decode sliding window reaches back
into the reused-prefix region. Adds a generic
`tree_cache.swa_reprefill_tail_tokens()` hook (base: no-op) used by the
scheduler's prefix-match paths to hold back the trailing sliding window from
reuse, and overrides it on `SWARadixCache` (plain radix, HiCache off).

## `dsv4_unified_kv_hicache.patch`

Reference mirror of upstream
[sgl-project/sglang#29417](https://github.com/sgl-project/sglang/pull/29417)
(open, not yet merged). Applied via `setup_deps.sh`'s
`patch_dsv4_unified_kv_hicache()`, gated on
`KV_OFFLOADING=dram` + `KV_OFFLOAD_BACKEND=hicache` +
`SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton`.

Enables unified-KV HiCache (host-offload) on DeepSeek-V4: removes the
previous hard fallback that disabled `unified_kv_triton` whenever
`--enable-hierarchical-cache` was set, teaches
`hybrid_pool_assembler.py`/`deepseek_v4_memory_pool.py` to page the
compressed (C4/C128) region of the unified buffer into a HiCache host pool
(unified_kv has no separate SWA host pool, so that pool/entry is skipped),
and adds `UnifiedRadixCache`'s own `swa_reprefill_tail_tokens()` override
(mutually exclusive with `SWARadixCache`'s — see above — since one is for
HiCache off, the other for HiCache on). **Depends on**
`patch_swa_reprefill_tail_unified_kv()` running first (provides the shared
base hook); `setup_deps.sh` always calls them in that order.

This is the largest/riskiest patch in this directory — upstream restructures
~150 lines of `hybrid_pool_assembler.py` across several hunks, and a couple
of the hunk boundaries in the upstream PR diff fell mid-statement with
unshown unchanged context on either side. Each hunk in
`patch_dsv4_unified_kv_hicache()` is applied independently and will
WARN-and-skip (not corrupt the file) if its anchor text doesn't match
byte-for-byte — check the container startup log for
`[SETUP] Patched: hybrid_pool_assembler.py` (and the other touched files) to
confirm it actually applied before relying on unified_kv + HiCache.

## `dsa_paged_mqa_logits_backend.patch`

Reference mirror of upstream
[sgl-project/sglang#30374](https://github.com/sgl-project/sglang/pull/30374).
Applied via `setup_deps.sh`'s `patch_dsa_paged_mqa_logits_backend()`,
unconditionally.

Adds the `sglang.jit_kernel.dsa` package (`paged_mqa_logits.py` +
`paged_mqa_logits_backend.py`'s `DSAPagedMQALogitsBackend` enum) and rewires
`dsa_indexer.py`'s paged-MQA-logits dispatch — previously an inline
`if _is_hip: ... elif use_dg_native: ... else: ...` chain — through it, plus
a new `--dsa-paged-mqa-logits-backend` server arg (default `"auto"`). On
ROCm this is a pure refactor: `DSAPagedMQALogitsBackend.resolve()` always
resolves to `AITER` on HIP (raising for anything else the arg could be set
to), and the new `is_aiter()` branch calls the exact same
`aiter.ops.triton.pa_mqa_logits.deepgemm_fp8_paged_mqa_logits` kernel the old
`if _is_hip:` branch called directly.

Needed because upstream landed this refactor between the `20260706` and
`20260708` `lmsysorg/sglang-rocm` image tags — images built from `20260706`
or earlier are missing it entirely (`sglang.jit_kernel.dsa` doesn't exist),
while `20260708`+ already has it baked in and this patch is a no-op
(gated by the `sglang.jit_kernel.dsa` package's presence).

The CUTE DSL backend (`jit_kernel/dsa/cutedsl_paged_mqa_logits.py`,
SM100/Blackwell-only) is intentionally **not** installed by this patch:
both `jit_kernel/dsa/__init__.py` and `paged_mqa_logits.py`'s
`cutedsl_paged_mqa_logits()` gate its import behind `is_hip()` / a local
(deferred) import, so it is never imported or executed on ROCm — shipping
~16KB of dead NVIDIA-only code would add risk (e.g. an import error if a
transitive dependency is missing) for zero behavioral benefit on our
hardware.

Verified byte-for-byte identical to the actual patched files baked into
`rocm/pytorch-private:sglang-0.5.14-rocm720-mi35x-mori-0706` by running the
patch function inside a fresh `lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260706`
container and diffing the result.

## `deepseek_v4_compress_state_coldstart.patch`

Reference mirror of upstream
[sgl-project/sglang#30333](https://github.com/sgl-project/sglang/pull/30333).
Applied via `setup_deps.sh`'s
`patch_deepseek_v4_compress_state_coldstart()`, unconditionally.

`CompressStatePool.clear_all_state()` only clears the last row of
`kv_score_buffer` (the "empty state" sentinel row) — correct for the
index-addressed C4 pool, but the ROCm/HIP C128 layout addresses
request-scoped state by `req_pool_idx` (a per-request ring, not
content-addressed). Since the pool is allocated via `torch.empty()`, a cold
server can read uninitialized garbage as a request's "previous" compress
state before that request's slot has ever been written. Initializes every
C128 row to the sentinel on HIP; C4 (and non-HIP) keep the original
last-row-only behavior.

Same version-drift situation as `dsa_paged_mqa_logits_backend.patch` above:
missing from images built before this PR merged upstream (between
`20260706` and `20260708`), a no-op once already present.

## How to enable

```bash
export EXTRA_DOCKER_MOUNTS="-v $DI_REPO_DIR/benchmarks/multi_node/amd_utils/patches/mori_conn.py:/sgl-workspace/sglang/python/sglang/srt/disaggregation/mori/conn.py:ro"
```

`$DI_REPO_DIR` is the InferenceX checkout root that `job.slurm`
already mounts into the container at `/workspace`.

When this env var is unset (CI default for runs that don't need the
patch), `${EXTRA_DOCKER_MOUNTS:-}` expands to the empty string and
container behavior is byte-identical to the unpatched path.

## When to use which patch

| Image / version | Need `mori_conn.py` overlay? |
|---|---|
| `lmsysorg/sglang-rocm:v0.5.12.post1-rocm720-mi35x-20260523` | yes (Qwen3.5-MoE-FP8, GLM-5, any hybrid model on this image) |
| `lmsysorg/sglang-rocm:v0.5.10.post1-rocm720-mi35x-*` (used by `dsr1-fp4-*-disagg`) | not validated; same code path likely affected — try with the overlay if you hit the same `struct.error` |
| `rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-*` (used by `dsr1-fp8-*-disagg`, `glm5-*-disagg`) | predates [PR #22665](https://github.com/sgl-project/sglang/pull/22665); different code paths; **do not** apply this overlay |

When upstream merges the proper fix (see
`scripts/sglang_disagg/docs/03-upstream-pr-proposal.md`) and that
fix lands in a published image, retire this overlay and the
`EXTRA_DOCKER_MOUNTS` knob can stay (still useful for future patches).
