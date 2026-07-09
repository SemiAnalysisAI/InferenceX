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
  diffs, kept as human-readable copies of the four DSv4 patches below for
  archaeology (upstream PR links, rationale). Not applied directly (no
  `git apply`) — `files/` below is what actually gets applied.
- `files/` — whole-file, known-good post-patch copies of every file touched
  by the four DSv4 patches (see "`files/` — whole-file overlay" below).
  This is what `../setup_deps.sh`'s `patch_*()` functions actually apply,
  via the `_copy_patched_file()` helper, idempotently, in-container, at
  startup.

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

## `files/` — whole-file overlay (replaces anchor-hunk patching)

`files/python/sglang/<rel_path>` mirrors
`/sgl-workspace/sglang/python/sglang/<rel_path>` inside the container, one
file per path touched by the four DSv4 patches above (14 files total).
Each holds the complete, known-good, POST-PATCH content of that file —
not a diff.

`setup_deps.sh`'s `_copy_patched_file()` helper drops these in directly
(`cp`, skipped if the target already matches — see the helper's own
comment block at the top of `setup_deps.sh`), instead of the previous
approach of applying anchor-matched string-replace hunks in place.

**Why:** the hunk-based approach is safe against *unrelated* upstream
drift (a non-matching hunk WARNs and skips instead of corrupting the
file), but fragile against *any* drift near an anchor. E.g. the
`20260706` image inserted a new `DEEPSEEK_V4_C4_INDEXER` pool
`build_pool_entry()` call in the middle of one of
`hybrid_pool_assembler.py`'s hunks in
`patch_dsv4_unified_kv_hicache()`, which silently no-op'd patching that
file entirely (the anchor text no longer matched anywhere) — the server
still started, but silently without HiCache paging for the compressed
region. A whole-file copy has no anchors to drift.

**Trade-off:** unlike hunk-based patching, this is all-or-nothing per
file and has no graceful degradation — it assumes the container's base
image is close enough to the one `files/` was captured from that
copying the whole file doesn't clobber unrelated upstream changes. Only
rely on it for the exact base image tag below; regenerate for anything
else.

Captured from: `lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260706`.

### Regenerating `files/` for a different base image

1. Start a throwaway container from the target base image with this repo
   bind-mounted (same setup as `job.slurm` uses), e.g.:

   ```bash
   docker run --rm -it -v "$DI_REPO_DIR:/workspace" \
     lmsysorg/sglang-rocm:<new-tag> bash
   ```

2. Inside the container, temporarily restore the anchor-hunk versions of
   the four `patch_*()` functions from git history (`git log -- \
   benchmarks/multi_node/amd_utils/setup_deps.sh`, find the commit before
   they were replaced by `_copy_patched_file()` calls) and source that
   version of `setup_deps.sh` with every relevant patch's gating env vars
   set (`KV_OFFLOADING=dram`, `KV_OFFLOAD_BACKEND=hicache`,
   `SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton`, etc. — so
   `patch_dsv4_unified_kv_hicache()` isn't skipped).
3. Watch the container log for `[SETUP] Patched: ...` / `[SETUP] Wrote:
   ...` lines confirming every hunk applied — any `WARN: anchor ... not
   found` means upstream drifted and the hunk needs manual reconciliation
   before trusting the output.
4. Copy each patched file at
   `/sgl-workspace/sglang/python/sglang/<rel_path>` out to
   `patches/files/python/sglang/<rel_path>` in this repo, overwriting the
   existing copy.
5. `diff` the new files against the old ones and sanity-check the diff
   only contains the expected upstream-version delta, then update the
   "Captured from" tag above.

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
