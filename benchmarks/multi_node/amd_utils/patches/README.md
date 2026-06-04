# In-tree sglang patches for the MoRI PD-disagg path

This directory carries small Python overlays that get bind-mounted over
the upstream sglang source inside the docker container at runtime.
They are needed because some sglang releases ship known bugs in the
MoRI disaggregation backend that block our benchmark + accuracy
configs.

The mount is wired through the `EXTRA_DOCKER_MOUNTS` env var that
`job.slurm` consumes (an opt-in `${EXTRA_DOCKER_MOUNTS:-}` after the
existing `-v` block). The local-test driver scripts under
`scripts/sglang_disagg/` pre-set this env var to the path of the
relevant overlay; CI runners that need the patch can do the same.

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

## `moriep.py`

Overlays
`/sgl-workspace/sglang/python/sglang/srt/layers/moe/token_dispatcher/moriep.py`.

Source: forked from `lmsysorg/sglang-rocm:v0.5.12.post1-*` (sglang
[v0.5.12.post1](https://github.com/sgl-project/sglang/tree/v0.5.12.post1)).
The base file is **byte-identical to the upstream tag**
(`md5 ac626f5459a699f9ac953d9d8e71d861`); the overlay is a single
+22-line insertion in `MoriTokenDispatcher.__init__`.

**Bug it fixes:** at low concurrency the MoRI EP dispatch path silently
corrupts output (decodes fine, acceptance length stays high, but gsm8k
drops to 0). The per-rank dispatch buffer
`num_max_dispatch_tokens_per_rank` (→ mori `max_num_inp_token_per_rank`)
is derived by the harness as `max(CONC_LIST)/TP*(MTP+1)`, which collapses
at low conc (conc-64 / TP8 / MTP3 → `64/8*4 = 32`). MoRI sizes its
receive buffer `MaxNumTokensToRecv() = worldSize * maxNumInpTokenPerRank`
(`max_total_recv_tokens` defaults to 0 → that fallback, and it is a *cap*
not a floor — `dispatch_combine.hpp:126-136`). The intra-node dispatch
kernel's per-dest atomic counter then runs past that buffer; the only
guard is `assert(destTokId < MaxNumTokensToRecv())`, compiled out under
`-DNDEBUG`, so the result is silent out-of-bounds writes
(`internode_v1.cpp` `DispatchIntraNodeBlock`).

The overlay floors `num_max_dispatch_tokens_per_rank` to **256** right at
its env read — the single source of truth that feeds both
`get_ep_dispatch_configs()` (kernel selection) and the buffer-sizing
arg. Empirically validated on MI355X (conc-64 DEP8+MTP3):
dispatch `32 → gsm8k 0.00`, `64 → 0.00` (one wavefront is not enough),
`256 → 0.94`.

This is a stop-gap. The proper upstream fix is in MoRI: size the receive
buffer from the routing fan-in and turn the compiled-out `assert` into a
real bounds guard (see [ROCm/mori#356](https://github.com/ROCm/mori/issues/356)).
The integration-level guard belongs in sglang's `moriep.py`
([sgl-project/sglang#27194](https://github.com/sgl-project/sglang/issues/27194)) —
this overlay is exactly that guard, pending upstream merge.

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
