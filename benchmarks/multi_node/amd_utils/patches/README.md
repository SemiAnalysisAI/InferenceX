# In-tree patches for the MoRI / MoRIIO PD-disagg path

This directory carries small overlays that fix up the engine source inside
the docker container at runtime. They are needed because some published
images ship known bugs in the (MoRI / MoRIIO) disaggregation backend that
block our benchmark + accuracy configs — so we can keep reusing the
**stock image** instead of rebuilding a patched one.

- `mori_conn.py` — single-file overlay (bind-mounted) for the **sglang**
  MoRI backend.
- `moriio/` — unified-diff overlay (applied with `patch` at container
  startup) for the **vLLM** MoRIIO connector (`minimax-m3` image). See its
  section below.

The `mori_conn.py` overlay is wired through the `EXTRA_DOCKER_MOUNTS` env
var that `job.slurm` consumes (an opt-in `${EXTRA_DOCKER_MOUNTS:-}` after
the existing `-v` block). The local-test driver scripts under
`scripts/sglang_disagg/` pre-set this env var to the path of the relevant
overlay; CI runners that need the patch can do the same. The `moriio/`
diff needs no extra mount — the repo (and thus the diff file) is already
bind-mounted into the container — `job.slurm` just runs `patch` against it
before launching the server; see "How to enable" in its section below.

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

## How to enable

```bash
export EXTRA_DOCKER_MOUNTS="-v $DI_REPO_DIR/benchmarks/multi_node/amd_utils/patches/mori_conn.py:/sgl-workspace/sglang/python/sglang/srt/disaggregation/mori/conn.py:ro"
```

`$DI_REPO_DIR` is the InferenceX checkout root that `job.slurm`
already mounts into the container at `/workspace`.

When this env var is unset (CI default for runs that don't need the
patch), `${EXTRA_DOCKER_MOUNTS:-}` expands to the empty string and
container behavior is byte-identical to the unpatched path.

## `moriio/` (vLLM MoRIIO connector, MiniMax-M3)

A single unified diff (`moriio-minimax-m3-disagg.diff`), applied with
`patch -p1` against the vLLM package dir inside the container, touching
three files:

```
/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/moriio/
  ├── moriio_connector.py
  ├── moriio_engine.py
  └── moriio_common.py
```

Source: forked from the stock `vllm/vllm-openai-rocm:minimax-m3` image
(vLLM `0.22.1rc1.dev490`).

**Bug (general MoRIIO, not M3-specific):** the connector assumed the
FlashAttention KV layout `[2, num_blocks, block_size, heads, head_dim]`
(K/V axis **outer**), but this vLLM's attention backends (standard
`TRITON_ATTN` **and** the M3 sparse backend) allocate
`[num_blocks, 2, block_size, heads, head_dim]` (K/V axis **inner**).
`_compute_block_transfer_offsets` indexed blocks with `stride[1]` (the
K/V stride) instead of `stride[0]` (the block stride), so every disagg
block transfer read the wrong region. Invisible to throughput
benchmarks (they don't check output); only the **gsm8k accuracy eval**
catches it. The connector was only ever correct for MLA models
(DeepSeek, rank-3 path); MiniMax-M3 is GQA + sparse lightning-indexer
→ broken (disagg gsm8k `0.0008` token salad).

**Fix** — axis-aware offset computation: detect the block axis + optional
size-2 K/V axis from each layer's real shape/stride, compute offsets per
distinct geometry (handles M3's 2nd geometry, the rank-3 bf16 key-only
indexer cache), `num_blocks = shape[0]`; the WRITE path memoizes offsets
per geometry. Result: disagg gsm8k `strict-match 0.9583 /
flexible-extract 0.9575` (matches single-node). Homogeneous models
(uniform layout) are unaffected — one geometry, one offset set, same
result. Full write-up in
`/apps/ditian12/m3_disagg_manual/moriio_hetkv_fix/README.md`.

The diff also bundles two heterogeneous-TP layers (no-op for homogeneous
TP, exercised by `nvidia/amd-master.yaml`'s TP4-prefill + TP8-decode
configs):

- **heterogeneous-TP addressing + guard:** stock MoRIIOConnector always
  addresses remote rank == local `tp_rank`, which has no listener once
  `DECODE_TP_SIZE > PREFILL_TP_SIZE`. `_remote_tp_rank` maps each decode
  rank to the correct single prefill rank. Two regimes, both requiring
  **replicated** KV heads (`tp_size >= total_kv_heads`, ≤1 distinct head
  per rank — MiniMax-M3 has 4 KV heads, so any TP≥4 is replicated):
  - `D-TP > P-TP` (e.g. P4/D8): `tp_rank // ratio`, mirroring NIXL's
    `TpKVTopology.get_target_remote_ranks`. Multiple decode ranks read
    from one prefill rank.
  - `P-TP > D-TP` (e.g. P8/D4): vLLM distributes heads across prefill
    ranks in consecutive pairs — (rank0,rank1)→head0, (rank2,rank3)→head1,
    etc. Decode rank k must connect to the **first** rank of its head group:
    `tp_rank * ratio`. Using `tp_rank` directly (as the original patch did)
    is wrong for ranks > 0: decode rank 1 lands on prefill rank 1 (holds
    head0) instead of prefill rank 2 (holds head1), producing garbage KV.
  The one unsupported case — KV-head **splitting** (`total_kv_heads >
  prefill_tp`, where each prefill rank holds a distinct head subset that
  a decode rank would need to slice from NHD layout, unrepresentable as a
  single `(offset,len)` per block) — **raises `NotImplementedError`** in
  `_compute_block_transfer_offsets`. (NIXL likewise only splits heads in
  HND layout and raises otherwise.)
- **dup-ack fan-in:** with `DECODE_TP_SIZE > PREFILL_TP_SIZE`, N decode
  ranks read from one prefill rank and each ACKs the same `transfer_id`.
  The producer now counts ACKs per `transfer_id` (consumer embeds its own
  `tp_size` in the notify payload) and only reports `finished_sending`
  once all expected consumers have ACKed — preventing both the late-ACK
  `EngineCore` crash and freeing/reusing KV blocks while a slower decode
  rank is still reading. Mirrors NIXL's
  `consumer_notification_counts_by_req`.

### How to enable

`job.slurm` auto-applies this diff when `DOCKER_IMAGE_NAME` contains
`minimax-m3` (and not the already-fixed `-hetkv` rebuild), unless the
caller sets `MORIIO_KV_PATCH=skip`. To wire it by hand (e.g. the
`m3_disagg_manual/run_manual_2node.sh` driver, which sets
`MORIIO_KV_PATCH`), run inside the container before the server starts:

```bash
patch -p1 -d /usr/local/lib/python3.12/dist-packages \
  < $DI_REPO_DIR/benchmarks/multi_node/amd_utils/patches/moriio/moriio-minimax-m3-disagg.diff
```

(`$DI_REPO_DIR` is the InferenceX checkout root that `job.slurm` already
mounts into the container at `/workspace`.)

This lets the **stock** `minimax-m3` image be reused for the E2E
accuracy run — no `-hetkv` rebuild needed. Retire the overlay once the
fix lands in a published image; it is not yet upstreamed.

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
