# Failure recovery — Run Sweep 37181441604 agentic c70

## Class

**recipe** (admission): `override_c70` `max-num-seqs` too high (32) for DCP PyNCCL `kv_gather` under AgentX c70 on B300 DSXE.

## Evidence

| Field | Value |
| --- | --- |
| Failed tip SHA | `141fbb8567c8dfc14616512fd74bc97806400ada` |
| Run | [37181441604](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/37181441604) attempt 1 RED |
| Job | `111386701892` (agentic c70) |
| Slurm | `7166` on `b300-dsxe_09` / `dsxe-sa-b300-prd0-gpu-09` |
| server_logs artifact | `11298073431` (`server_logs_kimik3_tp8_conc70_...`) |
| Knobs on tip | A2A=1, KV_GATHER=0, Q_GATHER=0, `max-num-seqs=32`, util 0.85, `load_async=true`, `lookup_async=true`, `max_load_batch_keys=1` |

Wrapper signals (`ProfileAborted`, `worker_crash:8`, `nccl_error:16`) are insufficient alone:

- `nccl_error:16` is init-only `ibv_query_port_speed` WARN at `07:28:28` (`first_ts=last_ts`; false positive for the kill).
- Real rail: `Mooncake rail: ibp198s0f0` / `Patched mooncake_store_config device_name='ibp198s0f0'`.
- `Application startup complete` at `07:35:50` window; runtime args confirm `max_num_seqs: 32`.
- Mooncake `failed_keys=0` through serve.
- Warmup progressed (`284/774` returned, `67` in flight at hang) then stalled; profiling never started (`warmup_failure`). Not a GHA wrapper-only fail.
- Canary + all agentic evals + collect-evals SUCCESS on this tip; sole agentic fail = c70 (fail-fast cancelled siblings).

### First real kill boundary

**Watchdog `_ALLGATHER_BASE` in `kv_gather` (dcp.py:1413)** at `08:18:49`:

```text
[Rank 5] Watchdog caught collective operation timeout:
WorkNCCL(SeqNum=224761, OpType=_ALLGATHER_BASE, NumelIn=3032064,
NumelOut=24256512, Timeout(ms)=600000) ran for 600011 ms
PG ID 3: last enqueued work: 224769, last started work: -1,
last completed work: 224760
stack: all_gather_into_tensor → kv_gather (dcp.py:1413) →
_context_parallel_compute_prefill_context → _forward_prefill_fused
→ DistBackendError / terminate / worker_crash:8 / EngineDead /
ProfileAborted (warmup_failure; 0 kept / 683 total)
```

Hang window starts ~`08:08:49` (600s before watchdog). Last live then stall:

- `08:05:41` peak GPU KV **94.6%** (Running: 0, Waiting: 70, Deferred: 29)
- `08:08:51` Running: 1, Waiting: 65, Deferred: 65, GPU KV: **89.9%**, gen 0.2 tok/s
- `08:09:01` Running: 1, Waiting: 65, Deferred: 65, GPU KV: 89.9%, gen **0.0** tok/s
- `08:09:50` first mid-hang `shm_broadcast` starvation line

No multimem timeout, no CUDA OOM, no `sample_tokens` timeout. SIGTERM only at shutdown after DistBackend (`08:19:54`), not the first boundary. Do not re-enable KV_GATHER or Q_GATHER.

Same PyNCCL `kv_gather` family as tip `cefbd2182` c48 / `71080871` c40 / `f3997a93` c24 / `c9ffe2b12` c70 / `fd61acb0` c56 / `ebc4eb499` c16. Tip already at max-num-seqs=32 for c70; further admission cut required.

Eval c32 xgrammar ISE on superseded tip `767f0b3b` / 37173109032 remains deferred; left alone.

## Fix

Smallest supported knob only:

- `override_c70.max-num-seqs`: **32 → 24**
- Match `cudagraph_capture_sizes` to max 24 (same shape as `override_c24`)
- Header comment: note c70 caps at 24
- Keep A2A=1; KV_GATHER=0; Q_GATHER=0; no compact_group_io / MC_MAX_MR_SIZE

## New tip

(filled after push)
