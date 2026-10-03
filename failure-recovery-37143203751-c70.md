# Failure recovery — Run Sweep 37143203751 agentic c70

## Class

**recipe** (admission): `override_c70` `max-num-seqs` too high for DCP PyNCCL `kv_gather` under AgentX c70 on B300 DSXE.

## Evidence

| Field | Value |
| --- | --- |
| Tip SHA | `c9ffe2b12d30d909b0e3ef97cdb0010ce6cb38d2` |
| Run | [37143203751](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/37143203751) attempt 1 RED |
| Job | `111276873706` (agentic c70) |
| Slurm | `7087` on `b300-dsxe_07` / `dsxe-sa-b300-prd0-gpu-15` |
| server_logs artifact | `11285248802` (`server_logs_kimik3_tp8_conc70_...`) |
| Knobs on tip | A2A=1, KV_GATHER=0, Q_GATHER=0, `max-num-seqs=48`, util 0.85, `load_async=true`, `lookup_async=true`, `max_load_batch_keys=1` |

Wrapper signals (`ProfileAborted`, `worker_crash:8`, `nccl_error:16`) are insufficient alone:

- `nccl_error:16` is init-only `ibv_query_port_speed` WARN at `20:01:07` (false positive).
- Real rail: `Mooncake rail: ibp198s0f0` / `Patched device_name='ibp198s0f0'`.
- `Application startup complete`; Mooncake `failed_keys=0` through serve.
- Warmup completed `772/774`; profiling started `21:06:27` and kept `115` requests (not warmup-only die).

### First real kill boundary

**Watchdog `_ALLGATHER_BASE` in `kv_gather` (dcp.py:1413)** at `21:17:09`:

```text
[Rank 7] Watchdog caught collective operation timeout:
WorkNCCL(SeqNum=353066, OpType=_ALLGATHER_BASE, NumelIn=3142656,
NumelOut=25141248, Timeout(ms)=600000) ran for 600016 ms
PG ID 3: last enqueued work: 353081, last started work: -1,
last completed work: 353065
stack: all_gather_into_tensor → kv_gather (dcp.py:1413) →
_context_parallel_compute_prefill_context → _forward_prefill_fused
→ DistBackendError / terminate / EngineDead / ProfileAborted
```

Hang window starts ~`21:07:09` (600s before watchdog). Last healthy engine line then stall:

- `21:07:12` Running: 10, Waiting: 2, Deferred: 2, GPU KV: **21.3%**, gen 232 tok/s
- `21:07:22` Running: 10, Waiting: 2, Deferred: 2, GPU KV: 21.3%, gen **0.0** tok/s

No multimem timeout, no CUDA OOM, no `sample_tokens` timeout.

### Same-run packing (context, not the kill line)

Earlier in the same profile, admission still packed the pool: max GPU KV **99.6%** at `21:00:12` (Running: 0, Waiting: 48, Deferred: 46); 99 windows with Running≈0 and KV≥85% from `20:38:42`–`21:01:32`. Engine recovered and resumed generating before the ALLGATHER hang. Distinct from tip `1d937c73e` c70 deferred-admission starve under `max-num-seqs=70` (0 kept, `sample_tokens` 1800s).

Same PyNCCL `kv_gather` family as tip `fd61acb0` c56 / `ebc4eb499` c16.

## Fix

Smallest supported knob only:

- `override_c70.max-num-seqs`: **48 → 32**
- Match `cudagraph_capture_sizes` to max 32 (same shape as `override_c32`)
- Do **not** re-enable `KV_GATHER` / `Q_GATHER`
- Do **not** set `compact_group_io`, `MC_MAX_MR_SIZE`, or `load_async=false`
- Do **not** touch #3632

`origin/main` already merged (0 behind); no merge required for this push.
