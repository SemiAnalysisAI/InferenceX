# Failure recovery — Run Sweep 37173109032 agentic c48

## Class

**recipe** (admission): `override_c48` `max-num-seqs` too high (1×=48) for DCP PyNCCL `kv_gather` under AgentX c48 on B300 DSXE.

## Evidence

| Field | Value |
| --- | --- |
| Tip SHA | `767f0b3b2eefc9c1058c52c87156fd4be7678882` |
| Run | [37173109032](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/37173109032) attempt 1 RED |
| Job | `111362024256` (agentic c48) |
| Slurm | `7147` on `b300-dsxe_04` / `dsxe-sa-b300-prd0-gpu-10` |
| server_logs artifact | `11295365223` |
| Knobs on tip | A2A=1, KV_GATHER=0, Q_GATHER=0, `max-num-seqs=48`, util 0.85, `load_async=true`, `lookup_async=true`, `max_load_batch_keys=1` |

Wrapper signals (`ProfileAborted`, `worker_crash:8`, `nccl_error:16`) are insufficient alone:

- `nccl_error:16` is init-only `ibv_query_port_speed` WARN at `04:35:49` (`first_ts=last_ts`; false positive for the kill).
- Real rail / Mooncake path healthy through serve.
- `Application startup complete`; runtime args confirm `--max-num-seqs 48`.
- Profiling kept `452/1033` (531 warmup, 467 error dropped) before kill; not warmup-only die.
- SIGTERM only at shutdown after DistBackend (`05:45:00`), not the first boundary.

### First real kill boundary

**Watchdog `_ALLGATHER_BASE` in `kv_gather` (dcp.py:1413)** at `05:43:54`:

```text
[Rank 3] Watchdog caught collective operation timeout:
WorkNCCL(SeqNum=334427, OpType=_ALLGATHER_BASE, NumelIn=4755456,
NumelOut=38043648, Timeout(ms)=600000) ran for 600017 ms
PG ID 3: last enqueued work: 334452, last started work: -1,
last completed work: 334426
stack: all_gather_into_tensor → kv_gather (dcp.py:1413) →
DistBackendError / terminate / worker_crash:8 / EngineDead /
ProfileAborted (50/499 = 10.0%)
```

Hang window starts ~`05:33:54` (600s before watchdog). Last live then stall:

- `05:33:18` Running: 10, Waiting: 39, Deferred: 28, GPU KV: **100.0%**
- `05:33:58` Running: 13, Waiting: 38, Deferred: 35, GPU KV: **98.1%**, gen 145.5 tok/s
- `05:34:08` Running: 13, Waiting: 38, Deferred: 35, GPU KV: 98.1%, gen **0.0** tok/s
- `05:34:56` first mid-profile `shm_broadcast` starvation line

Earlier packing in the same profile also hit GPU KV **99.8%** / **99.7%**. No multimem timeout, no CUDA OOM, no `sample_tokens` timeout. Do not re-enable KV_GATHER or Q_GATHER.

Same PyNCCL `kv_gather` family as tip `71080871` c40 / `f3997a93` c24 / `c9ffe2b12` c70 / `fd61acb0` c56 / `ebc4eb499` c16. Tip already at 1× CONC for c48; further admission cut required (same pattern as c40 40→32 and c70 48→32).

Eval c32 xgrammar ISE on this ledger remains infra/flake (prior dig); left alone.

## Fix

Smallest supported knob only:

- `override_c48.max-num-seqs`: **48 → 32**
- Match `cudagraph_capture_sizes` to max 32
- Header comment: note c48 caps at 32 (not 1×)
- Keep A2A=1; KV_GATHER=0; Q_GATHER=0; no compact_group_io / MC_MAX_MR_SIZE

## New tip

(filled after push)
