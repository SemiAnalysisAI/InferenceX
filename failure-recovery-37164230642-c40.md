# Failure recovery — Run Sweep 37164230642 agentic c40

## Class

**recipe** (admission): `override_c40` `max-num-seqs` too high (1×=40) for DCP PyNCCL `kv_gather` under AgentX c40 on B300 DSXE.

## Evidence

| Field | Value |
| --- | --- |
| Tip SHA | `71080871410ca24104005a5c4df313430726c1b5` |
| Run | [37164230642](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/37164230642) attempt 1 RED |
| Job | `111336354457` (agentic c40) |
| Slurm | `7123` on `b300-dsxe_06` / `dsxe-sa-b300-prd0-gpu-09` |
| server_logs artifact | `11292012067` (`server_logs_kimik3_tp8_conc40_...`) |
| Knobs on tip | A2A=1, KV_GATHER=0, Q_GATHER=0, `max-num-seqs=40`, util 0.85, `load_async=true`, `lookup_async=true`, `max_load_batch_keys=1` |

Wrapper signals (`ProfileAborted`, `worker_crash:8`, `nccl_error:16`) are insufficient alone:

- `nccl_error:16` is init-only `ibv_query_port_speed` WARN at `01:41:33` (false positive).
- Real rail / Mooncake path healthy through serve.
- `Application startup complete`; runtime args confirm `--max-num-seqs 40`.
- Profiling kept `838/1376` (444 warmup, 449 error dropped) before kill; not warmup-only die.
- Canary + all agentic evals SUCCESS on this tip; sole agentic fail = c40 (fail-fast cancelled siblings).

### First real kill boundary

**Watchdog `_ALLGATHER_BASE` in `kv_gather` (dcp.py:1413)** at `02:49:41`:

```text
[Rank 7] Watchdog caught collective operation timeout:
WorkNCCL(SeqNum=345118, OpType=_ALLGATHER_BASE, NumelIn=4755456,
NumelOut=38043648, Timeout(ms)=600000) ran for 600000 ms
PG ID 3: last enqueued work: 345144, last started work: -1,
last completed work: 345117
stack: all_gather_into_tensor → kv_gather (dcp.py:1413) →
DistBackendError / terminate / worker_crash:8 / ProfileAborted
(94/932 = 10.086%)
```

Hang window starts ~`02:39:41` (600s before watchdog). Last healthy then stall:

- `02:39:48` Running: 17, Waiting: 20, Deferred: 20, GPU KV: **71.5%**, gen 200 tok/s
- `02:39:58` Running: 17, Waiting: 20, Deferred: 20, GPU KV: 71.5%, gen **0.0** tok/s

Earlier packing in the same profile: peak GPU KV **~99.9%** with elevated Waiting/Deferred (e.g. Running: 2 / Waiting: 35 / Deferred: 21 at `02:33:28`; Running: 7 / Waiting: 27 / Deferred: 23 at `02:38:38`). Aggregate result also reported GPU KV usage **100.0%**.

No multimem timeout, no CUDA OOM, no `sample_tokens` timeout. Do not re-enable KV_GATHER or Q_GATHER.

Same PyNCCL `kv_gather` family as tip `f3997a93` c24 / `c9ffe2b12` c70 / `fd61acb0` c56 / `ebc4eb499` c16. Tip already at 1× CONC for c40; further admission cut required (same pattern as c56 56→48 and c70 48→32).

## Fix

Smallest supported knob only:

- `override_c40.max-num-seqs`: **40 → 32**
- Match `cudagraph_capture_sizes` to max 32
- Header comment: note c40 caps at 32 (not 1×)
- Keep A2A=1; KV_GATHER=0; Q_GATHER=0; no compact_group_io / MC_MAX_MR_SIZE

## New tip

(filled after push)
