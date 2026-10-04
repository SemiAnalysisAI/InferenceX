# Failure recovery — Run Sweep 37155636981 agentic c24

## Class

**recipe** (admission): `override_c24` `max-num-seqs` too high (2×=48) for DCP PyNCCL `kv_gather` under AgentX c24 on B300 DSXE.

## Evidence

| Field | Value |
| --- | --- |
| Tip SHA | `f3997a93aaea2663ead720910a9fbb178d006c64` |
| Run | [37155636981](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/37155636981) attempt 1 RED |
| Job | `111311962571` (agentic c24) |
| Slurm | `7107` on `b300-dsxe_07` / `dsxe-sa-b300-prd0-gpu-12` |
| server_logs artifact | `11288576572` (`server_logs_kimik3_tp8_conc24_...`) |
| Knobs on tip | A2A=1, KV_GATHER=0, Q_GATHER=0, `max-num-seqs=48`, util 0.85, `load_async=true`, `lookup_async=true`, `max_load_batch_keys=1` |

Wrapper signals (`ProfileAborted`, `worker_crash:8`, `nccl_error:16`) are insufficient alone:

- `nccl_error:16` is init-only `ibv_query_port_speed` WARN at `23:05:03` (false positive).
- Real rail / Mooncake path healthy through serve; `failed_keys=0` in KV transfer metrics.
- `Application startup complete`; startup args confirm `max_num_seqs: 48`.
- Profiling kept `730/1077` (265 warmup, 288 error dropped) before kill; not warmup-only die.

### First real kill boundary

**Watchdog `_ALLGATHER_BASE` in `kv_gather` (dcp.py:1413)** at `23:56:57`:

```text
[Rank 1] Watchdog caught collective operation timeout:
WorkNCCL(SeqNum=224109, OpType=_ALLGATHER_BASE, NumelIn=4755456,
NumelOut=38043648, Timeout(ms)=600000) ran for 600006 ms
PG ID 3: last enqueued work: 224148, last started work: -1,
last completed work: 224108
stack: all_gather_into_tensor → kv_gather (dcp.py:1413) →
_context_parallel_compute_prefill_context → _forward_prefill_fused
→ DistBackendError / terminate / worker_crash:8 / EngineDead /
ProfileAborted (82/812 = 10.099%)
```

Hang window starts ~`23:46:57` (600s before watchdog). Last healthy then stall:

- `23:46:50` Running: 22, Waiting: 0, GPU KV: **78.8%**, gen 202 tok/s
- `23:47:00` Running: 17, Waiting: 0, GPU KV: 75.7%, gen 402 tok/s
- `23:47:10` Running: 17, Waiting: 0, GPU KV: 75.7%, gen **0.0** tok/s

No multimem timeout, no CUDA OOM, no `sample_tokens` timeout. Do not re-enable KV_GATHER or Q_GATHER.

### Same-run packing (context)

Earlier in the same profile, peak GPU KV **91.1%** at `23:41:20` (Running: 19, Waiting: 1, Deferred: 1). Aggregate result also reported GPU KV usage **93.3%**. c24 was the last mid-conc still at 2× admission.

Same PyNCCL `kv_gather` family as tip `c9ffe2b12` c70 / `fd61acb0` c56 / `ebc4eb499` c16.

## Fix

Smallest supported knob only:

- `override_c24.max-num-seqs`: **48 → 24**
- Match `cudagraph_capture_sizes` to max 24
- Header comment: admission is 2× through CONC 8 only; CONC 16–48 use 1×
- Keep A2A=1; KV_GATHER=0; Q_GATHER=0; no compact_group_io / MC_MAX_MR_SIZE

## New tip

(filled after push)
