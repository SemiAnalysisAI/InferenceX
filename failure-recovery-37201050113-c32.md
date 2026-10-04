# Failure recovery — Run Sweep 37201050113 agentic c32

## Class

**recipe** (admission): `override_c32` `max-num-seqs` too high (32) for DCP PyNCCL `kv_gather` under AgentX c32 on B300 DSXE. Same packed-KV / PyNCCL family as prior mid- and high-conc cells; c32 at 1x was green on superseded tip `bdcf0a0b` and is further admission pressure on this tip.

## Evidence

| Field | Value |
| --- | --- |
| Failed tip SHA | `1e365aae596adc51039c54745b219c8eb4c7b6a2` |
| Run | [37201050113](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/37201050113) attempt 1 RED |
| Job | `111446959460` (agentic c32) |
| Slurm | `7216` on `b300-dsxe_04` / `dsxe-sa-b300-prd0-gpu-13` |
| server_logs artifact | `11306955758` (`server_logs_kimik3_tp8_conc32_...`) |
| Knobs on tip | A2A=1, KV_GATHER=0, Q_GATHER=0, `max-num-seqs=32`, util 0.85, `load_async=true`, `lookup_async=true`, `max_load_batch_keys=1` |

Wrapper signals (`ProfileAborted`, `worker_crash:8`, `nccl_error:16`) are insufficient alone:

- `nccl_error:16` is init-only `ibv_query_port_speed` WARN at `13:37:57` (`first_ts=last_ts`; false positive for the kill).
- Real rail: `Mooncake rail: ibp198s0f0` / `Patched mooncake_store_config device_name='ibp198s0f0'`.
- `Application startup complete` after `13:45:15`; runtime args confirm `max_num_seqs: 32`.
- Mooncake `failed_keys=0` through serve.
- Profiling progressed (`1162` successful / `1646` total; `354` warmup and `398` error dropped) then aborted at `130/1292 = 10.062%`. Not a GHA wrapper-only fail.
- Canary + agentic evals c1–c70 plus dup c70 + collect-evals/collect-results SUCCESS on this tip; sole agentic fail = c32 (fail-fast cancelled siblings, so c56@32 is unproven).

### First real kill boundary

**Watchdog `_ALLGATHER_BASE` in `kv_gather` (dcp.py:1413)** at `14:47:08`:

```text
[Rank 2] Watchdog caught collective operation timeout:
WorkNCCL(SeqNum=357100, OpType=_ALLGATHER_BASE, NumelIn=4755456,
NumelOut=38043648, Timeout(ms)=600000) ran for 600017 ms
PG ID 3: last enqueued work: 357132, last started work: -1,
last completed work: 357099
stack: all_gather_into_tensor → kv_gather (dcp.py:1413) →
_context_parallel_compute_prefill_context → _forward_prefill_fused
→ DistBackendError / terminate / worker_crash:8 / EngineDead /
ProfileAborted (130/1292 = 10.062%)
```

Hang window starts ~`14:37:08` (600s before watchdog). Last live then stall:

- `14:13:15` peak GPU KV **100.0%** (Running: 9, Waiting: 23, Deferred: 23)
- `14:37:15` Running: 25, Waiting: 0, GPU KV: **45.4%**, gen 28.4 tok/s
- `14:37:25` Running: 25, Waiting: 0, GPU KV: 45.4%, gen **0.0** tok/s
- `14:38:09` first mid-hang `shm_broadcast` starvation line

No multimem timeout, no CUDA OOM, no `sample_tokens` timeout. SIGTERM only at shutdown after DistBackend, not the first boundary. Do not re-enable KV_GATHER or Q_GATHER.

Same PyNCCL `kv_gather` family as tip `141fbb856` c70 / `767f0b3b2` c48 / `71080871` c40 / `f3997a93` c24. Tip already at max-num-seqs=32 for c32; further admission cut required. Do not change `override_c56` or other conc cells.

## Fix

Smallest supported knob only:

- `override_c32.max-num-seqs`: **32 → 24**
- Match `cudagraph_capture_sizes` to max 24 (same shape as `override_c24` / `override_c70`)
- Header comment: note c32 caps at 24
- Keep A2A=1; KV_GATHER=0; Q_GATHER=0; no compact_group_io / MC_MAX_MR_SIZE
- Do not touch `override_c16` / `override_c24` / `override_c40` / `override_c48` / `override_c56` / `override_c70`

## New tip

(filled after push)
