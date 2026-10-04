# Failure recovery — Run Sweep 37189444890 agentic c56

## Class

**recipe** (admission): `override_c56` `max-num-seqs` too high (48) for packed GPU KV under AgentX c56 on B300 DSXE. Same packed-KV family as prior c40@40 and c48@48 admissions.

## Evidence

| Field | Value |
| --- | --- |
| Failed tip SHA | `bdcf0a0b0beb049af94c950390985b93a235578b` |
| Run | [37189444890](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/37189444890) attempt 1 RED |
| Job | `111411443417` (agentic c56) |
| Slurm | `7189` on `b300-dsxe_15` / `dsxe-sa-b300-prd0-gpu-09` |
| server_logs artifact | `11302089911` (`server_logs_kimik3_tp8_conc56_...`) |
| Knobs on tip | A2A=1, KV_GATHER=0, Q_GATHER=0, `max-num-seqs=48`, util 0.85, `load_async=true`, `lookup_async=true`, `max_load_batch_keys=1` |

Wrapper signals (`ProfileMetricCoverageError`, `nccl_error:16`) are insufficient alone:

- `nccl_error:16` is init-only `ibv_query_port_speed` WARN at `10:04:38` (`first_ts=last_ts`; false positive for the kill).
- Real rail: `Mooncake rail: ibp198s0f0` / `Patched mooncake_store_config device_name='ibp198s0f0'`.
- `Application startup complete`; runtime args confirm `--max-num-seqs 48`.
- Mooncake `failed_keys=0` through serve.
- Warmup completed `619/619` in `2180.67s`; profiling then kept `1296` (619 warmup dropped from coverage accounting). Not a GHA wrapper-only fail.
- Canary + agentic evals on this tip are green; fail-fast RED is this c56 throughput job.

### First real kill boundary

**Packed-KV stall mid-profile** (no ALLGATHER watchdog this time). Last live engine log at `11:36:32`:

```text
Running: 23 reqs, Waiting: 32 reqs, Deferred: 32 reqs,
GPU KV cache usage: 98.2%, Avg generation throughput: 0.0 tokens/s
```

Then silence until SIGTERM at shutdown `11:55:07`. Peak GPU KV **100.0%** at `10:40:52` during warmup (Running: 0, Waiting: 54, Deferred: 51). Profiling duration 3600s timed out with `grace_period_timeout=True`; AIPerf then raised `ProfileMetricCoverageError` (TTFT 71.4% / ITL 71.5%; neither signal in the final 180s).

No multimem timeout, no CUDA OOM, no `sample_tokens` timeout, no DistBackendError. Do not re-enable KV_GATHER or Q_GATHER.

Same packed-KV family as tip `767f0b3b2` c48 / `71080871` c40 / `cefbd2182` c48 recovery. Tip already at max-num-seqs=48 for c56; further admission cut required.

## Fix

Smallest supported knob only:

- `override_c56.max-num-seqs`: **48 → 32**
- Match `cudagraph_capture_sizes` to max 32 (same shape as `override_c32` / `override_c48`)
- Header comment: note c56 caps at 32
- Keep A2A=1; KV_GATHER=0; Q_GATHER=0; no compact_group_io / MC_MAX_MR_SIZE
- Do not touch `override_c40` / `override_c48` / `override_c70`

## New tip

(filled after push)
