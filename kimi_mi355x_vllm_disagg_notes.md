# Kimi K2.5 MI355X vLLM Disagg Notes

## Goal

Make `kimik2.5-fp4-mi355x-vllm-disagg` beat the single-node MI355X baseline on per-GPU throughput, not just total throughput from using more GPUs.

## B200/GB200 Recipe Lessons

The B200/GB200 Kimi disagg recipes use worker counts, not 8 GPUs per `P` or `D`.

Examples from `benchmarks/multi_node/srt-slurm-recipes/vllm/kimi-k2.5-fp4`:

- `1p1d dep4-dep8`: prefill = 1 worker x 4 GPUs, decode = 1 worker x 8 GPUs.
- `1p1d dep4-dep16`: prefill = 1 worker x 4 GPUs, decode = 1 worker x 16 GPUs.
- `3p1d dep4-dep16`: prefill = 3 workers x 4 GPUs, decode = 1 worker x 16 GPUs.
- `6p1d dep4-dep16`: prefill = 6 workers x 4 GPUs, decode = 1 worker x 16 GPUs.
- `8p1d dep4-dep16`: prefill = 8 workers x 4 GPUs, decode = 1 worker x 16 GPUs.

So the useful lesson is the P:D GPU ratio and worker granularity, not the literal P/D count.

## MI355X Mapping

Current MI355X launcher appears node-granular: one worker normally consumes an 8-GPU MI355X node.

The first sub-node slicing layer is implemented by role-local visibility in
`benchmarks/multi_node/amd_utils/job.slurm`: prefill workers see
`0..PREFILL_TP_SIZE-1`, decode workers see `0..DECODE_TP_SIZE-1`. Benchmark
accounting in `server_vllm.sh` now reports `PREFILL_TP_SIZE * xP` and
`DECODE_TP_SIZE * yD`, instead of assuming 8 GPUs per worker.

With sub-node GPU slicing via `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES`,
the closest mapping is:

- GB200 `dep4` prefill -> MI355X 4-GPU prefill worker.
- GB200 `dep8` decode -> MI355X 8-GPU decode worker.
- GB200 `dep16` decode -> MI355X 2 x 8-GPU decode workers.

If sub-node slicing is not available, use coarser whole-node sweeps:

- `1k1k`: start with `1P1D` and `1P2D`; this is decode-heavy.
- `8k1k`: start with `2P1D`, then possibly `3P1D`; this is prefill-heavy.

Avoid interpreting `3P1D` as a universally good first target on MI355X. With whole nodes, `3P1D` already means 32 GPUs.

## DP-Attention / FP8 KV Finding

Validated on `mia1-p01-g07`:

- `DP8 + EP8 + TP1 + --kv-cache-dtype fp8` fails.
- The failure enters AITER MLA `mla_a8w8_qh64_qseqlen1_gqaratio64_v3_ps`.
- With vLLM persistent metadata enabled, this hits GPU memory access faults.
- With persistent metadata disabled via a local monkey patch, AITER reports:
  `fp8/fp8 with gqa_ratio=64 only supports decode_qlen=1 in persistent mode`.
- `DP8 + EP8 + TP1` with auto/bf16 KV starts successfully and returns a chat completion.

Therefore the current safe recipe rule is:

- DP-attn can be used for high-concurrency sweeps.
- DP-attn must strip `--kv-cache-dtype fp8` and run auto/bf16 KV on MI355X.
- Keep TP8 + fp8 KV as a separate non-DP baseline for low/mid concurrency.

## Candidate MI355X Recipe Sweep

Minimal first pass before DP-attn is stable:

1. `1k1k`, `1P1D`, prefill TP4/EP4 + decode TP8/EP8, conc `[512, 1024]`.
2. `1k1k`, `1P2D`, prefill TP4/EP4 + decode TP8/EP8, conc `[1024, 2048]`.
3. `8k1k`, `2P1D`, prefill TP4/EP4 + decode TP8/EP8, conc `[512, 1024]`.
4. `8k1k`, `4P1D`, prefill TP4/EP4 + decode TP8/EP8, conc `[1024]`.

The corresponding normal-only CI config is
`.github/configs/amd-kimi-mi355x-mpnd-normal.yaml`. It excludes DP-attn rows so
the run can finish green and produce ingestible artifacts.

Older whole-node fallback if sub-node slicing regresses:

1. `1k1k`, `1P1D`, TP8 prefill + TP8/EP8 decode, conc `[512, 1024]`.
2. `1k1k`, `1P2D`, TP8 prefill + TP8/EP8 decode, conc `[1024, 2048]`.
3. `1k1k`, `1P2D`, DP8/EP8 auto-KV, conc `[1024, 2048]`.
4. `8k1k`, `2P1D`, TP8 prefill + TP8/EP8 decode, conc `[512, 1024]`.
5. `8k1k`, `3P1D`, TP8 prefill + TP8/EP8 decode, conc `[1024, 2048]`.
6. `8k1k`, `3P1D`, DP8/EP8 auto-KV, conc `[1024, 2048]`.

If sub-node 4-GPU workers become available, prefer matching GB200 ratios more directly:

- `1k1k`: P=4G, D=8G or 16G.
- `8k1k`: P=3x4G, D=16G.

## CI Run 29200792444 / 29200792335 Diagnosis

Both pinned Kimi K2.5 MPND runs produced successful normal heterogeneous TP
jobs and failed only on DP-attn comparison jobs.

Normal successful rows were present in artifacts:

- `29200792444` (`1k1k`): TP4/EP4 prefill + TP8/EP8 decode succeeded for
  `1P1D` and `1P2D`.
- `29200792335` (`8k1k`): TP4/EP4 prefill + TP8/EP8 decode succeeded for
  `2P1D` and `4P1D`.

The missing 1k1k rows in the unofficial UI were not because artifacts were
absent. `collect-results` succeeded, but the whole workflow conclusion was
failure because DP-attn jobs failed, so the unofficial ingestion path likely
ignored or did not refresh partial data from the failed run.

The old artifacts also show why the first-layer accounting fix is needed:

- `1k1k 1P1D` reported `num_prefill_gpu=8`, but should be 4 for TP4.
- `1k1k 1P2D` reported `num_prefill_gpu=8`, but should be 4 for TP4.
- `8k1k 2P1D` reported `num_prefill_gpu=16`, but should be 8 for 2 x TP4.
- `8k1k 4P1D` reported `num_prefill_gpu=32`, but should be 16 for 4 x TP4.

DP-attn failures in those runs are separate from heterogeneous TP. Server and
MoRI proxy readiness completed, but benchmark reported:

```text
Successful requests: 0
FAIL: request failure rate 100.0% exceeds 5% threshold (0/10240 completed)
```

Those failed jobs did not preserve per-request error bodies, so the next DP-attn
debug pass should enable detailed benchmark output or server access/error logs.

## B200 Parameters Worth Adapting

Applicable ideas:

- Prefill: `enforce-eager=true`.
- Decode: `FULL_DECODE_ONLY` cudagraph.
- Disable prefix caching for fixed-seq throughput sweeps.
- Consider disabling chunked prefill for fixed-seq disagg.
- Split `max-num-seqs` by workload:
  - `1k1k`: larger, e.g. 512/1024.
  - `8k1k` prefill: smaller, e.g. 64/128.
  - `8k1k` decode: 256/512.
- Tune decode `max-cudagraph-capture-size` separately, starting with 256/512.
- Increase frontend/router capacity at high concurrency. B200/GB200 recipes often enable multiple frontends for high-throughput cases.

Not directly portable to MI355X:

- `FLASHINFER_MLA`.
- `flashinfer_nvlink_one_sided`.
- NVIDIA NCCL MNNVL/NVLS knobs.
- NIXL connector assumptions.

## CI Run 28083945960 Diagnosis

Run: `https://github.com/SemiAnalysisAI/InferenceX/actions/runs/28083945960`

Job `83145269559` is `multi-node eval`, not benchmark:

- The job name includes `eval-only`.
- `benchmark-multinode-tmpl.yml` skips benchmark result checks when `inputs.eval-only == true`.
- The log says `EVAL_ONLY mode: skipping throughput benchmark`.
- It then runs `lm_eval` on `utils/evals/gsm8k.yaml`.
- The job passed GSM8K:
  - strict match: `0.9310`
  - flexible extract: `0.9431`

Most failed jobs did not run servers or benchmarks. They failed during `actions/checkout` cleanup:

```text
File was unable to be removed Error: EACCES: permission denied,
rmdir '/it-share/gharunners2/gharunner06/actions-runner/_work/InferenceX/InferenceX/LOGS/agentic'
```

This is a stale root-owned or otherwise non-runner-owned workspace artifact. Because checkout runs before repo code is available, the existing pre/post launch cleanup in runner scripts cannot fix this class of failure.

The successful eval job also shows workspace permission problems after Slurm completion:

```text
cp .../eval_results/meta_env.json .../InferenceX/meta_env.json: Permission denied
cp .../eval_results/results_*.json .../InferenceX/results_*.json: Permission denied
cp .../eval_results/samples_gsm8k_*.jsonl .../InferenceX/samples_*.jsonl: Permission denied
```

Despite those errors, the result file was present later and upload/verification passed. The copy loop is misleading because it echoes "Copied" even when `cp` failed.

## CI Cleanup Recommendations

1. Add a pre-checkout cleanup step in the reusable workflow before `actions/checkout`, using `sudo rm -rf` for known stale paths:
   - `$GITHUB_WORKSPACE/LOGS`
   - `$GITHUB_WORKSPACE/benchmark_logs`
   - `$GITHUB_WORKSPACE/benchmark_artifacts`
   - root-level `results*.json`, `samples*.jsonl`, `meta_env.json`

2. Fix eval artifact extraction in `runners/launch_mi355x-amds.sh`:
   - Check `cp` return codes before printing "Copied".
   - If copying into `$GITHUB_WORKSPACE` needs sudo due to ownership drift, use `sudo cp` followed by `sudo chown "$USER":"$USER"` on copied artifacts.

3. Ensure containers do not write under `$GITHUB_WORKSPACE` as root where possible. Prefer `benchmark_logs` and then copy artifacts back with normalized ownership.

4. Manual immediate cleanup for current runners:
   - On affected runner hosts, remove stale workspace paths with sudo:
     `sudo rm -rf /it-share/gharunners2/gharunner*/actions-runner/_work/InferenceX/InferenceX/LOGS`
