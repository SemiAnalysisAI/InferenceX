# together_runner

Single-node (this 8×B200) multi-engine (SGLang + vLLM) benchmark harness for InferenceX: fast
**smoke** preflight, **externalized weight staging**, **staged startup
monitoring with ETA**, and **baseline/delta** comparison across machines.

Standalone helper scripts — does not touch the CI sweep (`run-sweep.yml`,
`benchmark_lib.sh`). Built on the patterns in `../benchmarks/benchmark_lib.sh`
and `../runners/launch_b200-dgxc.sh`.

## Layout

```
config.env            # ① the ONLY place you pre-set variables
sglang_lib.sh         # shared: env-check, stage timing/ETA, startup monitor
run_sglang_0_preflight.sh   # gate-1: machine/env checks (seconds)
prestage_weights.sh         # ② download weights once -> zero-download launches
run_sglang_1_start_container.sh
run_sglang_2_launch_server.sh
run_sglang_3_test_client.sh # health + chat + bench, emits result JSON
run_all.sh            # ③ one-click: --smoke | --full | --baseline
record_baselines.sh   # sweep CONC_LIST against one warm server, save baselines
compare.py            # ④ emit result schema / delta vs baseline / collect fleet
baselines/<hw>/<framework>/<profile>/<seqtag>/<tuned|untuned>/conc<N>.json   # committed golden
          (optional per-cluster override: baselines/<hw>/<cluster>/<framework>/...)
results/<hw>/<cluster>/<host>/<date>/<engine>_<profile>_<seqtag>_<tuned|untuned>_conc<N>.json  # gitignored
```

Directory scheme scales across **hw × cluster × host × date × recipe × conc**.
Results are provenance-first (a machine's whole history sits together); cross-date /
cross-hw queries go through `python3 compare.py collect` (reads embedded JSON fields,
not the path). Baselines are recipe-keyed (git history = the time axis).

## Quick start

```bash
cd InferenceX/together_runner
# 0. set token once (stored at ~/.cache/huggingface/token); edit config.env for the rest
hf auth login            # or: export HF_TOKEN=hf_xxx

# Smoke (seconds) — catches unset vars, wrong image, busy port, bad token, low disk:
bash run_all.sh --smoke

# Engine/profile is selected via ENGINE+PROFILE (default sglang/dsr1-fp4):
ENGINE=vllm PROFILE=gptoss-fp4 bash run_all.sh --full      # vLLM, gpt-oss-120b FP4

# Full run (preflight -> prestage -> start -> launch+monitor -> gate-2 -> bench -> compare):
bash run_all.sh --full

# Record the golden baseline on this reference machine (autotune ON), then commit it:
bash run_all.sh --baseline
git add baselines/ && git commit -m "baseline: b200 dsr1-fp4 1k1k conc16"
```

Scripts are composable — each runs on its own (`bash run_sglang_2_launch_server.sh`).

## ① Configuration (`config.env`)

Everything you might pre-set lives here, with safe defaults for this node:

| var | default | meaning |
|---|---|---|
| `HF_TOKEN_FILE` / `HF_TOKEN` | `~/.cache/huggingface/token` | HF auth (file preferred) |
| `HF_CACHE` | `/scratch/home/johnson/huggingface` | HF hub cache (blobs/hashes); on 14TB array |
| `MODELS_ROOT` | `/scratch/home/johnson/models` | pre-staged weights root |
| `FLASHINFER_CACHE` | `/scratch/home/johnson/flashinfer-cache` | persistent JIT kernel cache |
| `PROFILE` | `dsr1-fp4` | `smoke` (Llama-8B TP1) or `dsr1-fp4` (TP8/EP8) |
| `MODEL` / `MODEL_REVISION` | `nvidia/DeepSeek-R1-0528-FP4` / `main` | model + pinned revision |
| `TP` / `EP_SIZE` | `8` / `8` | parallelism |
| `ISL`/`OSL`/`CONC` | `1024`/`1024`/`16` | benchmark shape |
| **`ENABLE_TUNING`** | **`0`** | `0` = `--disable-flashinfer-autotune` (fast); `1` = keep (baselines) |
| `IMAGE` | `lmsysorg/sglang:dev-cu13` | container image |
| `HW` | `b200` | hardware tag in result/baseline paths |

## ② Weight externalization

`prestage_weights.sh` downloads `$MODEL@$MODEL_REVISION` once into
`$MODELS_ROOT/$PROFILE` and writes a `.ready` marker. Subsequent launches detect
`.ready` and start with **zero download** (mirrors the `MODEL_PATH` pattern in
`../runners/launch_b200-dgxc.sh`). Set `MODEL_PATH=/abs/path` to point at weights
staged elsewhere.

## ③ Two-tier smoke + one-click

- **Gate-1** (`--smoke`): seconds-level PASS/FAIL on vars, docker, image,
  GPU≥TP, free GPUs, port, HF token+reachability, disk, weights-staged.
- **Gate-2** (inside `--full`): the target model serves one real completion
  before the full benchmark — proves the recipe works on this hardware.

## ④ Baseline & metrics

Primary pass/fail metric: **`total_token_throughput` (tok/s)**, regression if it
drops **>5%** vs the committed baseline (`compare.py compare`, non-zero exit).
Reported alongside (info): `output_token_throughput`, `request_throughput`,
median/p99 TTFT, median/p99 TPOT, median ITL, median/p99 E2E latency, and
`tokens_per_kw` (from host GPU-power sampling — energy view à la InferenceX
tokens/MW). Result JSON keys mirror `../utils/process_result.py` so
`collect_results.py` / `summarize.py` can aggregate across machines.

Compare key = `(hw, framework, profile, seqtag, tuning, conc)`. `compare.py compare`
looks up the baseline cluster-first then hw-wide:
`baselines/<hw>/<cluster>/<framework>/...` → `baselines/<hw>/<framework>/<profile>/<seqtag>/<tuned|untuned>/conc<N>.json`.
Tuning is part of the key so tuned runs compare to tuned baselines (autotune alone
shifts throughput ~4–10%); keep `ENABLE_TUNING` consistent between reference and test.

## Baseline results (this node, `gpu-dp-96sjj-gkwph`)

Golden numbers measured here with the unified client. Regenerate this section with
`python3 compare.py table` after recording/updating baselines.

<!-- BEGIN baselines (compare.py table) -->
**sglang · dsr1-fp4** (nvidia/DeepSeek-R1-0528-FP4) — b200, 1k1k, TP8, tuned

| conc | total tok/s | output tok/s | median TPOT (ms) | median TTFT (ms) | tok/kW |
|---:|---:|---:|---:|---:|---:|
| 16 | 2,288 | 1,122 | 12.9 | 359 | 326 |
| 64 | 5,684 | 2,704 | 21.3 | 503 | 680 |
| 128 | 9,672 | 4,789 | 24.5 | 525 | 1,047 |
| 256 | 14,749 | 7,197 | 33.0 | 607 | 1,587 |

**vllm · gptoss-fp4** (openai/gpt-oss-120b) — b200, 1k1k, TP4, untuned

| conc | total tok/s | output tok/s | median TPOT (ms) | median TTFT (ms) | tok/kW |
|---:|---:|---:|---:|---:|---:|
| 16 | 13,952 | 3,416 | 3.7 | 26 | 1,503 |
| 64 | 37,333 | 8,296 | 6.2 | 41 | 3,273 |
| 128 | 49,232 | 13,227 | 8.1 | 102 | 4,942 |
| 256 | 61,760 | 15,650 | 14.6 | 369 | 5,790 |
| 512 | 75,788 | 18,708 | 22.5 | 687 | 6,595 |
<!-- END baselines -->

> sglang dsr1-fp4 and vllm gpt-oss-120b are **different models** — not a head-to-head;
> gpt-oss-120b is far lighter than DSR1's MoE, hence the much higher throughput.

## Startup stages & ETA

`bench_lib.sh` maps server-log markers (engine-aware) to named stages and prints
elapsed + a baseline ETA per stage:

| stage | marker | ~baseline | notes |
|---|---|---|---|
| container-start | — | ~10s | image already local |
| weight-load | `Load weight begin` | ~5–8 min | from cache/local |
| autotune | `Tuning fp4_gemm` | ~12 min | **skipped when `ENABLE_TUNING=0`** |
| graph-capture | `Capture cuda graph begin` | ~1 min | `cuda-graph-max-bs` sweep |
| ready | `fired up and ready to roll` | — | `/health` green |

`ENABLE_TUNING=0` typically cuts cold start from ~24 min to ~10 min.

## Troubleshooting

See `../KLAUD_DEBUG.md` for catalogued B200/B300 failure modes. Common:
- **OOM at load**: lower `--mem-fraction-static` (recipe in `run_sglang_2`).
- **Server died at startup**: the monitor dumps the last log lines; also
  `docker exec $CONTAINER_NAME tail -n 80 /root/server.log`.
- **Gated model 401**: refresh `hf auth login` / rotate `HF_TOKEN`.
- **Teardown**: `docker exec $CONTAINER_NAME pkill -f sglang.launch_server`
  (stop server) or `docker rm -f $CONTAINER_NAME` (remove container).

## Not in scope (yet)

Multi-node Slurm/srtctl (log format already carries `[node/rank]` for the
extension); CI changes; persisting autotune *results* across restarts (under
investigation — `ENABLE_TUNING=0` is the current fast lever).
