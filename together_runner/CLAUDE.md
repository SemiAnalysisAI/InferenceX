# together_runner — Claude Code memory

Single-node (this 8×B200) **multi-engine** (SGLang + vLLM) inference benchmark
harness for InferenceX. Standalone helper scripts; does NOT touch the CI sweep.
Built on patterns from `../benchmarks/benchmark_lib.sh`,
`../runners/launch_b200-dgxc.sh`, `../benchmarks/single_node/fixed_seq_len/`.

## What it does (3 pillars)
1. **Two-tier smoke** — `run_0_preflight.sh` (gate-1: env/machine, seconds) +
   gate-2 minimal real completion inside `--full`.
2. **Weight externalization** — `prestage_weights.sh` downloads once to
   `$MODELS_ROOT/$PROFILE` (+`.ready`); launches then do ZERO download.
3. **Staged monitoring + baseline/delta** — `bench_lib.sh` maps server-log
   markers (engine-aware) to named stages with per/next-stage + to-ready ETA;
   `compare.py` diffs results vs committed baselines (throughput ±5%) and
   `collect`s the whole fleet.

## Run it
```bash
cd together_runner
# SGLang (default engine), DSR1-FP4:
bash run_all.sh --smoke|--full|--baseline
# vLLM, gpt-oss-120b FP4:
ENGINE=vllm PROFILE=gptoss-fp4 bash run_all.sh --full
# sweep concurrencies into baselines (one warm server):
ENGINE=vllm PROFILE=gptoss-fp4 CONC_LIST="16 64 128" bash record_baselines.sh
python3 compare.py collect        # fleet table (+Δ vs baseline), framework column
python3 compare.py table --write-readme   # refresh the Baseline results table in README.md
```
All config in `config.env` (sourced everywhere; override by exporting first).
Each step also runs standalone (`bash run_2_launch_server.sh`).

## Key config (config.env)
- **`ENGINE`** = `sglang` (default) | `vllm`  → also the *framework* dimension.
- **`PROFILE`** = recipe: `dsr1-fp4` (sglang, TP8/EP8) · `gptoss-fp4`
  (vllm, openai/gpt-oss-120b, TP4) · `smoke` (Llama-8B, TP1). MODEL/TP/EP default
  per profile.
- Per-engine default `IMAGE`: sglang→`lmsysorg/sglang:dev-cu13`,
  vllm→`vllm/vllm-openai:v0.22.0`. `CONTAINER_NAME` defaults to `tr_<engine>`.
- **`ENABLE_TUNING`** default **0**. sglang: `--disable-flashinfer-autotune`
  (fast ~5m vs ~15-24m tuned). **vLLM: label only** (no separable fp4 autotune).
- Paths on the 14TB array under `/scratch/home/johnson/`: `HF_CACHE`,
  `MODELS_ROOT`, `FLASHINFER_CACHE`. NEVER use the 34GB root disk for weights.
- `REPO_ROOT` (repo root) is mounted read-only at `/inferencex` in the container
  so the **unified bench client** `utils/bench_serving/benchmark_serving.py` is
  available to BOTH engines (identical metric definitions → fair comparison).

## Directory layout (scales across hw × cluster × host × date × framework × recipe × conc)
- **results/** (gitignored, raw, provenance-first):
  `results/<hw>/<cluster>/<host>/<date>/<engine>_<profile>_<seqtag>_<tuned|untuned>_conc<N>.json`
  + companion `*.bench.json` (raw) and `*.gpu.csv`.
- **baselines/** (committed golden):
  `baselines/<hw>/<framework>/<profile>/<seqtag>/<tuned|untuned>/conc<N>.json`,
  optional per-cluster override `baselines/<hw>/<cluster>/<framework>/...`
  (compare prefers cluster-specific, falls back to hw-wide). `record_baselines.sh`
  saves hw-wide by default; `BASELINE_CLUSTER=1` scopes under the cluster.
- Query any axis with `compare.py collect` (reads embedded JSON fields, not path).

## Baselines & metrics
- Compare key = `(hw, framework, profile, seqtag, tuning, conc)`. **Keep ENGINE +
  ENABLE_TUNING consistent** between reference and test (framework keeps engines
  apart; tuning keeps tuned/untuned apart — autotune alone moves throughput ~4–10%).
- Primary pass/fail: `total_token_throughput`, regression if it drops **>5%** vs
  baseline (`compare.py compare` → non-zero exit). Result JSON mirrors
  `../utils/process_result.py` (reuse `collect_results.py`/`summarize.py`).
- Reference baselines on THIS node (`gpu-dp-96sjj-gkwph`), 1k1k, unified client
  (`benchmark_serving.py`), total tok/s:
  - **sglang dsr1-fp4 tuned**: conc16=2288 · 64=5684 · 128=9672 · 256=14749
  - **vllm gpt-oss-120b (TP4, untuned)**: conc16=13952 · 64=37333 · 128=49232 · 256=61760 · 512=75788
  (different models — not a head-to-head; gpt-oss-120b is far lighter than DSR1's MoE.)

## Startup stages
- sglang: engine-init → weight-load → [autotune ~12m if ENABLE_TUNING=1] →
  graph-capture → ready (markers: `Load weight begin`, `Tuning fp4_gemm`,
  `Capture cuda graph begin`).
- vllm: engine-init → weight-load → graph-capture → ready, no autotune (markers:
  `Loading model weights`, `Capturing CUDA graph`). Readiness via `/health` (both).
- Live: `docker exec tr_<engine> tail -f /root/server.log`.

## Gotchas
- **docker-proxy holds host PORT for the container's whole life** — preflight's
  port check is container-aware (only flags a *different* container/process).
  Only one engine can own PORT 8888 at a time; stop the other first.
- **Reuse container → relaunch fresh server**: `pkill -f "$(server_proc_pat)"`
  (sglang.launch_server / vllm serve) first; record_baselines.sh does this.
- vLLM specifics: image must be pulled first (`docker pull vllm/vllm-openai:v0.22.0`);
  OOM at load → set `VLLM_OOM_GUARD=1` or lower `GPU_MEM_UTIL` (KLAUD_DEBUG §2);
  both engines launch with `--served-model-name $MODEL` so bench/chat model id matches.
- See `../KLAUD_DEBUG.md` for B200/B300 OOM / DeepGemm / timeout modes.
- Teardown: `docker exec tr_<engine> pkill -f "$(...)"` or `docker rm -f tr_<engine>`.
- Commit identity for this repo: `Johnsonms <lizhaofu@gmail.com>`, no Claude trailer.
