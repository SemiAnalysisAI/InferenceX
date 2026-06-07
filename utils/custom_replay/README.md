# Custom multi-turn agentic replay (no weka / no aiperf)

A small, self-contained replayer for **Claude Code** multi-turn traces, used to
benchmark an OpenAI-compatible server (vLLM) and sweep **number of concurrent
sessions** for a latency/throughput **pareto frontier**.

This is an alternative to the aiperf + `inferencex-agentx-mvp` + weka path used
by the other `benchmarks/single_node/agentic/*.sh` recipes. It consumes a
standardized `*.replay.jsonl` instead of the weka HF datasets, and uses
**pre-canned assistant replay** (recorded turns drive the growing prefix; the
server's generation is timed and discarded). Deps are managed with **uv**.

## Scripts

- `make_replay_dataset.py` — raw Claude Code trace batch → `*.replay.jsonl`
- `replay_bench.py` — one concurrency point: N sessions, capture TTFT/TPOT/ITL/ISL/OSL/cache
- `sweep_pareto.py` — sweep `--concurrency`, write `pareto.csv` + `pareto.png`
- `analyze_replay.py` — trace data analysis (depth, ISL/OSL, cache reuse, fan-out)

## Entry point

`benchmarks/single_node/agentic/dsv4_fp4_vllm_replay.sh` launches DeepSeek-V4
**FP4** on vLLM (single node, pure TP, prefix caching on), waits for health, then
runs the concurrency sweep. Example:

```bash
MODEL=deepseek-ai/DeepSeek-V4-Pro TP=4 \
DATASET=/path/to/batch_1.replay.jsonl RESULT_DIR=/path/to/results \
CONCURRENCIES=1,2,4,8,16,32,64,128 DURATION=120 WARMUP=20 \
bash benchmarks/single_node/agentic/dsv4_fp4_vllm_replay.sh
```

Standalone sweep against an already-running server:

```bash
uv run --with aiohttp --with numpy --with matplotlib \
  python utils/custom_replay/sweep_pareto.py \
  --dataset batch_1.replay.jsonl --base-url http://0.0.0.0:8888 \
  --model deepseek-ai/DeepSeek-V4-Pro --concurrencies 1,4,16,64 \
  --duration 120 --warmup 20 --result-dir results
```

The `*.replay.jsonl` builder lives with the trace source (the gpumode-triton
project); `make_replay_dataset.py` is mirrored here so the fork is self-contained.
