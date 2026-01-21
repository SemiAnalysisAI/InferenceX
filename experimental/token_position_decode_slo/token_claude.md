# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Inference performance benchmarking suite for the Kimi K2 Thinking model on vLLM. Measures throughput and latency metrics under various Service Level Objectives (SLOs).

## Architecture

```
serve_kimi_k2_sbatch.sh (SLURM job)
    ↓
vLLM Server (port 8000, OpenAI-compatible API)
    ↑
benchmark_serving_random.py (async client)
    ↓
JSON results → plot_sla_frontier.py → PNG visualization
```

**Server-client model**: The server runs persistently on 8 GPUs via SLURM. Client jobs read the server URL from `/logs/server_info.txt` and send concurrent async requests.

## Key Components

- **`benchmark_serving_random.py`**: Async benchmarking client using aiohttp. Generates random prompts with controlled token lengths, sends requests to vLLM's `/v1/completions` endpoint, and measures latency/throughput metrics.
- **`serve_kimi_k2_sbatch.sh`**: SLURM script to start the vLLM inference server with tensor-parallel-8 configuration.
- **`bmk_kimi_k2_sbatch.sh`**: SLURM script that runs benchmark sweeps across input lengths (1024-14336) and concurrency levels (4-64).
- **`plot_sla_frontier.py`**: Visualization tool that loads benchmark results and generates SLA frontier plots.

## Key Metrics

- **TTFT**: Time to First Token (prefill latency)
- **TPOT**: Time Per Output Token (decode latency)
- **E2EL**: End-to-End Latency
- **Goodput**: Requests meeting SLA constraints

## Running Benchmarks

```bash
# Start server (allocates 8 GPUs, runs persistently)
sbatch serve_kimi_k2_sbatch.sh

# Wait for server to start, then run client sweep
sbatch bmk_kimi_k2_sbatch.sh

# Generate visualization from results
python plot_sla_frontier.py --results-dir results/
```

### Single Benchmark Run

```bash
python benchmark_serving_random.py \
    --model moonshotai/Kimi-K2-Thinking \
    --base-url "http://hostname:8000" \
    --random-input-len 4096 \
    --random-output-len 128 \
    --num-prompts 100 \
    --max-concurrency 16 \
    --request-rate inf \
    --ignore-eos \
    --result-filepath results/output.json
```

## Result File Format

JSON files in `results/` follow naming pattern: `kimi_k2_vllm_tp{TP}_isl{INPUT}_osl{OUTPUT}_conc{CONC}.json`

Key fields: `input_throughput`, `output_throughput`, `p99_ttft`, `p99_tpot`, `median_ttft`, `median_tpot`, `completed`, `total_input`, `total_output`

## Environment

- **Container**: `vllm/vllm-openai:nightly`
- **Model**: `moonshotai/Kimi-K2-Thinking` (MoE, 1T params, 32B active)
- **Dependencies**: transformers, aiohttp, numpy, matplotlib, tqdm (pre-configured in `.venv`)
