# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM inference benchmarking tools for MiniMax M2.1 model on vLLM, designed for Slurm clusters with pyxis container support.

## Repository Structure

```
inferperf_minimax-m2.1/
├── benchmark_serving_random.py       # Benchmark client (random prompt generation)
├── plot_results.py                   # Visualization script
├── serve_minimax_m2.1_sbatch.sh      # Slurm job: persistent vLLM server
├── bmk_minimax_m2.1_sbatch.sh        # Slurm job: benchmark client
├── results/                          # Benchmark result JSON files
└── model_weights/                    # Downloaded model weights (not committed)
```

## Running Benchmarks

### Two-Job Workflow (Server + Client)

1. **Start the server:**
   ```bash
   sbatch serve_minimax_m2.1_sbatch.sh
   ```

2. **Submit the benchmark client:**
   ```bash
   sbatch bmk_minimax_m2.1_sbatch.sh
   ```

The server writes its URL to `/workspace/server_info.txt`. The client reads this file to connect.

### Benchmark Client Sweep Configuration

The client script runs a sweep of ISL × Concurrency combinations:
- **Input lengths**: 1024, 6144, 10240, 12288, 14336
- **Output length**: 128 (fixed)
- **Concurrency**: 4, 8, 16, 32, 64
- **Total runs**: 25 (5 ISL × 5 concurrency)

### Using benchmark_serving_random.py directly

```bash
python benchmark_serving_random.py \
    --model MiniMaxAI/MiniMax-M2.1 \
    --base-url http://localhost:8000 \
    --random-input-len 2048 \
    --random-output-len 128 \
    --random-range-ratio 0.2 \
    --num-prompts 160 \
    --max-concurrency 16 \
    --num-warmups 32 \
    --request-rate inf \
    --ignore-eos \
    --result-filepath results/output.json
```

## Key Metrics

- **TTFT**: Time to first token (prefill latency)
- **TPOT**: Time per output token, excluding first (decode latency)
- **Interactivity**: 1000/TPOT_ms (tok/s/user) - user-perceived generation speed
- **Throughput**: Input/output tokens processed per second

## Visualizing Results

```bash
source .venv/bin/activate
python plot_results.py --results-dir results --output result_plot.png
```

### Plot Features
- **Two graphs**: Prefill and Decode (side by side)
- **X-axis**: Token Position (ISL values from data)
- **Y-axis**: Throughput (tok/s)
- **Lines**: One per concurrency level (legend: "conc X")

### Result File Naming Convention

Files must match: `*_tp{TP}_isl{ISL}_osl{OSL}_conc{CONC}.json`

Example: `minimax_m2.1_vllm_tp4_isl2048_osl128_conc16.json`

### Required JSON Fields for Plotting

- `input_throughput`: Input tokens/second (for Prefill graph)
- `output_throughput`: Output tokens/second (for Decode graph)

## Slurm Configuration

Both scripts use pyxis for containerized execution:
- Container: `vllm/vllm-openai:nightly-8711b216766bb5d3cbe15161061c3a7d9fffe59c`
- Server: 4 GPUs, 128GB memory, tensor parallelism 4
- Client: 0 GPUs, 16GB memory (CPU-only benchmark driver)