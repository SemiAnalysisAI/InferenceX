# vLLM Dynamic Scheduler Reconfiguration

InferenceX can optionally reconfigure selected vLLM scheduler limits between
benchmark runs without restarting the serving endpoint. This is useful for
single-server sweeps where the model, parallelism, quantization, max context,
and KV cache layout stay fixed, but each benchmark case uses different scheduler
admission limits.

## Requirements

This feature requires a vLLM build that exposes these HTTP endpoints:

- `POST /pause?mode=abort&clear_cache=true`
- `POST /reconfigure` (JSON body)
- `POST /resume`

The stock vLLM releases do not provide `/reconfigure`. You need either:

1. A Docker image built from the vLLM branch containing the reconfigure API
   (see [Building the patched image](#building-the-patched-image) below).
2. A runtime patch applied at container start from a mounted vLLM checkout.

## Reconfigurable Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `max_num_batched_tokens` | int > 0 | Max tokens scheduled per step |
| `max_num_seqs` | int > 0 | Max concurrent sequences |
| `enable_chunked_prefill` | bool | Toggle chunked prefill |
| `long_prefill_token_threshold` | int >= 0 | Cap prefill chunk size (0 = no cap) |

Everything else (TP, EP, GPU memory, KV cache dtype, block size, CUDA graphs,
compilation config, etc.) is baked in at startup and cannot be changed.

## Enabling

Set `VLLM_DYNAMIC_RECONFIGURE=1` and the desired parameter env vars before
calling `run_benchmark_serving` with `--backend vllm`:

```bash
export VLLM_DYNAMIC_RECONFIGURE=1
export VLLM_MAX_NUM_BATCHED_TOKENS=32768
export VLLM_MAX_NUM_SEQS=128
```

`run_benchmark_serving` calls `reconfigure_vllm_scheduler` before each benchmark
run. The helper pauses vLLM, sends a JSON body to `/reconfigure`, and resumes.

## A/B Test Script

`benchmarks/test_reconfigure_sweep.sh` runs back-to-back comparisons:

- **Phase A (baseline):** N cold starts, one per parameter combo
- **Phase B (reconfigure):** 1 cold start, N reconfigure cycles

```bash
export MODEL=openai/gpt-oss-120b TP=8 CONC=32 ISL=1024 OSL=1024
bash benchmarks/test_reconfigure_sweep.sh
```

## Building the Patched Image

The changes are pure Python -- no C++/CUDA recompilation needed. Overlay them
onto the stock vLLM image:

```bash
# From the vllm repo root (on the branch with the reconfigure patch)
docker build -f docker/Dockerfile.reconfigure-overlay \
  -t ghcr.io/semianalysisai/vllm-reconfigure:test1 .

docker push ghcr.io/semianalysisai/vllm-reconfigure:test1
```

Or patch at runtime by mounting the vLLM checkout and running the overlay script
at the top of the benchmark:

```bash
docker run --gpus all --rm -it --network host --shm-size 64g \
  -v /path/to/vllm:/workspace/vllm-patch:ro \
  -v /path/to/inferencex:/workspace \
  vllm/vllm-openai:v0.15.1 \
  bash -c '
    bash /workspace/vllm-patch/docker/apply-reconfigure-overlay.sh
    export MODEL=openai/gpt-oss-120b TP=8 CONC=32 ISL=1024 OSL=1024
    bash /workspace/benchmarks/test_reconfigure_sweep.sh
  '
```

## Safety Notes

Do not use this mechanism to change startup-time engine settings such as model,
quantization, tensor/data/expert parallelism, KV cache dtype, block size,
`gpu_memory_utilization`, or `max_model_len`. Launch vLLM with the largest static
capacity required by the sweep and use dynamic reconfiguration only for scheduler
limits.
