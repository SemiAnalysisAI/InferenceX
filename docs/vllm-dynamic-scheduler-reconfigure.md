# vLLM Dynamic Scheduler Reconfiguration

InferenceX can optionally reconfigure selected vLLM scheduler limits between
benchmark runs without restarting the serving endpoint. This is useful for
single-server sweeps where the model, parallelism, quantization, max context,
and KV cache layout stay fixed, but each benchmark case uses different scheduler
admission limits.

## Requirements

This feature requires a vLLM build that exposes these HTTP endpoints:

- `POST /pause?mode=keep`
- `POST /reconfigure_scheduler`
- `POST /resume`

The stock vLLM releases do not provide `/reconfigure_scheduler` unless the
runtime scheduler reconfiguration patch has been included in the installed vLLM
package or container image.

## Enabling

Set `VLLM_DYNAMIC_RECONFIGURE=1` before calling `run_benchmark_serving` with
`--backend vllm`.

Supported environment variables:

```bash
export VLLM_DYNAMIC_RECONFIGURE=1
export VLLM_MAX_NUM_BATCHED_TOKENS=32768
export VLLM_MAX_NUM_SEQS=128
export VLLM_MAX_NUM_SCHEDULED_TOKENS=32768
```

`run_benchmark_serving` calls the reconfiguration helper before each benchmark
run. The helper pauses vLLM, applies the requested scheduler limits, and resumes
serving.

## Example Sweep

Launch vLLM once with the largest static capacity needed by the sweep, then vary
scheduler limits between benchmark cases:

```bash
vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-num-seqs 256 \
  --max-num-batched-tokens 32768 \
  > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

for conc in 1 2 4 8 16 32 64 128; do
    export VLLM_DYNAMIC_RECONFIGURE=1
    export VLLM_MAX_NUM_SEQS="$conc"
    export VLLM_MAX_NUM_BATCHED_TOKENS=32768
    export VLLM_MAX_NUM_SCHEDULED_TOKENS=32768

    run_benchmark_serving \
      --model "$MODEL" \
      --port "$PORT" \
      --backend vllm \
      --input-len "$ISL" \
      --output-len "$OSL" \
      --random-range-ratio "$RANDOM_RANGE_RATIO" \
      --num-prompts "$((conc * 10))" \
      --max-concurrency "$conc" \
      --result-filename "${RESULT_FILENAME}_conc${conc}" \
      --result-dir /workspace/ \
      --server-pid "$SERVER_PID"
done
```

## Distribution of the vLLM Patch

Cluster runs must use a vLLM package or image that includes the dynamic scheduler
API. Practical options are:

1. Build a custom benchmark container from the vLLM branch that contains the API.
2. Install the patched vLLM wheel in the InferenceX job before starting `vllm serve`.
3. Mount a patched vLLM checkout and install it editable in the benchmark image.

For reproducible cluster results, prefer a custom container or pinned wheel and
record the vLLM commit SHA in the benchmark metadata.

## Safety Notes

Do not use this mechanism to change startup-time engine settings such as model,
quantization, tensor/data/expert parallelism, KV cache dtype, block size,
`gpu_memory_utilization`, or `max_model_len`. Launch vLLM with the largest static
capacity required by the sweep and use dynamic reconfiguration only for scheduler
limits.
