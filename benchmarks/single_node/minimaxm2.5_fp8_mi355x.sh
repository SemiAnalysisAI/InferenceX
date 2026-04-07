#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    EP_SIZE \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

hf download "$MODEL"

# Set HIP_VISIBLE_DEVICES to match ROCR_VISIBLE_DEVICES for Ray compatibility in vLLM 0.14+
if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

export VLLM_ROCM_USE_AITER=1
# export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1

export HSA_ENABLE_DEBUG=1
export HSA_TOOLS_LIB="/opt/rocm/lib/librocm-debug-agent.so.2"
export ROCM_DEBUG_AGENT_OPTIONS="--all"
ls -la /opt/rocm/lib/librocm-debug-agent.so.2  
SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

if [ "$EP_SIZE" -gt 1 ]; then
  EP=" --enable-expert-parallel"
else
  EP=" "
fi

amd-smi

python3 -c "
import pathlib
f = pathlib.Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py')
code = f.read_text()

# 点A: forward pass 完成后立即 sync
code = code.replace(
    '''        hidden_states, last_hidden_states = self._dummy_run(
            self.max_num_tokens, is_profile=True
        )
        if get_pp_group().is_last_rank:''',
    '''        hidden_states, last_hidden_states = self._dummy_run(
            self.max_num_tokens, is_profile=True
        )
        import torch as _torch_dbg; _torch_dbg.cuda.synchronize(); print('[DEBUG-A] dummy_run sync OK', flush=True)
        if get_pp_group().is_last_rank:'''
)

# 点B: sampler 完成后 sync
code = code.replace(
    '''                output = self._dummy_sampler_run(last_hidden_states)
        else:
            output = None
        self._sync_device()''',
    '''                output = self._dummy_sampler_run(last_hidden_states)
                import torch as _torch_dbg; _torch_dbg.cuda.synchronize(); print('[DEBUG-B] sampler_run sync OK', flush=True)
        else:
            output = None
        self._sync_device()'''
)

# 点D: memory_profiling 退出后 (在 gpu_worker.py)
f.write_text(code)
print('gpu_model_runner.py patched OK')

# patch gpu_worker.py
f2 = pathlib.Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py')
code2 = f2.read_text()
code2 = code2.replace(
    '''            self.model_runner.profile_run()

            profile_torch_peak = torch.accelerator.memory_stats(self.device).get(''',
    '''            self.model_runner.profile_run()
            print('[DEBUG-D] profile_run returned OK', flush=True)

            profile_torch_peak = torch.accelerator.memory_stats(self.device).get('''
)
f2.write_text(code2)
print('gpu_worker.py patched OK')
"

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
vllm serve $MODEL --port $PORT \
--tensor-parallel-size=$TP \
$EP \
--gpu-memory-utilization 0.80 \
--max-model-len $MAX_MODEL_LEN \
--kv-cache-dtype fp8 \
--block-size=32 \
--no-enable-prefix-caching \
--attention-backend "ROCM_AITER_FA" \
--trust-remote-code > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --trust-remote-code

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
