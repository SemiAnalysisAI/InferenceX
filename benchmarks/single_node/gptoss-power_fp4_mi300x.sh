#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
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

# If the machine runs a MEC FW older than 177, RCCL
# cannot reclaim some memory.
version=`rocm-smi --showfw | grep MEC | head -n 1 |  awk '{print $NF}'`
if [[ "$version" == "" || $version -lt 177 ]]; then
  export HSA_NO_SCRATCH_RECLAIM=1
fi

# Set HIP_VISIBLE_DEVICES to match ROCR_VISIBLE_DEVICES for Ray compatibility in vLLM 0.14+
if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

# --- GPU Power Monitoring (rocm-smi, 1-second intervals) ---
GPU_POWER_CSV="/workspace/gpu_power_${RESULT_FILENAME}.csv"

echo "Starting GPU power monitoring -> $GPU_POWER_CSV"

python3 -c "
import subprocess, time, csv, sys, re

csv_file = '${GPU_POWER_CSV}'

with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'timestamp', 'index', 'power.draw [W]', 'power.limit [W]',
        'temperature.edge [C]', 'temperature.junction [C]',
        'utilization.gpu [%]', 'memory.used [MiB]', 'memory.total [MiB]'
    ])

def parse_val(s):
    \"\"\"Extract numeric value from rocm-smi output string.\"\"\"
    if not s:
        return 0.0
    m = re.search(r'[\d.]+', str(s))
    return float(m.group()) if m else 0.0

while True:
    try:
        ts = time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())
        # Get power info
        power_out = subprocess.run(
            ['rocm-smi', '--showpower', '--csv'],
            capture_output=True, text=True, timeout=5
        ).stdout.strip().split('\n')
        # Get temperature info
        temp_out = subprocess.run(
            ['rocm-smi', '--showtemp', '--csv'],
            capture_output=True, text=True, timeout=5
        ).stdout.strip().split('\n')
        # Get utilization info
        use_out = subprocess.run(
            ['rocm-smi', '--showuse', '--csv'],
            capture_output=True, text=True, timeout=5
        ).stdout.strip().split('\n')
        # Get memory info
        mem_out = subprocess.run(
            ['rocm-smi', '--showmeminfo', 'vram', '--csv'],
            capture_output=True, text=True, timeout=5
        ).stdout.strip().split('\n')

        # Parse power (skip header)
        power_data = {}
        for line in power_out[1:]:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 2:
                gpu_id = parts[0].replace('card', '')
                power_data[gpu_id] = parse_val(parts[1])

        # Parse power cap
        powercap_out = subprocess.run(
            ['rocm-smi', '--showmaxpower', '--csv'],
            capture_output=True, text=True, timeout=5
        ).stdout.strip().split('\n')
        powercap_data = {}
        for line in powercap_out[1:]:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 2:
                gpu_id = parts[0].replace('card', '')
                powercap_data[gpu_id] = parse_val(parts[1])

        # Parse temperature (edge and junction/hotspot)
        temp_edge = {}
        temp_junction = {}
        for line in temp_out[1:]:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                gpu_id = parts[0].replace('card', '')
                temp_edge[gpu_id] = parse_val(parts[1])
                temp_junction[gpu_id] = parse_val(parts[2]) if len(parts) > 2 else parse_val(parts[1])

        # Parse GPU utilization
        gpu_use = {}
        for line in use_out[1:]:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 2:
                gpu_id = parts[0].replace('card', '')
                gpu_use[gpu_id] = parse_val(parts[1])

        # Parse memory (VRAM used and total)
        mem_used = {}
        mem_total = {}
        for line in mem_out[1:]:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                gpu_id = parts[0].replace('card', '')
                # rocm-smi reports memory in bytes
                used_bytes = parse_val(parts[1])
                total_bytes = parse_val(parts[2])
                mem_used[gpu_id] = used_bytes / (1024*1024)  # Convert to MiB
                mem_total[gpu_id] = total_bytes / (1024*1024)

        # Write rows
        all_gpus = sorted(set(power_data.keys()) | set(gpu_use.keys()))
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for gpu_id in all_gpus:
                writer.writerow([
                    ts,
                    gpu_id,
                    f'{power_data.get(gpu_id, 0):.1f}',
                    f'{powercap_data.get(gpu_id, 0):.1f}',
                    f'{temp_edge.get(gpu_id, 0):.0f}',
                    f'{temp_junction.get(gpu_id, 0):.0f}',
                    f'{gpu_use.get(gpu_id, 0):.1f}',
                    f'{mem_used.get(gpu_id, 0):.0f}',
                    f'{mem_total.get(gpu_id, 0):.0f}'
                ])

    except Exception as e:
        print(f'Power monitor warning: {e}', file=sys.stderr)

    time.sleep(1)
" &
POWER_MONITOR_PID=$!
echo "Power monitor PID: $POWER_MONITOR_PID"

set -x
vllm serve $MODEL --port $PORT \
--tensor-parallel-size=$TP \
--gpu-memory-utilization 0.95 \
--max-model-len $MAX_MODEL_LEN \
--compilation-config  '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
--block-size=64 \
--no-enable-prefix-caching \
--disable-log-requests > $SERVER_LOG 2>&1 &

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
    --num-prompts $(( $CONC * 10 )) \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/

# Stop power monitoring
kill $POWER_MONITOR_PID 2>/dev/null
wait $POWER_MONITOR_PID 2>/dev/null

# Print power summary
echo ""
echo "=== GPU Power Summary ==="
POWER_ROWS=$(wc -l < "$GPU_POWER_CSV")
echo "Collected $((POWER_ROWS - 1)) power samples"
echo "CSV saved to: $GPU_POWER_CSV"
python3 -c "
import csv, sys
from collections import defaultdict

data = defaultdict(lambda: {'power': [], 'temp_edge': [], 'temp_junction': [], 'util': [], 'mem_used': [], 'mem_total': []})

with open('${GPU_POWER_CSV}') as f:
    reader = csv.DictReader(f)
    for row in reader:
        gpu = row['index']
        data[gpu]['power'].append(float(row['power.draw [W]']))
        data[gpu]['temp_edge'].append(float(row['temperature.edge [C]']))
        data[gpu]['temp_junction'].append(float(row['temperature.junction [C]']))
        data[gpu]['util'].append(float(row['utilization.gpu [%]']))
        data[gpu]['mem_used'].append(float(row['memory.used [MiB]']))
        data[gpu]['mem_total'].append(float(row['memory.total [MiB]']))

total_avg = 0
for gpu in sorted(data.keys()):
    d = data[gpu]
    avg_p = sum(d['power'])/len(d['power'])
    max_p = max(d['power'])
    avg_te = sum(d['temp_edge'])/len(d['temp_edge'])
    max_te = max(d['temp_edge'])
    avg_tj = sum(d['temp_junction'])/len(d['temp_junction'])
    max_tj = max(d['temp_junction'])
    avg_u = sum(d['util'])/len(d['util'])
    max_mem = max(d['mem_used'])
    total_avg += avg_p
    print(f'GPU {gpu}: Avg Power={avg_p:.1f}W  Peak={max_p:.1f}W  Avg TEdge={avg_te:.0f}C  Avg TJunction={avg_tj:.0f}C  Peak TJ={max_tj:.0f}C  Avg Util={avg_u:.1f}%  Peak Mem={max_mem:.0f} MiB')
print(f'Total Avg Power: {total_avg:.0f} W ({len(data)} GPUs, {len(list(data.values())[0][\"power\"])} samples/GPU)')
" 2>/dev/null || echo "(Power summary generation failed)"

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
    append_lm_eval_summary
fi
set +x
