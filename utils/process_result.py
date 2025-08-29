import sys
import json
from pathlib import Path


framework = sys.argv[1]  # First arg is the framework (TRT-LLM, vLLM, SGLang, etc.)
tp_size = int(sys.argv[2])
result_filename = sys.argv[3]
precision = sys.argv[4] if len(sys.argv) > 4 else 'fp8'  # Fourth arg is precision, default to fp8

with open(f'{result_filename}.json') as f:
    bmk_result = json.load(f)

# Extract hardware from result filename or runner name
# Result filename format: {exp-name}_tp{tp}_conc{conc}_{runner}
# We need to extract the hardware type from the runner
result_parts = result_filename.split('_')
if len(result_parts) >= 4:
    runner_part = result_parts[-1]  # Last part is the runner
    # Extract hardware type (e.g., 'b200' from 'b200-nv_0')
    hw = runner_part.split('-')[0].upper()  # Convert to uppercase for consistency
else:
    hw = "UNKNOWN"

data = {
    'hw': hw,           # Hardware (B200, H200, etc.)
    'framework': framework,  # Framework (TRT-LLM, vLLM, SGLang, etc.)
    'precision': precision,  # Precision (fp8, fp4, etc.)
    'tp': tp_size,
    'conc': int(bmk_result['max_concurrency']),
    'model': bmk_result['model_id'],
    'tput_per_gpu': float(bmk_result['total_token_throughput']) / tp_size
}

for key, value in bmk_result.items():
    if key.endswith('ms'):
        data[key.replace('_ms', '')] = float(value) / 1000.0
    if 'tpot' in key:
        data[key.replace('_ms', '').replace('tpot', 'intvty')] = 1000.0 / float(value)

print(json.dumps(data, indent=2))

with open(f'agg_{result_filename}.json', 'w') as f:
    json.dump(data, f, indent=2)
