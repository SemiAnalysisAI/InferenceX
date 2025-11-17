import sys
import json
import os
from pathlib import Path


hw = os.environ.get('RUNNER_TYPE')
tp_size = int(os.environ.get('TP'))
ep_size = int(os.environ.get('EP_SIZE'))
prefill_gpus_str = os.environ.get('PREFILL_GPUS', '')
decode_gpus_str = os.environ.get('DECODE_GPUS', '')

# If empty string (aggregated runs), assign to tp_size (total gpus), otherwise convert to int
prefill_gpus = tp_size if not prefill_gpus_str else int(prefill_gpus_str)
decode_gpus = tp_size if not decode_gpus_str else int(decode_gpus_str)
dp_attention = os.environ.get('DP_ATTENTION')
result_filename = os.environ.get('RESULT_FILENAME')
framework = os.environ.get('FRAMEWORK')
precision = os.environ.get('PRECISION')
mtp_mode = os.environ.get('MTP_MODE')

with open(f'{result_filename}.json') as f:
    bmk_result = json.load(f)

data = {
    'hw': hw,
    'tp': tp_size,
    'ep': ep_size,
    'dp_attention': dp_attention, # true or false
    'conc': int(bmk_result['max_concurrency']),
    'model': bmk_result['model_id'],
    'framework': framework,
    'precision': precision,
    'tput_per_gpu': float(bmk_result['total_token_throughput']) / tp_size,
    'output_tput_per_gpu': float(bmk_result['output_throughput']) / decode_gpus,
    'input_tput_per_gpu': (float(bmk_result['total_token_throughput']) - float(bmk_result['output_throughput']) )/ prefill_gpus
}

# Check if both PREFILL_GPUS and DECODE_GPUS env vars exist and are not empty
if prefill_gpus_str and decode_gpus_str:
    data['disagg'] = True
    data['num_prefill_gpu'] = prefill_gpus
    data['num_decode_gpu'] = decode_gpus
else:
    data['disagg'] = False

if mtp_mode:  # MTP
    data['mtp'] = mtp_mode

for key, value in bmk_result.items():
    if key.endswith('ms'):
        data[key.replace('_ms', '')] = float(value) / 1000.0
    if 'tpot' in key:
        data[key.replace('_ms', '').replace('tpot', 'intvty')] = 1000.0 / float(value)

print(json.dumps(data, indent=2))

with open(f'agg_{result_filename}.json', 'w') as f:
    json.dump(data, f, indent=2)
