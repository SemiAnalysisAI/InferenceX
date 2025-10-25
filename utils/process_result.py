import sys
import json
from pathlib import Path


hw = sys.argv[1]
tp_size = int(sys.argv[2])
ep_size = int(sys.argv[3])
dp_attention = sys.argv[4]
result_filename = sys.argv[5]
framework = sys.argv[6]
precision = sys.argv[7]

with open(f'{result_filename}.json') as f:
    bmk_result = json.load(f)

data = {
    'hw': hw,
    'tp': tp_size,
    'ep': ep_size,
    'conc': int(bmk_result['max_concurrency']),
    'dp_attention': dp_attention, # true or false
    'model': bmk_result['model_id'],
    'framework': framework,
    'precision': precision,
    'tput_per_gpu': float(bmk_result['total_token_throughput']) / tp_size,
    'output_tput_per_gpu': float(bmk_result['output_throughput']) / tp_size
}

if len(sys.argv) == 9:  # MTP
    data['mtp'] = sys.argv[8]

for key, value in bmk_result.items():
    if key.endswith('ms'):
        data[key.replace('_ms', '')] = float(value) / 1000.0
    if 'tpot' in key:
        data[key.replace('_ms', '').replace('tpot', 'intvty')] = 1000.0 / float(value)

print(json.dumps(data, indent=2))

with open(f'agg_{result_filename}.json', 'w') as f:
    json.dump(data, f, indent=2)
