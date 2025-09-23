import sys
import json
from pathlib import Path


hw = sys.argv[1]
tp_size = int(sys.argv[2])
result_filename = sys.argv[3]
framework = sys.argv[4]
precision = sys.argv[5]

with open(f'{result_filename}.json') as f:
    bmk_result = json.load(f)

data = {
    'hw': hw,
    'tp': tp_size,
    'conc': int(bmk_result['max_concurrency']),
    'model': bmk_result['model_id'],
    'framework': framework,
    'precision': precision,
    'tput_per_gpu': float(bmk_result['total_token_throughput']) / tp_size
}

if len(sys.argv) == 7:  # MTP
    data['mtp'] = sys.argv[6]

for key, value in bmk_result.items():
    if key.endswith('ms'):
        data[key.replace('_ms', '')] = float(value) / 1000.0
    if 'tpot' in key:
        data[key.replace('_ms', '').replace('tpot', 'intvty')] = 1000.0 / float(value)

print(json.dumps(data, indent=2))

with open(f'agg_{result_filename}.json', 'w') as f:
    json.dump(data, f, indent=2)
