import sys
import json
from pathlib import Path


results = []
results_dir = Path(sys.argv[1])
for result_path in results_dir.rglob(f'*.json'):
    with open(result_path) as f:
        result = json.load(f)
    results.append(result)
results.sort(key=lambda r: (r['hw'], r['tp'], r['conc']))

summary_header = f'''\
| Hardware | Framework | TP | Conc | TTFT (ms) | TPOT (ms) | E2EL (s) | TPUT per GPU |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |\
'''
print(summary_header)

for result in results:
    # Extract framework from experiment name or runner
    framework = 'vLLM'  # default
    if 'trt' in result.get('exp_name', '').lower() or 'trt' in result.get('runner', '').lower():
        framework = 'TRT-LLM'
    
    print(
        f"| {result['hw'].upper()} "
        f"| {framework} "
        f"| {result['tp']} "
        f"| {result['conc']} "
        f"| {(result['median_ttft'] * 1000):.4f} "
        f"| {(result['median_tpot'] * 1000):.4f} "
        f"| {result['median_e2el']:.4f} "
        f"| {result['tput_per_gpu']:.4f} |"
    )
