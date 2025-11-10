import sys
import json
from pathlib import Path


results = []
results_dir = Path(sys.argv[1])
for result_path in results_dir.rglob(f'*.json'):
    with open(result_path) as f:
        result = json.load(f)
    results.append(result)
results.sort(key=lambda r: (r.get('model', 'unknown'), r['hw'], r.get('framework', 'vllm'), r.get('precision', 'fp8'), r['tp'], r['ep'], r['conc']))

summary_header = f'''\
| Model | Hardware | Framework | Precision | TP | EP | DP Attention | Conc | TTFT (ms) | TPOT (ms) | Interactivity (tok/s/user) | E2EL (s) | TPUT per GPU | Output TPUT per GPU | Input TPUT per GPU |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |\
'''
print(summary_header)

for result in results:
    framework = result.get('framework', 'vllm')
    precision = result.get('precision', 'fp8')
    model = result.get('model', 'unknown')
    print(
        f"| {model} "
        f"| {result['hw'].upper()} "
        f"| {framework.upper()} "
        f"| {precision.upper()} "
        f"| {result['tp']} "
        f"| {result['ep']} "
        f"| {result['dp_attention']} "
        f"| {result['conc']} "
        f"| {(result['median_ttft'] * 1000):.4f} "
        f"| {(result['median_tpot'] * 1000):.4f} "
        f"| {result['median_intvty']:.4f} "
        f"| {result['median_e2el']:.4f} "
        f"| {result['tput_per_gpu']:.4f} "
        f"| {result['output_tput_per_gpu']:.4f} "
        f"| {result['input_tput_per_gpu']:.4f} |"
    )
