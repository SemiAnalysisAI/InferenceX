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
| Hardware | Framework | Precision | TP | Conc | TTFT (ms) | TPOT (ms) | E2EL (s) | TPUT per GPU |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |\
'''
print(summary_header)

for result in results:
    # Extract framework - prefer explicit framework field, fallback to detection
    framework = result.get('framework', 'vLLM')  # default to vLLM if not specified
    
    # If no explicit framework field, try to detect from other fields
    if framework == 'vLLM':
        exp_name = result.get('exp_name', '')
        runner = result.get('runner', '')
        
        # Check for TRT-LLM indicators
        if ('trt' in exp_name.lower() or 'trt' in runner.lower() or 
            'trt-llm' in exp_name.lower() or 'trt-llm' in runner.lower() or
            'tensorrt' in exp_name.lower() or 'tensorrt' in runner.lower()):
            framework = 'TRT-LLM'
    
    # Get precision, default to 'fp8' if not present
    precision = result.get('precision', 'fp8')
    
    # Get metrics with fallbacks for missing fields
    ttft = result.get('ttft', result.get('median_ttft', 0))
    tpot = result.get('tpot', result.get('median_tpot', 0))
    e2el = result.get('e2el', result.get('median_e2el', 0))
    
    print(
        f"| {result['hw'].upper()} "
        f"| {framework} "
        f"| {precision.upper()} "
        f"| {result['tp']} "
        f"| {result['conc']} "
        f"| {(ttft * 1000):.4f} "
        f"| {(tpot * 1000):.4f} "
        f"| {e2el:.4f} "
        f"| {result['tput_per_gpu']:.4f} |"
    )
