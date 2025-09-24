import sys
import json
from pathlib import Path

n_iosl = 3   # [1k1k, 8k1k, 1k8k]
n_concs = 5  # [4, 8, 16, 32, 64]

# H200: (70b-tp: [1, 2, 4, 8], dsr1-tp: [8], gptoss-tp: [1, 2, 4, 8]) x [vllm/sglang, trt] + 70b-tp x trt extra conc: [128]
h200_runs = (4 + 1 + 4) * 2 * n_iosl * n_concs + 4 * n_iosl * 1

# B200:
# 70b = [tp1, tp8] x [fp4, fp8] x n_concs
# 70b-trt = [tp1, tp8] x [fp4, fp8] x conc:[4, 8, 16, 32, 64, 128, 256]
# dsr1 = [tp8] x (fp8: n_concs + fp4: [4, 8, 16, 32, 64, 128, 256])
# dsr1-trt = fp8: ([tp8] x n_concs) + fp4: ([tp4, tp8] x conc:[4, 8, 16, 32, 64, 128, 256])
b200_runs = (2 * 2 * n_concs + 2 * 2 * 7 + 1 * (n_concs + 7) + (1 * n_concs + 2 * 7)) * n_iosl

total_runs = {
    'h100': (3 + 4) * n_iosl * n_concs,              # 70b-tp: [2, 4, 8], gptoss-tp: [1, 2, 4, 8]
    'h200': h200_runs,
    'b200': b200_runs,
    'mi300x': (4 + 1 + 4) * n_iosl * n_concs,        # 70b-tp: [1, 2, 4, 8], dsr1-tp: [8], gptoss-tp: [1, 2, 4, 8]
    'mi325x': (4 + 1 + 4) * n_iosl * n_concs,        # 70b-tp: [1, 2, 4, 8], dsr1-tp: [8], gptoss-tp: [1, 2, 4, 8]
    'mi355x': ((4 + 1) * 2 + 2) * n_iosl * n_concs,  # (70b-tp: [1, 2, 4, 8], dsr1-tp: [8]) x [fp4, fp8], gptoss-tp: [1, 2, 4, 8]
    'gb200': 65,                                     # 45 runs (trtllm) + 20 runs (sglang)
}
success_runs = {'h100': 0, 'h200': 0, 'b200': 0, 'mi300x': 0, 'mi325x': 0, 'mi355x': 0, 'gb200': 0}


for results_filepath in Path(sys.argv[1]).rglob('*.json'):
    with open(results_filepath) as f:
        results = json.load(f)

    for result in results:
        hw_type = result['hw'].replace('-trt', '')
        success_runs[hw_type] += 1

run_stats = {}
for hw, n_success in success_runs.items():
    run_stats[hw] = {'n_success': n_success, 'total': total_runs[hw]}
with open(f'{sys.argv[2]}.json', 'w') as f:
    json.dump(run_stats, f, indent=2)
