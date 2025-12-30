import sys
import json
from pathlib import Path


results_dir = Path(sys.argv[1])
exp_name = sys.argv[2]

agg_results = []
for result_path in results_dir.rglob(f'*.json'):
    with open(result_path) as f:
        result = json.load(f)
    # Skip aggregated result files (lists) and only process individual results (dicts)
    if isinstance(result, dict) and 'is_multinode' in result:
        agg_results.append(result)

with open(f'agg_{exp_name}.json', 'w') as f:
    json.dump(agg_results, f, indent=2)
