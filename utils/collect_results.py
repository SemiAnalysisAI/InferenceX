import sys
import json
from pathlib import Path


results_dir = Path(sys.argv[1])
exp_name = sys.argv[2]

agg_results = []
for result_path in results_dir.rglob(f'*.json'):
    if result_path.stat().st_size == 0:
        print(f"Skipping empty JSON artifact: {result_path}", file=sys.stderr)
        continue
    try:
        with open(result_path) as f:
            result = json.load(f)
    except json.JSONDecodeError as exc:
        print(
            f"Skipping invalid JSON artifact: {result_path}: {exc}",
            file=sys.stderr,
        )
        continue
    agg_results.append(result)

if not agg_results:
    raise SystemExit(f"No valid JSON results found under {results_dir}")

with open(f'agg_{exp_name}.json', 'w') as f:
    json.dump(agg_results, f, indent=2)
