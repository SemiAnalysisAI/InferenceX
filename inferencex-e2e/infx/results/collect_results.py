import json
import sys
from pathlib import Path

from infx.results.schema.models import benchmark_row
from infx.results.schema.quarantine import quarantine, report


def main() -> int:
    results_dir = Path(sys.argv[1])
    exp_name = sys.argv[2]

    candidates = []
    for result_path in results_dir.rglob("*.json"):
        with open(result_path) as f:
            result = json.load(f)
        source = str(result_path.relative_to(results_dir))
        candidates.append((source, result, benchmark_row(result)))
    accepted, rejected = quarantine(candidates)

    with open(f"agg_{exp_name}.json", "w") as f:
        json.dump([result for _, result in accepted], f, indent=2)
    return report(rejected)


if __name__ == "__main__":
    sys.exit(main())
