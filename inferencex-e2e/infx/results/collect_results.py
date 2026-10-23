import json
import sys
from pathlib import Path


def main() -> None:
    """Concatenate every JSON result under the results dir in argv into agg_<exp_name>.json."""
    results_dir = Path(sys.argv[1])
    exp_name = sys.argv[2]

    agg_results = []
    for result_path in sorted(results_dir.rglob("*.json")):
        with open(result_path) as f:
            result = json.load(f)
        agg_results.append(result)

    with open(f"agg_{exp_name}.json", "w") as f:
        json.dump(agg_results, f, indent=2)


if __name__ == "__main__":
    main()
