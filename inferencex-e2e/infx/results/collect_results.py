import json
import sys
from pathlib import Path

from infx.results.request_records import dumps


def main() -> None:
    """Concatenate every JSON result under the results dir in argv into agg_<exp_name>.json."""
    results_dir = Path(sys.argv[1])
    exp_name = sys.argv[2]

    agg_results = []
    for result_path in sorted(results_dir.rglob("*.json")):
        with open(result_path) as f:
            result = json.load(f)
        agg_results.append(result)

    if not agg_results:
        print("No benchmark JSON artifacts to aggregate.")
        return

    Path(f"agg_{exp_name}.json").write_text(dumps(agg_results))


if __name__ == "__main__":
    main()
