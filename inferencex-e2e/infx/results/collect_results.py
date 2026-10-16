import json
import sys
from pathlib import Path
from typing import Any

# Fields that identify one benchmark point within a run. Retries of a point
# share them and differ only in run_attempt and measured values.
POINT_FIELDS = (
    "hw",
    "framework",
    "model",
    "precision",
    "scenario_type",
    "isl",
    "osl",
    "conc",
    "kv_offloading",
    "recipe_fingerprint",
)


def keep_latest_attempts(results: list[Any]) -> list[Any]:
    """Keep the highest run_attempt of each point; rows without one are kept as-is."""
    latest: dict[str, int] = {}
    for index, result in enumerate(results):
        if not isinstance(result, dict) or not isinstance(result.get("run_attempt"), int):
            continue
        if not result.get("recipe_fingerprint"):
            continue
        key = json.dumps([result.get(field) for field in POINT_FIELDS])
        prior = latest.get(key)
        if prior is None or result["run_attempt"] >= results[prior]["run_attempt"]:
            latest[key] = index
    keep = set(latest.values())
    return [
        result
        for index, result in enumerate(results)
        if index in keep
        or not isinstance(result, dict)
        or not isinstance(result.get("run_attempt"), int)
        or not result.get("recipe_fingerprint")
    ]


def main() -> None:
    results_dir = Path(sys.argv[1])
    exp_name = sys.argv[2]

    agg_results = []
    for result_path in sorted(results_dir.rglob("*.json")):
        with open(result_path) as f:
            result = json.load(f)
        agg_results.append(result)

    kept = keep_latest_attempts(agg_results)
    if len(kept) != len(agg_results):
        print(
            f"Dropped {len(agg_results) - len(kept)} result(s) superseded by a later run attempt",
            file=sys.stderr,
        )

    with open(f"agg_{exp_name}.json", "w") as f:
        json.dump(kept, f, indent=2)


if __name__ == "__main__":
    main()
