import json
import sys
from pathlib import Path


def main() -> None:
    results_dir = Path(sys.argv[1])
    exp_name = sys.argv[2]

    agg_results = []
    for result_path in results_dir.rglob("*.json"):
        with open(result_path) as f:
            result = json.load(f)
        if exp_name == "bmk" and isinstance(result, dict) and "benchmark_outcome" in result:
            outcome = result["benchmark_outcome"]
            if not isinstance(outcome, dict) or outcome.get("status") not in ("passed", "failed"):
                raise ValueError(f"{result_path}: malformed benchmark_outcome")
            if ("attempt_id" in outcome or "finalized" in outcome) and (
                not isinstance(outcome.get("attempt_id"), str)
                or not outcome["attempt_id"].strip()
                or not isinstance(outcome.get("finalized"), bool)
                or (outcome["status"] == "passed" and not outcome["finalized"])
            ):
                raise ValueError(f"{result_path}: malformed AgentX benchmark_outcome")
            if outcome["status"] == "failed":
                print(f"Skipping failed benchmark result: {result_path}", file=sys.stderr)
                continue
        agg_results.append(result)

    with open(f"agg_{exp_name}.json", "w") as f:
        json.dump(agg_results, f, indent=2)


if __name__ == "__main__":
    main()
