from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

KERNEL = "QuantBatchMatmulV3_ND_ND_fp8e4m3_fp8e4m3_fp32_bf16_high_performance"


def validate_observation(
    row: dict[str, Any], case: dict[str, Any], worker_sha: str
) -> str:
    if any(row.get(k) != case.get(k) for k in ("case_id", "opType", "args")):
        raise ValueError("Guard case identity mismatch")
    provenance = row.get("provenance", {})
    error = row.get("launch_exception", "")
    guard = row.get("guard", {})
    if (
        row.get("status") != "inspected"
        or "100000" not in error
        or any(code in error for code in ("507015", "507035"))
        or provenance.get("worker_sha256") != worker_sha
        or provenance.get("task_queue_enable") != "0"
        or guard.get("observed") is not True
        or guard.get("mode") != 1
        or guard.get("forward_allowed") is not False
    ):
        raise ValueError("Guard did not establish an isolated prelaunch rejection")
    if guard.get("decision") == "unverified":
        return "skipped_unverified_tiling"
    if guard.get("dispatch_key") != 5124 or guard.get("kernel_name") not in {
        KERNEL,
        KERNEL + "_5124",
        KERNEL + "_5124_mix_aic",
        KERNEL + "_5124_mix_aiv",
    }:
        raise ValueError("Unknown FP8 kernel ABI/key")
    for dimension in ("m", "n", "k"):
        if any(
            guard.get(prefix + dimension) != case["args"][dimension]
            for prefix in ("", "expected_")
        ):
            raise ValueError(
                "Observed tiling dimensions differ from the requested case"
            )
    for name, alignment in (("base_m", 16), ("base_n", 16), ("base_k", 32)):
        value = guard.get(name)
        if type(value) is not int or value <= 0 or value % alignment:
            raise ValueError("Invalid local tile")
    required = 2 * guard["base_m"] * guard["base_k"]
    if guard.get("l0a_bytes") != required or guard.get("l0a_capacity_bytes") != 65536:
        raise ValueError("Invalid L0A capacity calculation")
    expected = "within_capacity" if required <= 65536 else "over_capacity"
    if guard.get("decision") != expected:
        raise ValueError("Guard decision contradicts its local tile")
    return "new" if required <= 65536 else "skipped_l0a_capacity"


def apply_guard(plan: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    if plan.get("kind") != "fp8":
        raise ValueError("Only FP8 plans require this guard")
    selected = {c["case_id"]: c for c in plan["cases"] if c["phase"] == "new"}
    observations = {}
    identities = set()
    for row in rows:
        cid = row.get("case_id")
        if cid not in selected or cid in observations:
            raise ValueError("Unexpected or duplicate guard case")
        phase = validate_observation(row, selected[cid], plan["worker_sha256"])
        identity = row.get("hardware_identity")
        if not isinstance(identity, str) or len(identity) != 64:
            raise ValueError("Missing guard hardware identity")
        identities.add(identity)
        observations[cid] = phase
    if set(observations) != set(selected) or len(identities) != 1:
        raise ValueError("Incomplete guard coverage or mixed hardware identities")
    result = json.loads(json.dumps(plan))
    for case in result["cases"]:
        if case["case_id"] in observations:
            case["phase"] = observations[case["case_id"]]
    valid = {cid for cid, phase in observations.items() if phase == "new"}
    result["smoke_case_ids"] = [cid for cid in plan["smoke_case_ids"] if cid in valid]
    if valid and not result["smoke_case_ids"]:
        result["smoke_case_ids"] = [sorted(valid)[0]]
    result["counts"] = dict(Counter(c["phase"] for c in result["cases"]))
    result["fp8_l0a"] = {"hardware_identity": identities.pop(), "observations": rows}
    return result


def verify_plan(plan: dict[str, Any]) -> None:
    proof = plan.get("fp8_l0a")
    if not isinstance(proof, dict):
        raise ValueError(
            "FP8 requires an approved prelaunch L0A plan; run the guard first"
        )
    original = json.loads(json.dumps(plan))
    for case in original["cases"]:
        if case["phase"] in ("skipped_l0a_capacity", "skipped_unverified_tiling"):
            case["phase"] = "new"
    checked = apply_guard(original, proof["observations"])
    if (
        checked["cases"] != plan["cases"]
        or checked["fp8_l0a"]["hardware_identity"] != proof["hardware_identity"]
    ):
        raise ValueError("FP8 plan differs from the validated tiling observations")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = apply_guard(
        json.loads(args.plan.read_text()),
        [
            json.loads(line)
            for line in args.observations.read_text().splitlines()
            if line.strip()
        ],
    )
    plan["fp8_l0a"]["source_sha256"] = hashlib.sha256(
        args.observations.read_bytes()
    ).hexdigest()
    with args.output.open("x") as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(plan["counts"]))


if __name__ == "__main__":
    main()
