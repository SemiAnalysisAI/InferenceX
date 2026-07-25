#!/usr/bin/env python3
"""Remove machine-local artifact paths and raw samples from TPU GEMM results."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def compact_results(results: dict[str, Any]) -> dict[str, Any]:
    compact = copy.deepcopy(results)
    run = compact.get("run", {})
    run.pop("hostname", None)
    run.pop("argv", None)
    for row in compact["rows"]:
        placement = row.get("placement", {})
        placement.pop("input_shardings", None)
        placement.pop("output_shardings", None)

        hlo = row.get("hlo")
        if hlo:
            hlo.pop("stablehlo_path", None)
            hlo.pop("optimized_hlo_path", None)

        device_execution = row.get("device_execution_us")
        if device_execution:
            device_execution.pop("raw_us", None)
            device_execution.pop("trace_path", None)
    compact["dataset"] = {
        "format": "operatorx_tpu_gemm_validation_summary",
        "machine_identity_removed": True,
        "raw_artifact_paths_removed": True,
        "raw_xprof_samples_removed": True,
    }
    return compact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    compact = compact_results(json.loads(args.input.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(compact, indent=2))
    print(f"wrote {args.output} ({len(compact['rows'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
