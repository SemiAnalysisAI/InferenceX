"""Delete only explicitly registered, identity-verified stale study scratch."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from runtime import cache_usage, write_json

SCRATCH_ROOT = Path("/offload-scratch")
TARGETS: dict[str, dict[str, Any]] = {
    "35476050409": {
        "node": "im-b200-c002",
        "name": "inferencex-offload-35476050409-2-16690-ccf4bb49cf4d",
        "owner": {
            "study": "agentx-offload-v1",
            "run": "35476050409",
            "attempt": "2",
            "job": "16690",
        },
    },
    "35476043213": {
        "node": "im-b200-c008",
        "name": "inferencex-offload-35476043213-3-16692-bbaface6dcdc",
        "owner": {
            "study": "agentx-offload-v1",
            "run": "35476043213",
            "attempt": "3",
            "job": "16692",
        },
    },
}


def cleanup_owned(root: Path, target: dict[str, Any], receipt_path: Path) -> None:
    scratch = root / target["name"]
    if scratch.parent != root or scratch.is_symlink():
        raise RuntimeError("Refusing unsafe stale-scratch path")
    if not scratch.exists():
        write_json(
            receipt_path,
            {
                "node": target["node"],
                "scratch": str(scratch),
                "owner": target["owner"],
                "usage": None,
                "already_absent": True,
                "deleted": True,
            },
        )
        return
    owner_path = scratch / "owner.json"
    if not owner_path.is_file() or json.loads(owner_path.read_text()) != target["owner"]:
        raise RuntimeError("Stale-scratch identity mismatch")
    usage = cache_usage(scratch / "cache")
    receipt = {
        "node": target["node"],
        "scratch": str(scratch),
        "owner": target["owner"],
        "usage": usage,
        "deleted": False,
    }
    write_json(receipt_path, receipt)
    shutil.rmtree(scratch)
    receipt["deleted"] = not scratch.exists()
    write_json(receipt_path, receipt)
    if not receipt["deleted"]:
        raise RuntimeError("Stale scratch still exists after cleanup")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", choices=sorted(TARGETS))
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    cleanup_owned(SCRATCH_ROOT, TARGETS[args.run_id], args.receipt)


if __name__ == "__main__":
    main()
