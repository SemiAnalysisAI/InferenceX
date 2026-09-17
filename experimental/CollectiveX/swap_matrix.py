"""Build one isolated swap-blocks execution cell per selected registered GPU pool."""

import argparse
import json
from pathlib import Path


def build_matrix(platforms: dict, only_sku: str, exclude_skus: str) -> dict:
    excluded = set(filter(None, exclude_skus.split(",")))
    unknown = ({only_sku} if only_sku else set()) | excluded
    unknown -= platforms.keys()
    if unknown:
        raise ValueError(f"Unknown GPU pools: {sorted(unknown)}")
    if only_sku in excluded:
        raise ValueError("only-sku and exclude-skus must be disjoint")
    cells = []
    for sku, platform in platforms.items():
        if not only_sku and sku.endswith("-tw"):
            continue
        if (only_sku and sku != only_sku) or sku in excluded:
            continue
        cells.append(
            {
                "id": f"swap-{sku}",
                "sku": sku,
                "backend": "swap-blocks",
                "nodes": 1,
                "gpus_per_node": 1,
                "scale_up_domain": 1,
                "launcher": "swap-blocks",
                "vendor": "amd" if platform["arch"].startswith("gfx") else "nvidia",
            }
        )
    return {"include": cells}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-sku", default="")
    parser.add_argument("--exclude-skus", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    platforms = json.loads(
        (Path(__file__).parent / "configs/platform_config.json").read_text()
    )["platforms"]
    Path(args.out).write_text(
        json.dumps(build_matrix(platforms, args.only_sku, args.exclude_skus)) + "\n"
    )


if __name__ == "__main__":
    main()
