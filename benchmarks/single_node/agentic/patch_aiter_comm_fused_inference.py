"""Guarded hotfix for the pinned AITER TP MoE inference-buffer writes."""

import argparse
import hashlib
import json
from importlib.util import find_spec
from pathlib import Path


SOURCE_SHA256 = "131a3e0be4c39b2d47474c8a06572e83adce1806fc0d4201ddeaef13f6019f49"
BEFORE = b"        def stage2_override(**kwargs: Any):\n            def launch():\n"
AFTER = (
    b"        def stage2_override(**kwargs: Any):\n"
    b"            # Compiled custom ops may enter outside InferenceMode. The\n"
    b"            # runner owns inference tensors, including its padding buffer.\n"
    b"            @torch.inference_mode()\n"
    b"            def launch():\n"
)


def apply_fix(path: Path) -> dict:
    original = path.read_bytes()
    before_hash = hashlib.sha256(original).hexdigest()
    if before_hash == SOURCE_SHA256 and original.count(BEFORE) == 1:
        patched = original.replace(BEFORE, AFTER, 1)
        path.write_bytes(patched)
        status = "applied"
    elif original.count(AFTER) == 1 and hashlib.sha256(
        original.replace(AFTER, BEFORE, 1)
    ).hexdigest() == SOURCE_SHA256:
        patched = original
        status = "already_applied"
    else:
        raise ValueError(f"Refusing to patch unexpected AITER runtime: {path} ({before_hash})")
    return {
        "name": "aiter_comm_fused_stage2_inference_mode",
        "path": str(path), "status": status,
        "upstream_sha256": SOURCE_SHA256,
        "patched_sha256": hashlib.sha256(patched).hexdigest(),
        "scope": "TP-only Stage2 callback; preserve communication fusion and CUDA graphs",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = find_spec("aiter")
    if spec is None or spec.origin is None:
        raise RuntimeError("AITER package source was not found")
    path = Path(spec.origin).parent / "ops/comm_fused_moe_runtime.py"
    record = apply_fix(path)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"AITER TP inference-mode fix: {record['status']} ({record['patched_sha256']})")


if __name__ == "__main__":
    main()
