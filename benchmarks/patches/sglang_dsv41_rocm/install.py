"""Install the digest-pinned, V4.1-only ROCm attention source backport."""

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

ANCHOR = """    elif _is_hip:
        from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
"""
REPLACEMENT = """    elif _is_hip:
        if runner.model_config.hf_text_config.model_type == "deepseek_v41":
            from sglang.srt.layers.attention.dsv41_rocm.backend import (
                DeepseekV4HipRadixBackend as DeepseekV41BackportBackend,
            )

            logger.info("Using digest-pinned V4.1 ROCm source backport on latest nightly.")
            return DeepseekV41BackportBackend(runner)
        from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
"""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def install(package: Path, evidence: Path) -> None:
    source = Path(__file__).resolve().parent
    manifest = json.loads((source / "provenance.json").read_text())
    registry = package / "srt/layers/attention/attention_registry.py"
    original = registry.read_text()
    if REPLACEMENT in original:
        original = original.replace(REPLACEMENT, ANCHOR, 1)
    if sha256(original.encode()) != manifest["registry_sha256"]:
        raise RuntimeError("Unexpected SGLang attention registry; refusing to patch another revision")
    if original.count(ANCHOR) != 1:
        raise RuntimeError("Expected exactly one HIP dsv4 registry anchor")
    for item in manifest["files"]:
        data = (source / "dsv41_rocm" / item["installed_name"]).read_bytes()
        if sha256(data) != item["adapted_sha256"]:
            raise RuntimeError(f"Backport source hash mismatch: {item['installed_name']}")
    destination = registry.parent / "dsv41_rocm"
    destination.mkdir(exist_ok=True)
    for path in (source / "dsv41_rocm").glob("*.py"):
        shutil.copy2(path, destination / path.name)
    patched = original.replace(ANCHOR, REPLACEMENT, 1)
    registry.write_text(patched)
    manifest["installed_registry_sha256"] = sha256(patched.encode())
    evidence.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"V4.1 ROCm backport installed; exact source evidence: {evidence}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    args = parser.parse_args()
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Cannot locate installed SGLang package")
    install(Path(next(iter(spec.submodule_search_locations))), args.evidence)
