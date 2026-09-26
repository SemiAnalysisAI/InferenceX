#!/usr/bin/env python3
"""Install the reviewed pair into the task-owned native C48 runtime only.

Run with /opt/sglang/bin/python3 after the build receipt has been verified.
No package resolution, GPU allocation, environment activation, or launch occurs.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

BASE_CORE = "297e426733dc73c881ba2d6aa79d56fb5b4c8ac27b0c25e1b47bb411d45ec747"
VERSION = "1.5.0.dev20260909"
FEATURES = "kv-indexer,slot-tracker,select-service,mm-routing,aic-forward-pass,request-trace-s3"
DYNAMO = Path("/opt/sglang/lib/python3.12/site-packages/dynamo")
SGLANG = Path("/sgl-workspace/sglang/python/sglang")


def sha(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--wheels", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    require(platform.system() == "Linux" and platform.machine() == "x86_64", "Expected native Linux x86_64")
    require(sys.version_info[:3] == (3, 12, 3), "Expected original Python 3.12.3")
    require(Path(sys.prefix) == Path("/opt/sglang"), "Use the workload Python, not the build venv")
    receipt_path = args.wheels / "build-output-receipt.json"
    require(sha(receipt_path) == args.receipt_sha256, "Build receipt identity mismatch")
    receipt = json.loads(receipt_path.read_text())
    records = json.loads((args.inputs / "integrated-candidate-file-hashes.json").read_text())
    require(receipt["inputs"]["reviewed_files"] == records and len(records) == 15, "Reviewed source identity mismatch")
    require(receipt["inputs"]["source"] == "c3e05f0244ae6264d7953f68e2499c6dc2f54723", "Unexpected Dynamo source")
    require(receipt["features"] == FEATURES and receipt["manylinux_policy"] == "manylinux_2_39", "Unexpected build policy")
    for package in ["ai-dynamo", "ai-dynamo-runtime"]:
        require(importlib.metadata.version(package) == VERSION, "Unexpected installed " + package)
    require(importlib.metadata.version("sglang") == "0.0.0.dev1+g008403017", "Unexpected installed SGLang")

    wheels = []
    runtime_core = None
    for item in receipt["wheels"]:
        require(Path(item["wheel"]).name == item["wheel"], "Wheel must be a basename")
        wheel = args.wheels / item["wheel"]
        require(sha(wheel) == item["sha256"], "Wheel hash mismatch: " + wheel.name)
        with zipfile.ZipFile(wheel) as archive:
            if wheel.name.startswith("ai_dynamo_runtime-"):
                require("cp310-abi3-manylinux_2_39_x86_64" in wheel.name, "Wrong native ABI")
                runtime_core = hashlib.sha256(archive.read("dynamo/_core.abi3.so")).hexdigest()
        wheels.append(wheel)
    require(len(wheels) == 2 and runtime_core is not None, "Expected one Python and one native wheel")
    targets = []
    for item in records:
        if not item["path"].endswith(".py"):
            continue
        if item["tree"] == "Dynamo":
            target = DYNAMO / Path(item["path"]).relative_to("components/src/dynamo")
            source = args.inputs / "dynamo-c3e05f-full" / item["path"]
        else:
            target = SGLANG / Path(item["path"]).relative_to("python/sglang")
            source = args.inputs / "sglang-008403017-candidate" / item["path"]
        require(sha(source) == item["candidate_sha256"], "Companion payload mismatch: " + item["path"])
        targets.append((item, source, target))
    require(len(targets) == 10, "Expected all ten Python source/absence checks")
    current = [sha(target) if target.exists() else None for _, _, target in targets]
    core = DYNAMO / "_core.abi3.so"
    already_installed = sha(core) == runtime_core and all(value == item["candidate_sha256"] for value, (item, _, _) in zip(current, targets))
    if not already_installed:
        require(sha(core) == BASE_CORE, "Original native core mismatch; refuse partial/foreign runtime")
        for value, (item, _, _) in zip(current, targets):
            require(value == item["base_sha256"], "Original source mismatch: " + item["path"])
    if args.check_only:
        print(json.dumps({"preflight": "passed", "already_installed": already_installed, "mutated": False}))
        return
    if not already_installed:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--force-reinstall", *map(str, wheels)], check=True)
        for item, source, target in targets:
            if item["tree"] != "SGLang":
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            staged = target.with_suffix(target.suffix + ".containment-tmp")
            staged.write_bytes(source.read_bytes())
            staged.chmod(target.stat().st_mode & 0o777 if target.exists() else 0o644)
            staged.replace(target)
    require(sha(core) == runtime_core, "Installed native core mismatch")
    installed = []
    for item, _, target in targets:
        require(sha(target) == item["candidate_sha256"], "Installed source mismatch: " + item["path"])
        installed.append({"source": item["path"], "path": str(target), "sha256": sha(target)})
    for package in ["ai-dynamo", "ai-dynamo-runtime"]:
        require(importlib.metadata.version(package) == VERSION, "Installed version changed")
    args.output.write_text(json.dumps({"status": "installed_hash_verified", "already_installed": already_installed, "build_receipt_sha256": args.receipt_sha256, "runtime_core_sha256": runtime_core, "sources": installed, "gpu_validation": False, "activation": "The launcher must set the containment flag on frontend and both worker roles, and deferred release on both workers."}, indent=2) + "\n")
    print(json.dumps({"installed": True, "sources_verified": len(installed), "gpu_validation": False}))


if __name__ == "__main__":
    main()
