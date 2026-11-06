#!/usr/bin/env bash
set -eo pipefail

source /infmax-workspace/benchmarks/check_env.sh
check_env_vars IMX_SGLANG_COMMIT IMX_MEGA_COMMIT

python3 - <<'PY_SETUP'
import fcntl
import json
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

runtime = Path("/opt/inferencex-qwen35-runtime")
with (runtime / "plugin-install.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    revisions = {}
    for key, path in (("IMX_SGLANG_COMMIT", "/opt/sglang-main"), ("IMX_MEGA_COMMIT", "/opt/sgl-nvidia-mega")):
        expected = os.environ[key]
        actual = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
        if actual != expected:
            raise RuntimeError(f"{path}: expected {expected}, got {actual}")
        revisions[key] = actual
    image_source = Path(importlib.util.find_spec("sglang").origin).resolve().parents[2]
    image_revision = subprocess.check_output(["git", "-C", str(image_source), "rev-parse", "HEAD"], text=True).strip()
    if image_revision != revisions["IMX_SGLANG_COMMIT"]:
        raise RuntimeError(f"Image SGLang revision does not match the source pin: {image_revision}")
    receipt = runtime / "plugin-installed.json"
    if not receipt.exists():
        build_source = runtime / "plugin-build-source"
        shutil.copytree("/opt/sgl-nvidia-mega", build_source)
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--target", str(runtime / "python"), str(build_source)], check=True)
        receipt.write_text(json.dumps(revisions, sort_keys=True) + "\n")
    if json.loads(receipt.read_text()) != revisions:
        raise RuntimeError("Plugin installation does not match the recipe's pinned sources")
    print(json.dumps(dict(sources=revisions, tree_core="rust", source_mounts="read-only"), sort_keys=True), flush=True)
PY_SETUP
