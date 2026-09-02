#!/usr/bin/env bash
set -euo pipefail

result_dir="${1:?usage: run_m7_tier0_collectives.sh <result-dir>}"
overlay_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
aiter_repo="https://github.com/ROCm/aiter.git"
aiter_ref="v0.1.19"
aiter_commit="31350226161346314b3d8882c8085bd31dce6a34"
scratch="$(mktemp -d /tmp/k3-m7-tier0-collectives.XXXXXX)"
trap 'rm -rf "$scratch"' EXIT

mkdir -p "$result_dir"
benchmark_log="$result_dir/k3_m7_tier0_collectives.log"
benchmark_json="$result_dir/k3_m7_tier0_collectives_benchmark.json"
provenance_json="$result_dir/k3_m7_tier0_collectives_provenance.json"
sha256_file="$result_dir/k3_m7_tier0_collectives_SHA256SUMS"

git init --quiet "$scratch/aiter"
git -C "$scratch/aiter" remote add origin "$aiter_repo"
git -C "$scratch/aiter" fetch --quiet --depth 1 origin "$aiter_ref"
git -C "$scratch/aiter" checkout --quiet --detach FETCH_HEAD
actual_commit="$(git -C "$scratch/aiter" rev-parse HEAD)"
if [[ "$actual_commit" != "$aiter_commit" ]]; then
    printf 'ERROR: expected AITER %s, got %s\n' "$aiter_commit" "$actual_commit" >&2
    exit 1
fi
git -C "$scratch/aiter" submodule update --init --depth 1 3rdparty/composable_kernel

export PREBUILD_KERNELS=0
export AITER_USE_SYSTEM_TRITON=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
python3 -m pip uninstall -y aiter amd-aiter >/dev/null 2>&1 || true
python3 -m pip install -e "$scratch/aiter" --no-build-isolation --no-deps

export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_CUSTOM_AR=1
export OMP_NUM_THREADS=1

AITER_CHECKOUT="$scratch/aiter" AITER_COMMIT="$actual_commit" \
python3 - "$provenance_json" "$overlay_dir/bench_m7_tier0_collectives.py" <<'PY'
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import aiter
import torch
import vllm
import vllm._aiter_ops as vllm_aiter_ops
from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

output = Path(sys.argv[1])
benchmark = Path(sys.argv[2]).resolve()
checkout = Path(os.environ["AITER_CHECKOUT"]).resolve()
installed = Path(aiter.__file__).resolve()
vllm_aiter_ops_path = Path(vllm_aiter_ops.__file__).resolve()
expected_vllm_aiter_ops_sha256 = (
    "3ea7b700fe3dba5eb4dfbe533d96651d64d5fe028b9a2dabf76d8360c0c7bf15"
)
if checkout not in installed.parents:
    raise SystemExit(f"expected editable AITER under {checkout}, got {installed}")

properties = torch.cuda.get_device_properties(0)
arch = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
if not torch.version.hip or arch != "gfx950":
    raise SystemExit(f"expected ROCm gfx950, got hip={torch.version.hip!r} arch={arch!r}")
if properties.multi_processor_count != 256:
    raise SystemExit(f"expected 256 CUs, got {properties.multi_processor_count}")
for symbol in ("should_custom_rs", "custom_reduce_scatter", "should_custom_ag", "custom_all_gather"):
    if not hasattr(CustomAllreduce, symbol):
        raise SystemExit(f"installed AITER lacks CustomAllreduce.{symbol}")
if not hasattr(vllm_aiter_ops.rocm_aiter_ops, "get_fused_allreduce_rmsnorm_op"):
    raise SystemExit("installed vLLM lacks the AITER fused all-reduce RMSNorm op")


def distribution_version() -> str:
    for name in ("aiter", "amd-aiter"):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    # Some pinned vLLM ROCm images install AITER directly from source without
    # retaining dist-info metadata. The exact checkout commit, imported source
    # path, source hashes, and /app/versions.txt remain authoritative here.
    return "source-install-without-dist-info"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if sha256(vllm_aiter_ops_path) != expected_vllm_aiter_ops_sha256:
    raise SystemExit(
        "installed vLLM AITER-op source differs from the pinned baseline: "
        f"{vllm_aiter_ops_path}"
    )


tracked = (
    "aiter/dist/device_communicators/custom_all_reduce.py",
    "aiter/ops/custom_all_reduce.py",
    "csrc/include/custom_all_reduce.cuh",
)
payload = {
    "aiter_repo": "https://github.com/ROCm/aiter.git",
    "aiter_commit": os.environ["AITER_COMMIT"],
    "aiter_installed_file": str(installed),
    "aiter_version": distribution_version(),
    "aiter_source_sha256": {
        relative: sha256(checkout / relative) for relative in tracked
    },
    "benchmark_file": str(benchmark),
    "benchmark_sha256": sha256(benchmark),
    "aiter_status_porcelain": subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--short"], text=True
    ).splitlines(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "device": properties.name,
    "arch": arch,
    "cu_num": properties.multi_processor_count,
    "vllm_version": vllm.__version__,
    "vllm_file": str(Path(vllm.__file__).resolve()),
    "vllm_aiter_ops_file": str(vllm_aiter_ops_path),
    "vllm_aiter_ops_sha256": sha256(vllm_aiter_ops_path),
}
versions = Path("/app/versions.txt")
if versions.is_file():
    payload["image_versions"] = versions.read_text(encoding="utf-8").splitlines()
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

python3 -m torch.distributed.run \
    --standalone \
    --nproc-per-node=8 \
    "$overlay_dir/bench_m7_tier0_collectives.py" \
    --output "$benchmark_json" \
    2>&1 | tee "$benchmark_log"

python3 - "$benchmark_json" <<'PY'
import json
import math
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
runtime = payload.get("runtime", {})
if runtime.get("arch") != "gfx950" or int(runtime.get("cu_num", -1)) != 256:
    raise SystemExit("Tier-0 collective benchmark did not run on a 256-CU gfx950")
if int(runtime.get("world_size", -1)) != 8:
    raise SystemExit("Tier-0 collective benchmark did not run with TP=8")
if int(payload.get("rotations", -1)) != 48:
    raise SystemExit("Tier-0 collective benchmark did not rotate 48 weight shards")

routes = payload.get("route_support", {})
required_routes = {
    "routed_custom_all_reduce",
    "shared_custom_all_reduce",
    "shared_custom_reduce_scatter_last_dim",
    "local_custom_all_gather_last_dim",
    "fully_connected",
    "dual_communicator_overlap",
    "fused_allreduce_rmsnorm",
    "dynamic_fused_ar_rms_hidden_dim",
}
if set(routes) != required_routes or not all(routes.values()):
    raise SystemExit(f"required AITER collective route was not active: {routes}")

expected_paths = {
    "tier2_baseline",
    "tier2_fused_ar_rms",
    "tier0_sequential",
    "tier0_overlap",
}
timings = payload.get("timings", {})
if set(timings) != expected_paths:
    raise SystemExit(f"missing Tier-0 timing path: {timings}")
for name, result in timings.items():
    median = float(result.get("median_us", 0))
    if not math.isfinite(median) or median <= 0:
        raise SystemExit(f"invalid timing for {name}: {result}")
    if len(result.get("samples_us", [])) != int(payload.get("samples", -1)):
        raise SystemExit(f"incomplete timing samples for {name}: {result}")

replay = payload.get("changed_input_graph_replay", {})
if set(replay) != expected_paths:
    raise SystemExit(f"missing changed-input replay result: {replay}")
for name, errors in replay.items():
    if not all(math.isfinite(float(value)) for value in errors.values()):
        raise SystemExit(f"invalid correctness metrics for {name}: {errors}")

print(
    "M=7 collective and fused-AR-RMS graph replay completed: "
    + ", ".join(
        f"{name}={timings[name]['median_us']:.3f} us" for name in sorted(timings)
    )
)
PY

(
    cd "$result_dir"
    sha256sum \
        "$(basename "$benchmark_log")" \
        "$(basename "$benchmark_json")" \
        "$(basename "$provenance_json")" \
        >"$(basename "$sha256_file")"
)
