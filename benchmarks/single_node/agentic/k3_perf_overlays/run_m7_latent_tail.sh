#!/usr/bin/env bash
set -euo pipefail

result_dir="${1:?usage: run_m7_latent_tail.sh <result-dir>}"
overlay_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
aiter_repo="https://github.com/andyluo7/aiter.git"
aiter_ref="codex/k3-m7-latent-tail-20260901"
aiter_commit="d7755c2f75f70ba0b3e81e6720b9529244cffb3f"
scratch="$(mktemp -d /tmp/k3-m7-latent-tail.XXXXXX)"
trap 'rm -rf "$scratch"' EXIT

mkdir -p "$result_dir"
test_log="$result_dir/k3_m7_latent_tail_tests.log"
benchmark_json="$result_dir/k3_m7_latent_tail_benchmark.json"
provenance_json="$result_dir/k3_m7_latent_tail_provenance.json"
sha256_file="$result_dir/k3_m7_latent_tail_SHA256SUMS"

git init --quiet "$scratch/aiter"
git -C "$scratch/aiter" remote add origin "$aiter_repo"
git -C "$scratch/aiter" fetch --quiet --depth 1 origin "$aiter_ref"
git -C "$scratch/aiter" checkout --quiet --detach FETCH_HEAD
actual_commit="$(git -C "$scratch/aiter" rev-parse HEAD)"
if [[ "$actual_commit" != "$aiter_commit" ]]; then
    echo "Error: expected AITER $aiter_commit, got $actual_commit" >&2
    exit 1
fi
git -C "$scratch/aiter" submodule update --init --depth 1 3rdparty/composable_kernel

export PREBUILD_KERNELS=0
export AITER_USE_SYSTEM_TRITON=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
python3 -m pip uninstall -y aiter amd-aiter >/dev/null 2>&1 || true
python3 -m pip install -e "$scratch/aiter" --no-build-isolation --no-deps

# The Kimi launcher normally sets this below the diagnostic early exit. Set it
# here before importing vLLM so rocm_unquantized_gemm_impl exercises AITER's
# pinned Kimi-K3 tuned table instead of silently falling back to torch linear.
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_LINEAR=1
export AITER_CONFIG_GEMM_BF16="$scratch/aiter/aiter/configs/model_configs/kimik3_bf16_tuned_gemm.csv"
export AITER_LOG_TUNED_CONFIG=1

AITER_CHECKOUT="$scratch/aiter" AITER_COMMIT="$actual_commit" \
python3 - "$provenance_json" <<'PY'
import csv
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
from aiter.tuned_gemm import get_GEMM_A16W16_config
from vllm._aiter_ops import rocm_aiter_ops

output = Path(sys.argv[1])
checkout = Path(os.environ["AITER_CHECKOUT"]).resolve()
installed = Path(aiter.__file__).resolve()
if checkout not in installed.parents:
    raise SystemExit(f"expected editable AITER under {checkout}, got {installed}")

properties = torch.cuda.get_device_properties(0)
arch = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
if not torch.version.hip or arch != "gfx950":
    raise SystemExit(f"expected ROCm gfx950, got hip={torch.version.hip!r} arch={arch!r}")
if not rocm_aiter_ops.is_tgemm_enabled():
    raise SystemExit("expected the vLLM AITER tuned-GEMM route to be enabled")

tracked = (
    "aiter/configs/model_configs/kimik3_bf16_tuned_gemm.csv",
    "aiter/ops/flydsl/kernels/latent_moe_tail_gfx950.py",
    "aiter/ops/flydsl/latent_moe_tail.py",
    "op_tests/flydsl_tests/test_latent_moe_tail.py",
)
tuned_config = Path(os.environ["AITER_CONFIG_GEMM_BF16"]).resolve()
if checkout not in tuned_config.parents:
    raise SystemExit(f"expected tuned config under {checkout}, got {tuned_config}")

with tuned_config.open(encoding="utf-8", newline="") as handle:
    projection_rows = [
        row
        for row in csv.DictReader(handle)
        if row["gfx"] == "gfx950"
        and int(row["cu_num"]) == properties.multi_processor_count
        and int(row["M"]) in {1, 2, 8, 16}
        and int(row["N"]) == 7168
        and int(row["K"]) == 3584
        and row["bias"] == "False"
        and row["dtype"] == "torch.bfloat16"
        and row["outdtype"] == "torch.bfloat16"
    ]
if {int(row["M"]) for row in projection_rows} != {1, 2, 8, 16}:
    raise SystemExit(
        "missing the expected M=1,2,8,16 Kimi-K3 tuned projection rows"
    )

selected_projection_routes = []
for requested_m in (1, 2, 7, 14):
    config = get_GEMM_A16W16_config(
        M=requested_m,
        N=7168,
        K=3584,
        bias=False,
        dtype=str(torch.bfloat16),
        otype=str(torch.bfloat16),
    )
    if config["libtype"] == "torch":
        raise SystemExit(
            f"M={requested_m} selected the untuned torch GEMM fallback: {config}"
        )
    selected_projection_routes.append(
        {
            "requested_M": requested_m,
            "libtype": config["libtype"],
            "kernelName": config.get("kernelName"),
            "solidx": int(config["solidx"]),
            "splitK": None
            if config.get("splitK") is None
            else int(config["splitK"]),
        }
    )


def distribution_version() -> str:
    for name in ("aiter", "amd-aiter"):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    raise SystemExit("editable AITER install has no importlib distribution metadata")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


payload = {
    "aiter_repo": "https://github.com/andyluo7/aiter.git",
    "aiter_commit": os.environ["AITER_COMMIT"],
    "aiter_installed_file": str(installed),
    "aiter_version": distribution_version(),
    "aiter_source_sha256": {
        relative: sha256(checkout / relative) for relative in tracked
    },
    "aiter_tuned_gemm_enabled": True,
    "aiter_tuned_gemm_config": str(tuned_config),
    "aiter_tuned_projection_rows": [
        {
            "M": int(row["M"]),
            "libtype": row["libtype"],
            "kernelName": row["kernelName"],
            "us": float(row["us"]),
        }
        for row in sorted(projection_rows, key=lambda value: int(value["M"]))
    ],
    "selected_projection_routes": selected_projection_routes,
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
}
versions = Path("/app/versions.txt")
if versions.is_file():
    payload["image_versions"] = versions.read_text(encoding="utf-8").splitlines()
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
python3 -m pytest -q \
    "$scratch/aiter/op_tests/flydsl_tests/test_latent_moe_tail.py" \
    2>&1 | tee "$test_log"

HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
python3 "$overlay_dir/bench_latent_moe_tail_small_m.py" \
    --output "$benchmark_json"

python3 - "$benchmark_json" <<'PY'
import json
import math
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("runtime", {}).get("arch") != "gfx950":
    raise SystemExit("latent-tail benchmark did not run on gfx950")
if not payload.get("all_changed_input_graph_replays_passed"):
    raise SystemExit("changed-input graph replay validation failed")
results = payload.get("results", [])
if [row.get("num_tokens") for row in results] != [1, 2, 7, 14]:
    raise SystemExit("latent-tail benchmark did not cover M=1,2,7,14")
for row in results:
    timings = row.get("p50_us", {})
    if set(timings) != {"torch_mm", "control", "candidate"}:
        raise SystemExit(
            f"missing three-way projection timing for M={row.get('num_tokens')}: {row}"
        )
    if not all(
        math.isfinite(float(value)) and float(value) > 0
        for value in timings.values()
    ):
        raise SystemExit(f"invalid timing for M={row.get('num_tokens')}: {row}")
    replay_paths = row.get("changed_input_graph_replay", {})
    if set(replay_paths) != {"torch_mm", "control", "candidate"}:
        raise SystemExit(
            f"missing three-way changed-input replay for M={row.get('num_tokens')}: {row}"
        )
    candidate_speedup = float(row.get("speedup", 0))
    if not math.isfinite(candidate_speedup) or candidate_speedup <= 0:
        raise SystemExit(f"invalid speedup for M={row.get('num_tokens')}: {row}")
    torch_mm_speedup = float(row.get("torch_mm_over_control_speedup", 0))
    if not math.isfinite(torch_mm_speedup) or torch_mm_speedup <= 0:
        raise SystemExit(
            f"invalid torch_mm/control speedup for M={row.get('num_tokens')}: {row}"
        )
print(
    "Three-way latent-tail graph replay completed: "
    + ", ".join(
        (
            f"M={row['num_tokens']} "
            f"candidate/control={row['speedup']:.4f}x "
            f"torch_mm/control={row['torch_mm_over_control_speedup']:.4f}x"
        )
        for row in results
    )
)
PY

(
    cd "$result_dir"
    sha256sum \
        "$(basename "$test_log")" \
        "$(basename "$benchmark_json")" \
        "$(basename "$provenance_json")" \
        > "$(basename "$sha256_file")"
)
