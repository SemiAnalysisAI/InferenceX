#!/usr/bin/env bash
set -euo pipefail

result_dir="${1:?usage: run_grouped_topk_radix.sh <result-dir>}"
overlay_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
aiter_repo="https://github.com/andyluo7/aiter.git"
aiter_ref="codex/k3-c1-radix-router-20260901"
candidate_commit="d68332357e17e93f0f07d4deb1fba3144239466f"
stock_commit="7f184691e35627b3a672974687e617d057164836"
scratch="$(mktemp -d /tmp/k3-radix-router.XXXXXX)"
trap 'rm -rf "$scratch"' EXIT

mkdir -p "$result_dir"
test_log="$result_dir/k3_radix_router_tests.log"
comparison_json="$result_dir/k3_radix_router_comparison.json"
comparison_tsv="$result_dir/k3_radix_router_comparison.tsv"
provenance_json="$result_dir/k3_radix_router_provenance.json"
sha256_file="$result_dir/k3_radix_router_SHA256SUMS"
candidate_checkout="$scratch/aiter-candidate"
stock_checkout="$scratch/aiter-stock"
candidate_jit="$scratch/jit-candidate"
stock_jit="$scratch/jit-stock"

git init --quiet "$candidate_checkout"
git -C "$candidate_checkout" remote add origin "$aiter_repo"
# Keep enough first-parent history to include the pinned stock revision.  A
# separate depth-1 fetch of stock_commit leaves it as a disconnected shallow
# root, which makes merge-base reject a valid ancestor relationship.
git -C "$candidate_checkout" fetch --quiet --depth 8 origin "$aiter_ref"
git -C "$candidate_checkout" checkout --quiet --detach FETCH_HEAD
actual_candidate_commit="$(git -C "$candidate_checkout" rev-parse HEAD)"
if [[ "$actual_candidate_commit" != "$candidate_commit" ]]; then
    printf 'ERROR: expected candidate AITER %s, got %s\n' \
        "$candidate_commit" "$actual_candidate_commit" >&2
    exit 1
fi
if ! git -C "$candidate_checkout" cat-file -e "${stock_commit}^{commit}"; then
    git -C "$candidate_checkout" fetch --quiet --deepen 64 origin "$aiter_ref"
fi
if ! git -C "$candidate_checkout" merge-base --is-ancestor \
    "$stock_commit" "$candidate_commit"; then
    echo "ERROR: stock AITER commit is not an ancestor of the candidate" >&2
    exit 1
fi

git clone --quiet --shared --no-checkout "$candidate_checkout" "$stock_checkout"
git -C "$stock_checkout" checkout --quiet --detach "$stock_commit"
git -C "$candidate_checkout" submodule update --init --depth 1 \
    3rdparty/composable_kernel
git -C "$stock_checkout" submodule update --init --depth 1 \
    3rdparty/composable_kernel

export PREBUILD_KERNELS=0
export AITER_USE_SYSTEM_TRITON=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
python3 -m pip uninstall -y aiter amd-aiter >/dev/null 2>&1 || true
python3 -m pip install -e "$candidate_checkout" --no-build-isolation --no-deps

AITER_CANDIDATE="$candidate_checkout" \
AITER_STOCK="$stock_checkout" \
AITER_CANDIDATE_COMMIT="$candidate_commit" \
AITER_STOCK_COMMIT="$stock_commit" \
python3 - "$provenance_json" <<'PY'
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

output = Path(sys.argv[1])
candidate = Path(os.environ["AITER_CANDIDATE"]).resolve()
stock = Path(os.environ["AITER_STOCK"]).resolve()
installed = Path(aiter.__file__).resolve()
if candidate not in installed.parents:
    raise SystemExit(f"expected editable AITER under {candidate}, got {installed}")

properties = torch.cuda.get_device_properties(0)
arch = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
if not torch.version.hip or arch != "gfx950":
    raise SystemExit(f"expected ROCm gfx950, got hip={torch.version.hip!r} arch={arch!r}")
if properties.multi_processor_count != 256:
    raise SystemExit(f"expected 256 CUs, got {properties.multi_processor_count}")

tracked = (
    "aiter/ops/topk.py",
    "csrc/include/moe_op.h",
    "csrc/include/rocm_ops.hpp",
    "csrc/kernels/topk_softmax_kernels_group.cu",
)
candidate_only = ("op_tests/test_grouped_topk_radix.py",)


def distribution_version() -> str:
    for name in ("aiter", "amd-aiter"):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    raise SystemExit("editable AITER install has no distribution metadata")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def status(path: Path) -> list[str]:
    return subprocess.check_output(
        ["git", "-C", str(path), "status", "--short"], text=True
    ).splitlines()


payload = {
    "aiter_repo": "https://github.com/andyluo7/aiter.git",
    "candidate_commit": os.environ["AITER_CANDIDATE_COMMIT"],
    "stock_commit": os.environ["AITER_STOCK_COMMIT"],
    "aiter_installed_file": str(installed),
    "aiter_version": distribution_version(),
    "candidate_source_sha256": {
        relative: sha256(candidate / relative)
        for relative in tracked + candidate_only
    },
    "stock_source_sha256": {
        relative: sha256(stock / relative) for relative in tracked
    },
    "candidate_status_porcelain": status(candidate),
    "stock_status_porcelain": status(stock),
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

mkdir -p "$candidate_jit" "$stock_jit"
AITER_META_DIR="$candidate_checkout" \
AITER_JIT_DIR="$candidate_jit" \
AITER_REBUILD=1 \
HIP_VISIBLE_DEVICES=0 \
ROCR_VISIBLE_DEVICES=0 \
python3 -m pytest -q \
    "$candidate_checkout/op_tests/test_grouped_topk_radix.py" \
    2>&1 | tee "$test_log"

run_round() {
    local implementation="$1"
    local checkout="$2"
    local jit_dir="$3"
    local rebuild="$4"
    local output="$5"
    shift 5
    AITER_META_DIR="$checkout" \
    AITER_JIT_DIR="$jit_dir" \
    AITER_REBUILD="$rebuild" \
    HIP_VISIBLE_DEVICES=0 \
    ROCR_VISIBLE_DEVICES=0 \
    python3 "$overlay_dir/bench_grouped_topk_radix.py" \
        --implementation "$implementation" \
        --aiter-commit "$(git -C "$checkout" rev-parse HEAD)" \
        --output "$output" \
        --num-tokens "$@"
}

# Alternate implementations and reverse the shape order on the second pass.
# Compilation and graph construction are outside all timed regions.
run_round stock "$stock_checkout" "$stock_jit" 1 \
    "$result_dir/k3_radix_router_stock_round1.json" 1 2 4 7 14
run_round radix "$candidate_checkout" "$candidate_jit" 0 \
    "$result_dir/k3_radix_router_candidate_round1.json" 14 7 4 2 1
run_round stock "$stock_checkout" "$stock_jit" 0 \
    "$result_dir/k3_radix_router_stock_round2.json" 14 7 4 2 1
run_round radix "$candidate_checkout" "$candidate_jit" 0 \
    "$result_dir/k3_radix_router_candidate_round2.json" 1 2 4 7 14

python3 "$overlay_dir/summarize_grouped_topk_radix.py" \
    --stock \
        "$result_dir/k3_radix_router_stock_round1.json" \
        "$result_dir/k3_radix_router_stock_round2.json" \
    --candidate \
        "$result_dir/k3_radix_router_candidate_round1.json" \
        "$result_dir/k3_radix_router_candidate_round2.json" \
    --stock-commit "$stock_commit" \
    --candidate-commit "$candidate_commit" \
    --output-json "$comparison_json" \
    --output-tsv "$comparison_tsv"

(
    cd "$result_dir"
    sha256sum \
        "$(basename "$test_log")" \
        "$(basename "$comparison_json")" \
        "$(basename "$comparison_tsv")" \
        "$(basename "$provenance_json")" \
        k3_radix_router_stock_round1.json \
        k3_radix_router_candidate_round1.json \
        k3_radix_router_stock_round2.json \
        k3_radix_router_candidate_round2.json \
        >"$(basename "$sha256_file")"
)
