#!/usr/bin/env bash
set -euo pipefail

result_dir="${1:?usage: tune_m7_bf16_gemms.sh <result-dir> [M]}"
gemm_m="${2:-7}"
case "$gemm_m" in
    4|5|6|7) ;;
    *)
        echo "Error: supported Kimi-K3 BF16 GEMM M values are 4, 5, 6, and 7; got $gemm_m" >&2
        exit 1
        ;;
esac
overlay_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
aiter_repo="https://github.com/ROCm/aiter.git"
aiter_tag="v0.1.19"
aiter_commit="31350226161346314b3d8882c8085bd31dce6a34"
artifact_prefix="k3_m${gemm_m}_bf16"
scratch="$(mktemp -d "/tmp/k3-m${gemm_m}-bf16-tune.XXXXXX")"
trap 'rm -rf "$scratch"' EXIT

# The default arm must use the image's bundled configuration. Do not inherit a
# caller-provided override into either tuning or the default/candidate/default
# replay comparison.
unset AITER_CONFIG_GEMM_BF16

mkdir -p "$result_dir"
input_csv="$overlay_dir/${artifact_prefix}_untuned_gemm.csv"
candidate_csv="$result_dir/${artifact_prefix}_tuned_candidates.csv"
profile_csv="$result_dir/${artifact_prefix}_tuning_profile.csv"
graph_json="$result_dir/${artifact_prefix}_graph_comparison.json"
selected_csv="$result_dir/${artifact_prefix}_selected.csv"
tuner_log="$result_dir/${artifact_prefix}_tuner.log"
provenance_json="$result_dir/${artifact_prefix}_provenance.json"
sha256_file="$result_dir/${artifact_prefix}_SHA256SUMS"

case "$gemm_m" in
    4)
        tune_gpus="${K3_M4_GEMM_TUNE_GPUS:-6}"
        tune_timeout="${K3_M4_GEMM_TUNE_TIMEOUT:-1800}"
        ;;
    5)
        tune_gpus="${K3_M5_GEMM_TUNE_GPUS:-6}"
        tune_timeout="${K3_M5_GEMM_TUNE_TIMEOUT:-1800}"
        ;;
    6)
        tune_gpus="${K3_M6_GEMM_TUNE_GPUS:-6}"
        tune_timeout="${K3_M6_GEMM_TUNE_TIMEOUT:-1800}"
        ;;
    7)
        tune_gpus="${K3_M7_GEMM_TUNE_GPUS:-6}"
        tune_timeout="${K3_M7_GEMM_TUNE_TIMEOUT:-1800}"
        ;;
esac

cp "$input_csv" "$result_dir/${artifact_prefix}_untuned_gemm.csv"
git clone --filter=blob:none --branch "$aiter_tag" --depth 1 "$aiter_repo" "$scratch/aiter"
actual_commit="$(git -C "$scratch/aiter" rev-parse HEAD)"
if [[ "$actual_commit" != "$aiter_commit" ]]; then
    echo "Error: expected AITER $aiter_commit, got $actual_commit" >&2
    exit 1
fi

AITER_SOURCE_COMMIT="$actual_commit" AITER_SOURCE_TAG="$aiter_tag" \
AITER_GEMM_M="$gemm_m" \
AITER_SOURCE_DIR="$scratch/aiter" \
python3 - "$provenance_json" <<'PY'
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

import aiter
import torch
from aiter.jit.core import AITER_CONFIGS

output = Path(sys.argv[1])
properties = torch.cuda.get_device_properties(0)
installed_root = Path(aiter.__file__).resolve().parent
source_root = Path(os.environ["AITER_SOURCE_DIR"])
tracked_files = ("aiter/tuned_gemm.py", "aiter/jit/core.py")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


try:
    installed_version = importlib.metadata.version("aiter")
except importlib.metadata.PackageNotFoundError:
    # The pinned vLLM image installs AITER directly into site-packages without
    # retaining wheel metadata. Source hashes and /app/versions.txt below are
    # the authoritative provenance checks for that image layout.
    installed_version = "source-install-without-dist-info"


payload = {
    "gemm_m": int(os.environ["AITER_GEMM_M"]),
    "source_commit": os.environ["AITER_SOURCE_COMMIT"],
    "source_tag": os.environ["AITER_SOURCE_TAG"],
    "installed_aiter_file": str(installed_root / "__init__.py"),
    "installed_aiter_version": installed_version,
    "source_file_sha256": {
        relative: sha256(source_root / relative) for relative in tracked_files
    },
    "installed_file_sha256": {
        relative: sha256(installed_root.parent / relative) for relative in tracked_files
    },
    "default_merged_bf16_config": AITER_CONFIGS.AITER_CONFIG_GEMM_BF16_FILE,
    "python": platform.python_version(),
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "device": properties.name,
    "arch": str(getattr(properties, "gcnArchName", "")).split(":", 1)[0],
    "cu_num": properties.multi_processor_count,
}
versions = Path("/app/versions.txt")
if versions.is_file():
    payload["image_versions"] = versions.read_text(encoding="utf-8").splitlines()
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

python3 - "$provenance_json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload["arch"] != "gfx950":
    raise SystemExit(f"expected gfx950, got {payload['arch']!r}")
if payload["source_tag"] != "v0.1.19":
    raise SystemExit(f"unexpected AITER source tag: {payload['source_tag']!r}")
if payload["source_file_sha256"] != payload["installed_file_sha256"]:
    raise SystemExit("installed AITER Python sources do not match the pinned source")
image_versions = payload.get("image_versions", [])
if "AITER_BRANCH: v0.1.19" not in image_versions:
    raise SystemExit("pinned image does not report AITER_BRANCH: v0.1.19")
PY

(
    cd "$scratch/aiter"
    python3 csrc/gemm_a16w16/gemm_tuner.py \
        --input_file "$input_csv" \
        --tuned_file "$candidate_csv" \
        --profile_file "$profile_csv" \
        --libtype all \
        --with-hipblaslt \
        --shape_grouped \
        --mp "$tune_gpus" \
        --timeout "$tune_timeout" \
        --warmup 20 \
        --iters 201 \
        --verbose
) 2>&1 | tee "$tuner_log"

HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
python3 "$overlay_dir/bench_m7_bf16_gemms.py" \
    --input-csv "$input_csv" \
    --candidate-csv "$candidate_csv" \
    --output "$graph_json" \
    --selected-csv "$selected_csv"

python3 - "$input_csv" "$candidate_csv" "$graph_json" "$gemm_m" <<'PY'
import csv
import json
import sys
from pathlib import Path


def rows_and_keys(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keys = {
        (int(row["M"]), int(row["N"]), int(row["K"]))
        for row in rows
    }
    return rows, keys


expected_rows, expected = rows_and_keys(sys.argv[1])
tuned_rows, tuned = rows_and_keys(sys.argv[2])
expected_m = int(sys.argv[4])
if len(expected_rows) != len(expected):
    raise SystemExit("input contains duplicate M/N/K shapes")
if {key[0] for key in expected} != {expected_m}:
    raise SystemExit(
        f"input does not contain exactly M={expected_m}: "
        f"observed={sorted({key[0] for key in expected})}"
    )
if len(tuned_rows) != len(tuned):
    raise SystemExit("tuner output contains duplicate M/N/K shapes")
if tuned != expected:
    raise SystemExit(
        f"tuner did not produce exactly the six requested shapes: "
        f"missing={sorted(expected - tuned)}, extra={sorted(tuned - expected)}"
    )
summary = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
comparisons = summary.get("comparisons", [])
if len(comparisons) != len(expected):
    raise SystemExit(
        f"graph benchmark produced {len(comparisons)} comparisons, expected {len(expected)}"
    )
if not summary.get("all_changed_input_graph_replays_passed"):
    raise SystemExit("changed-input HIP graph replay did not pass")
print(
    f"Selected {summary['selected_shape_count']} of {len(expected)} "
    f"exact M={expected_m} shapes"
)
PY

(
    cd "$result_dir"
    sha256sum \
        "$(basename "$input_csv")" \
        "$(basename "$candidate_csv")" \
        "$(basename "$profile_csv")" \
        "$(basename "$graph_json")" \
        "$(basename "$selected_csv")" \
        "$(basename "$tuner_log")" \
        "$(basename "$provenance_json")" \
        >"$(basename "$sha256_file")"
)
