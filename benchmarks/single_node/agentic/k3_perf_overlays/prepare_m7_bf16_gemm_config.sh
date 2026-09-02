#!/usr/bin/env bash
set -euo pipefail

result_dir="${1:?usage: prepare_m7_bf16_gemm_config.sh <result-dir>}"
overlay_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
selected_csv="$overlay_dir/k3_m7_bf16_selected.csv"
runtime_csv="$result_dir/k3_m7_bf16_runtime_config.csv"
provenance_json="$result_dir/k3_m7_bf16_runtime_config_provenance.json"

mkdir -p "$result_dir"
unset AITER_CONFIG_GEMM_BF16

python3 - \
    "$selected_csv" \
    "$runtime_csv" \
    "$provenance_json" <<'PY'
import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import aiter
import torch
from aiter.jit.core import AITER_CONFIGS

selected_path = Path(sys.argv[1]).resolve()
runtime_path = Path(sys.argv[2]).resolve()
provenance_path = Path(sys.argv[3]).resolve()

expected_selected_sha256 = (
    "f06db8549787cfca0cc8d9e6c44dedf92ff32d9e42b678002877eaa262d59d93"
)
expected_source_hashes = {
    "aiter/tuned_gemm.py": (
        "090b4bef2c5a7b58b137da68ca039eb09835f0ea78a4eaed21716c7b738e7750"
    ),
    "aiter/jit/core.py": (
        "9dd781706ab18258bd95199c96329e6b12abcce1bbd6e975f62dd39fa976e979"
    ),
}
expected_rows = {
    ("gfx950", 256, 7, 1536, 128): ("flydsl", 29, 1),
    ("gfx950", 256, 7, 20480, 7168): ("hipblaslt", 440308, 0),
    ("gfx950", 256, 7, 2880, 7168): ("flydsl", 2239, 4),
    ("gfx950", 256, 7, 3584, 7168): ("flydsl", 3889, 4),
    ("gfx950", 256, 7, 6288, 7168): ("hipblaslt", 440306, 0),
}
key_fields = (
    "gfx",
    "cu_num",
    "M",
    "N",
    "K",
    "bias",
    "dtype",
    "outdtype",
    "scaleAB",
    "bpreshuffle",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def key(row: dict[str, str]) -> tuple[str, int, int, int, int]:
    return (
        row["gfx"],
        int(row["cu_num"]),
        int(row["M"]),
        int(row["N"]),
        int(row["K"]),
    )


if sha256(selected_path) != expected_selected_sha256:
    raise SystemExit("selected M=7 config does not match the adjudicated artifact")

installed_root = Path(aiter.__file__).resolve().parent
for relative, expected_hash in expected_source_hashes.items():
    installed_path = installed_root.parent / relative
    if not installed_path.is_file() or sha256(installed_path) != expected_hash:
        raise SystemExit(
            f"installed AITER source mismatch for {relative}: {installed_path}"
        )

versions_path = Path("/app/versions.txt")
versions = (
    versions_path.read_text(encoding="utf-8").splitlines()
    if versions_path.is_file()
    else []
)
if "AITER_BRANCH: v0.1.19" not in versions:
    raise SystemExit("image does not report the validated AITER v0.1.19 source")

properties = torch.cuda.get_device_properties(0)
arch = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
if arch != "gfx950" or properties.multi_processor_count != 256:
    raise SystemExit(
        f"expected a 256-CU gfx950, got arch={arch!r} "
        f"cu_num={properties.multi_processor_count}"
    )

base_path = Path(AITER_CONFIGS.AITER_CONFIG_GEMM_BF16_FILE).resolve()
if not base_path.is_file():
    raise SystemExit(f"default merged AITER BF16 config is missing: {base_path}")

base_fields, base_rows = load_csv(base_path)
selected_fields, selected_rows = load_csv(selected_path)
if base_fields != selected_fields:
    raise SystemExit(
        f"selected config schema differs from the installed config: "
        f"base={base_fields}, selected={selected_fields}"
    )

selected_keys = {key(row) for row in selected_rows}
if len(selected_rows) != 5 or selected_keys != set(expected_rows):
    raise SystemExit(f"unexpected selected M=7 shapes: {sorted(selected_keys)}")
for row in selected_rows:
    expected_route = expected_rows[key(row)]
    actual_route = (row["libtype"], int(row["solidx"]), int(row["splitK"]))
    if actual_route != expected_route:
        raise SystemExit(
            f"unexpected route for {key(row)}: {actual_route}, expected {expected_route}"
        )

base_keys = {
    tuple(row[field] for field in key_fields)
    for row in base_rows
}
selected_full_keys = {
    tuple(row[field] for field in key_fields)
    for row in selected_rows
}
overlap = base_keys & selected_full_keys
if overlap:
    raise SystemExit(f"installed config already contains selected rows: {sorted(overlap)}")

runtime_path.parent.mkdir(parents=True, exist_ok=True)
temporary_path = runtime_path.with_suffix(runtime_path.suffix + ".tmp")
with temporary_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=base_fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(base_rows)
    writer.writerows(selected_rows)
os.replace(temporary_path, runtime_path)

runtime_fields, runtime_rows = load_csv(runtime_path)
if runtime_fields != base_fields or len(runtime_rows) != len(base_rows) + 5:
    raise SystemExit("runtime config did not preserve the installed rows plus five additions")
runtime_keys = [tuple(row[field] for field in key_fields) for row in runtime_rows]
if len(runtime_keys) != len(set(runtime_keys)):
    raise SystemExit("runtime config contains duplicate dispatch keys")

payload = {
    "aiter_commit": "31350226161346314b3d8882c8085bd31dce6a34",
    "aiter_tag": "v0.1.19",
    "arch": arch,
    "cu_num": properties.multi_processor_count,
    "base_config": str(base_path),
    "base_config_sha256": sha256(base_path),
    "base_row_count": len(base_rows),
    "selected_config": str(selected_path),
    "selected_config_sha256": sha256(selected_path),
    "selected_row_count": len(selected_rows),
    "runtime_config": str(runtime_path),
    "runtime_config_sha256": sha256(runtime_path),
    "runtime_row_count": len(runtime_rows),
    "installed_aiter": str(installed_root),
    "installed_source_sha256": expected_source_hashes,
    "image_versions": versions,
    "python": platform.python_version(),
    "torch": torch.__version__,
    "hip": torch.version.hip,
}
provenance_path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

printf 'Prepared exact M=7 BF16 runtime config: %s\n' "$runtime_csv"
