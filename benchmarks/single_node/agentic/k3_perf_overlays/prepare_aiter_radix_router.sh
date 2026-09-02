#!/usr/bin/env bash
set -euo pipefail

mode="${1:?usage: prepare_aiter_radix_router.sh <prepare|verify|record-binary> <session-root> [stock|radix]}"
session_root="${2:?radix-router session root is required}"
implementation="${3:-}"
result_dir="${RESULT_DIR:?RESULT_DIR must be set}"

aiter_repo="https://github.com/andyluo7/aiter.git"
aiter_ref="codex/k3-c1-radix-router-20260901"
stock_commit="7f184691e35627b3a672974687e617d057164836"
candidate_commit="d68332357e17e93f0f07d4deb1fba3144239466f"
candidate_checkout="$session_root/aiter-candidate"
stock_checkout="$session_root/aiter-stock"
jit_seed="$session_root/installed-jit-seed"

mkdir -p "$result_dir"

expected_checkout() {
    case "$1" in
        stock)
            printf '%s\n' "$stock_checkout"
            ;;
        radix)
            printf '%s\n' "$candidate_checkout"
            ;;
        *)
            echo "Error: unsupported radix-router implementation '$1'" >&2
            return 1
            ;;
    esac
}

expected_commit() {
    case "$1" in
        stock)
            printf '%s\n' "$stock_commit"
            ;;
        radix)
            printf '%s\n' "$candidate_commit"
            ;;
        *)
            return 1
            ;;
    esac
}

case "$mode" in
    prepare)
        if [[ -e "$candidate_checkout" || -e "$stock_checkout" || -e "$jit_seed" ]]; then
            echo "Error: radix-router session root is not fresh: $session_root" >&2
            exit 1
        fi
        mkdir -p "$session_root" "$jit_seed"

        python3 - "$result_dir/original_aiter.json" <<'PY'
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

spec = importlib.util.find_spec("aiter")
if spec is None or spec.origin is None:
    raise SystemExit("cannot locate the image's installed AITER package")
version = None
for name in ("aiter", "amd-aiter"):
    try:
        version = importlib.metadata.version(name)
        break
    except importlib.metadata.PackageNotFoundError:
        pass
payload = {
    "installed_file": str(Path(spec.origin).resolve()),
    "installed_version": version,
}
Path(sys.argv[1]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

        original_jit="$(python3 - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("aiter")
if spec is None or spec.origin is None:
    raise SystemExit("cannot locate installed AITER")
print(Path(spec.origin).resolve().parent / "jit")
PY
)"
        if [[ ! -d "$original_jit" ]]; then
            echo "Error: installed AITER JIT directory is missing: $original_jit" >&2
            exit 1
        fi
        while IFS= read -r -d '' module; do
            cp -L "$module" "$jit_seed/$(basename "$module")"
        done < <(find -L "$original_jit" -maxdepth 1 -type f -name '*.so' -print0)
        if ! find "$jit_seed" -maxdepth 1 -type f -name '*.so' -print -quit \
                | grep -q .; then
            echo "Error: the image AITER JIT seed contains no shared objects" >&2
            exit 1
        fi
        if ! find "$jit_seed" -maxdepth 1 -type f -name 'module_moe_asm*.so' \
                -print -quit | grep -q .; then
            echo "Error: the image AITER JIT seed has no module_moe_asm binary" >&2
            exit 1
        fi
        if [[ -d "$original_jit/flydsl_cache" ]]; then
            cp -a "$original_jit/flydsl_cache" "$jit_seed/"
        fi

        git init --quiet "$candidate_checkout"
        git -C "$candidate_checkout" remote add origin "$aiter_repo"
        git -C "$candidate_checkout" fetch --quiet --depth 8 origin "$aiter_ref"
        git -C "$candidate_checkout" checkout --quiet --detach FETCH_HEAD
        if [[ "$(git -C "$candidate_checkout" rev-parse HEAD)" != "$candidate_commit" ]]; then
            echo "Error: fetched AITER candidate does not match $candidate_commit" >&2
            exit 1
        fi
        if ! git -C "$candidate_checkout" cat-file -e "${stock_commit}^{commit}"; then
            git -C "$candidate_checkout" fetch --quiet --deepen 64 origin "$aiter_ref"
        fi
        if ! git -C "$candidate_checkout" merge-base --is-ancestor \
                "$stock_commit" "$candidate_commit"; then
            echo "Error: pinned stock AITER commit is not an ancestor of the candidate" >&2
            exit 1
        fi

        git clone --quiet --shared --no-checkout "$candidate_checkout" "$stock_checkout"
        git -C "$stock_checkout" checkout --quiet --detach "$stock_commit"
        git -C "$candidate_checkout" submodule update --init --depth 1 \
            3rdparty/composable_kernel
        git -C "$stock_checkout" submodule update --init --depth 1 \
            3rdparty/composable_kernel

        git -C "$candidate_checkout" diff --name-only \
            "$stock_commit..$candidate_commit" \
            >"$result_dir/candidate_changed_files.txt"
        cat >"$result_dir/expected_changed_files.txt" <<'EOF'
.github/scripts/split_tests.sh
aiter/ops/topk.py
csrc/include/moe_op.h
csrc/include/rocm_ops.hpp
csrc/kernels/topk_softmax_kernels_group.cu
op_tests/test_grouped_topk_radix.py
op_tests/test_moeTopkSoftmax.py
EOF
        if ! cmp -s \
                "$result_dir/expected_changed_files.txt" \
                "$result_dir/candidate_changed_files.txt"; then
            echo "Error: unexpected files in the pinned AITER radix delta" >&2
            diff -u \
                "$result_dir/expected_changed_files.txt" \
                "$result_dir/candidate_changed_files.txt" >&2 || true
            exit 1
        fi

        python3 -m pip uninstall -y aiter amd-aiter >/dev/null 2>&1 || true
        PREBUILD_KERNELS=0 AITER_USE_SYSTEM_TRITON=1 \
            python3 -m pip install -e "$candidate_checkout" \
                --no-build-isolation --no-deps

        AITER_CANDIDATE="$candidate_checkout" \
        AITER_STOCK="$stock_checkout" \
        AITER_JIT_SEED="$jit_seed" \
        AITER_REPO="$aiter_repo" \
        AITER_REF="$aiter_ref" \
        AITER_CANDIDATE_COMMIT="$candidate_commit" \
        AITER_STOCK_COMMIT="$stock_commit" \
        python3 - "$result_dir/radix_router_prepare_provenance.json" <<'PY'
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import aiter

output = Path(sys.argv[1])
candidate = Path(os.environ["AITER_CANDIDATE"]).resolve()
stock = Path(os.environ["AITER_STOCK"]).resolve()
installed = Path(aiter.__file__).resolve()
if candidate not in installed.parents:
    raise SystemExit(f"expected editable AITER under {candidate}, got {installed}")

runtime_paths = (
    "aiter/ops/topk.py",
    "csrc/include/moe_op.h",
    "csrc/include/rocm_ops.hpp",
    "csrc/kernels/topk_softmax_kernels_group.cu",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def version() -> str | None:
    for name in ("aiter", "amd-aiter"):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return None


seed = Path(os.environ["AITER_JIT_SEED"])
payload = {
    "repository": os.environ["AITER_REPO"],
    "ref": os.environ["AITER_REF"],
    "stock_commit": os.environ["AITER_STOCK_COMMIT"],
    "candidate_commit": os.environ["AITER_CANDIDATE_COMMIT"],
    "installed_file": str(installed),
    "installed_version": version(),
    "stock_status": subprocess.check_output(
        ["git", "-C", str(stock), "status", "--short"], text=True
    ).splitlines(),
    "candidate_status": subprocess.check_output(
        ["git", "-C", str(candidate), "status", "--short"], text=True
    ).splitlines(),
    "stock_source_sha256": {path: sha256(stock / path) for path in runtime_paths},
    "candidate_source_sha256": {
        path: sha256(candidate / path) for path in runtime_paths
    },
    "jit_seed_files": {
        path.name: sha256(path)
        for path in sorted(seed.glob("*.so"))
        if path.is_file()
    },
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
        ;;

    verify)
        checkout="$(expected_checkout "$implementation")"
        commit="$(expected_commit "$implementation")"
        if [[ "$(git -C "$checkout" rev-parse HEAD)" != "$commit" ]]; then
            echo "Error: $implementation AITER checkout does not match $commit" >&2
            exit 1
        fi
        if [[ -n "$(git -C "$checkout" status --short)" ]]; then
            echo "Error: $implementation AITER checkout is dirty" >&2
            git -C "$checkout" status --short >&2
            exit 1
        fi
        AITER_CHECKOUT="$checkout" \
        AITER_COMMIT="$commit" \
        AITER_IMPLEMENTATION="$implementation" \
        python3 - "$result_dir/aiter_radix_router_provenance.json" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

import aiter
import torch

output = Path(sys.argv[1])
checkout = Path(os.environ["AITER_CHECKOUT"]).resolve()
installed = Path(aiter.__file__).resolve()
candidate = Path(os.environ["K3_RADIX_SESSION_ROOT"]).resolve() / "aiter-candidate"
if candidate not in installed.parents:
    raise SystemExit(f"expected editable AITER under {candidate}, got {installed}")
if Path(os.environ.get("AITER_META_DIR", "")).resolve() != checkout:
    raise SystemExit("AITER_META_DIR does not match the selected checkout")
props = torch.cuda.get_device_properties(0)
arch = str(getattr(props, "gcnArchName", "")).split(":", 1)[0]
if not torch.version.hip or arch != "gfx950" or props.multi_processor_count != 256:
    raise SystemExit(
        f"expected 256-CU ROCm gfx950, got hip={torch.version.hip!r} "
        f"arch={arch!r} cu={props.multi_processor_count}"
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


runtime_paths = (
    "aiter/ops/topk.py",
    "csrc/include/moe_op.h",
    "csrc/include/rocm_ops.hpp",
    "csrc/kernels/topk_softmax_kernels_group.cu",
)
payload = {
    "implementation": os.environ["AITER_IMPLEMENTATION"],
    "commit": os.environ["AITER_COMMIT"],
    "aiter_meta_dir": str(checkout),
    "aiter_installed_file": str(installed),
    "aiter_rebuild": os.environ.get("AITER_REBUILD"),
    "aiter_jit_dir": os.environ.get("AITER_JIT_DIR"),
    "hip": torch.version.hip,
    "arch": arch,
    "cu_num": props.multi_processor_count,
    "source_sha256": {path: sha256(checkout / path) for path in runtime_paths},
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
        ;;

    record-binary)
        mapfile -t modules < <(
            find -L "${AITER_JIT_DIR:?AITER_JIT_DIR must be set}" -maxdepth 1 \
                -type f -name 'module_moe_asm*.so' -print | LC_ALL=C sort
        )
        if [[ "${#modules[@]}" -ne 1 ]]; then
            printf 'Error: expected one rebuilt module_moe_asm binary, found %s\n' \
                "${#modules[@]}" >&2
            printf '%s\n' "${modules[@]}" >&2
            exit 1
        fi
        has_radix=false
        if strings "${modules[0]}" | grep -Fq 'grouped_topk_radix_kernel'; then
            has_radix=true
        fi
        if [[ "$implementation" == "radix" && "$has_radix" != "true" ]]; then
            echo "Error: candidate module does not contain the radix kernel" >&2
            exit 1
        fi
        if [[ "$implementation" == "stock" && "$has_radix" != "false" ]]; then
            echo "Error: stock module unexpectedly contains the radix kernel" >&2
            exit 1
        fi
        {
            printf 'implementation\t%s\n' "$implementation"
            printf 'module\t%s\n' "${modules[0]}"
            printf 'sha256\t%s\n' "$(sha256sum "${modules[0]}" | awk '{print $1}')"
            printf 'size_bytes\t%s\n' "$(stat -c %s "${modules[0]}")"
            printf 'contains_grouped_topk_radix_kernel\t%s\n' "$has_radix"
        } >"$result_dir/aiter_radix_router_binary.tsv"
        ;;

    *)
        echo "Error: unsupported mode '$mode'" >&2
        exit 1
        ;;
esac
