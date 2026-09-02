#!/usr/bin/env bash
set -euo pipefail

action="${1:?usage: prepare_deferred_finalize.sh <prepare|activate-base|activate-candidate|restore>}"
overlay_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
session_root="${K3_DEFERRED_SESSION_ROOT:?K3_DEFERRED_SESSION_ROOT must be set}"
result_dir="${RESULT_DIR:?RESULT_DIR must be set}"
arm_cache_root="${K3_ARM_CACHE_ROOT:?K3_ARM_CACHE_ROOT must be set}"
validation_dir="${K3_DEFERRED_VALIDATION_DIR:?K3_DEFERRED_VALIDATION_DIR must be set}"

aiter_repo="https://github.com/ROCm/aiter.git"
aiter_commit="31350226161346314b3d8882c8085bd31dce6a34"
vllm_commit="1dc464d42681d22f38caf1fdc1eb632dc4421c45"
aiter_patch="$overlay_dir/aiter_deferred_finalize.patch"
vllm_patch="$overlay_dir/vllm_deferred_finalize.patch"
hardware_test="$overlay_dir/test_fused_route_reduce_ar_rms.py"
aiter_patch_sha="eaa1a84330715f01c5cddbb626f7caa54d84c27fc1aff0380b2927eedb18bcf8"
aiter_patch_id="0c4666c9b1ff4c45feb1c84fdb672b6d3aea9697"
vllm_patch_sha="e1a2048c710a3e1981ed82990ed953f1f529a031184e586f316d5cc84cb6754a"
hardware_test_sha="ab09a6680407704ed1370a05c618f944a2b6643160916eca1640edfaab343df6"
aiter_base="$session_root/aiter-base"
aiter_candidate="$session_root/aiter-candidate"
backup_root="$arm_cache_root/vllm-deferred-finalize-backup"

vllm_relpaths=(
    "vllm/_aiter_ops.py"
    "vllm/distributed/device_communicators/aiter_custom_all_reduce.py"
    "vllm/model_executor/layers/fused_moe/config.py"
    "vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py"
    "vllm/model_executor/layers/fused_moe/fused_moe_method_base.py"
    "vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py"
    "vllm/model_executor/layers/fused_moe/modular_kernel.py"
    "vllm/model_executor/layers/fused_moe/moe_output.py"
    "vllm/model_executor/layers/fused_moe/prepare_finalize/no_dp_ep.py"
    "vllm/model_executor/layers/fused_moe/routed_experts.py"
    "vllm/model_executor/layers/fused_moe/runner/moe_runner.py"
    "vllm/models/kimi_k3/amd/latent_moe_runner.py"
)
vllm_before_shas=(
    "3ea7b700fe3dba5eb4dfbe533d96651d64d5fe028b9a2dabf76d8360c0c7bf15"
    "266d606b2378b50bc3d1017fd6ae56f3d15ae61508ba35562048cd1bd0ac9319"
    "5520e19a6e329e75d69e24744024460649a9eee4471f0a54c02a74dbf3b48288"
    "8e570b2db04dfa2e2519b1df6c0bb402fb67c739af99c81c3da972687eba4b92"
    "18dcfc469d3252749801bbf943857f58da85de1f2db8fdff5a9f4986a925dc68"
    "321c744eac1fcaffd6b1b2971204c8baa57dfec53e91e5d24140014d72d64f83"
    "1e60aca6ed0dd4fcb46d577897ff1651f27a6130b3449d22265c0c791beec5d5"
    "6ea4cbd78f9efc7f842c5df246f553f08b991c3479b8b04b89902ed08aaf8e8f"
    "adbbf529cd22059c95b5380952943a70cf11e05e321dcd707f296bebde61a6e7"
    "db9fd011aa5f4f09eb86aa2ec55c113b88815ed9c0c617092a667e144d0bd57b"
    "80d4b53fdbbbaf2aa9408f22ea476bacbdbb9db5cef0e51fca62c36a4f384d25"
    "e6235f947b0a1d89327a7e11391a4d26c3f4fa57c1af804384acf55b0c1041ee"
)
vllm_after_shas=(
    "6ddd915c8d4769f945e5e535757905cfe91357ed64365fc843cb9658f2c4bb8d"
    "95b761d8730b122769256798727136e8a8a590cf9e863bf442aa1b9981fd05bf"
    "ec6cd2a8e29c6d7a17b728a8a1e5e8cdd98717d66567528ac7d7a85720a7dfef"
    "d4de24026780bb0c47079b6a6de1404b31d28c9b77fee86776f7a715a4c0117f"
    "ab9d5367bc94ee56393f0806689dc457fda17075c48381e61e5df6bf76109ef1"
    "f265818a2f15607d780cdce9e0c3d725200c9d1939d0138740f110953f2accd5"
    "40b84b6db0b8bd1aaf388c89ab10c41a00fcbd2ed85fd17874bf73acae6ce924"
    "707c6bd0577f3e927bf63a5ae7aefe9c9ab9c8683a00685f320b5ad5ed3dfc9a"
    "5a90d1a19c018afbbdaadfe1500631d6150b05037863e594366433255d84db62"
    "1c1dab1c4be9b1bd178643bbbff31d2a48eb7687bc9c4e122921cc5d582d4c5e"
    "c9305ff138ed9ea8878a96dd225b73599588ef69886097d9968f3a62282a2fef"
    "7db08d85106988a3b87833026232131122a8407113d41a570bb6acef911ed101"
)

file_sha() {
    sha256sum "$1" | awk '{print $1}'
}

verify_input_hashes() {
    local actual
    command -v patch >/dev/null 2>&1 || {
        echo "Error: patch is required for the deferred-finalization overlay" >&2
        return 1
    }
    actual="$(file_sha "$aiter_patch")"
    [[ "$actual" == "$aiter_patch_sha" ]] || {
        echo "Error: AITER patch checksum mismatch: $actual" >&2
        return 1
    }
    actual="$(file_sha "$vllm_patch")"
    [[ "$actual" == "$vllm_patch_sha" ]] || {
        echo "Error: vLLM patch checksum mismatch: $actual" >&2
        return 1
    }
    actual="$(file_sha "$hardware_test")"
    [[ "$actual" == "$hardware_test_sha" ]] || {
        echo "Error: fused-route hardware-test checksum mismatch: $actual" >&2
        return 1
    }
}

ensure_aiter_sources() {
    local actual_commit candidate_patch_id
    verify_input_hashes
    mkdir -p "$session_root"

    if [[ ! -d "$aiter_base/.git" ]]; then
        if [[ -e "$aiter_base" ]]; then
            echo "Error: incomplete AITER base checkout exists at $aiter_base" >&2
            return 1
        fi
        git init --quiet "$aiter_base"
        git -C "$aiter_base" remote add origin "$aiter_repo"
        git -C "$aiter_base" fetch --quiet --depth 1 origin "$aiter_commit"
        git -C "$aiter_base" checkout --quiet --detach FETCH_HEAD
        git -C "$aiter_base" submodule update --init --depth 1 \
            3rdparty/composable_kernel
    fi
    actual_commit="$(git -C "$aiter_base" rev-parse HEAD)"
    [[ "$actual_commit" == "$aiter_commit" ]] || {
        echo "Error: expected AITER base $aiter_commit, got $actual_commit" >&2
        return 1
    }
    if [[ -n "$(git -C "$aiter_base" status --short)" ]]; then
        echo "Error: AITER base checkout is dirty" >&2
        git -C "$aiter_base" status --short >&2
        return 1
    fi

    if [[ ! -e "$aiter_candidate/.git" ]]; then
        if [[ -e "$aiter_candidate" ]]; then
            echo "Error: incomplete AITER candidate checkout exists at $aiter_candidate" >&2
            return 1
        fi
        git -C "$aiter_base" worktree add --quiet --detach \
            "$aiter_candidate" "$aiter_commit"
        git -C "$aiter_candidate" submodule update --init --depth 1 \
            3rdparty/composable_kernel
        git -C "$aiter_candidate" apply --check "$aiter_patch"
        git -C "$aiter_candidate" apply "$aiter_patch"
    fi
    actual_commit="$(git -C "$aiter_candidate" rev-parse HEAD)"
    [[ "$actual_commit" == "$aiter_commit" ]] || {
        echo "Error: expected AITER candidate base $aiter_commit, got $actual_commit" >&2
        return 1
    }
    candidate_patch_id="$(git -C "$aiter_candidate" diff --binary | git patch-id --stable | awk '{print $1}')"
    [[ "$candidate_patch_id" == "$aiter_patch_id" ]] || {
        echo "Error: AITER candidate patch-id mismatch: $candidate_patch_id" >&2
        return 1
    }

    mkdir -p "$validation_dir"
    {
        printf 'aiter_repo\t%s\n' "$aiter_repo"
        printf 'aiter_base_commit\t%s\n' "$aiter_commit"
        printf 'aiter_patch_sha256\t%s\n' "$aiter_patch_sha"
        printf 'aiter_patch_id\t%s\n' "$aiter_patch_id"
        printf 'vllm_base_commit\t%s\n' "$vllm_commit"
        printf 'vllm_patch_sha256\t%s\n' "$vllm_patch_sha"
        printf 'hardware_test_sha256\t%s\n' "$hardware_test_sha"
    } >"$validation_dir/source_manifest.tsv"
}

site_packages() {
    python3 - <<'PY'
import sysconfig

print(sysconfig.get_paths()["purelib"])
PY
}

verify_vllm_files() {
    local expected_kind="$1"
    local root="$2"
    local i target actual expected
    for i in "${!vllm_relpaths[@]}"; do
        target="$root/${vllm_relpaths[$i]}"
        [[ -f "$target" ]] || {
            echo "Error: installed vLLM source is missing $target" >&2
            return 1
        }
        if [[ "$expected_kind" == "before" ]]; then
            expected="${vllm_before_shas[$i]}"
        else
            expected="${vllm_after_shas[$i]}"
        fi
        actual="$(file_sha "$target")"
        [[ "$actual" == "$expected" ]] || {
            echo "Error: vLLM $expected_kind checksum mismatch for $target" >&2
            echo "Expected $expected, got $actual" >&2
            return 1
        }
    done
}

install_aiter_checkout() {
    local checkout="$1"
    local expected_mode="$2"
    local provenance="$result_dir/aiter_${expected_mode}_provenance.tsv"

    PREBUILD_KERNELS=0 \
        AITER_USE_SYSTEM_TRITON=1 \
        PIP_DISABLE_PIP_VERSION_CHECK=1 \
        python3 -m pip install --quiet --force-reinstall --no-deps \
            --no-build-isolation -e "$checkout"

    AITER_EXPECTED_ROOT="$checkout" \
        AITER_EXPECTED_COMMIT="$aiter_commit" \
        AITER_EXPECTED_MODE="$expected_mode" \
        AITER_PROVENANCE="$provenance" \
        python3 - <<'PY'
import hashlib
import os
from pathlib import Path

import aiter

expected_root = Path(os.environ["AITER_EXPECTED_ROOT"]).resolve()
installed = Path(aiter.__file__).resolve()
if expected_root not in installed.parents:
    raise SystemExit(f"expected editable AITER under {expected_root}, got {installed}")

tracked = (
    "aiter/dist/device_communicators/custom_all_reduce.py",
    "aiter/fused_moe.py",
    "aiter/jit/utils/torch_guard.py",
    "aiter/ops/custom_all_reduce.py",
    "csrc/include/custom_all_reduce.cuh",
    "csrc/include/custom_all_reduce.h",
    "csrc/include/rocm_ops.hpp",
    "csrc/kernels/custom_all_reduce.cu",
)

with Path(os.environ["AITER_PROVENANCE"]).open("w", encoding="utf-8") as handle:
    handle.write(f"mode\t{os.environ['AITER_EXPECTED_MODE']}\n")
    handle.write(f"commit\t{os.environ['AITER_EXPECTED_COMMIT']}\n")
    handle.write(f"installed_file\t{installed}\n")
    for relative in tracked:
        path = expected_root / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        handle.write(f"file\t{relative}\t{digest}\n")
PY
}

restore_vllm() {
    local purelib root i target backup actual
    purelib="$(site_packages)"
    root="$purelib"
    if [[ ! -d "$backup_root" ]]; then
        verify_vllm_files before "$root"
        return 0
    fi
    for i in "${!vllm_relpaths[@]}"; do
        target="$root/${vllm_relpaths[$i]}"
        backup="$backup_root/${vllm_relpaths[$i]}"
        [[ -f "$backup" ]] || {
            echo "Error: vLLM deferred-finalize backup is missing $backup" >&2
            return 1
        }
        actual="$(file_sha "$backup")"
        [[ "$actual" == "${vllm_before_shas[$i]}" ]] || {
            echo "Error: vLLM backup checksum mismatch for $backup" >&2
            return 1
        }
    done
    for i in "${!vllm_relpaths[@]}"; do
        target="$root/${vllm_relpaths[$i]}"
        backup="$backup_root/${vllm_relpaths[$i]}"
        cp -p "$backup" "$target"
    done
    verify_vllm_files before "$root"
}

apply_candidate() {
    local purelib root i target backup
    local -a compile_paths=()
    ensure_aiter_sources
    install_aiter_checkout "$aiter_candidate" candidate

    purelib="$(site_packages)"
    root="$purelib"
    verify_vllm_files before "$root"
    [[ ! -e "$backup_root" ]] || {
        echo "Error: vLLM deferred-finalize backup already exists: $backup_root" >&2
        return 1
    }
    mkdir -p "$backup_root"
    for i in "${!vllm_relpaths[@]}"; do
        target="$root/${vllm_relpaths[$i]}"
        backup="$backup_root/${vllm_relpaths[$i]}"
        mkdir -p "$(dirname "$backup")"
        cp -p "$target" "$backup"
    done

    rollback_on_error() {
        local rc=$?
        if ((rc != 0)); then
            restore_vllm || true
            install_aiter_checkout "$aiter_base" rollback || true
        fi
        exit "$rc"
    }
    trap rollback_on_error EXIT

    (
        cd "$purelib"
        patch --batch --forward --strip=1 --input="$vllm_patch"
    )
    verify_vllm_files after "$root"

    for i in "${!vllm_relpaths[@]}"; do
        compile_paths+=("$root/${vllm_relpaths[$i]}")
    done
    python3 -m compileall -q "${compile_paths[@]}"

    python3 - <<'PY'
import inspect

from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
from aiter.fused_moe import fused_moe
from vllm._aiter_ops import rocm_aiter_ops

assert "return_per_slot" in inspect.signature(fused_moe).parameters
assert hasattr(CustomAllreduce, "custom_fused_route_reduce_ar_rms")
assert rocm_aiter_ops.fused_moe_supports_return_per_slot()
assert rocm_aiter_ops.supports_fused_route_reduce_allreduce_rmsnorm()
print("Kimi-K3 deferred-finalization import contract passed")
PY

    mkdir -p "$validation_dir"
    if [[ ! -f "$validation_dir/hardware_test_passed.tsv" ]]; then
        PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
            python3 -m pytest -q "$hardware_test" \
            2>&1 | tee "$validation_dir/hardware_test.log"
        {
            printf 'status\tpassed\n'
            printf 'aiter_patch_sha256\t%s\n' "$aiter_patch_sha"
            printf 'vllm_patch_sha256\t%s\n' "$vllm_patch_sha"
            printf 'hardware_test_sha256\t%s\n' "$hardware_test_sha"
        } >"$validation_dir/hardware_test_passed.tsv"
    fi
    grep -Fxq $'status\tpassed' "$validation_dir/hardware_test_passed.tsv"
    grep -Fxq $'aiter_patch_sha256\t'"$aiter_patch_sha" \
        "$validation_dir/hardware_test_passed.tsv"
    grep -Fxq $'vllm_patch_sha256\t'"$vllm_patch_sha" \
        "$validation_dir/hardware_test_passed.tsv"
    grep -Fxq $'hardware_test_sha256\t'"$hardware_test_sha" \
        "$validation_dir/hardware_test_passed.tsv"

    {
        printf 'status\tapplied\n'
        printf 'aiter_base_commit\t%s\n' "$aiter_commit"
        printf 'aiter_patch_sha256\t%s\n' "$aiter_patch_sha"
        printf 'vllm_base_commit\t%s\n' "$vllm_commit"
        printf 'vllm_patch_sha256\t%s\n' "$vllm_patch_sha"
        printf 'hardware_test_sha256\t%s\n' "$hardware_test_sha"
        printf 'hardware_test_marker\t%s\n' \
            "$validation_dir/hardware_test_passed.tsv"
        for i in "${!vllm_relpaths[@]}"; do
            printf 'vllm_file\t%s\t%s\t%s\n' \
                "${vllm_relpaths[$i]}" \
                "${vllm_before_shas[$i]}" \
                "${vllm_after_shas[$i]}"
        done
    } >"$result_dir/deferred_finalize_provenance.tsv"

    trap - EXIT
}

case "$action" in
    prepare)
        ensure_aiter_sources
        ;;
    activate-base)
        ensure_aiter_sources
        restore_vllm
        install_aiter_checkout "$aiter_base" baseline
        ;;
    activate-candidate)
        apply_candidate
        ;;
    restore)
        ensure_aiter_sources
        restore_vllm
        install_aiter_checkout "$aiter_base" restored
        {
            printf 'status\trestored\n'
            printf 'aiter_base_commit\t%s\n' "$aiter_commit"
            printf 'aiter_patch_sha256\t%s\n' "$aiter_patch_sha"
            printf 'vllm_base_commit\t%s\n' "$vllm_commit"
            printf 'vllm_patch_sha256\t%s\n' "$vllm_patch_sha"
        } >"$result_dir/deferred_finalize_restore.tsv"
        ;;
    *)
        echo "Error: unsupported deferred-finalize action '$action'" >&2
        exit 1
        ;;
esac
