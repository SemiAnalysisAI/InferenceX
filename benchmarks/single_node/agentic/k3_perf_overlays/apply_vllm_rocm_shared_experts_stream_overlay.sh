#!/usr/bin/env bash
set -euo pipefail

action="${1:?usage: apply_vllm_rocm_shared_experts_stream_overlay.sh <apply|restore>}"
overlay_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
candidate_base="1dc464d42681d22f38caf1fdc1eb632dc4421c45"
runtime_patch="$overlay_dir/vllm_rocm_shared_experts_stream.patch"
hardware_test="$overlay_dir/test_rocm_shared_experts_stream.py"
runtime_patch_sha="310c62b7192be306bff8d45a6a570e213207b3636522d7c3dbcbba0c62db7567"
runtime_patch_id="c3d60b85b56d7dd8fff2492c14616882cb3f0e7d"
hardware_test_sha="a71f7efc559c47d85e186471550408dfce0c83360d7c991c6475ab266acff2ca"
target_relpath="vllm/model_executor/layers/fused_moe/runner/shared_experts.py"
expected_before_sha="13ec6272802ceeca754c4907c738eed23afc093c2b7b93658983a211e51c754d"
expected_after_sha="c74478803591208274b78840e8834cd82f2173c56904c460523153659e1ce816"

site_packages="$(python3 - <<'PY'
import sysconfig

print(sysconfig.get_paths()["purelib"])
PY
)"
target="$site_packages/$target_relpath"
cache_root="${K3_ARM_CACHE_ROOT:?K3_ARM_CACHE_ROOT must be set}"
result_dir="${RESULT_DIR:?RESULT_DIR must be set}"
backup="$cache_root/vllm-rocm-shared-experts-stream-backup/$target_relpath"
provenance_file="$result_dir/vllm_rocm_shared_experts_stream_provenance.tsv"
restore_file="$result_dir/vllm_rocm_shared_experts_stream_restore.tsv"
test_result_dir="$result_dir/rocm_shared_experts_stream_hardware_test"

file_sha() {
    sha256sum "$1" | awk '{print $1}'
}

verify_input_hashes() {
    local actual
    command -v patch >/dev/null 2>&1 || {
        echo "Error: patch is required for the ROCm shared-expert stream overlay" >&2
        return 1
    }
    actual="$(file_sha "$runtime_patch")"
    [[ "$actual" == "$runtime_patch_sha" ]] || {
        echo "Error: ROCm shared-expert stream patch checksum mismatch: $actual" >&2
        return 1
    }
    actual="$(file_sha "$hardware_test")"
    [[ "$actual" == "$hardware_test_sha" ]] || {
        echo "Error: ROCm shared-expert stream hardware-test checksum mismatch: $actual" >&2
        return 1
    }
}

verify_target() {
    local expected="$1"
    local actual
    [[ -f "$target" ]] || {
        echo "Error: installed vLLM source is missing $target" >&2
        return 1
    }
    actual="$(file_sha "$target")"
    [[ "$actual" == "$expected" ]] || {
        echo "Error: installed vLLM source mismatch for $target" >&2
        echo "Expected $expected, got $actual" >&2
        return 1
    }
}

restore_overlay() {
    local actual
    if [[ ! -f "$backup" ]]; then
        verify_target "$expected_before_sha"
        printf 'status\tnot-applied\nbase\t%s\n' "$candidate_base" \
            >"$restore_file"
        return 0
    fi
    actual="$(file_sha "$backup")"
    [[ "$actual" == "$expected_before_sha" ]] || {
        echo "Error: ROCm shared-expert stream backup checksum mismatch: $actual" >&2
        return 1
    }
    cp -p "$backup" "$target"
    verify_target "$expected_before_sha"
    printf 'status\trestored\nbase\t%s\npatch_sha256\t%s\npatch_id\t%s\n' \
        "$candidate_base" "$runtime_patch_sha" "$runtime_patch_id" \
        >"$restore_file"
    echo "Restored vLLM ROCm shared-expert stream overlay to base $candidate_base"
}

case "$action" in
    restore)
        restore_overlay
        exit 0
        ;;
    apply)
        ;;
    *)
        echo "Error: unsupported overlay action '$action'" >&2
        exit 1
        ;;
esac

verify_input_hashes
verify_target "$expected_before_sha"
if [[ -e "$backup" ]]; then
    echo "Error: ROCm shared-expert stream backup already exists: $backup" >&2
    exit 1
fi
mkdir -p "$(dirname "$backup")" "$result_dir"
cp -p "$target" "$backup"

rollback_on_error() {
    local rc=$?
    if ((rc != 0)); then
        restore_overlay || true
    fi
    exit "$rc"
}
trap rollback_on_error EXIT

(
    cd "$site_packages"
    patch --batch --forward --strip=1 --input="$runtime_patch"
)
verify_target "$expected_after_sha"
python3 -m compileall -q "$target"

K3_SHARED_STREAM_TEST_RESULT_DIR="$test_result_dir" \
    python3 -m torch.distributed.run --standalone --nproc-per-node="${TP:?TP must be set}" \
        "$hardware_test"

python3 - "$test_result_dir" "$TP" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
world_size = int(sys.argv[2])
paths = sorted(root.glob("rank_*.json"))
if len(paths) != world_size:
    raise SystemExit(f"expected {world_size} rank results, found {len(paths)}")
rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
if sorted(row["rank"] for row in rows) != list(range(world_size)):
    raise SystemExit(f"unexpected rank set: {rows}")
for row in rows:
    if row.get("status") != "passed":
        raise SystemExit(f"hardware test failed: {row}")
    if row.get("selected_order") != "MULTI_STREAM_OVERLAPPED":
        raise SystemExit(f"unexpected shared-expert order: {row}")
    if row.get("world_size") != world_size:
        raise SystemExit(f"unexpected world size: {row}")
summary = {
    "all_ranks_passed": True,
    "max_abs_error": max(row["max_abs_error"] for row in rows),
    "rank_count": len(rows),
    "selected_order": "MULTI_STREAM_OVERLAPPED",
}
(root / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

{
    printf 'status\tapplied\n'
    printf 'base\t%s\n' "$candidate_base"
    printf 'patch_sha256\t%s\n' "$runtime_patch_sha"
    printf 'patch_id\t%s\n' "$runtime_patch_id"
    printf 'hardware_test_sha256\t%s\n' "$hardware_test_sha"
    printf 'file\t%s\t%s\t%s\n' \
        "$target_relpath" "$expected_before_sha" "$expected_after_sha"
} >"$provenance_file"

trap - EXIT
echo "K3 ROCm shared-expert stream overlay: patch $runtime_patch_id"
