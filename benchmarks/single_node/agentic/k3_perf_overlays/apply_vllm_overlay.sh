#!/usr/bin/env bash
set -euo pipefail

variant="${1:?usage: apply_vllm_overlay.sh <variant>}"
overlay_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$variant" in
    pr52494)
        patch_file="$overlay_dir/vllm_pr52494.patch"
        expected_patch_sha="9efc1feee0dd5a4adaa8ce0229aec71c9620572b8cc49f290c1adab3ba4f195e"
        expected_before_linear_sha="f06a6408231150c797e99d8ae6d91d2993d90d69a061230747217386bb7d9d15"
        expected_after_linear_sha="3d093e916016dbda0b3df8eadd19da0242b0762a2a5576d6324b0584af35a5d9"
        expected_after_mla_sha="669fa7e02bb4b53b1171a48a29d97f1b3c424d80bcc516efff41f6c048cfc466"
        ;;
    *)
        echo "Error: unsupported Kimi-K3 vLLM overlay '$variant'" >&2
        exit 1
        ;;
esac

site_packages="$(python3 - <<'PY'
import sysconfig

print(sysconfig.get_paths()["purelib"])
PY
)"
vllm_root="$site_packages/vllm"
linear_file="$vllm_root/models/kimi_k3/amd/linear.py"
mla_file="$vllm_root/models/kimi_k3/amd/mla.py"

if [[ ! -f "$linear_file" ]]; then
    echo "Error: installed vLLM source is missing $linear_file" >&2
    exit 1
fi

actual_patch_sha="$(sha256sum "$patch_file" | awk '{print $1}')"
if [[ "$actual_patch_sha" != "$expected_patch_sha" ]]; then
    echo "Error: overlay patch checksum mismatch: got $actual_patch_sha" >&2
    exit 1
fi

actual_linear_sha="$(sha256sum "$linear_file" | awk '{print $1}')"
if [[ "$actual_linear_sha" == "$expected_after_linear_sha" ]]; then
    actual_mla_sha="$(sha256sum "$mla_file" 2>/dev/null | awk '{print $1}')"
    if [[ "$actual_mla_sha" == "$expected_after_mla_sha" ]]; then
        echo "Kimi-K3 vLLM overlay '$variant' is already applied"
        exit 0
    fi
fi

if [[ "$actual_linear_sha" != "$expected_before_linear_sha" ]]; then
    echo "Error: vLLM baseline source mismatch for $linear_file" >&2
    echo "Expected $expected_before_linear_sha, got $actual_linear_sha" >&2
    exit 1
fi
if [[ -e "$mla_file" ]]; then
    echo "Error: unexpected pre-existing AMD Kimi-K3 MLA wrapper: $mla_file" >&2
    exit 1
fi

(
    cd "$site_packages"
    patch --dry-run --batch --forward -p1 < "$patch_file"
    patch --batch --forward -p1 < "$patch_file"
)

actual_linear_sha="$(sha256sum "$linear_file" | awk '{print $1}')"
actual_mla_sha="$(sha256sum "$mla_file" | awk '{print $1}')"
if [[ "$actual_linear_sha" != "$expected_after_linear_sha" ]] || \
        [[ "$actual_mla_sha" != "$expected_after_mla_sha" ]]; then
    echo "Error: vLLM overlay '$variant' did not produce the pinned source" >&2
    echo "linear.py: $actual_linear_sha" >&2
    echo "mla.py: $actual_mla_sha" >&2
    exit 1
fi

echo "Applied Kimi-K3 vLLM overlay '$variant' to $site_packages"
