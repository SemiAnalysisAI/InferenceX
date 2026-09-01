#!/usr/bin/env bash
set -euo pipefail

variant="${1:?usage: apply_vllm_overlay.sh <variant>}"
overlay_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$variant" in
    pr52494)
        patch_file="$overlay_dir/vllm_pr52494.patch"
        expected_patch_sha="9efc1feee0dd5a4adaa8ce0229aec71c9620572b8cc49f290c1adab3ba4f195e"
        target_relpaths=(
            "vllm/models/kimi_k3/amd/linear.py"
            "vllm/models/kimi_k3/amd/mla.py"
        )
        expected_before_shas=(
            "f06a6408231150c797e99d8ae6d91d2993d90d69a061230747217386bb7d9d15"
            "missing"
        )
        expected_after_shas=(
            "3d093e916016dbda0b3df8eadd19da0242b0762a2a5576d6324b0584af35a5d9"
            "669fa7e02bb4b53b1171a48a29d97f1b3c424d80bcc516efff41f6c048cfc466"
        )
        ;;
    tier1)
        patch_file="$overlay_dir/vllm_k3_tier1.patch"
        expected_patch_sha="d96268e8bf06c1572863ab18ac97c0e524ce1fa121fc5579a26926dd76a37434"
        target_relpaths=("vllm/models/kimi_k3/amd/latent_moe_runner.py")
        expected_before_shas=(
            "e6235f947b0a1d89327a7e11391a4d26c3f4fa57c1af804384acf55b0c1041ee"
        )
        expected_after_shas=(
            "29f057131bc4970cc44a0727a802cb0f29789da0e26629a222b5784aadaaea27"
        )
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

actual_patch_sha="$(sha256sum "$patch_file" | awk '{print $1}')"
if [[ "$actual_patch_sha" != "$expected_patch_sha" ]]; then
    echo "Error: overlay patch checksum mismatch: got $actual_patch_sha" >&2
    exit 1
fi

already_applied=1
for i in "${!target_relpaths[@]}"; do
    target="$site_packages/${target_relpaths[$i]}"
    if [[ ! -f "$target" ]] || \
            [[ "$(sha256sum "$target" | awk '{print $1}')" != "${expected_after_shas[$i]}" ]]; then
        already_applied=0
        break
    fi
done
if [[ "$already_applied" == "1" ]]; then
    echo "Kimi-K3 vLLM overlay '$variant' is already applied"
    exit 0
fi

for i in "${!target_relpaths[@]}"; do
    target="$site_packages/${target_relpaths[$i]}"
    expected="${expected_before_shas[$i]}"
    if [[ "$expected" == "missing" ]]; then
        if [[ -e "$target" ]]; then
            echo "Error: unexpected pre-existing vLLM source: $target" >&2
            exit 1
        fi
        continue
    fi
    if [[ ! -f "$target" ]]; then
        echo "Error: installed vLLM source is missing $target" >&2
        exit 1
    fi
    actual="$(sha256sum "$target" | awk '{print $1}')"
    if [[ "$actual" != "$expected" ]]; then
        echo "Error: vLLM baseline source mismatch for $target" >&2
        echo "Expected $expected, got $actual" >&2
        exit 1
    fi
done

(
    cd "$site_packages"
    patch --dry-run --batch --forward -p1 < "$patch_file"
    patch --batch --forward -p1 < "$patch_file"
)

for i in "${!target_relpaths[@]}"; do
    target="$site_packages/${target_relpaths[$i]}"
    if [[ ! -f "$target" ]]; then
        echo "Error: vLLM overlay '$variant' did not create $target" >&2
        exit 1
    fi
    actual="$(sha256sum "$target" | awk '{print $1}')"
    expected="${expected_after_shas[$i]}"
    if [[ "$actual" != "$expected" ]]; then
        echo "Error: vLLM overlay '$variant' produced the wrong source for $target" >&2
        echo "Expected $expected, got $actual" >&2
        exit 1
    fi
done

echo "Applied Kimi-K3 vLLM overlay '$variant' to $site_packages"
