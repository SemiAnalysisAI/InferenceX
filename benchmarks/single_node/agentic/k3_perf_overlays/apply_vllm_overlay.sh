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
    compile52190)
        patch_file="$overlay_dir/vllm_k3_compile52190.patch"
        expected_patch_sha="164509dae2072ceb6260ffd9b5695fa2efc2a55a4ef12df083f5c110390741ff"
        target_relpaths=(
            "vllm/config/compilation.py"
            "vllm/models/kimi_k3/amd/kda.py"
            "vllm/models/kimi_k3/amd/latent_moe_runner.py"
            "vllm/models/kimi_k3/amd/linear.py"
            "vllm/models/kimi_k3/amd/ops/attn_res.py"
        )
        expected_before_shas=(
            "255c802c30fd1b116eb4bb477f816af70872fc3dd92b88477fe54b5f595da5e2"
            "6b8ad0fd1ebe626245a35cf3be598883d8c543b0cd288c383ede40b4c82821a7"
            "e6235f947b0a1d89327a7e11391a4d26c3f4fa57c1af804384acf55b0c1041ee"
            "f06a6408231150c797e99d8ae6d91d2993d90d69a061230747217386bb7d9d15"
            "ade95f1859ac26569f2833487490d672b3831364f7fe492153c005e181ff3a1e"
        )
        expected_after_shas=(
            "9d17df12823a62d345ff08fac118ef0b1d281ebbdb6cd9e8cc9ad41221196458"
            "5bcb8996181396cb019a95447089fc0f55adf3a545a4165b2b43700583b17a86"
            "f05c9a1d79b1e8697a5f66007a5c502a878d6c86fc9f884acae0deb5f9341f42"
            "326adb617d8772ce99fbcc894bb4e3e5687e24ad3212591eceda447cc23774b9"
            "0a779ad4ecfc6e2489edd67dd328b6ff064a574cf605803ede35f27d10b69278"
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
