#!/usr/bin/env bash
set -euo pipefail

action="${1:?usage: apply_vllm_metadata_reuse_overlay.sh <apply|restore> [mla|kda-specialized|kda-reuse]}"
stage="${2:-kda-reuse}"

candidate_repo="andyluo7/vllm"
candidate_base="1dc464d42681d22f38caf1fdc1eb632dc4421c45"

target_relpaths=(
    "vllm/models/kimi_k3/amd/kda.py"
    "vllm/models/kimi_k3/amd/kda_metadata.py"
    "vllm/models/kimi_k3/nvidia/kda_metadata.py"
    "vllm/v1/attention/backend.py"
    "vllm/v1/attention/backends/mla/rocm_aiter_mla.py"
    "vllm/v1/worker/gpu/attn_utils.py"
)
expected_before_shas=(
    "6b8ad0fd1ebe626245a35cf3be598883d8c543b0cd288c383ede40b4c82821a7"
    "f193a312333086aab7bd43fa68ecd4bf31b2ade95546e3482615deb896ad9ed6"
    "7bea662e79cd2e402a22568cc3aeb80276cde3b2fd271b3c839c196fa7f7ee60"
    "40b05381736376b130202bfc82d8028f81fec10e23970e579ddcf93e56a3fa5b"
    "d7c10346c5df7cc88732784f61e0c68f153bc5a60f0071df760fcef1780a0b6e"
    "55bb9f034bd91f09358ba405974a622b9fe5bfcc4834a6483f985a49a674915f"
)
case "$stage" in
    mla)
        candidate_commit="747d03051581638884d5454b2eba7719bcdd2f44"
        expected_after_shas=(
            "6b8ad0fd1ebe626245a35cf3be598883d8c543b0cd288c383ede40b4c82821a7"
            "f193a312333086aab7bd43fa68ecd4bf31b2ade95546e3482615deb896ad9ed6"
            "7bea662e79cd2e402a22568cc3aeb80276cde3b2fd271b3c839c196fa7f7ee60"
            "40b05381736376b130202bfc82d8028f81fec10e23970e579ddcf93e56a3fa5b"
            "d760dc97587a13ac24657e6d7af1ad8af73170bad31848f84cdd75cb0466ee26"
            "691b8e552dcff7f935dad2d646c4a05a092c6b99965630b522131530bd4c545f"
        )
        ;;
    kda-specialized)
        candidate_commit="5c76ec1906d86352e0dcb58333d2775493595584"
        expected_after_shas=(
            "4ba1f79a695ea65b49aa040d497ea3da3a6a84832bd31baf90fde6b165dd2455"
            "682943380aef388a6618dd76d73bbc465b2a2acdd4ac5d966fa49e040a4b60b1"
            "e9d41934e73ae64cb926f64808513d46be5e76075606fe3fde7f7fcf9a72d32c"
            "40b05381736376b130202bfc82d8028f81fec10e23970e579ddcf93e56a3fa5b"
            "d760dc97587a13ac24657e6d7af1ad8af73170bad31848f84cdd75cb0466ee26"
            "691b8e552dcff7f935dad2d646c4a05a092c6b99965630b522131530bd4c545f"
        )
        ;;
    kda-reuse)
        candidate_commit="bbb59bf5a529a863e4ebe3fff6abb01027ffddcf"
        expected_after_shas=(
            "4ba1f79a695ea65b49aa040d497ea3da3a6a84832bd31baf90fde6b165dd2455"
            "682943380aef388a6618dd76d73bbc465b2a2acdd4ac5d966fa49e040a4b60b1"
            "62a9bcc30c9c17b9a824e1a9566292a444378eeb98e41deeea1bdf6589513915"
            "f993da614df3680f0ff5872a40b4d9a36d4337569e27e8759b06c281b11ae979"
            "f765ad868abf432f7737030aa4250ff13212bf5f8252ebf6c3c5073b119253fc"
            "e121c870ff447edf15bc63f94f12e49b0e41e348465a6e577ecee85e0f13b9b2"
        )
        ;;
    *)
        echo "Error: unsupported metadata-reuse stage '$stage'" >&2
        exit 1
        ;;
esac

site_packages="$(python3 - <<'PY'
import sysconfig

print(sysconfig.get_paths()["purelib"])
PY
)"
vllm_root="$site_packages/vllm"
cache_root="${K3_ARM_CACHE_ROOT:?K3_ARM_CACHE_ROOT must be set}"
result_dir="${RESULT_DIR:?RESULT_DIR must be set}"
download_root="$cache_root/vllm_metadata_reuse_candidate/$stage"
backup_root="$cache_root/vllm_metadata_reuse_backup/$stage"
provenance_file="$result_dir/vllm_metadata_reuse_overlay_provenance.tsv"
restore_file="$result_dir/vllm_metadata_reuse_overlay_restore.tsv"

file_sha() {
    sha256sum "$1" | awk '{print $1}'
}

verify_baseline_or_fail() {
    local i target actual
    for i in "${!target_relpaths[@]}"; do
        target="$site_packages/${target_relpaths[$i]}"
        if [[ ! -f "$target" ]]; then
            echo "Error: installed vLLM source is missing $target" >&2
            return 1
        fi
        actual="$(file_sha "$target")"
        if [[ "$actual" != "${expected_before_shas[$i]}" ]]; then
            echo "Error: vLLM baseline source mismatch for $target" >&2
            echo "Expected ${expected_before_shas[$i]}, got $actual" >&2
            return 1
        fi
    done
}

restore_overlay() {
    local i target backup actual

    if [[ ! -d "$backup_root" ]]; then
        verify_baseline_or_fail
        printf 'status\tnot-applied\nbase\t%s\n' "$candidate_base" \
            >"$restore_file"
        return 0
    fi

    for i in "${!target_relpaths[@]}"; do
        target="$site_packages/${target_relpaths[$i]}"
        backup="$backup_root/${target_relpaths[$i]}"
        if [[ ! -f "$backup" ]]; then
            echo "Error: $stage metadata backup is missing $backup" >&2
            return 1
        fi
        actual="$(file_sha "$backup")"
        if [[ "$actual" != "${expected_before_shas[$i]}" ]]; then
            echo "Error: $stage metadata backup checksum mismatch for $backup" >&2
            return 1
        fi
    done

    for i in "${!target_relpaths[@]}"; do
        target="$site_packages/${target_relpaths[$i]}"
        backup="$backup_root/${target_relpaths[$i]}"
        cp -p "$backup" "$target"
        actual="$(file_sha "$target")"
        if [[ "$actual" != "${expected_before_shas[$i]}" ]]; then
            echo "Error: failed to restore baseline source for $target" >&2
            return 1
        fi
    done

    printf 'status\trestored\nstage\t%s\nbase\t%s\ncandidate\t%s\n' \
        "$stage" "$candidate_base" "$candidate_commit" >"$restore_file"
    echo "Restored vLLM $stage metadata overlay to base $candidate_base"
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

verify_baseline_or_fail
mkdir -p "$download_root" "$backup_root" "$result_dir"

for i in "${!target_relpaths[@]}"; do
    relpath="${target_relpaths[$i]}"
    candidate="$download_root/$relpath"
    mkdir -p "$(dirname "$candidate")"
    curl --fail --location --retry 3 --retry-all-errors \
        "https://raw.githubusercontent.com/$candidate_repo/$candidate_commit/$relpath" \
        --output "$candidate"
    actual="$(file_sha "$candidate")"
    if [[ "$actual" != "${expected_after_shas[$i]}" ]]; then
        echo "Error: downloaded candidate checksum mismatch for $relpath" >&2
        echo "Expected ${expected_after_shas[$i]}, got $actual" >&2
        exit 1
    fi
done

for i in "${!target_relpaths[@]}"; do
    relpath="${target_relpaths[$i]}"
    target="$site_packages/$relpath"
    backup="$backup_root/$relpath"
    mkdir -p "$(dirname "$backup")"
    cp -p "$target" "$backup"
done

applied=0
rollback_on_error() {
    local rc=$?
    if ((rc != 0 && applied != 0)); then
        restore_overlay || true
    fi
    exit "$rc"
}
trap rollback_on_error EXIT

applied=1
for i in "${!target_relpaths[@]}"; do
    relpath="${target_relpaths[$i]}"
    target="$site_packages/$relpath"
    candidate="$download_root/$relpath"
    cp -p "$candidate" "$target"
    actual="$(file_sha "$target")"
    if [[ "$actual" != "${expected_after_shas[$i]}" ]]; then
        echo "Error: failed to install metadata-reuse candidate for $target" >&2
        exit 1
    fi
done

python3 -m compileall -q \
    "$vllm_root/models/kimi_k3/amd/kda.py" \
    "$vllm_root/models/kimi_k3/amd/kda_metadata.py" \
    "$vllm_root/models/kimi_k3/nvidia/kda_metadata.py" \
    "$vllm_root/v1/attention/backend.py" \
    "$vllm_root/v1/attention/backends/mla/rocm_aiter_mla.py" \
    "$vllm_root/v1/worker/gpu/attn_utils.py"

K3_METADATA_STAGE="$stage" python3 - <<'PY'
import os

from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAMetadataBuilder

stage = os.environ["K3_METADATA_STAGE"]
assert AiterMLAMetadataBuilder.supports_metadata_reuse
assert "paged_kv_indptr" in AiterMLAMetadataBuilder.reusable_metadata_buffers

if stage in {"kda-specialized", "kda-reuse"}:
    from vllm.models.kimi_k3.amd.kda_metadata import (
        KimiK3ROCmKDAMetadataBuilder,
    )
    from vllm.models.kimi_k3.nvidia.kda_metadata import (
        KimiK3KDAMetadataBuilder,
    )

    assert issubclass(KimiK3ROCmKDAMetadataBuilder, KimiK3KDAMetadataBuilder)
    if stage == "kda-reuse":
        assert KimiK3KDAMetadataBuilder.supports_metadata_reuse
        assert (
            "spec_query_start_loc"
            in KimiK3KDAMetadataBuilder.reusable_metadata_buffers
        )

print(f"Kimi-K3 {stage} metadata import contract passed")
PY

{
    printf 'status\tapplied\n'
    printf 'stage\t%s\n' "$stage"
    printf 'repo\t%s\n' "$candidate_repo"
    printf 'base\t%s\n' "$candidate_base"
    printf 'candidate\t%s\n' "$candidate_commit"
    printf 'site_packages\t%s\n' "$site_packages"
    for i in "${!target_relpaths[@]}"; do
        printf 'file\t%s\t%s\t%s\n' \
            "${target_relpaths[$i]}" \
            "${expected_before_shas[$i]}" \
            "${expected_after_shas[$i]}"
    done
} >"$provenance_file"

trap - EXIT
echo "K3 $stage metadata overlay: commit $candidate_commit"
