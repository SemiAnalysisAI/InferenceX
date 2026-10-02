#!/usr/bin/env bash
set -eo pipefail

apply_verified_patch() {
    local package_root="$1" patch_file="$2" checksum="$3"
    printf '%s  %s\n' "$checksum" "$patch_file" | sha256sum --check --status
    git -C "$package_root" apply --check --include='vllm/*' "$patch_file"
    git -C "$package_root" apply --include='vllm/*' "$patch_file"
}

main() {
    # Native srt-slurm runs this preamble in both worker and router containers.
    # Only the standalone router image is exempt; a worker with missing vLLM must fail.
    if command -v vllm-router >/dev/null 2>&1 && ! command -v vllm >/dev/null 2>&1; then
        echo 'Standalone vLLM Router image: no engine backport required'
        exit 0
    fi

    # Keep the pinned #59164 backport plus the synthetic-only
    # compatibility experiment in the installed wheel; no compiled replacement.
    package_root=$(python3 -c 'import importlib.util; from pathlib import Path; spec = importlib.util.find_spec("vllm"); assert spec and spec.submodule_search_locations, "vLLM package not found"; print(Path(next(iter(spec.submodule_search_locations))).parent)')
    # Pin the latest reviewed #59164 head; its runtime diff is equivalent to
    # the previously validated d5e6faa9 extraction.
    zeroing_patch="$(dirname -- "${BASH_SOURCE[0]}")/k3-moriio-sync-zeroing.patch"
    apply_verified_patch "$package_root" "$zeroing_patch" 3da3746e85d53e4a3b17062b4113475d31a86cc07418e87ef5b4bb0c906cad20
    echo 'Applied vLLM#59164 at c5b1350f1f2bf10a128127a9b85e93d5f9f18e62'
    synthetic_patch="$(dirname -- "${BASH_SOURCE[0]}")/k3-synthetic-unproposed-drafts.patch"
    apply_verified_patch "$package_root" "$synthetic_patch" 33a6f792b13d39705a50562ca037a1d3c49dc054a8bdd8539fd8a154667f39df
    echo 'Applied synthetic-only pre-#58784 draft gathering; real verification retains placeholder rejection'
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
