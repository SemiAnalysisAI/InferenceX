#!/usr/bin/env bash

# Shared srt-slurm checkout setup for multinode launchers.

SRT_SLURM_INFERENCEX_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 || exit 1
    pwd
)"
SRT_SLURM_VERSION_FILE="${SRT_SLURM_INFERENCEX_ROOT}/configs/srt-slurm-version.txt"
SRT_SLURM_REPOSITORY="https://github.com/NVIDIA/srt-slurm.git"

if [[ ! -r "$SRT_SLURM_VERSION_FILE" ]]; then
    echo "Unable to read the srt-slurm version from $SRT_SLURM_VERSION_FILE" >&2
    return 1
fi

read -r SRT_SLURM_VERSION < "$SRT_SLURM_VERSION_FILE"
if [[ -z "$SRT_SLURM_VERSION" ]]; then
    echo "The srt-slurm version file is empty: $SRT_SLURM_VERSION_FILE" >&2
    return 1
fi

prepare_srt_slurm_checkout() {
    local destination=$1
    local overlay_root="${SRT_SLURM_INFERENCEX_ROOT}/benchmarks/multi_node/srt-slurm-recipes"
    local source_dir
    local source_name
    local target_dir

    if [[ -z "$destination" || "$destination" == "/" ]]; then
        echo "Refusing to prepare an invalid srt-slurm destination: '$destination'" >&2
        return 1
    fi

    echo "Cloning srt-slurm ${SRT_SLURM_VERSION}..."
    rm -rf "$destination"
    git clone \
        --branch "$SRT_SLURM_VERSION" \
        --single-branch \
        "$SRT_SLURM_REPOSITORY" \
        "$destination" || return 1

    # Overlay all InferenceX-owned recipes and setup scripts. Keeping these in
    # InferenceX lets every launcher use the same tagged srt-slurm code while
    # preserving configurations that have not been released upstream.
    for source_dir in "$overlay_root"/*; do
        [[ -d "$source_dir" ]] || continue
        source_name=${source_dir##*/}
        if [[ "$source_name" == "configs" ]]; then
            target_dir="$destination/configs"
        else
            target_dir="$destination/recipes/$source_name"
        fi
        mkdir -p "$target_dir"
        cp -R "$source_dir/." "$target_dir/" || return 1
    done

    # The B200 TRT-LLM Kimi K2.5 configs predate the upstream recipe naming
    # convention used by CONFIG_FILE, so expose the checked-in recipes at the
    # expected path without maintaining a second copy.
    source_dir="$overlay_root/trtllm/kimi-k2.5/disagg/trtllm_dynamo/b200-fp4"
    if [[ -d "$source_dir" ]]; then
        target_dir="$destination/recipes/trtllm/kimi-k25-nvfp4/b200-fp4"
        mkdir -p "$target_dir"
        cp -R "$source_dir/." "$target_dir/" || return 1
    fi
}
