#!/usr/bin/env bash
# Canonical srt-slurm checkout, recipe overlay, and InferenceX benchmark bridge.

source "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE must be set}/runners/lib/runner_config.sh"
source "$GITHUB_WORKSPACE/runners/lib/slurm.sh"

load_srt_slurm_settings() {
    local python_bin
    python_bin="$(runner_config_python)" || return 1
    local exports
    if ! exports="$(
        "$python_bin" "$GITHUB_WORKSPACE/utils/runner_config.py" \
            --config "$GITHUB_WORKSPACE/configs/runners.yaml" \
            srt --shell
    )"; then
        return 1
    fi
    eval "$exports"
    export SRT_SLURM_REPOSITORY SRT_SLURM_BRANCH SRT_SLURM_REVISION
    export SRT_SLURM_LEGACY_RECIPES_REVISION
}

apply_srt_compatibility_patches() {
    local destination="$1"
    local patch
    for patch in "$GITHUB_WORKSPACE"/runners/srt-slurm-patches/*.patch; do
        [[ -f "$patch" ]] || continue
        git -C "$destination" apply --unidiff-zero --check "$patch" || return 1
        git -C "$destination" apply --unidiff-zero "$patch" || return 1
    done
}

clone_srt_slurm() {
    local destination="$1"
    load_srt_slurm_settings || return 1
    git clone \
        --branch "$SRT_SLURM_BRANCH" \
        --single-branch \
        "$SRT_SLURM_REPOSITORY" \
        "$destination" || return 1
    git -C "$destination" checkout --detach "$SRT_SLURM_REVISION" || return 1
    local actual_revision
    actual_revision="$(git -C "$destination" rev-parse HEAD)"
    if [[ "$actual_revision" != "$SRT_SLURM_REVISION" ]]; then
        echo "Error: expected srt-slurm $SRT_SLURM_REVISION, got $actual_revision" >&2
        return 1
    fi
    apply_srt_compatibility_patches "$destination" || return 1
}

stage_srt_recipe() {
    local config_arg="$1"
    if [[ -z "$config_arg" ]]; then
        echo "Error: CONFIG_FILE is required for an srt-slurm benchmark" >&2
        return 1
    fi
    local relative_path="${config_arg%%:*}"
    local selector=""
    if [[ "$config_arg" == *:* ]]; then
        selector=":${config_arg#*:}"
    fi
    if [[ "$relative_path" == /* || "$relative_path" != recipes/* ]]; then
        echo "Error: CONFIG_FILE must be relative to recipes/: $config_arg" >&2
        return 1
    fi
    case "/$relative_path/" in
        */../*|*/./*)
            echo "Error: CONFIG_FILE cannot contain path traversal: $config_arg" >&2
            return 1
            ;;
    esac

    local local_recipe="$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/${relative_path#recipes/}"
    if [[ -f "$local_recipe" ]]; then
        mkdir -p "$(dirname "$relative_path")" || return 1
        cp "$local_recipe" "$relative_path" || return 1
        echo "Using InferenceX recipe overlay: $relative_path" >&2
    elif [[ ! -f "$relative_path" ]]; then
        echo "Fetching data-only legacy recipe snapshot for $relative_path" >&2
        git fetch --depth 1 origin "$SRT_SLURM_LEGACY_RECIPES_REVISION" || return 1
        mkdir -p "$(dirname "$relative_path")" || return 1
        local staged_recipe="${relative_path}.tmp.$$"
        if ! git show \
            "$SRT_SLURM_LEGACY_RECIPES_REVISION:$relative_path" \
            > "$staged_recipe"; then
            rm -f "$staged_recipe"
            return 1
        fi
        mv "$staged_recipe" "$relative_path" || return 1
    fi

    if [[ ! -f "$relative_path" ]]; then
        echo "Error: srt-slurm recipe not found: $relative_path" >&2
        return 1
    fi
    printf '%s%s\n' "$relative_path" "$selector"
}

prepare_srt_benchmark() {
    local config_arg="$1"
    local output
    output=".inferencex-$(basename "${config_arg%%:*}")"
    local args=(
        "$config_arg"
        --output "$output"
    )
    if [[ -n "${ISL:-}" ]]; then
        args+=(--isl "$ISL")
    fi
    if [[ -n "${OSL:-}" ]]; then
        args+=(--osl "$OSL")
    fi
    if [[ -n "${CONC_LIST:-}" ]]; then
        args+=(--concurrencies "${CONC_LIST// /x}")
    fi
    if [[ -n "${RANDOM_RANGE_RATIO:-}" ]]; then
        args+=(--random-range-ratio "$RANDOM_RANGE_RATIO")
    fi
    if [[ -n "${MODEL_PATH:-}" ]]; then
        args+=(--default-served-model-name "$(basename "${MODEL_PATH%/}")")
    fi
    local python_bin
    python_bin="$(runner_config_python)" || return 1
    "$python_bin" "$GITHUB_WORKSPACE/utils/prepare_srt_config.py" "${args[@]}"
}
