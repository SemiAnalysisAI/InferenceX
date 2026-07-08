#!/usr/bin/env bash
# Shell bridge to the validated runtime registry in configs/runners.yaml.

runner_config_python() {
    local python_bin="${INFERENCEX_RUNNER_PYTHON:-python3}"
    if ! command -v "$python_bin" >/dev/null 2>&1; then
        echo "Error: runner registry Python not found: $python_bin" >&2
        return 1
    fi
    if ! "$python_bin" -c 'import yaml' >/dev/null 2>&1; then
        echo "Error: $python_bin cannot import PyYAML; run the multi-node dependency bootstrap or set INFERENCEX_RUNNER_PYTHON" >&2
        return 1
    fi
    printf '%s\n' "$python_bin"
}

load_runner_paths() {
    local cluster="$1"
    local python_bin
    python_bin="$(runner_config_python)" || return 1
    local exports
    if ! exports="$(
        "$python_bin" "$GITHUB_WORKSPACE/utils/runner_config.py" \
            --config "$GITHUB_WORKSPACE/configs/runners.yaml" \
            paths \
            --cluster "$cluster" \
            --workspace "$GITHUB_WORKSPACE" \
            --shell
    )"; then
        return 1
    fi
    eval "$exports"
}

load_runner_model() {
    local cluster="$1"
    local python_bin
    python_bin="$(runner_config_python)" || return 1
    local exports
    if ! exports="$(
        "$python_bin" "$GITHUB_WORKSPACE/utils/runner_config.py" \
            --config "$GITHUB_WORKSPACE/configs/runners.yaml" \
            model \
            --cluster "$cluster" \
            --model-prefix "$MODEL_PREFIX" \
            --precision "$PRECISION" \
            --framework "$FRAMEWORK" \
            --model "$MODEL" \
            --shell
    )"; then
        return 1
    fi
    eval "$exports"
    export MODEL_PATH MODEL_PATH_LAYOUT SRT_SLURM_MODEL_PREFIX SERVED_MODEL_NAME
    load_runner_paths "$cluster"
}
