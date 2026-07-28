#!/usr/bin/bash

# Resolve a launcher's cluster-local model environment from configs/runners.yaml.
resolve_runner_model_config() {
    local runner_label="$1"
    shift

    local runner_utils_dir repo_root resolved_model_env
    runner_utils_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(cd "$runner_utils_dir/.." && pwd)"

    if ! resolved_model_env=$(python3 "$repo_root/utils/resolve_runner_model.py" \
        --runner-config "$repo_root/configs/runners.yaml" \
        --runner-label "$runner_label" \
        --framework "$FRAMEWORK" \
        --model-prefix "$MODEL_PREFIX" \
        --precision "$PRECISION" \
        "$@"); then
        return 1
    fi

    eval "$resolved_model_env"
}
