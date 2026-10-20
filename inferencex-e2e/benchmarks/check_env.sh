#!/usr/bin/env bash

# Required-input validation for Bash callers. Sourcing this file only defines
# check_env_vars; it has no other side effects.

# Usage: check_env_vars VAR1 VAR2 ...; exits 1 listing every name that is unset or empty.
check_env_vars() {
    local missing_vars=()
    local var_name

    for var_name in "$@"; do
        if [[ -z "${!var_name:-}" ]]; then
            missing_vars+=("$var_name")
        fi
    done

    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        echo "Error: The following required environment variables are not set:"
        for var_name in "${missing_vars[@]}"; do
            echo "  - $var_name"
        done
        exit 1
    fi
}
