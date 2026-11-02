#!/usr/bin/env bash
set -eo pipefail
source /infmax-workspace/benchmarks/check_env.sh
check_env_vars DYNAMO_VENV_ROOT DYNAMO_VERSION DYNAMO_ENGINE_GRPC_VERSION SLURM_JOB_ID MORI_SOURCE_COMMIT MORI_BUILD_TIMEOUT_SECONDS
venv_path="${DYNAMO_VENV_ROOT}/${SLURM_JOB_ID}"
mkdir -p "$DYNAMO_VENV_ROOT" "$venv_path/home"
(
    flock -x 200
    if [[ ! -f "$venv_path/.mori-source-complete" ]]; then
        python3 -m venv --system-site-packages "$venv_path"
        log="/logs/mori-build-$(hostname -s).log"
        timeout "$MORI_BUILD_TIMEOUT_SECONDS" bash /configs/k3-mori810-build.sh "$venv_path" > "$log" 2>&1 || { tail -80 "$log"; exit 1; }
        printf '%s\n' "$MORI_SOURCE_COMMIT" > "$venv_path/.mori-source-complete"
    fi
    [[ $(cat "$venv_path/.mori-source-complete") == "$MORI_SOURCE_COMMIT" ]]
) 200>"${venv_path}.lock"
"$venv_path/bin/python" /configs/k3-mori810-verify.py "$venv_path"
