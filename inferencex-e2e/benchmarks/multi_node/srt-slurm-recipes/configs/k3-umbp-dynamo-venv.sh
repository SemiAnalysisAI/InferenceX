#!/usr/bin/env bash
set -eo pipefail
source /infmax-workspace/benchmarks/check_env.sh
check_env_vars DYNAMO_VENV_ROOT DYNAMO_VERSION DYNAMO_ENGINE_GRPC_VERSION SLURM_JOB_ID
bash /configs/k3-umbp-pool.sh
venv_path="${DYNAMO_VENV_ROOT}/${SLURM_JOB_ID}"
mkdir -p "$DYNAMO_VENV_ROOT" "$venv_path/home"
(
    flock -x 200
    if [[ ! -f "$venv_path/.install-complete" ]]; then
        python3 -m venv --system-site-packages "$venv_path"
        "$venv_path/bin/python" -m pip install --disable-pip-version-check "ai-dynamo==${DYNAMO_VERSION}" "ai-dynamo-runtime==${DYNAMO_VERSION}" "grpcio==${DYNAMO_ENGINE_GRPC_VERSION}" "grpcio-tools==${DYNAMO_ENGINE_GRPC_VERSION}"
        "$venv_path/bin/python" -c 'from importlib.metadata import version; print({p: version(p) for p in ("ai-dynamo", "ai-dynamo-runtime", "vllm", "torch")})'
        touch "$venv_path/.install-complete"
    fi
) 200>"${venv_path}.lock"
"$venv_path/bin/python" -c 'import sys; from dynamo.frontend.main import parse_args; sys.argv = ["dynamo.frontend", "--dyn-chat-processor", "dynamo", "--router-mode", "least-loaded", "--router-session-affinity-ttl-secs", "900"]; cfg, vllm_flags, sglang_flags = parse_args(); assert cfg.chat_processor == "dynamo" and vllm_flags is None and sglang_flags is None; print("Dynamo Rust chat preflight passed")'
