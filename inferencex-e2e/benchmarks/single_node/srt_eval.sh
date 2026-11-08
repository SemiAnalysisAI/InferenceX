#!/usr/bin/env bash

# srt-slurm single-node post_eval: srt_eval.sh ENDPOINT STATUS_FILE. The artifacts land in this
# checkout; srt-slurm ignores a failed eval, so its exit code goes to STATUS_FILE for the launcher.
set -eo pipefail
if [[ $# != 2 || -z "$1" || -z "$2" ]]; then
    echo "Usage: $0 endpoint status-file" >&2
    exit 1
fi
SRT_EVAL_STATUS_FILE="$2"
trap 'rc=$?; printf "%s\n" "$rc" > "$SRT_EVAL_STATUS_FILE"' EXIT

# An agentic eval's acceptance gate ran as this job's benchmark stage, which srt-slurm runs only
# without EVAL_ONLY; the eval itself is still eval-only.
if [[ "${EVAL_AL_GATE:-}" == true ]]; then
    export EVAL_ONLY=true RUN_EVAL=false
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$root/benchmarks/check_env.sh"
check_env_vars MODEL MODEL_NAME CONC TP EP_SIZE DP_ATTENTION IS_MULTINODE IS_AGENTIC
# Fixed-sequence evals fit the benchmark context; AgentX evaluates at the native context.
if [[ "$IS_AGENTIC" != 1 ]]; then
    check_env_vars MAX_MODEL_LEN
fi
if [[ "$IS_MULTINODE" != false ]]; then
    echo "ERROR: single-node eval requires IS_MULTINODE=false" >&2
    exit 1
fi
# srt-slurm mounts the served checkpoint here; the workflow's MODEL_PATH is a host path.
if [[ -d /model ]]; then
    export MODEL_PATH=/model
fi
cd "$root"
PYTHONSAFEPATH=1 PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m infx.bench eval --endpoint "$1" --concurrency "$CONC" --stage-to "$root"
