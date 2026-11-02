#!/usr/bin/env bash
set -eo pipefail
bash /configs/k3-mori810-prepare.sh
bash /configs/k3-mori810-pool.sh
source /infmax-workspace/benchmarks/check_env.sh
check_env_vars DYNAMO_VENV_ROOT SLURM_JOB_ID
"${DYNAMO_VENV_ROOT}/${SLURM_JOB_ID}/bin/python" -c 'import sys; from dynamo.frontend.main import parse_args; sys.argv = ["dynamo.frontend", "--dyn-chat-processor", "dynamo", "--router-mode", "least-loaded", "--router-session-affinity-ttl-secs", "900"]; cfg, vllm_flags, sglang_flags = parse_args(); assert cfg.chat_processor == "dynamo" and vllm_flags is None and sglang_flags is None'
