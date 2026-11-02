#!/usr/bin/env bash
set -eo pipefail
bash /configs/k3-mori810-prepare.sh
source /infmax-workspace/benchmarks/check_env.sh
check_env_vars DYNAMO_VENV_ROOT SLURM_JOB_ID
server=$("${DYNAMO_VENV_ROOT}/${SLURM_JOB_ID}/bin/python" -c 'import pathlib,mori; print(pathlib.Path(mori.__file__).parent / "umbp_master")')
exec "$server" 0.0.0.0:25190
