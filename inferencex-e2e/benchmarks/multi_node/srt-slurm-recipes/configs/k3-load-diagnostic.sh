#!/usr/bin/env bash
set -eo pipefail
if [[ -z ${UMBP_ENDPOINT-} ]]; then
    exit 0
fi
source /infmax-workspace/benchmarks/check_env.sh
check_env_vars UMBP_ENDPOINT
diagnostic=/infmax-workspace/infx/bench/load_diagnostic.py
python3 "$diagnostic" snapshot --phase before-pool
python3 "$diagnostic" install
python3 -u "$diagnostic" watch --seconds 3300 > "/logs/load-watch-$(hostname -s).jsonl" 2>&1 < /dev/null &
bash /configs/k3-umbp-pool.sh
python3 "$diagnostic" snapshot --phase after-pool
