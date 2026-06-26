#!/usr/bin/env bash
# The H200 GHA self-hosted runner is named h200-dgxc-slurm_NN, so the workflow's
# launch_${RUNNER_NAME%%_*}.sh convention resolves to THIS name. Thin alias to the real
# H200 adapter (launch_h200.sh) — no logic here, just the name the runner expects.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/launch_h200.sh" "$@"
