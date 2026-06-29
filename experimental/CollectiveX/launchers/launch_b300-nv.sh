#!/usr/bin/env bash
# CollectiveX — B300 (b300-nv GH runner) adapter. The self-hosted runner is named
# `b300-nv_NN`, so runner.name's prefix resolves to this file via
# launch_${RUNNER_NAME%%_*}.sh. Identical B300 settings to launch_b300.sh (the
# canonical/manual entry point) — delegate so there is a single source of truth.
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/launch_b300.sh" "$@"
