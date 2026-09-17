#!/usr/bin/env bash

set -eo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <node-local-log-dir> <shared-log-dir>" >&2
    exit 2
fi

SOURCE_LOGS=$1
SHARED_LOGS=$2

if [[ ! -d "$SOURCE_LOGS" ]]; then
    echo "[logs][ERROR] no node-local logs found on $(hostname): $SOURCE_LOGS" >&2
    exit 1
fi

# Server containers create the source tree as root, and node 0 may have already
# created the shared destination as root. The Slurm nodes provide passwordless
# sudo for the same Docker lifecycle used by job.slurm.
sudo mkdir -p "$SHARED_LOGS"
sudo cp -r "$SOURCE_LOGS"/. "$SHARED_LOGS"/
echo "[logs] staged $(hostname):$SOURCE_LOGS -> $SHARED_LOGS"
