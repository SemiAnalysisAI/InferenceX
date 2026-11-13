#!/bin/bash
# Push ubenchx/ to the trn3 work node through the control node (the work node only trusts the
# control node's key). Usage: cluster/sync.sh [node-ip]   (default: UltraServer node 0)
set -e
CTL=${TRN3_CTL:?set TRN3_CTL=user@control-node}
NODE=${1:-${TRN3_NODE0:?set TRN3_NODE0 or pass a node ip}}
HERE=$(cd "$(dirname "$0")/.." && pwd)
rsync -a --delete --exclude data --exclude figures --exclude __pycache__ "$HERE/" "$CTL:trn3ub-stage/ubenchx/" 2>/dev/null
ssh -o BatchMode=yes $CTL "rsync -a --delete ~/trn3ub-stage/ubenchx/ $NODE:/opt/dlami/nvme/trn3ub/src/" 2>/dev/null
