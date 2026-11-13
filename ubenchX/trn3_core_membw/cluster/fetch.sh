#!/bin/bash
# Pull results (small files only) from the three work nodes into ubenchx/data/raw/<node>/.
set -e
CTL=${TRN3_CTL:?set TRN3_CTL=user@control-node}
HERE=$(cd "$(dirname "$0")/.." && pwd)
for NODE in ${TRN3_NODES:?set TRN3_NODES="ip ip ..." (the work nodes)}; do
  ssh -o BatchMode=yes $CTL "mkdir -p ~/trn3ub-stage/raw/$NODE && rsync -a --prune-empty-dirs \
    --include='*/' --include='result.json' --include='physical_cores.txt' --include='device.json' --include='meta.json' --include='hop_ns.npy' \
    --include='runs/distance/**.json' --include='runs/nccom/**.json' --include='runs/xcoll/**/result.json' --include='runs/nccom/**.txt' --include='runs/nccom/**.err*' --include='runs/torch/*.json' --include='logs/*.log' \
    --exclude='*' $NODE:/opt/dlami/nvme/trn3ub/ ~/trn3ub-stage/raw/$NODE/" 2>/dev/null
  mkdir -p "$HERE/data/raw/$NODE"
  rsync -a "$CTL:trn3ub-stage/raw/$NODE/" "$HERE/data/raw/$NODE/" 2>/dev/null
done
du -sh "$HERE/data/raw"/*
