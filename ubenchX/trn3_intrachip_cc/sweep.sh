#!/bin/bash
# Launch every intra-chip placement at once, each on its own chip in its own container (own /tmp).
#   ubenchX/trn3_intrachip_cc/sweep.sh [OUT_DIR]     (from the repo root, on the host)
cd "$(dirname "$0")/../.."
OUT=${1:-ubenchX/trn3_intrachip_cc/runs/sweep_r0}
PLACEMENTS=(lnc1_r2_same lnc1_r4_same2 lnc1_r4_cross2 lnc1_r4 lnc1_r8 lnc2_r2 lnc2_r4)
for i in "${!PLACEMENTS[@]}"; do
  ubenchX/trn3_dma_bw/run.sh ubenchX/trn3_intrachip_cc/run_placement.sh "${PLACEMENTS[$i]}" "$i" "$OUT" &
done
wait
