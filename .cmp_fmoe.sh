#!/bin/bash
# Inspect aiter fused_moe tuned tables for the inter_dim=384 / tile_n=256 mismatch.
for f in /usr/local/lib/python3.12/dist-packages/aiter/configs/tuned_fmoe.csv \
         /usr/local/lib/python3.12/dist-packages/aiter/configs/model_configs/dsv4_tuned_fmoe.csv \
         /tmp/aiter_configs/tuned_fmoe.csv; do
  if [ -f "$f" ]; then
    n384=$(awk -F, '$4==384 && $5==384' "$f" | wc -l)
    n256=$(awk -F, '$4==384 && $5==384' "$f" | grep -c 'x256x256')
    echo "$f  rows384=$n384  naming_tile256=$n256"
  else
    echo "$f  MISSING"
  fi
done
echo "=== aiter version ==="
python -c "import aiter; print(getattr(aiter,'__version__','?'), aiter.__file__)" 2>&1 | tail -1
