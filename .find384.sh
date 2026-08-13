#!/bin/bash
# Find which shipped aiter tuned-fmoe table contributes the inter_dim=384 rows.
echo "--- shipped csvs mentioning fmoe ---"
find /usr/local/lib/python3.12/dist-packages/aiter/configs -name '*fmoe*' | while read -r f; do
  n=$(awk -F, '$4==384 && $5==384' "$f" | wc -l)
  t=$(awk -F, '$4==384 && $5==384' "$f" | grep -c 'x256x256')
  echo "$(basename "$f")  rows384=$n  tile256=$t  total=$(wc -l < "$f")"
done
echo "--- /tmp/aiter_configs contents ---"
ls -la /tmp/aiter_configs 2>/dev/null | head -20
