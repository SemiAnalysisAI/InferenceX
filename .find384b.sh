#!/bin/bash
# Report only the tuned-fmoe tables that actually carry inter_dim=384 rows.
find /usr/local/lib/python3.12/dist-packages/aiter/configs -name '*fmoe*' | sort | while read -r f; do
  n=$(awk -F, '$4==384 && $5==384' "$f" | wc -l)
  [ "$n" -gt 0 ] || continue
  t=$(awk -F, '$4==384 && $5==384' "$f" | grep -c 'x256x256')
  echo "HIT $f rows384=$n tile256=$t"
done
echo "--- dsv4-named tables ---"
find /usr/local/lib/python3.12/dist-packages/aiter/configs -name '*dsv4*' | sort
