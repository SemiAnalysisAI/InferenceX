#!/usr/bin/env bash
# Which tuned-GEMM csv files does the vendor image have that the target nightly
# lacks, and which of those exist upstream on aiter/main?
#
# Tuned-GEMM configs are pure data. A missing csv does not crash -- it silently
# falls back to untuned dispatch, which is exactly the kind of regression that
# would show up as "the CI number is lower than the measured number" with no
# error in the log. So this list has to be complete, not spot-checked.
set -u
D=/home/jiacao/3way-20260812-2214
T=$(find "$D/target" -maxdepth 9 -type d -name aiter -path '*dist-packages*' | head -1)
V=$(find "$D/vendor" -maxdepth 9 -type d -name aiter -path '*dist-packages*' | head -1)

ls "$T/configs/model_configs/" > /tmp/csv_target.txt
ls "$V/configs/model_configs/" > /tmp/csv_vendor.txt

echo "target csv count: $(wc -l < /tmp/csv_target.txt)"
echo "vendor csv count: $(wc -l < /tmp/csv_vendor.txt)"
echo
echo "=== in VENDOR but not in TARGET (must be added) ==="
comm -13 /tmp/csv_target.txt /tmp/csv_vendor.txt | sed 's/^/  /'
echo
echo "=== in TARGET but not in VENDOR (target is newer here) ==="
comm -23 /tmp/csv_target.txt /tmp/csv_vendor.txt | sed 's/^/  /'
echo
echo "=== upstream availability of the vendor-only ones (aiter/main) ==="
comm -13 /tmp/csv_target.txt /tmp/csv_vendor.txt | while read -r f; do
    [ -z "$f" ] && continue
    s=$(gh api "/repos/ROCm/aiter/contents/aiter/configs/model_configs/$f?ref=main" --jq '.size' 2>/dev/null)
    if [[ "$s" =~ ^[0-9]+$ ]]; then printf "  %-9s %-9s %s\n" PRESENT "${s}B" "$f"
    else                            printf "  %-9s %-9s %s\n" ABSENT  "-"    "$f"; fi
done
echo
echo "=== dsv4 csvs: vendor line counts vs target ==="
for f in $(ls "$V/configs/model_configs/" | grep -i dsv4); do
    lv=$(wc -l < "$V/configs/model_configs/$f")
    if [ -f "$T/configs/model_configs/$f" ]; then
        lt=$(wc -l < "$T/configs/model_configs/$f")
    else
        lt="<absent>"
    fi
    printf "  %-52s vendor=%-6s target=%s\n" "$f" "$lv" "$lt"
done
