#!/usr/bin/env bash
# 'tgemm' as a bare string appears in exactly 1 file on all three sides, which
# means it is not the identifier the DSv4 arm actually uses. Widen the search:
# check aiter's own version, the vendor's aiter provenance, and the plausible
# tuned-GEMM entry points by name.
set -u
D="${1:-/home/jiacao/3way-20260812-2214}"

for side in ref vendor target; do
    A=$(find "$D/$side" -maxdepth 9 -type d -name aiter -path '*dist-packages*' 2>/dev/null | head -1)
    echo "=== $side  ${A:-<absent>}"
    [ -z "$A" ] && continue

    echo -n "    aiter version: "
    grep -hE "^__version__|^version" "$A/_version.py" 2>/dev/null | head -2 | tr '\n' ' '; echo

    echo "    -- tgemm hits (any case, with context) --"
    grep -rn "tgemm" "$A" --include='*.py' 2>/dev/null | head -5 | sed 's|'"$A"'|.|' | sed 's/^/       /'

    echo "    -- tuned-GEMM surface --"
    for pat in "tuned_gemm" "TunedGemm" "tuned_gemm_dsv4" "gemm_a8w8_blockscale" "gemm_tune"; do
        n=$(grep -rl -- "$pat" "$A" --include='*.py' 2>/dev/null | wc -l)
        printf "       %-28s %s file(s)\n" "$pat" "$n"
    done

    echo "    -- tuned-GEMM csv configs --"
    ls "$A/configs/model_configs/" 2>/dev/null | grep -iE "gemm|dsv4" | head -6 | sed 's/^/       /'
    echo -n "       total csv: "; ls "$A/configs/model_configs/"*.csv 2>/dev/null | wc -l
done
