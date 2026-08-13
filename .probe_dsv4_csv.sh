#!/usr/bin/env bash
# The imgdiff run flagged 'configs/model_configs/a8w8_blockscale_tuned_gemm_dsv4.csv'
# as +46 lines in the vendor tree. Tuned-GEMM configs are data, not code: if the
# dsv4 csv is vendor-only, the tuned shapes for this model simply do not exist
# in the nightly line and the GEMMs fall back to untuned dispatch.
set -u
D="${1:-/home/jiacao/3way-20260812-2214}"

for side in ref vendor target; do
    A=$(find "$D/$side" -maxdepth 9 -type d -name aiter -path '*dist-packages*' 2>/dev/null | head -1)
    echo "=== $side"
    [ -z "$A" ] && { echo "    <absent>"; continue; }
    echo "    -- dsv4 csv files --"
    ls "$A/configs/model_configs/" 2>/dev/null | grep -i dsv4 | sed 's/^/       /' || true
    n=$(ls "$A/configs/model_configs/" 2>/dev/null | grep -ci dsv4)
    echo "       count: $n"
    f="$A/configs/model_configs/a8w8_blockscale_tuned_gemm_dsv4.csv"
    if [ -f "$f" ]; then
        echo "       a8w8_blockscale_tuned_gemm_dsv4.csv: $(wc -l < "$f") lines"
    fi
    # gluon / sparse attention entry points, by name rather than by path
    echo "    -- gluon sparse attention --"
    for pat in "pa_decode_sparse" "_gluon_kernels" "mla_gluon"; do
        c=$(grep -rl -- "$pat" "$A" --include='*.py' 2>/dev/null | wc -l)
        printf "       %-20s %s file(s)\n" "$pat" "$c"
    done
done
