#!/bin/bash
D=/usr/local/lib/python3.12/dist-packages/aiter/ops/flydsl
echo "--- flydsl file md5s ---"
find "$D" -name '*.py' | sort | xargs md5sum 2>/dev/null | sed "s#$D/##"
echo "--- dsv4 tuned table, correct columns (gfx,cu,token,model_dim,inter_dim,expert) ---"
T=/usr/local/lib/python3.12/dist-packages/aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv
awk -F, 'NR>1{print $5}' "$T" | sort -u | tr '\n' ' '; echo
echo "rows inter_dim=384: $(awk -F, 'NR>1 && $5==384' "$T" | wc -l)"
echo "  of which naming x256x256: $(awk -F, 'NR>1 && $5==384' "$T" | grep -c 'x256x256')"
