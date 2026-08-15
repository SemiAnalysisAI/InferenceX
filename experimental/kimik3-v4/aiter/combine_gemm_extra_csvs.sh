#!/usr/bin/env bash
# Merge offline M614 + M6144 tune rows into one extra CSV for smoke/merge.
set -euo pipefail

REPO="${REPO:-$HOME/InferenceX}"
AITER_DIR="${REPO}/experimental/kimik3-v4/aiter"
M614="${M614:-${AITER_DIR}/kimik3_bf16_tuned_gemm.m614.csv}"
M6144="${M6144:-${AITER_DIR}/kimik3_bf16_tuned_gemm.m6144.csv}"
N6288="${N6288:-${AITER_DIR}/kimik3_bf16_tuned_gemm.n6288.csv}"
SMALL_M="${SMALL_M:-${AITER_DIR}/kimik3_bf16_tuned_gemm.cudagraph_small_m.csv}"
BULK="${BULK:-${AITER_DIR}/kimik3_bf16_tuned_gemm.extra.csv}"
OUT="${OUT:-${AITER_DIR}/kimik3_bf16_tuned_gemm.combined.csv}"
HEADER="gfx,cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,libtype,solidx,splitK,us,kernelName,err_ratio,tflops,bw"

inputs=()
for f in "$M614" "$M6144" "$N6288" "$SMALL_M" "$BULK"; do
  [[ -f "$f" ]] && inputs+=("$f")
done
[[ ${#inputs[@]} -gt 0 ]] || { echo "No tune CSV inputs found under ${AITER_DIR}" >&2; exit 1; }

{
  echo "$HEADER"
  for f in "${inputs[@]}"; do
    awk -F, 'NR>1 && NF>=10 && $1!~/^#/{print}' "$f"
  done
} | awk -F, '
  NF >= 10 && $1 !~ /^#/ {
    key = $3 "," $4 "," $5
    rows[key] = $0
  }
  END {
    for (k in rows) print rows[k]
  }
' > "$OUT"

rows=$(($(wc -l < "$OUT") - 1))
echo "Wrote ${OUT} (${rows} unique rows from ${#inputs[@]} input file(s), deduped by M,N,K)"
