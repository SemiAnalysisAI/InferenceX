#!/usr/bin/env bash
# Merge aiter BF16 GEMM tuned CSVs into /tmp/aiter_configs/bf16_tuned_gemm.csv
# before vLLM workers start. Appends optional extra rows (offline tune output).
#
# Usage (inside ROCm container):
#   AITER_GEMM_EXTRA_CSV=/workspace/experimental/kimik3-v4/aiter/kimik3_bf16_tuned_gemm.extra.csv \
#     bash experimental/kimik3-v4/aiter/merge_aiter_gemm_configs.sh
#
set -euo pipefail

AITER_PKG="${AITER_PKG:-/usr/local/lib/python3.12/dist-packages/aiter}"
OUT_DIR="${AITER_CONFIG_DIR:-/tmp/aiter_configs}"
OUT_CSV="${OUT_DIR}/bf16_tuned_gemm.csv"
EXTRA_CSV="${AITER_GEMM_EXTRA_CSV:-}"

mkdir -p "$OUT_DIR"

HEADER="gfx,cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,libtype,solidx,splitK,us,kernelName,err_ratio,tflops,bw"

emit_csv_body() {
  local f="$1"
  [[ -f "$f" ]] || return 0
  awk -F, 'NR==1{next} NF>=10 && $1!~/^#/{print}' "$f"
}

{
  echo "$HEADER"
  emit_csv_body "${AITER_PKG}/configs/bf16_tuned_gemm.csv"
  for f in "${AITER_PKG}"/configs/model_configs/*_bf16_tuned_gemm.csv; do
    emit_csv_body "$f"
  done
  if [[ -n "$EXTRA_CSV" && -f "$EXTRA_CSV" ]]; then
    emit_csv_body "$EXTRA_CSV"
  fi
} | awk -F, '
  NR == 1 { print; next }
  NF >= 10 && $1 !~ /^#/ {
    key = $3 "," $4 "," $5
    rows[key] = $0
  }
  END {
    for (k in rows) print rows[k]
  }
' > "$OUT_CSV"

rows=$(($(wc -l < "$OUT_CSV") - 1))
export AITER_CONFIG_GEMM_BF16="$OUT_CSV"
echo "[merge_aiter_gemm_configs] wrote ${OUT_CSV} (${rows} tuned rows)"
if [[ -n "$EXTRA_CSV" && -f "$EXTRA_CSV" ]]; then
  extra_rows=$(awk -F, 'NR>1 && NF>=10 && $1!~/^#/{c++} END{print c+0}' "$EXTRA_CSV")
  echo "[merge_aiter_gemm_configs] extra csv=${EXTRA_CSV} (${extra_rows} rows)"
fi
