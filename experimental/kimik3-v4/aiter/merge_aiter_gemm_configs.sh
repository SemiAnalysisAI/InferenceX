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
  # Skip header lines (including duplicate mid-file headers that force pandas
  # StringDtype and break aiter's int/bool MultiIndex lookup).
  awk -F, 'NR==1{next} NF>=10 && $1!~/^#/ && $1!="gfx" && $3!="M"{print}' "$f"
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
  NF >= 10 && $1 !~ /^#/ && $1 != "gfx" && $3 != "M" {
    key = $3 "," $4 "," $5
    rows[key] = $0
  }
  END {
    for (k in rows) print rows[k]
  }
' > "$OUT_CSV"

# Normalize bias/scaleAB/bpreshuffle to False/True so pandas reads bool and
# matches aiter get_GEMM_A16W16_config(..., bias: bool, ...). Mixed 0/False
# strings make the whole index StringDtype and every lookup misses → torch
# fallback → SIGSEGV on some untuned shapes.
python3 - "$OUT_CSV" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows: list[dict[str, str]] = []
with path.open() as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames or []
    for row in reader:
        if row.get("M") in (None, "M") or row.get("gfx") == "gfx":
            continue
        for col in ("bias", "scaleAB", "bpreshuffle"):
            v = str(row.get(col, "False")).strip().lower()
            row[col] = "True" if v in ("1", "true", "t", "yes") else "False"
        rows.append(row)
with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"[merge_aiter_gemm_configs] normalized bool cols ({len(rows)} rows)")
PY

rows=$(($(wc -l < "$OUT_CSV") - 1))
export AITER_CONFIG_GEMM_BF16="$OUT_CSV"
echo "[merge_aiter_gemm_configs] wrote ${OUT_CSV} (${rows} tuned rows)"
if [[ -n "$EXTRA_CSV" && -f "$EXTRA_CSV" ]]; then
  extra_rows=$(awk -F, 'NR>1 && NF>=10 && $1!~/^#/ && $3!="M"{c++} END{print c+0}' "$EXTRA_CSV")
  echo "[merge_aiter_gemm_configs] extra csv=${EXTRA_CSV} (${extra_rows} rows)"
fi
