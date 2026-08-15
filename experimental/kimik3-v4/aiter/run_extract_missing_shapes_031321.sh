#!/usr/bin/env bash
# Extract missing GEMM shapes from the 031321 tp8pp2 smoke logs (on ln).
set -euo pipefail
REPO="${REPO:-$HOME/InferenceX}"
LOG_ROOT="${LOG_ROOT:-$HOME/kimik3_tp8pp2_smoke_logs}"
python3 "$REPO/experimental/kimik3-v4/aiter/extract_missing_gemm_shapes.py" \
  "$LOG_ROOT/conc4_20260815_031321/server.log" \
  "$LOG_ROOT/rank1_20260815_031321/server.log" \
  -o "$LOG_ROOT/missing_shapes_031321.csv"
python3 "$REPO/experimental/kimik3-v4/aiter/extract_missing_gemm_shapes.py" \
  "$LOG_ROOT/conc4_20260815_031321/server.log" \
  "$LOG_ROOT/rank1_20260815_031321/server.log" \
  --max-m 44 \
  -o "$LOG_ROOT/missing_shapes_031321_m_le_44.csv"
echo "N=6288 rows:"
awk -F, 'NR>1 && $2==6288 {print}' "$LOG_ROOT/missing_shapes_031321.csv" | head -20
echo "Top M<=44:"
head -25 "$LOG_ROOT/missing_shapes_031321_m_le_44.csv"
