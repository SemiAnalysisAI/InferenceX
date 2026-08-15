#!/usr/bin/env bash
# Offline tune cudagraph-relevant small-M GEMM shapes (M<=44) from a missing-shape CSV.
# Addresses historical PIECEWISE capture faults near small M with torch fallback.
#
#   SLURM_REUSE_JOBID=16758 \\
#     SHAPES_CSV=$HOME/kimik3_tp8pp2_smoke_logs/missing_shapes_031321_m_le_44.csv \\
#     bash experimental/kimik3-v4/aiter/run_gemm_tune_cudagraph_small_m.sh
#
set -euo pipefail

REPO="${REPO:-$HOME/InferenceX}"
SLURM_REUSE_JOBID="${SLURM_REUSE_JOBID:-${SLURM_JOB_ID:-}}"
NODE="${NODE:-mia1-p01-g06}"
IMAGE="${IMAGE:-vllm/vllm-openai-rocm:nightly-cb8104839c141609d99f1254459ef3a4f1bd4263}"
SHAPES_CSV="${SHAPES_CSV:-$HOME/kimik3_tp8pp2_smoke_logs/missing_shapes_031321_m_le_44.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-${REPO}/experimental/kimik3-v4/aiter/kimik3_bf16_tuned_gemm.cudagraph_small_m.csv}"
TOP_N="${TOP_N:-15}"
MAX_CANDIDATES="${MAX_CANDIDATES:-100}"
LIBTYPE="${LIBTYPE:-flydsl}"
LOG_ROOT="${LOG_ROOT:-$HOME/kimik3_gemm_repro_logs}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_ROOT}/tune_cudagraph_small_m_${TS}.log"

[[ -n "$SLURM_REUSE_JOBID" ]] || { echo "Set SLURM_REUSE_JOBID" >&2; exit 1; }
[[ -f "$SHAPES_CSV" ]] || { echo "Missing $SHAPES_CSV — run run_extract_missing_shapes_031321.sh first" >&2; exit 1; }
mkdir -p "$LOG_ROOT"

echo "=== cudagraph small-M GEMM tune on ${NODE} top=${TOP_N} lib=${LIBTYPE} cand=${MAX_CANDIDATES} ===" | tee "$LOG"
echo "shapes=$SHAPES_CSV -> $OUTPUT_CSV" | tee -a "$LOG"

srun --overlap --jobid="$SLURM_REUSE_JOBID" -w "$NODE" -N1 -n1 bash -s <<EOF | tee -a "$LOG"
set -eo pipefail
if docker ps &>/dev/null 2>&1; then D=docker; else D="sudo docker"; fi
\$D run --rm \\
  --device /dev/dri --device /dev/kfd \\
  --ulimit memlock=-1 --network host --ipc host --group-add video --privileged \\
  -v ${REPO}:/workspace \\
  -v ${SHAPES_CSV}:/shapes.csv:ro \\
  -v /tmp:/tmp \\
  -e ROCR_VISIBLE_DEVICES=0 \\
  -e HIP_VISIBLE_DEVICES=0 \\
  --entrypoint bash ${IMAGE} -lc '
    python3 /workspace/experimental/kimik3-v4/aiter/tune_gemm_from_missing_csv.py \\
      --shapes-csv /shapes.csv \\
      --top ${TOP_N} \\
      --exclude-n 6288 \\
      --libtype ${LIBTYPE} \\
      --max-candidates ${MAX_CANDIDATES} \\
      -o /workspace/experimental/kimik3-v4/aiter/kimik3_bf16_tuned_gemm.cudagraph_small_m.csv
    wc -l /workspace/experimental/kimik3-v4/aiter/kimik3_bf16_tuned_gemm.cudagraph_small_m.csv
    head -5 /workspace/experimental/kimik3-v4/aiter/kimik3_bf16_tuned_gemm.cudagraph_small_m.csv
  '
EOF

echo "Done. Log: ${LOG}"
echo "Next: bash experimental/kimik3-v4/aiter/combine_gemm_extra_csvs.sh"
