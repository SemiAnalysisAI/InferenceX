#!/usr/bin/env bash
# Tune the padded-M bucket GEMM shapes that agentic prefill falls back to torch on.
#
# Agentic prefill batches produce arbitrary M (6664, 7121, 7615, ...), so tuning the
# literal M values observed in one run never covers the next. aiter looks up
# exact M -> get_padded_m(gl=0) -> get_padded_m(gl=1), so tuning the bucket M values
# generalises: every large M collapses onto the same gl=1 key (8192 on gfx950).
#
#   SLURM_REUSE_JOBID=16758 \
#     LOGS="$HOME/kimik3_tp8pp2_smoke_logs/rank1_20260815_152108/server.log \
#           $HOME/kimik3_tp8pp2_smoke_logs/conc20_dspark_20260815_152108/server.log" \
#     bash experimental/kimik3-v4/aiter/run_gemm_tune_padded_buckets.sh
#
set -euo pipefail

REPO="${REPO:-$HOME/InferenceX}"
SLURM_REUSE_JOBID="${SLURM_REUSE_JOBID:-${SLURM_JOB_ID:-}}"
NODE="${NODE:-mia1-p01-g06}"
IMAGE="${IMAGE:-vllm/vllm-openai-rocm:nightly-cb8104839c141609d99f1254459ef3a4f1bd4263}"
LOGS="${LOGS:-}"
# gl=1 is the broadest bucket and is listed first so --top keeps it.
GL="${GL:-1,0}"
TOP_N="${TOP_N:-40}"
MAX_CANDIDATES="${MAX_CANDIDATES:-200}"
LIBTYPE="${LIBTYPE:-flydsl,asm}"
SHAPES_CSV="${SHAPES_CSV:-$HOME/kimik3_tp8pp2_smoke_logs/missing_bucket_shapes.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-${REPO}/experimental/kimik3-v4/aiter/kimik3_bf16_tuned_gemm.padded_buckets.csv}"
LOG_ROOT="${LOG_ROOT:-$HOME/kimik3_gemm_repro_logs}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_ROOT}/tune_padded_buckets_${TS}.log"

[[ -n "$SLURM_REUSE_JOBID" ]] || { echo "Set SLURM_REUSE_JOBID" >&2; exit 1; }
[[ -n "$LOGS" ]] || { echo "Set LOGS to one or more vLLM server.log paths" >&2; exit 1; }
mkdir -p "$LOG_ROOT" "$(dirname "$SHAPES_CSV")"

# Mount each server.log read-only at a stable in-container path.
LOG_MOUNTS=()
IN_LOGS=()
i=0
for f in $LOGS; do
  [[ -f "$f" ]] || { echo "Missing log $f" >&2; exit 1; }
  LOG_MOUNTS+=("-v" "${f}:/logs/server_${i}.log:ro")
  IN_LOGS+=("/logs/server_${i}.log")
  i=$((i + 1))
done

echo "=== padded-bucket GEMM tune on ${NODE} gl=${GL} top=${TOP_N} lib=${LIBTYPE} ===" | tee "$LOG"
echo "logs=${LOGS}" | tee -a "$LOG"
echo "shapes=${SHAPES_CSV} -> ${OUTPUT_CSV}" | tee -a "$LOG"

srun --overlap --jobid="$SLURM_REUSE_JOBID" -w "$NODE" -N1 -n1 bash -s <<EOF | tee -a "$LOG"
set -eo pipefail
if docker ps &>/dev/null 2>&1; then D=docker; else D="sudo docker"; fi
\$D run --rm \\
  --device /dev/dri --device /dev/kfd \\
  --ulimit memlock=-1 --network host --ipc host --group-add video --privileged \\
  -v ${REPO}:/workspace \\
  ${LOG_MOUNTS[@]} \\
  -v /tmp:/tmp \\
  -e ROCR_VISIBLE_DEVICES=0 \\
  -e HIP_VISIBLE_DEVICES=0 \\
  -e PYTHONPATH=/workspace/experimental/kimik3-v4/aiter \\
  --entrypoint bash ${IMAGE} -lc '
    set -e
    python3 /workspace/experimental/kimik3-v4/aiter/extract_padded_bucket_shapes.py \\
      ${IN_LOGS[@]} \\
      --gl ${GL} \\
      -o /tmp/missing_bucket_shapes.csv
    python3 /workspace/experimental/kimik3-v4/aiter/tune_gemm_from_missing_csv.py \\
      --shapes-csv /tmp/missing_bucket_shapes.csv \\
      --top ${TOP_N} \\
      --libtype ${LIBTYPE} \\
      --max-candidates ${MAX_CANDIDATES} \\
      --allow-torch-winner \\
      -o /workspace/experimental/kimik3-v4/aiter/kimik3_bf16_tuned_gemm.padded_buckets.csv
    cp /tmp/missing_bucket_shapes.csv ${SHAPES_CSV} 2>/dev/null || true
    wc -l /workspace/experimental/kimik3-v4/aiter/kimik3_bf16_tuned_gemm.padded_buckets.csv
  '
EOF

echo "Done. Log: ${LOG}"
echo "Next: bash experimental/kimik3-v4/aiter/combine_gemm_extra_csvs.sh"
