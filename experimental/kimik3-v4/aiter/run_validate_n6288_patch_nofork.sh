#!/usr/bin/env bash
# Validate N=6288 chunk patch install does NOT fork-bomb (no GPU needed for the
# process-count check; uses the ROCm image so aiter is importable).
#
#   SLURM_REUSE_JOBID=16758 bash experimental/kimik3-v4/aiter/run_validate_n6288_patch_nofork.sh
#
set -euo pipefail

REPO="${REPO:-$HOME/InferenceX}"
SLURM_REUSE_JOBID="${SLURM_REUSE_JOBID:-${SLURM_JOB_ID:-}}"
NODE="${NODE:-mia1-p01-g06}"
IMAGE="${IMAGE:-vllm/vllm-openai-rocm:nightly-cb8104839c141609d99f1254459ef3a4f1bd4263}"
LOG_ROOT="${LOG_ROOT:-$HOME/kimik3_gemm_repro_logs}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_ROOT}/validate_n6288_nofork_${TS}.log"

[[ -n "$SLURM_REUSE_JOBID" ]] || { echo "Set SLURM_REUSE_JOBID" >&2; exit 1; }
mkdir -p "$LOG_ROOT"

echo "=== validate N6288 patch no-fork on ${NODE} ===" | tee "$LOG"

srun --overlap --jobid="$SLURM_REUSE_JOBID" -w "$NODE" -N1 -n1 bash -s <<EOF | tee -a "$LOG"
set -eo pipefail
if docker ps &>/dev/null 2>&1; then D=docker; else D="sudo docker"; fi
\$D run --rm \\
  --device /dev/dri --device /dev/kfd \\
  --ulimit memlock=-1 --network host --ipc host --group-add video --privileged \\
  -v ${REPO}:/workspace \\
  -e ROCR_VISIBLE_DEVICES=0 \\
  -e HIP_VISIBLE_DEVICES=0 \\
  -e GITHUB_WORKSPACE=/workspace \\
  -e AITER_N6288_CHUNK_PATCH=1 \\
  -e PYTHONPATH=/workspace/experimental/kimik3-v4/aiter/aiter_site \\
  --entrypoint bash ${IMAGE} -lc '
    set -e
    echo "--- sitecustomize deny: rocm_agent_enumerator ---"
    python3 /opt/rocm/bin/rocm_agent_enumerator -name | head -5
    echo "enumerator_ok"
    echo "--- simulate vllm argv install ---"
    timeout 120 python3 -c "import sys; sys.argv[0]=\"vllm\"; import sitecustomize" 2>&1 | tail -20
    echo "--- direct install ---"
    timeout 120 python3 /workspace/experimental/kimik3-v4/aiter/patch_gemm_n6288_chunk.py
    echo "direct_install_ok"
    # Process count sanity: should not explode
    N=\$(ps -u root -o cmd= 2>/dev/null | grep -c rocm_agent_enumerator || true)
    echo "rocm_agent_enumerator_procs=\$N"
    if [ "\${N:-0}" -gt 20 ]; then
      echo "FAIL: enumerator fork-storm (\$N procs)" >&2
      exit 1
    fi
    echo PASS
  '
EOF

echo "Log: ${LOG}"
