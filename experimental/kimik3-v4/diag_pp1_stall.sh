#!/usr/bin/env bash
# Diagnose a PP1 stall: which TP ranks lag, their host CPU state, and py-spy stacks.
#
#   SLURM_REUSE_JOBID=16758 TS=20260815_042213 bash experimental/kimik3-v4/diag_pp1_stall.sh
#
set -uo pipefail

SLURM_REUSE_JOBID="${SLURM_REUSE_JOBID:-${SLURM_JOB_ID:-}}"
TS="${TS:?set TS=<launcher timestamp>}"
NODE1="${NODE1:-mia1-p02-g17}"
NODE0="${NODE0:-mia1-p01-g06}"
LOG_ROOT="${LOG_ROOT:-$HOME/kimik3_tp8pp2_smoke_logs}"
R1_LOG="${LOG_ROOT}/rank1_${TS}/server.log"
CONT1="kimik3_tp8pp2_r1_${TS}"
CONT0="kimik3_tp8pp2_r0_${TS}"

[[ -n "$SLURM_REUSE_JOBID" ]] || { echo "Set SLURM_REUSE_JOBID" >&2; exit 1; }

echo "=== $(date -u) PP1 stall diagnosis (TS=${TS}) ==="

echo
echo "--- PP1 per-rank capture progress ---"
for t in 0 1 2 3 4 5 6 7; do
  reg=$(grep -c "Worker_PP1_TP${t} .*Registering .* cuda graph addresses" "$R1_LOG" 2>/dev/null)
  fin=$(grep -c "Worker_PP1_TP${t} .*Graph capturing finished" "$R1_LOG" 2>/dev/null)
  mem=$(grep -c "Worker_PP1_TP${t} .*gpu_worker.py:803" "$R1_LOG" 2>/dev/null)
  last=$(grep "Worker_PP1_TP${t} " "$R1_LOG" 2>/dev/null | tail -1 | cut -c1-150)
  printf 'PP1_TP%s registered=%s capture_finished=%s mem_report=%s\n  last: %s\n' \
    "$t" "$reg" "$fin" "$mem" "$last"
done

echo
echo "--- GPU state ${NODE1} ---"
srun --overlap --jobid="$SLURM_REUSE_JOBID" -w "$NODE1" -N1 -n1 bash -lc 'amd-smi monitor 2>&1 | head -12'

echo
echo "--- GPU state ${NODE0} ---"
srun --overlap --jobid="$SLURM_REUSE_JOBID" -w "$NODE0" -N1 -n1 bash -lc 'amd-smi monitor 2>&1 | head -12'

echo
echo "--- rank1 worker host CPU / state ---"
srun --overlap --jobid="$SLURM_REUSE_JOBID" -w "$NODE1" -N1 -n1 bash -lc \
  "docker exec ${CONT1} bash -lc 'ps -eo pid,stat,pcpu,etime,wchan:24,cmd | grep -E \"VLLM|EngineCore|python\" | grep -v grep | head -20'" 2>&1

echo
echo "--- rank1 py-spy stacks (install if needed) ---"
srun --overlap --jobid="$SLURM_REUSE_JOBID" -w "$NODE1" -N1 -n1 bash -lc \
  "docker exec ${CONT1} bash -lc 'command -v py-spy >/dev/null || pip install -q py-spy >/dev/null 2>&1; for p in \$(pgrep -f VllmWorker | head -8) \$(pgrep -f EngineCore | head -2); do echo \"### pid=\$p\"; timeout 25 py-spy dump --pid \$p 2>&1 | head -30; done'" 2>&1

echo
echo "--- rank0 (PP0) last engine lines ---"
srun --overlap --jobid="$SLURM_REUSE_JOBID" -w "$NODE0" -N1 -n1 bash -lc \
  "docker logs --tail 15 ${CONT0} 2>&1" 2>&1

echo
echo "=== done ==="
