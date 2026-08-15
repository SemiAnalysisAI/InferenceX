#!/usr/bin/env bash
# Dump py-spy stacks for specific PP1 worker pids inside the rank1 container,
# plus any in-flight JIT compiler processes (hipcc/clang) and aiter build locks.
#
#   SLURM_REUSE_JOBID=16758 TS=20260815_042213 PIDS="2025 2027 2023" \
#     bash experimental/kimik3-v4/diag_pp1_stacks.sh
#
set -uo pipefail

SLURM_REUSE_JOBID="${SLURM_REUSE_JOBID:-${SLURM_JOB_ID:-}}"
TS="${TS:?set TS}"
NODE1="${NODE1:-mia1-p02-g17}"
CONT1="kimik3_tp8pp2_r1_${TS}"
PIDS="${PIDS:-2025 2027 2023}"

[[ -n "$SLURM_REUSE_JOBID" ]] || { echo "Set SLURM_REUSE_JOBID" >&2; exit 1; }

srun --overlap --jobid="$SLURM_REUSE_JOBID" -w "$NODE1" -N1 -n1 bash -lc "
docker exec ${CONT1} bash -lc '
  command -v py-spy >/dev/null 2>&1 || pip install -q py-spy >/dev/null 2>&1
  echo \"=== JIT / compiler processes ===\"
  ps -eo pid,stat,pcpu,etime,cmd | grep -Ei \"hipcc|clang|ninja|cmake|rocm_agent|setup.py\" | grep -v grep | head -20 || echo none
  echo
  echo \"=== aiter jit build dirs (recent) ===\"
  ls -lt /usr/local/lib/python3.12/dist-packages/aiter/jit/ 2>/dev/null | head -8
  find /usr/local/lib/python3.12/dist-packages/aiter/jit -name \"*.lock\" -newermt \"-30 minutes\" 2>/dev/null | head -10 || true
  echo
  for p in ${PIDS}; do
    echo \"##### py-spy dump pid=\$p (\$(cat /proc/\$p/comm 2>/dev/null)) #####\"
    timeout 40 py-spy dump --pid \$p --nonblocking 2>&1 | head -60
    echo \"--- /proc/\$p/status ---\"
    grep -E \"^(State|Threads|VmRSS)\" /proc/\$p/status 2>/dev/null
    echo
  done
'
" 2>&1
