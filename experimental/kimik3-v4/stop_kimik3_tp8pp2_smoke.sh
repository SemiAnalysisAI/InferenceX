#!/usr/bin/env bash
# Stop a running TP8xPP2 smoke: kill the launcher tree, then force-remove the rank
# containers on both nodes. A wedged rank-0 container sometimes refuses `docker rm`
# on the first try ("did not receive an exit event"), so retry after killing its
# host-side pids.
#
#   SLURM_REUSE_JOBID=16758 TS=20260815_044741 LAUNCHER_PID=2476798 \
#     bash experimental/kimik3-v4/stop_kimik3_tp8pp2_smoke.sh
#
set -uo pipefail

SLURM_REUSE_JOBID="${SLURM_REUSE_JOBID:-${SLURM_JOB_ID:-}}"
TS="${TS:?set TS=<launcher timestamp>}"
LAUNCHER_PID="${LAUNCHER_PID:-}"
NODE0="${NODE0:-mia1-p01-g06}"
NODE1="${NODE1:-mia1-p02-g17}"
LOG_ROOT="${LOG_ROOT:-$HOME/kimik3_tp8pp2_smoke_logs}"

[[ -n "$SLURM_REUSE_JOBID" ]] || { echo "Set SLURM_REUSE_JOBID" >&2; exit 1; }

echo "=== stopping smoke TS=${TS} ==="

if [[ -n "$LAUNCHER_PID" ]]; then
  kill -TERM "$LAUNCHER_PID" 2>/dev/null
  sleep 3
  kill -KILL "$LAUNCHER_PID" 2>/dev/null
fi
for pat in "rank0_launch_${TS}" "rank1_launch_${TS}" "kimik3_tp8pp2_c.*_${TS}"; do
  pkill -9 -f "$pat" 2>/dev/null
done
sleep 2

for item in "${NODE0}:kimik3_tp8pp2_r0_${TS}" "${NODE1}:kimik3_tp8pp2_r1_${TS}"; do
  node="${item%%:*}"
  cont="${item#*:}"
  echo "--- ${node} / ${cont} ---"
  srun --overlap --jobid="$SLURM_REUSE_JOBID" -w "$node" -N1 -n1 bash -lc "
    docker rm -f ${cont} 2>&1 | tail -1
    if docker ps -a --format '{{.Names}}' | grep -qx '${cont}'; then
      pids=\$(docker top ${cont} -eo pid 2>/dev/null | tail -n +2)
      echo \"retry after killing host pids: \$pids\"
      for p in \$pids; do kill -9 \$p 2>/dev/null; done
      sleep 5
      docker rm -f ${cont} 2>&1 | tail -1
    fi
    docker ps -a --format '{{.Names}}' | grep kimik3 || echo no_kimik3_containers
  "
done

echo
echo "--- launcher / smoke procs ---"
pgrep -af "run_kimik3_tp8pp2|rank0_launch_${TS}" || echo none_left
echo "=== stopped (logs under ${LOG_ROOT}) ==="
