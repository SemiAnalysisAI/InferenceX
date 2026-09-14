#!/usr/bin/env bash
set -euo pipefail
out="${1:?task output directory required}"
mkdir -p "$out"
exec > >(tee "$out/capabilities.log") 2>&1
printf 'at=%s host=%s user=%s\n' "$(date -u +%FT%TZ)" "$(hostname)" "$(id -un)"
printf 'run=%s attempt=%s head=%s\n' "$GITHUB_RUN_ID" "$GITHUB_RUN_ATTEMPT" "$(git rev-parse HEAD)"
# Each independent read is bounded and retained even when another read fails.
read_probe() {
  local label="$1" limit="$2" rc
  shift 2
  if timeout --signal=TERM --kill-after=2s "$limit" "$@" > "$out/$label.txt" 2> "$out/$label.stderr"; then
    rc=0
  else
    rc=$?
  fi
  printf '%s\n' "$rc" > "$out/$label.rc"
  printf 'read=%s rc=%s\n' "$label" "$rc"
}
# Read-only control-node facts; never srun/sbatch/salloc or node SSH.
read_probe old-accounting 15s sacct -X -j 2740,2714,2715,2716 -P --format=JobID,JobName,User,State,ExitCode,NodeList,Start,End,AllocTRES
read_probe old-active 10s squeue -j 2740,2714,2715,2716 --noheader -o '%i|%u|%T|%N|%L|%j'
read_probe enroot-list 10s enroot list
read_probe ptrace-scope 2s cat /proc/sys/kernel/yama/ptrace_scope
for tool in py-spy gdb python3 timeout; do
  printf 'tool=%s path=' "$tool"
  command -v "$tool" || true
done
# Preserve existing runtime inventory; do not enter or recreate a container.
read_probe image-stat 5s stat --printf='sqsh_size=%s mtime=%y\n' "$SQUASH_FILE"
# Inspect only candidate executable paths in the exact cached image, no extraction/import.
read_probe image-tool-list 25s unsquashfs -ll "$SQUASH_FILE"   'usr/bin/gdb' 'usr/local/bin/gdb' 'usr/bin/py-spy' 'usr/local/bin/py-spy'   'usr/bin/python3.12' 'usr/local/bin/python3.12'
# Probe attach only to our own short-lived CPU child, never a serving process.
read_probe attach-probe 15s python3 "$(dirname "$0")/attach_capability.py" "$out"
echo 'Resource query failure/timeout leaves reuse unresolved: do not launch GPU from this receipt alone.'
