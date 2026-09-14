#!/usr/bin/env bash
set -euo pipefail
out="${1:?task output directory required}"
mkdir -p "$out"
exec > >(tee "$out/native2746-state.log") 2>&1
printf 'at=%s host=%s user=%s uid=%s\n' "$(date -u +%FT%TZ)" "$(hostname)" "$(id -un)" "$(id -u)"
printf 'run=%s attempt=%s head=%s native_job=2746\n' "$GITHUB_RUN_ID" "$GITHUB_RUN_ATTEMPT" "$(git rev-parse HEAD)"
# A failed accounting read must not hide scontrol or the existing runtime inventory.
read_probe() {
  local label="$1" limit="$2" rc
  shift 2
  if timeout --signal=TERM --kill-after=2s "$limit" "$@" > "$out/$label.txt" 2> "$out/$label.stderr"; then
    rc=0
  else
    rc=$?
  fi
  printf '%s\n' "$rc" > "$out/$label.rc"
  printf 'at=%s read=%s rc=%s\n' "$(date -u +%FT%TZ)" "$label" "$rc"
}
read_probe native-queue 10s squeue -j 2746 --noheader -o '%i|%u|%a|%P|%T|%r|%S|%V|%M|%L|%l|%D|%N|%j'
read_probe native-control 10s scontrol show job 2746
read_probe native-accounting 15s sacct -j 2746 --starttime 2026-09-13 -P --format=JobID,JobName,User,Account,Partition,State,Reason,ExitCode,DerivedExitCode,NodeList,Submit,Eligible,Start,End,Elapsed,Timelimit,ReqTRES,AllocTRES
read_probe enroot-list 10s enroot list
read_probe image-stat 5s stat --printf='path=%n size=%s mtime=%y owner=%U uid=%u mode=%a\n' "$SQUASH_FILE"
read_probe image-superblock 5s unsquashfs -s "$SQUASH_FILE"
# Restrict environment output to paths so credentials cannot enter the artifact.
for name in HOME XDG_RUNTIME_DIR XDG_DATA_HOME ENROOT_CACHE_PATH ENROOT_DATA_PATH ENROOT_RUNTIME_PATH ENROOT_CONFIG_PATH ENROOT_LIBRARY_PATH; do
  printf '%s=%s\n' "$name" "${!name-<unset>}"
done > "$out/runtime-paths.txt"
printf 'at=%s finished_read_only=true\n' "$(date -u +%FT%TZ)"
echo 'Empty or failed reads are unknown, not proof of terminal state or available allocation.'
