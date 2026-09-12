#!/usr/bin/env bash
set -euo pipefail

[[ -z "${SLURM_JOB_ID:-}" ]] || { echo "Refusing inherited allocation" >&2; exit 1; }
root=${GITHUB_WORKSPACE:?}
identity="gb300-power-${GITHUB_RUN_ID:?}-${GITHUB_RUN_ATTEMPT:?}"
receipt="${RUNNER_TEMP:?}/${identity}.job"
out="$root/LOGS/power-listener-diagnostic"
mkdir -p "$out"
git -C "$root" rev-parse HEAD > "$out/source-sha.txt"
printf '%s\n' "$identity" > "$out/identity.txt"
[[ ! -e "$receipt" ]] || { echo "Existing submission receipt: refusing duplicate" >&2; exit 1; }
[[ -z "$(squeue -h --name="$identity" --user="$USER" -o '%i')" ]] || {
    echo "Existing diagnostic job: reconcile before submitting" >&2
    exit 1
}

cat > "$out/node.sh" <<'NODE'
#!/usr/bin/env bash
set -u
date -u
hostname -f
id
printf 'job=%s step=%s nodes=%s gpu_indices=%s cpus=%s\n' \
    "${SLURM_JOB_ID:-}" "${SLURM_STEP_ID:-}" "${SLURM_JOB_NODELIST:-}" \
    "${SLURM_JOB_GPUS:-}" "${SLURM_CPUS_ON_NODE:-}"
taskset -pc $$
nvidia-smi --query-gpu=uuid,name,index --format=csv,noheader
ss -lntp '( sport = :9401 )'
if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:9401 -sTCP:LISTEN
fi
python3 - <<'PY'
import pathlib
inodes = set()
for name in ('tcp', 'tcp6'):
    for line in pathlib.Path('/proc/net', name).read_text().splitlines()[1:]:
        fields = line.split()
        if fields[1].split(':')[-1] == f'{9401:04X}' and fields[3] == '0A':
            inodes.add(fields[9])
            print('listener_uid_inode', fields[7], fields[9])
found = set()
for proc in pathlib.Path('/proc').iterdir():
    if not proc.name.isdecimal():
        continue
    try:
        owned = any(fd.readlink().as_posix() in {f'socket:[{i}]' for i in inodes}
                    for fd in (proc / 'fd').iterdir())
        if not owned:
            continue
        found.add(proc.name)
        print('listener_pid', proc.name)
        for name in ('comm', 'cgroup'):
            print(name, (proc / name).read_text().strip())
        print('exe', (proc / 'exe').readlink())
        print('status', '\n'.join(line for line in (proc / 'status').read_text().splitlines()
                                  if line.startswith(('Name:', 'Uid:', 'PPid:'))))
    except (OSError, PermissionError):
        continue
print('listener_inodes', sorted(inodes), 'visible_owner_pids', sorted(found))
if inodes and not found:
    print('ownership_blocker: listener exists but process file access did not expose its owner')
PY
curl --max-time 5 --max-filesize 1048576 --silent --show-error --output "$1/metrics-${SLURM_PROCID}.txt" \
    --write-out 'metrics_http_status=%{http_code}\n' http://127.0.0.1:9401/metrics
enroot list
printf 'ENROOT_DATA_PATH=%s\n' "${ENROOT_DATA_PATH:-unset}"
ls -ld "${ENROOT_DATA_PATH:-$HOME/.local/share/enroot}" /run/enroot/user-"$(id -u)" 2>/dev/null
exit 0
NODE

cat > "$out/job.sh" <<'JOB'
#!/usr/bin/env bash
set -euo pipefail
out=$1
[[ "${SLURM_JOB_NUM_NODES:?}" == 2 && "${SLURM_GPUS_PER_NODE:?}" == 4 ]] || {
    echo "Unexpected native allocation size" >&2
    exit 1
}
date -u > "$out/native-start.txt"
scontrol show job "$SLURM_JOB_ID" > "$out/native-job.txt"
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --exact \
    --output="$out/node-%t.txt" bash "$out/node.sh" "$out"
date -u > "$out/native-end.txt"
JOB

package() {
    tar -czf "$root/multinode_server_logs.tar.gz" -C "$root/LOGS" power-listener-diagnostic
}
trap package EXIT
job_id=$(sbatch --parsable --account=benchmark --partition="${SLURM_PARTITION:-batch_1}" \
    --job-name="$identity" --comment="$identity" --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    --gpus-per-node=4 --cpus-per-task=144 --exclusive --mem=0 --time=00:10:00 \
    --output="$out/batch-%j.txt" "$out/job.sh" "$out")
job_id=${job_id%%;*}
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Ambiguous submission; do not retry" >&2; exit 1; }
printf '%s\n' "$job_id" | tee "$receipt" "$out/job-id.txt"
scontrol show job "$job_id" > "$out/submission-job.txt"
while [[ -n "$(squeue -j "$job_id" -h -o '%i')" ]]; do
    sleep 5
done
sacct -j "$job_id" -n -P --format=JobID,JobName,User,State,ExitCode,Start,End,Elapsed,AllocTRES,NodeList \
    > "$out/native-accounting.txt"
echo "Diagnostic completed. No AgentX benchmark was run; benchmark result guards are expected to fail."
