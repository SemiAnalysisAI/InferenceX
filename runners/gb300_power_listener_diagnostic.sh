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

base=/data/home/sa-shared/gharunners/powerx-agentx-20260909/gb300-power-lifecycle
export PYTHONPATH="$base/producer/src" PYTHONDONTWRITEBYTECODE=1
python="$base/venv/bin/python"
"$python" -c 'from srtctl.cli.mixins.telemetry_stage import read_producer_commit; assert read_producer_commit() == "80d7203e424f903c9017de4608ee2044afce9574"'
[[ -z "$(git -C "$base/producer" status --porcelain)" ]] || exit 1
image=/data/home/sa-shared/gharunners/squash/nvcr.io_nvidia_k8s_dcgm-exporter_4.6.0-4.8.3-distroless.sqsh
[[ "$(sha256sum "$image" | cut -d' ' -f1)" == b726092000b9dbbec030d6326c514bc33d049f08ef4db04ea2fe2eac95ae3ebf ]] || exit 1
cp "$root/runners/gb300_power_lifecycle.py" "$out/lifecycle.py"
cp "$base/requirements.lock" "$out/requirements.lock"
cat > "$out/job.sh" <<'JOB'
#!/usr/bin/env bash
set -euo pipefail
out=$1
python=$2
[[ "${SLURM_JOB_NUM_NODES:?}" == 2 && "${SLURM_GPUS_PER_NODE:?}" == 4 ]] || {
    echo "Unexpected native allocation size" >&2
    exit 1
}
date -u > "$out/native-start.txt"
scontrol show job "$SLURM_JOB_ID" > "$out/native-job.txt"
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --exact \
    --output="$out/preflight-%t.txt" "$python" "$out/lifecycle.py" before "$out"
rc=0
"$python" "$out/lifecycle.py" collect "$out" > "$out/shared-lifecycle.log" 2>&1 || rc=$?
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --exact \
    --output="$out/postflight-%t.txt" "$python" "$out/lifecycle.py" after "$out" || rc=$?
printf '%s\n' "$rc" > "$out/lifecycle-exit.txt"
date -u > "$out/native-end.txt"
exit "$rc"
JOB

package() {
    tar -czf "$root/multinode_server_logs.tar.gz" -C "$root/LOGS" power-listener-diagnostic
}
job_id=
cleanup() {
    local rc=$? actual
    trap - EXIT
    if [[ "$job_id" =~ ^[0-9]+$ ]]; then
        actual=$(squeue -j "$job_id" -h -o '%j|%u') || rc=1
        if [[ "$actual" == "$identity|$USER" ]]; then
            scancel "$job_id" || rc=1
        elif [[ -n "$actual" ]]; then
            echo "Refusing cleanup: native job ownership mismatch" >&2
            rc=1
        fi
        sacct -j "$job_id" -n -P --format=JobID,JobName,User,State,ExitCode,Start,End,Elapsed,AllocTRES,NodeList \
            > "$out/cleanup-accounting.txt" || true
    fi
    package || rc=1
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM HUP INT
job_id=$(sbatch --parsable --account=benchmark --partition="${SLURM_PARTITION:-batch_1}" \
    --job-name="$identity" --comment="$identity" --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    --gpus-per-node=4 --cpus-per-task=144 --exclusive --mem=0 --time=00:09:00 --no-requeue \
    --output="$out/batch-%j.txt" "$out/job.sh" "$out" "$python")
job_id=${job_id%%;*}
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Ambiguous submission; do not retry" >&2; exit 1; }
printf '%s\n' "$job_id" | tee "$receipt" "$out/job-id.txt"
scontrol show job "$job_id" > "$out/submission-job.txt"
while [[ -n "$(squeue -j "$job_id" -h -o '%i')" ]]; do
    sleep 5
done
sacct -j "$job_id" -n -P --format=JobID,JobName,User,State,ExitCode,Start,End,Elapsed,AllocTRES,NodeList \
    > "$out/native-accounting.txt"
echo "Shared telemetry diagnostic completed. No AgentX benchmark was run; benchmark result guards are expected to fail."
