#!/usr/bin/env bash
set -euo pipefail
set +m
: "${SLURM_JOB_ID:?native Slurm allocation required}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || exit 64
command -v setsid >/dev/null
: "${RESULT_DIR:?}"
[[ "${HANG_STACK_MODE:-}" == faulthandler ]] || exit 64
[[ "$MODEL" == moonshotai/Kimi-K3 && "$TP" == 8 && "$CONC" == 40 && "$DCP_SIZE" == 8 ]]
[[ "$KV_OFFLOADING" == dram && "$KV_OFFLOAD_BACKEND" == mooncake && "$TOTAL_CPU_DRAM_GB" == 2249 ]]
[[ "$DURATION" == 3600 && "$EVAL_ONLY" == false && "${AIPERF_EXPERIMENTAL_FAST:-0}" == 0 ]]
[[ "${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-300}" == 300 ]]
script=benchmarks/single_node/agentic/kimik3_fp4_b300_vllm_mtp.sh
printf '%s  %s\n' d29a83d797e0a64c05fedd2d58dd5e5c5f5a4677bd4fa1fbde4e2790583b8a6d "$script" | sha256sum --check --status
out="$RESULT_DIR/hang-diagnostic"
mkdir -p "$out"
diag="$(cd "$(dirname "$0")" && pwd)"
observer=""
bench=""
bench_rc=""
cleanup() {
    rc=$?
    trap - EXIT
    if [[ -n "$observer" ]]; then
        python3 "$diag/owned_process.py" stop "$out/observer-identity.json" "$out/observer-cleanup.json" || true
        if ! kill -0 "$observer" 2>/dev/null; then wait "$observer" 2>/dev/null || true; fi
        observer=""
    fi
    if [[ -n "$bench" ]]; then
        python3 "$diag/owned_process.py" stop "$out/benchmark-process-group.json" "$out/benchmark-cleanup.json" --group || true
        if [[ -z "$bench_rc" ]] && ! kill -0 "$bench" 2>/dev/null; then
            if wait "$bench" 2>/dev/null; then bench_rc=0; else bench_rc=$?; fi
        fi
    fi
    # Actions must never open a FIFO. After bounded owned cleanup, unlink only
    # task-owned FIFOs; live descriptors remain safe until native cleanup finishes.
    python3 - "$out/registered" "$out/preflight/registered" <<'PYFIFOS'
import pathlib, stat, sys
for directory in sys.argv[1:]:
    root = pathlib.Path(directory)
    if root.is_dir():
        for path in root.glob('*.fifo'):
            if stat.S_ISFIFO(path.lstat().st_mode):
                path.unlink()
PYFIFOS
    printf '%s\n' "$rc" > "$out/wrapper-exit-code.txt"
    if [[ -n "$bench_rc" ]]; then printf '%s\n' "$bench_rc" > "$out/original-script-exit-code.txt"; fi
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
hook="$(cd "$(dirname "$0")/faulthandler_hook" && pwd)"
# Do not shadow unknown image initialization; the capability result justifies
# this branch only when the original image has no pre-existing sitecustomize.
python3 - <<'PYSITE'
import importlib.util
if importlib.util.find_spec('sitecustomize') is not None:
    raise SystemExit('Existing sitecustomize must be reviewed before diagnostic injection')
PYSITE
# Fresh own-child registration and PID-safe signal proof; no external attach tool.
timeout --signal=TERM --kill-after=2s 15s env \
    PYTHONPATH="$hook${PYTHONPATH:+:$PYTHONPATH}" \
    POWERX_STACK_DIR="$out/preflight/registered" POWERX_DIAG_JOB="$SLURM_JOB_ID" \
    python3 "$(dirname "$0")/preflight.py" "$out/preflight"
python3 - "$out/runtime.json" <<'PY'
import json, os, socket, sys, vllm
version = vllm.__version__
if '3696c772a' not in version:
    raise SystemExit('Unexpected vLLM revision: ' + version)
json.dump({'version': version, 'hostname': socket.gethostname(), 'pid': os.getpid(),
           'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
           'run_id': os.environ.get('GITHUB_RUN_ID'), 'attempt': os.environ.get('GITHUB_RUN_ATTEMPT'),
           'diagnostic_only': True, 'native_execute_model_timeout_seconds': 300},
          open(sys.argv[1], 'w'), indent=2)
PY
env PYTHONPATH="$hook${PYTHONPATH:+:$PYTHONPATH}" \
    POWERX_STACK_DIR="$out/registered" POWERX_DIAG_JOB="$SLURM_JOB_ID" \
    setsid bash "$script" &
bench=$!
printf '%s\n' "$bench" > "$out/benchmark-shell.pid"
python3 "$diag/owned_process.py" record "$bench" "$out/benchmark-process-group.json" --group
python3 "$(dirname "$0")/observe.py" --root "$bench" --log "$RESULT_DIR/server.log" \
    --output "$out/stacks" --registrations "$out/registered" --job "$SLURM_JOB_ID" > "$out/observer.log" 2>&1 &
observer=$!
printf '%s\n' "$observer" > "$out/observer.pid"
python3 "$diag/owned_process.py" record "$observer" "$out/observer-identity.json"
# A failed registration observer must stop this diagnostic, not waste a full run.
while kill -0 "$bench" 2>/dev/null; do
    if ! kill -0 "$observer" 2>/dev/null; then
        wait "$observer" || true
        observer=""
        printf 'Stack observer exited before benchmark completion\n' >&2
        exit 70
    fi
    sleep 1
done
if wait "$bench"; then
    bench_rc=0
else
    bench_rc=$?
fi
exit "$bench_rc"
