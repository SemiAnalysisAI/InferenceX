#!/usr/bin/env bash
# One native SMI collector per serving node. Raw files stay on node-local scratch;
# the launcher stages them as its host user after containers stop.
set -uo pipefail
power_dir=$1
control_dir=$2
vendor=$3
rank=$4
role=$5
gpu_indices=$6
num_nodes=$7
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export TZ=UTC
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
source "$repo_root/benchmarks/benchmark_lib.sh"
mkdir -p "$power_dir"
collector_rc=0
finished=0

write_control() {
    local path="$control_dir/$1"
    local pending="$path.tmp"
    printf '%s\n' "$2" > "$pending" || return
    # The control directory is created by the host user before Docker starts.
    # Containers must not strand root-owned files in the shared runner tree.
    if [[ -n "${POWERX_HOST_UID:-}" && -n "${POWERX_HOST_GID:-}" ]]; then
        chown "$POWERX_HOST_UID:$POWERX_HOST_GID" "$pending" || return
    fi
    mv -f "$pending" "$path"
}

finish() {
    local incoming_rc=$?
    [[ "$finished" == 0 ]] || return
    if [[ "$incoming_rc" != 0 ]]; then collector_rc=$incoming_rc; fi
    if ! _background_process_is_running "${GPU_MONITOR_PID:-}"; then collector_rc=1; fi
    stop_gpu_monitor
    if [[ "$vendor" == amd ]]; then
        amd-smi list --json > "$power_dir/gpu_metrics_devices_end.json" || collector_rc=1
    else
        nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,driver_version --format=csv \
            > "$power_dir/gpu_metrics_identity_end.csv" || collector_rc=1
    fi
    python3 -m infx.results.power.native_multinode end --directory "$power_dir" \
        --collector-exit-code "$collector_rc" || collector_rc=1
    if [[ -n "${POWERX_HOST_UID:-}" && -n "${POWERX_HOST_GID:-}" ]]; then
        chown -R "$POWERX_HOST_UID:$POWERX_HOST_GID" "$power_dir" || collector_rc=1
    fi
    write_control "done-$rank" "$collector_rc"
    finished=1
}
trap finish EXIT
trap 'collector_rc=130; AMD_MONITOR_STOP_TIMEOUT_S=0; exit 130' INT
trap 'collector_rc=143; AMD_MONITOR_STOP_TIMEOUT_S=0; exit 143' TERM HUP

case "${POWERX_CLOCK_SYNCHRONIZED:-false}" in
    yes|true) clock_synchronized=true ;;
    *) clock_synchronized=false ;;
esac

python3 -m infx.results.power.native_multinode begin --directory "$power_dir" \
    --vendor "$vendor" --rank "$rank" --role "$role" --gpu-indices "$gpu_indices" \
    --num-nodes "$num_nodes" --clock-synchronized "$clock_synchronized" || exit 1
printf '{"timestamp_timezone":"UTC"}\n' > "$power_dir/gpu_metrics_context.json" || exit 1
start_gpu_monitor --output "$power_dir/gpu_metrics.csv" || exit 1
[[ "$GPU_MONITOR_VENDOR" == "$vendor" ]] || exit 1
if [[ "$vendor" == amd ]]; then
    _write_amd_smi_sidecar "$power_dir/gpu_metrics_devices.json" list --json
fi
_background_process_is_running "$GPU_MONITOR_PID" || exit 1
write_control "ready-$rank" ready
while [[ ! -f "$control_dir/stop" ]]; do
    _background_process_is_running "$GPU_MONITOR_PID" || exit 1
    sleep 1 &
    wait $! || true
done
