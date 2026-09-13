#!/usr/bin/env bash
# Shared barriers for native collectors. Launchers own host scratch and mounts.

powerx_start_collector() {
    local power_dir="$1" control_dir="$2" vendor="$3" rank="$4" role="$5" gpus="$6" nodes="$7"
    local indices
    indices=$(seq -s, 0 "$((gpus - 1))") || return 1
    POWERX_CONTROL_DIR="$control_dir"
    POWERX_NUM_NODES="$nodes"
    bash "$(dirname "${BASH_SOURCE[0]}")/native_power_collect.sh" \
        "$power_dir" "$control_dir" "$vendor" "$rank" "$role" "$indices" "$nodes" &
    POWERX_COLLECTOR_PID=$!
}

powerx_write_control() {
    local path="$POWERX_CONTROL_DIR/$1"
    local pending="$path.tmp"
    printf '%s\n' "$2" > "$pending" || return
    if [[ -n "${POWERX_HOST_UID:-}" && -n "${POWERX_HOST_GID:-}" ]]; then
        chown "$POWERX_HOST_UID:$POWERX_HOST_GID" "$pending" || return
    fi
    mv -f "$pending" "$path"
}

powerx_wait_collectors() {
    local phase="$1" deadline=$((SECONDS + ${POWERX_BARRIER_TIMEOUT_S:-60})) rank pending failed
    while :; do
        pending=0
        failed=0
        for ((rank=0; rank<POWERX_NUM_NODES; rank++)); do
            if [[ "$phase" == ready && -f "$POWERX_CONTROL_DIR/done-$rank" ]]; then
                echo "PowerX collector $rank stopped before benchmark readiness" >&2
                return 1
            fi
            if [[ ! -f "$POWERX_CONTROL_DIR/$phase-$rank" ]]; then
                pending=1
            elif [[ "$phase" == done && "$(cat "$POWERX_CONTROL_DIR/done-$rank")" != 0 ]]; then
                failed=1
            fi
        done
        if [[ "$pending" == 0 ]]; then
            [[ "$failed" == 0 ]] || echo "One or more PowerX collectors failed" >&2
            return "$failed"
        fi
        if (( SECONDS >= deadline )); then
            echo "Timed out waiting for PowerX $phase receipts" >&2
            return 1
        fi
        sleep 1
    done
}

powerx_stop_collectors() {
    local rc=0
    powerx_write_control stop stop || rc=$?
    powerx_wait_collectors done || rc=$?
    powerx_reap_collector || rc=$?
    return "$rc"
}

powerx_reap_collector() {
    [[ -n "${POWERX_COLLECTOR_PID:-}" ]] || return 0
    local deadline=$((SECONDS + ${POWERX_BARRIER_TIMEOUT_S:-60})) rc=0
    while kill -0 "$POWERX_COLLECTOR_PID" 2>/dev/null; do
        if (( SECONDS >= deadline )); then
            kill -TERM "$POWERX_COLLECTOR_PID" 2>/dev/null || true
            # The shared AMD monitor drains for three seconds before writing receipts.
            local grace_deadline=$((SECONDS + 5))
            while kill -0 "$POWERX_COLLECTOR_PID" 2>/dev/null && (( SECONDS < grace_deadline )); do
                sleep 1
            done
            kill -KILL "$POWERX_COLLECTOR_PID" 2>/dev/null || true
            rc=1
            break
        fi
        sleep 1
    done
    wait "$POWERX_COLLECTOR_PID" || rc=$?
    POWERX_COLLECTOR_PID=""
    return "$rc"
}
