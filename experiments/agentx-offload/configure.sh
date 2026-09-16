#!/usr/bin/env bash
# Sourced by the canonical MiniMax recipe only for experiment=agentx-offload.
configure_offload_experiment() {
    check_env_vars KV_OFFLOADING TOTAL_CPU_DRAM_GB TP CONC DURATION RESULT_DIR SLURM_JOB_ID
    check_env_vars AIPERF_EXPERIMENTAL_FAST AIPERF_UNSAFE_OVERRIDE AIPERF_WARMUP_REQUESTS_PER_LANE
    if [[ "$AIPERF_EXPERIMENTAL_FAST" != 0 || "$AIPERF_UNSAFE_OVERRIDE" != false || "$AIPERF_WARMUP_REQUESTS_PER_LANE" != 10 || "$DURATION" != 3600 ]]; then
        echo 'Offload crossover evidence requires canonical duration and warmup.' >&2
        return 1
    fi
    mkdir -p "$RESULT_DIR"
    python3 experiments/agentx-offload/runtime.py prepare
    OFFLOAD_SCRATCH=$(python3 -c 'import json,os; print(json.load(open(os.path.join(os.environ["RESULT_DIR"],"offload_config.json")))["scratch"])')
    export OFFLOAD_SCRATCH
    trap 'finish_offload_experiment "$?"' EXIT
    OFFLOAD_CONFIG=$(python3 -c 'import json,os; v=json.load(open(os.path.join(os.environ["RESULT_DIR"],"offload_config.json")))["connector"]; print(json.dumps(v) if v else "")')
    OFFLOAD_ARGS=()
    if [[ -n "$OFFLOAD_CONFIG" ]]; then
        OFFLOAD_ARGS=(--kv-transfer-config "$OFFLOAD_CONFIG")
    fi
    EXPERIMENT_ARGS=(--kv-cache-memory-bytes 85899345920)
    if [[ "$KV_OFFLOADING" == dram || "$KV_OFFLOADING" == nvme ]]; then
        python3 runners/patch_vllm_simple_kv_offload.py
        export VLLM_USE_SIMPLE_KV_OFFLOAD=1
    fi
    python3 experiments/agentx-offload/runtime.py monitor --parent "$$" &
    OFFLOAD_MONITOR_PID=$!
}

finish_offload_experiment() {
    local result_code="$1"
    if [[ -n "${OFFLOAD_MONITOR_PID-}" ]]; then
        if ! kill -0 "$OFFLOAD_MONITOR_PID" 2>/dev/null; then
            echo 'Offload monitor exited unexpectedly; invalidating this run.' >&2
            if [[ ! -f "$RESULT_DIR/offload-guard.json" ]]; then
                printf '%s\n' '{"reason":"Offload monitor exited unexpectedly"}' > "$RESULT_DIR/offload-guard.json"
            fi
            result_code=1
        fi
        kill "$OFFLOAD_MONITOR_PID" 2>/dev/null || true
        wait "$OFFLOAD_MONITOR_PID" 2>/dev/null || true
    fi
    if [[ -n "${SERVER_PID-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo 'Server still alive; preserving its active cache.' >&2
        return 1
    fi
    python3 experiments/agentx-offload/runtime.py finish --exit-code "$result_code" || return 1
    return "$result_code"
}
