#!/usr/bin/env bash
start_amd_multinode_power() {
    [[ "${BENCH_INPUT_LEN:-}" == 8192 && "${BENCH_OUTPUT_LEN:-}" == 1024 &&
       "${EVAL_ONLY:-false}" != true && "${IS_AGENTIC:-0}" != 1 &&
       "${IS_AGENTIC:-false}" != true && "${DRY_RUN:-0}" != 1 ]] || return 0
    local prefill_nodes_per_worker decode_nodes_per_worker prefill_nodes total_nodes role tp worker_nodes gpu_count gpu_indices
    prefill_nodes_per_worker=$(( (PREFILL_TP_SIZE + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
    decode_nodes_per_worker=$(( (DECODE_TP_SIZE + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
    prefill_nodes=$(( prefill_nodes_per_worker * xP ))
    total_nodes=$(( prefill_nodes + decode_nodes_per_worker * yD ))
    [[ "$total_nodes" == "$NNODES" ]] || { echo 'PowerX: inconsistent AMD node topology' >&2; return 1; }
    if (( NODE_RANK < prefill_nodes )); then
        role=prefill; tp=$PREFILL_TP_SIZE; worker_nodes=$prefill_nodes_per_worker
    else
        role=decode; tp=$DECODE_TP_SIZE; worker_nodes=$decode_nodes_per_worker
    fi
    # Distributed tensor parallelism assigns equal local ranks to every node,
    # e.g. TP12 across two nodes uses GPU0..5 on each, not 8 GPUs plus 4 GPUs.
    (( tp % worker_nodes == 0 )) || { echo 'PowerX: uneven per-node TP layout' >&2; return 1; }
    gpu_count=$(( tp / worker_nodes ))
    gpu_indices=$(seq 0 $((gpu_count - 1)) | paste -sd, -)
    export POWERX_CONTROL_DIR="${BENCHMARK_LOGS_DIR}/power-control-${SLURM_JOB_ID}"
    local native_dir="/run_logs/slurm_job-${SLURM_JOB_ID}/native_power/node-${NODE_RANK}"
    bash "$WS_PATH/../../native_power_collect.sh" "$native_dir" "$POWERX_CONTROL_DIR" \
        amd "$NODE_RANK" "$role" "$gpu_indices" "$total_nodes" &
    POWERX_COLLECTOR_PID=$!
    # EXIT remains independent of serving-engine INT/TERM handlers.
    trap 'if [[ -n "${POWERX_COLLECTOR_PID:-}" ]]; then kill "$POWERX_COLLECTOR_PID" 2>/dev/null || true; wait "$POWERX_COLLECTOR_PID" 2>/dev/null || true; fi' EXIT
}

wait_amd_multinode_power() {
    local stage=$1 deadline=$((SECONDS + 60)) ready rank
    [[ -n "${POWERX_CONTROL_DIR:-}" ]] || return 0
    if [[ "$stage" == done ]]; then
        printf 'stop\n' > "$POWERX_CONTROL_DIR/stop"
        chown "$POWERX_HOST_UID:$POWERX_HOST_GID" "$POWERX_CONTROL_DIR/stop"
    fi
    while (( SECONDS < deadline )); do
        ready=1
        for ((rank=0; rank<NNODES; rank++)); do
            if [[ "$stage" == ready && -f "$POWERX_CONTROL_DIR/done-$rank" ]]; then
                echo "PowerX: node $rank collector exited before benchmark" >&2; return 1
            fi
            [[ -f "$POWERX_CONTROL_DIR/$stage-$rank" ]] || ready=0
        done
        if [[ "$ready" == 1 ]]; then
            if [[ "$stage" == done ]]; then
                for ((rank=0; rank<NNODES; rank++)); do
                    [[ "$(cat "$POWERX_CONTROL_DIR/done-$rank")" == 0 ]] || return 1
                done
            fi
            return 0
        fi
        sleep 1
    done
    echo "PowerX: timed out waiting for collector $stage receipts" >&2
    return 1
}
