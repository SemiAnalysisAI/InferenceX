#!/usr/bin/env bash

powerx_fixed_8k1k() {
    [[ "${REQUIRE_POWER:-0}" =~ ^(1|true|TRUE|yes|YES)$ &&
       "${SCENARIO_TYPE:-fixed-seq-len}" == "fixed-seq-len" && "${ISL:-}" == "8192" && "${OSL:-}" == "1024" &&
       "${IS_AGENTIC:-0}" != "1" && "${EVAL_ONLY:-false}" != "true" ]]
}

powerx_clone_srt() {
    local destination="$1"
    local revision="3f3b7af26e34acc8b62b39971bec839a19ac57a2"
    git clone https://github.com/edwingao28/srt-slurm.git "$destination" || return
    cd "$destination" || return
    git checkout --detach "$revision" || return
    [[ "$(git rev-parse HEAD)" == "$revision" ]] || return 1
    cp -a "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/." recipes/ || return
    git rev-parse HEAD > "$GITHUB_WORKSPACE/power-producer-sha.txt"
}

powerx_prepare_srt() {
    powerx_fixed_8k1k || return 0
    # Deliberately split the workflow's validated integer concurrency list.
    # shellcheck disable=SC2086
    CONFIG_FILE=$(python "$GITHUB_WORKSPACE/runners/prepare_srt_power.py" \
        "$CONFIG_FILE" ${CONC_LIST:?Missing matrix concurrencies}) || return
    export CONFIG_FILE
}

powerx_snapshot_srt() {
    [[ -n "${LOGS_DIR:-}" && -d "$LOGS_DIR" ]] || return 0
    if [[ "${USES_DCGM_POWER:-0}" == "1" ]]; then
        mkdir -p "$LOGS_DIR/power"
        cp "$GITHUB_WORKSPACE/exporter-image.sha256" "$LOGS_DIR/power/" 2>/dev/null || true
        cp "$GITHUB_WORKSPACE/power-producer-sha.txt" "$LOGS_DIR/power/" 2>/dev/null || true
    fi
    mkdir -p "$GITHUB_WORKSPACE/LOGS"
    cp -a "$LOGS_DIR/." "$GITHUB_WORKSPACE/LOGS/" 2>/dev/null || true
    bundle_server_logs "$LOGS_DIR" "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz" || true
}
