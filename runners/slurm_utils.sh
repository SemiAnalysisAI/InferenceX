#!/usr/bin/env bash

# Launchers source this file before changing into srt-slurm.
INFERENCEX_SLURM_UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$INFERENCEX_SLURM_UTILS_DIR/../benchmarks/benchmark_lib.sh" --validation-only || return 1

SRTCTL_EVAL_ARGS=(
    --set 'post_eval.command=["bash", "{infmax_workspace}/benchmarks/multi_node/srt_eval.sh", "{endpoint}", "{infmax_workspace}"]'
)

# Write a job-local cluster config; profiles contain only native srt-slurm settings.
write_srt_cluster_config() {
    if [[ $# -lt 3 || -z "$1" || -z "$2" || ( "$3" != 0 && "$3" != 1 ) ]]; then
        echo "Usage: write_srt_cluster_config profile output uses_power (0 or 1) [overrides...]" >&2
        return 1
    fi
    check_env_vars SLURM_ACCOUNT SLURM_PARTITION SRTCTL_ROOT SQUASH_FILE NGINX_SQUASH_FILE IMAGE
    local profile="$1" output="$2" uses_power="$3"
    shift 3
    local power_args=()
    if [[ "$uses_power" == 1 ]]; then
        check_env_vars DCGM_EXPORTER_SQSH
        power_args=(--container dcgm-exporter "$DCGM_EXPORTER_SQSH")
    fi
    PYTHONPATH="$INFERENCEX_SLURM_UTILS_DIR/..${PYTHONPATH:+:$PYTHONPATH}" \
        python3 -m infx.srt_slurm.cluster_config \
        "$INFERENCEX_SLURM_UTILS_DIR/srt-slurm/${profile}.yaml" "$output" \
        --var SLURM_ACCOUNT "$SLURM_ACCOUNT" --var SLURM_PARTITION "$SLURM_PARTITION" \
        --var SRTCTL_ROOT "$SRTCTL_ROOT" --var SQUASH_FILE "$SQUASH_FILE" \
        --var NGINX_SQUASH_FILE "$NGINX_SQUASH_FILE" --var IMAGE "$IMAGE" \
        "$@" "${power_args[@]}"
}

# Leaves the caller in the checkout, matching the launchers' installation flow.
# Every recipe is owned by InferenceX; srt-slurm 2 no longer ships recipes/.
setup_srt_slurm() {
    if [[ $# -ne 3 || -z "$1" || -z "$2" || ( "$3" != 0 && "$3" != 1 ) ]]; then
        echo "Usage: setup_srt_slurm destination framework uses_power (0 or 1)" >&2
        return 1
    fi
    local destination="$1" framework="$2" uses_power="$3"
    check_env_vars INFERENCEX_RUNTIME_ENV_VARS EVAL_ONLY
    SRT_EVAL_PASSTHROUGH=$(python3 - <<'PYENV'
import json
import os

names = [
    "EVAL_FRAMEWORK", "EVAL_CONC", "EVAL_LIMIT", "EVAL_SUITE",
    "SWEBENCH_GEN_MODE", "SWEBENCH_USE_MODAL", "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET", "IS_AGENTIC", "SCENARIO_TYPE",
    "TP", "EP_SIZE", "DP_ATTENTION", "PP_SIZE", "DCP_SIZE", "PCP_SIZE", "CONC",
]
print(json.dumps(names + os.environ["INFERENCEX_RUNTIME_ENV_VARS"].split()))
PYENV
    ) || return 1
    SRTCTL_EVAL_ARGS+=(--set "post_eval.passthrough_env=$SRT_EVAL_PASSTHROUGH")
    # Custom benchmarks inherit exported workflow settings through sbatch/srun;
    # native recipe environment and benchmark.env retain their override priority.
    local source="$INFERENCEX_SLURM_UTILS_DIR/../utils/srt-slurm"
    if [[ "$framework" == "tilert" ]]; then
        # TileRT still needs its legacy runtime until the native backend and router land.
        SRT_SLURM_COMMIT=6bc3f306bdafa1edfb5dded2fcda8f1ccede1bde
        git init "$destination" || return 1
        git -C "$destination" remote add origin https://github.com/SemiAnalysisAI/srt-slurm.git || return 1
        git -C "$destination" fetch --depth=1 origin "$SRT_SLURM_COMMIT" || return 1
        git -C "$destination" checkout --detach "$SRT_SLURM_COMMIT" || return 1
    else
        if [[ ! -e "$source/.git" ]]; then
            echo "Missing srt-slurm submodule; run git submodule update --init before launching." >&2
            return 1
        fi
        SRT_SLURM_COMMIT=$(git -C "$source" rev-parse HEAD) || return 1
        SRTCTL_EVAL_ARGS+=(--set benchmark.stream_output=true)
        # A local clone keeps job writes isolated and preserves upstream Git provenance.
        git clone --no-hardlinks "$source" "$destination" || return 1
        # Temporary fixes awaiting upstream merge; see runners/srt-slurm/patches/README.md.
        local patch
        for patch in "$GITHUB_WORKSPACE"/runners/srt-slurm/patches/*.patch; do
            [[ -e "$patch" ]] || continue
            git -C "$destination" apply "$patch" || return 1
        done
    fi
    cd "$destination" || return 1
    [[ "$(git rev-parse HEAD)" == "$SRT_SLURM_COMMIT" ]] || return 1
    git rev-parse HEAD > "$GITHUB_WORKSPACE/srt-slurm-sha.txt" || return 1
    if [[ "$uses_power" == "1" ]]; then
        cp "$GITHUB_WORKSPACE/srt-slurm-sha.txt" "$GITHUB_WORKSPACE/power-producer-sha.txt" || return 1
    fi
    mkdir -p recipes benchmarks/multi_node || return 1
    cp -R "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/." recipes/ || return 1
    # Both CONFIG_FILE spellings currently occur in master configs.
    ln -s ../../recipes benchmarks/multi_node/srt-slurm-recipes || return 1
    cp -R "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/configs/." configs/ || return 1
}

# Use the requested image's cache identity, never a convenient older squash file.
resolve_h100_srt_container() {
    local image="$1" framework="$2"
    [[ -n "$image" && "$image" != *[[:space:]]* ]] || return 1
    CONTAINER_KEY="${image/nvcr.io\//nvcr.io#}"
    case "$framework" in
        dynamo-sglang)
            SQUASH_FILE="/mnt/nfs/lustre/containers/$(printf '%s' "$image" | sed 's/[\/:@#]/_/g').sqsh"
            ;;
        dynamo-trt)
            SQUASH_FILE="/mnt/nfs/sa-shared/containers/$(printf '%s' "${image#nvcr.io/}" | sed 's/[\/:@#]/+/g').sqsh"
            ;;
        *) return 1 ;;
    esac
}

check_staged_srt_assets() {
    local model="$1" image="$2"
    if [[ ! -r "$model/config.json" ]] || ! unsquashfs -s "$image" >/dev/null 2>&1; then
        echo 'ERROR: readiness-blocked: staged model/config or requested container is unavailable' >&2
        return 1
    fi
}

# AgentX acceptance comes from the committed golden curve; evals use real verification.
apply_srt_recipe() {
    if [[ $# -lt 2 || -z "$1" || -z "$2" ]]; then
        echo "Usage: apply_srt_recipe config framework [srtctl arguments...]" >&2
        return 1
    fi
    check_env_vars MODEL_PREFIX IS_AGENTIC EVAL_ONLY SPEC_DECODING
    if [[ "$IS_AGENTIC" == 1 || "$IS_AGENTIC" == true ]] && [[ "$EVAL_ONLY" != true && "$SPEC_DECODING" != none ]]; then
        check_env_vars THINKING_MODE
    fi
    local config="$1" framework="$2"
    shift 2
    PYTHONPATH="$INFERENCEX_SLURM_UTILS_DIR/..${PYTHONPATH:+:$PYTHONPATH}" \
        python3 -m infx.srt_slurm.synthetic_acceptance \
        "$config" "$framework" -- "$@"
}

# One native submission per fixed-sequence matrix point, shared across Slurm pools.
launch_srt_single_node() {
    set -eo pipefail
    local profile="$1"
    shift
    check_env_vars GITHUB_WORKSPACE SRT_RECIPE FRAMEWORK MODEL MODEL_PREFIX IMAGE PRECISION \
        TP PP_SIZE DCP_SIZE PCP_SIZE EP_SIZE DP_ATTENTION GPU_COUNT IS_AGENTIC SPEC_DECODING \
        CONC ISL OSL RANDOM_RANGE_RATIO RESULT_FILENAME GPU_MONITOR_INTERVAL SRT_MODEL_PATH \
        HF_HUB_CACHE_MOUNT HF_HUB_CACHE SALLOC_TIME_LIMIT
    SRT_SINGLE_NODE_ROOT=$(mktemp -d "$GITHUB_WORKSPACE/srt-single.XXXXXX")
    SRTCTL_ROOT="$SRT_SINGLE_NODE_ROOT/checkout"
    export INFMAX_WORKSPACE="$GITHUB_WORKSPACE"
    setup_srt_slurm "$SRTCTL_ROOT" "$FRAMEWORK" 0
    if ! command -v uv >/dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source "$HOME/.local/bin/env"
    fi
    uv venv .venv
    source .venv/bin/activate
    uv pip install -e .
    export PYTHONPATH="$GITHUB_WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"

    python3 -m infx.srt_slurm.single_node prepare "$GITHUB_WORKSPACE/$SRT_RECIPE" "$SRT_SINGLE_NODE_ROOT/arguments"
    mapfile -d '' -t SRT_RUNTIME_ARGS < "$SRT_SINGLE_NODE_ROOT/arguments"
    SRT_SELECTED_RECIPE="${SRT_RUNTIME_ARGS[0]}"
    SRT_RUNTIME_ARGS=("${SRT_RUNTIME_ARGS[@]:1}")
    SRT_RUNTIME_ARGS+=(
        --set 'post_eval.command=["bash", "{infmax_workspace}/benchmarks/single_node/srt_eval.sh", "{endpoint}", "/logs/infx-eval-exit-code"]'
        --set "post_eval.passthrough_env=$SRT_EVAL_PASSTHROUGH"
    )
    # Reuse only a valid cache for this exact image. Missing caches are imported
    # by native Pyxis inside the same benchmark allocation.
    SRT_CONTAINER="$IMAGE"
    if [[ -n "${SRT_SQUASH_FILE:-}" && -r "$SRT_SQUASH_FILE" ]] && unsquashfs -s "$SRT_SQUASH_FILE" >/dev/null 2>&1; then
        SRT_CONTAINER="$SRT_SQUASH_FILE"
    fi
    python3 -m infx.srt_slurm.cluster_config \
        "$INFERENCEX_SLURM_UTILS_DIR/srt-slurm/${profile}.yaml" srtslurm.yaml \
        --var SRTCTL_ROOT "$SRTCTL_ROOT" --var SQUASH_FILE "$SRT_CONTAINER" \
        --var IMAGE "$IMAGE" --var NGINX_SQUASH_FILE nginx:1.27.4 \
        --var SRT_DEFAULT_TIME_LIMIT "$SALLOC_TIME_LIMIT" \
        --model "hf:$MODEL" "$SRT_MODEL_PATH" --container "$IMAGE" "$SRT_CONTAINER" \
        --mount "$HF_HUB_CACHE_MOUNT" "$HF_HUB_CACHE" --exclusive "$@"
    make setup ARCH=x86_64

    SRT_JOB_ID=""
    SRT_JOB_OUTPUT=""
    finish_native_single_node() {
        local rc=$? artifact
        trap - EXIT
        # Submission may succeed immediately before cancellation or a client error.
        if [[ -z "$SRT_JOB_ID" ]] && python3 -m infx.srt_slurm.single_node submission \
            "$GITHUB_WORKSPACE/srt-single-node-submission.json" > "$SRT_SINGLE_NODE_ROOT/submission-fields" 2>/dev/null; then
            mapfile -t SRT_SUBMISSION < "$SRT_SINGLE_NODE_ROOT/submission-fields"
            SRT_JOB_ID="${SRT_SUBMISSION[0]}"
            SRT_JOB_OUTPUT="${SRT_SUBMISSION[1]}"
        fi
        if [[ -n "$SRT_JOB_ID" ]] && slurm_job_is_active "$SRT_JOB_ID"; then
            scancel "$SRT_JOB_ID" || true
        fi
        if [[ -n "$SRT_JOB_OUTPUT" && -d "$SRT_JOB_OUTPUT" ]]; then
            bundle_server_logs "$SRT_JOB_OUTPUT" "$GITHUB_WORKSPACE/srt-single-node-logs.tar.gz"
            for artifact in "$SRT_JOB_OUTPUT/logs/$RESULT_FILENAME.json" "$SRT_JOB_OUTPUT"/logs/gpu_metrics*; do
                [[ -f "$artifact" ]] || continue
                copy_to_workspace "$artifact" "$GITHUB_WORKSPACE/$(basename "$artifact")" || rc=1
            done
        fi
        exit "$rc"
    }
    trap finish_native_single_node EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    local submission_rc=0
    apply_srt_recipe "$SRT_SELECTED_RECIPE" "$FRAMEWORK" \
        --json --yes --output "$SRT_SINGLE_NODE_ROOT/outputs" "${SRT_RUNTIME_ARGS[@]}" \
        > "$GITHUB_WORKSPACE/srt-single-node-submission.json" || submission_rc=$?
    if (( submission_rc != 0 )); then
        cat "$GITHUB_WORKSPACE/srt-single-node-submission.json" >&2
        return "$submission_rc"
    fi
    python3 -m infx.srt_slurm.single_node submission "$GITHUB_WORKSPACE/srt-single-node-submission.json" \
        > "$SRT_SINGLE_NODE_ROOT/submission-fields"
    mapfile -t SRT_SUBMISSION < "$SRT_SINGLE_NODE_ROOT/submission-fields"
    SRT_JOB_ID="${SRT_SUBMISSION[0]}"
    SRT_JOB_OUTPUT="${SRT_SUBMISSION[1]}"
    stream_slurm_job_log "$SRT_JOB_ID" "$SRT_JOB_OUTPUT/logs/sweep_${SRT_JOB_ID}.log"
    verify_slurm_job_status "$SRT_JOB_ID"
    # Native SRT treats post-throughput eval failure as non-fatal. InferenceX
    # requires every requested eval to finish successfully, including staging.
    if [[ "$RUN_EVAL" == true || "$EVAL_ONLY" == true ]]; then
        test -f "$SRT_JOB_OUTPUT/logs/infx-eval-exit-code"
        test "$(cat "$SRT_JOB_OUTPUT/logs/infx-eval-exit-code")" = 0
    fi
    if [[ "$EVAL_ONLY" != true ]]; then
        test -s "$SRT_JOB_OUTPUT/logs/$RESULT_FILENAME.json"
    fi

}

slurm_job_is_active() {
    local job_id="$1"
    squeue -j "$job_id" --noheader 2>/dev/null | grep -q "$job_id"
}

stream_slurm_job_log() {
    local job_id="$1"
    local log_file="$2"

    while [[ ! -f "$log_file" ]]; do
        if ! slurm_job_is_active "$job_id"; then
            echo "ERROR: job $job_id failed before creating $log_file" >&2
            scontrol show job "$job_id" || true
            return 1
        fi
        sleep 5
    done

    (
        while slurm_job_is_active "$job_id"; do
            sleep 10
        done
    ) &
    local poll_pid=$!

    echo "Tailing $log_file"
    tail -F -s 2 -n+1 "$log_file" --pid="$poll_pid" 2>/dev/null
    wait "$poll_pid"
}

verify_slurm_job_status() {
    local job_id="$1"
    # Disappearing from squeue means terminal, not successful. Accounting can
    # lag briefly; inspect only the allocation, never successful service steps.
    local attempt accounting state exit_code controller field controller_job_id
    local -a controller_fields
    for attempt in {1..10}; do
        accounting=$(sacct -X -n -P -j "$job_id" --format=State,ExitCode 2>/dev/null) || accounting=""
        IFS='|' read -r state exit_code <<< "$accounting"
        if [[ -z "$state" ]]; then
            # Some pools do not expose slurmdbd. The controller retains recent
            # terminal allocations; require its state and exit code, too.
            controller=$(scontrol show job -o "$job_id" 2>/dev/null) || controller=""
            controller_job_id=""
            read -r -a controller_fields <<< "$controller"
            for field in "${controller_fields[@]}"; do
                case "$field" in
                    JobId=*) controller_job_id="${field#JobId=}" ;;
                    JobState=*) state="${field#JobState=}" ;;
                    ExitCode=*) exit_code="${field#ExitCode=}" ;;
                esac
            done
            if [[ "$controller_job_id" != "$job_id" ]]; then
                state=""
                exit_code=""
            fi
        fi
        case "$state" in
            COMPLETED)
                if [[ "$exit_code" == "0:0" ]]; then
                    return 0
                fi
                ;;
            ""|PENDING|RUNNING|CONFIGURING|COMPLETING)
                sleep 1
                continue
                ;;
        esac
        echo "ERROR: Slurm job $job_id ended with state=$state exit_code=$exit_code" >&2
        return 1
    done
    echo "ERROR: could not verify terminal Slurm status for job $job_id" >&2
    return 1
}

copy_to_workspace() {
    local source_file="$1"
    local destination_file="$2"

    # When the runner workspace is mounted into the container the staged result
    # already is the artifact, and cp onto itself fails with "same file".
    if [[ -e "$destination_file" && "$source_file" -ef "$destination_file" ]]; then
        echo "Result already present at $destination_file"
        return 0
    fi

    if ! cp "$source_file" "$destination_file"; then
        echo "ERROR: failed to copy $source_file to $destination_file" >&2
        return 1
    fi
    echo "Copied $(basename "$source_file") to $destination_file"
}

# Preserve short SRT filenames and report failures even inside an `if`/`||` caller.
copy_fixed_sequence_results() {
    local logs_dir="$1" workspace="$2" result_filename="$3"
    local result_subdirs result_subdir result_files result_file config_name
    local filename concurrency gpus ctx gen workspace_result_file

    result_subdirs=$(find "$logs_dir" -maxdepth 1 -type d -name "*isl*osl*" 2>/dev/null) || return 1

    if [ -z "$result_subdirs" ]; then
        echo "Warning: No result subdirectories found in $logs_dir"
    else
        for result_subdir in $result_subdirs; do
            echo "Processing result subdirectory: $result_subdir"
            config_name=$(basename "$result_subdir")
            result_files=$(find "$result_subdir" -name "results_concurrency_*.json" 2>/dev/null) || return 1

            for result_file in $result_files; do
                if [ -f "$result_file" ]; then
                    # Both disaggregated (_ctx_C_gen_D) and aggregated names occur.
                    filename=$(basename "$result_file")
                    concurrency=$(echo "$filename" | sed -n 's/results_concurrency_\([0-9]*\)_gpus_.*/\1/p')
                    gpus=$(echo "$filename" | sed -n 's/results_concurrency_[0-9]*_gpus_\([0-9][0-9]*\).*/\1/p')
                    ctx=$(echo "$filename" | sed -n 's/.*_ctx_\([0-9]*\)_gen_.*/\1/p')
                    gen=$(echo "$filename" | sed -n 's/.*_gen_\([0-9]*\)\.json/\1/p')

                    echo "Processing concurrency $concurrency with $gpus GPUs (ctx: $ctx, gen: $gen): $result_file"

                    workspace_result_file=$(PYTHONPATH="$INFERENCEX_SLURM_UTILS_DIR/..${PYTHONPATH:+:$PYTHONPATH}" python3 -m infx.results.result_filename \
                        --point "$result_filename" "$config_name" "$concurrency" "$gpus" "$ctx" "$gen") || return 1
                    workspace_result_file="$workspace/$workspace_result_file"
                    copy_to_workspace "$result_file" "$workspace_result_file" || return 1

                    echo "Copied result file to: $workspace_result_file"
                fi
            done
        done
    fi

    echo "All result files processed"
}

copy_agentic_results() {
    local source_dir="$1"
    local workspace="$2"
    local result_filename="$3"
    local result_file
    local copied=0

    if [[ ! -d "$source_dir" ]]; then
        echo "ERROR: agentic result directory not found at $source_dir" >&2
        return 1
    fi

    while IFS= read -r -d '' result_file; do
        copy_to_workspace \
            "$result_file" \
            "$workspace/$(basename "$result_file")" || return 1
        copied=$((copied + 1))
    done < <(
        find "$source_dir" -maxdepth 1 -type f \
            -name "${result_filename}_conc*.json" -print0
    )

    if [[ "$copied" -eq 0 ]]; then
        echo "ERROR: no ${result_filename}_conc*.json results found in $source_dir" >&2
        return 1
    fi

    echo "Copied $copied agentic result file(s)"
}

collect_agentic_power_results() {
    local job_id="$1" logs_dir="$2" source_dir="$3" workspace="$4"
    local result_filename="$5" producer_sha="$6"
    shift 6
    local rc=0 concurrency attempt
    [[ "$#" -gt 0 ]] || return 1
    mkdir -p "$logs_dir/power" || return 1
    logs_dir="$(cd "$logs_dir" && pwd -P)" || return 1
    workspace="$(cd "$workspace" && pwd -P)" || return 1

    # Accounting can lag squeue removal; retry only missing or nonterminal rows.
    for attempt in 1 2 3; do
        echo "$attempt" > "$logs_dir/power/native-job-status-attempts.txt"
        sacct -X -n -P -j "$job_id" --format=JobIDRaw,State,ExitCode \
            > "$logs_dir/power/native-job-status.txt" \
            2>> "$logs_dir/power/native-job-status.stderr" || true
        if awk -F'|' -v job="$job_id" '
            $1 == job && $2 !~ /^(PENDING|RUNNING|COMPLETING)$/ { found = 1 }
            END { exit !found }
        ' "$logs_dir/power/native-job-status.txt"; then
            break
        fi
        if [[ "$attempt" != "3" ]]; then sleep 5; fi
    done
    if ! awk -F'|' -v job="$job_id" '
        $1 == job { found = 1; if ($2 != "COMPLETED" || $3 != "0:0") failed = 1 }
        END { exit (!found || failed) }
    ' "$logs_dir/power/native-job-status.txt"; then
        rc=1
    fi
    copy_agentic_results "$source_dir" "$workspace" "$result_filename" || rc=$?
    for concurrency in "$@"; do
        (
            check_env_vars INFERENCEX_RESULTS_PYTHON
            cd "$workspace" || exit 1
            PYTHONPATH="$INFERENCEX_SLURM_UTILS_DIR/..${PYTHONPATH:+:$PYTHONPATH}" "$INFERENCEX_RESULTS_PYTHON" -m infx.results.agentic.power_adapter \
                --result-dir "$logs_dir/agentic/conc_${concurrency}" \
                --agg-result "$workspace/${result_filename}_conc${concurrency}.json" \
                --power-dir "$logs_dir/power" \
                --logs-root "$logs_dir" \
                --expected-producer-sha "$producer_sha" \
                --require-power
        ) || rc=$?
    done
    return "$rc"
}

copy_eval_artifacts() {
    local eval_dir="$1"
    local workspace="$2"

    if [[ ! -d "$eval_dir" ]]; then
        echo "WARNING: eval results not found at $eval_dir" >&2
        return 0
    fi

    local eval_file
    while IFS= read -r -d '' eval_file; do
        copy_to_workspace "$eval_file" "$workspace/$(basename "$eval_file")" || return 1
    done < <(find "$eval_dir" -maxdepth 1 -type f -print0)
}

bundle_server_logs() {
    local logs_dir="$1"
    local archive="$2"

    if [[ ! -d "$logs_dir" ]] || ! find "$logs_dir" -mindepth 1 -print -quit | grep -q .; then
        return 0
    fi

    tar czf "$archive" -C "$logs_dir" . 2>/dev/null || {
        echo "WARNING: failed to bundle $archive" >&2
        return 0
    }
}
