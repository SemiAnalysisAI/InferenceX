#!/usr/bin/env bash

# Launchers source this file before changing into srt-slurm.
INFERENCEX_SLURM_UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

# Optionally inject synthetic acceptance into a recipe's speculative-config when
# SYNTHETIC_ACCEPTANCE=true (no-op otherwise). Call after the job-name override
# and before `srtctl apply` so the rendered job picks it up. Returns non-zero if
# the injector fails, so a broken opt-in never reaches srtctl with an unrewritten
# recipe; callers should propagate that rather than continuing.
inject_synthetic_acceptance() {
    local config_path="$1"
    local framework="$2"

    python3 "$GITHUB_WORKSPACE/runners/inject_synthetic_acceptance.py" \
        "$config_path" "$framework"
}

slurm_job_is_active() {
    local job_id="$1"
    squeue -j "$job_id" --noheader 2>/dev/null | grep -q "$job_id"
}

verify_slurm_job_completion() {
    local job_id="$1" records record state exit_code
    # Disappearance from squeue only means the job is no longer active. Read
    # the allocation's terminal record, not a successful batch/extern step.
    records=$(sacct -j "$job_id" --noheader --parsable2 --format=JobIDRaw,State,ExitCode 2>/dev/null) || records=""
    record=$(printf '%s\n' "$records" | awk -F'|' -v job="$job_id" '$1 == job {print; exit}')
    if [[ -n "$record" ]]; then
        IFS='|' read -r _ state exit_code <<< "$record"
    else
        # Recently completed jobs can precede accounting visibility.
        record=$(scontrol show job "$job_id" --oneliner 2>/dev/null) || record=""
        state=$(printf '%s\n' "$record" | sed -n 's/.*JobState=\([^ ]*\).*/\1/p')
        exit_code=$(printf '%s\n' "$record" | sed -n 's/.*ExitCode=\([^ ]*\).*/\1/p')
    fi
    printf '%s\n' "$record" > "${GITHUB_WORKSPACE:-.}/slurm_job_${job_id}_outcome.txt"
    if [[ "$state" != "COMPLETED" || "$exit_code" != "0:0" ]]; then
        echo "ERROR: Slurm job $job_id ended with state=${state:-unknown} exit=${exit_code:-unknown}" >&2
        return 1
    fi
}

stream_slurm_job_log() {
    local job_id="$1"
    local log_file="$2"

    while [[ ! -f "$log_file" ]]; do
        if ! slurm_job_is_active "$job_id"; then
            echo "ERROR: job $job_id failed before creating $log_file" >&2
            verify_slurm_job_completion "$job_id" || true
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
    verify_slurm_job_completion "$job_id"
}

copy_to_workspace() {
    local source_file="$1"
    local destination_file="$2"

    # A compute-visible runner workspace may be mounted directly into the
    # benchmark container. In that case the staged result already is the
    # workflow artifact, so copying it onto itself would fail with cp's
    # "same file" error even though the benchmark succeeded.
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

                    workspace_result_file=$(python3 "$INFERENCEX_SLURM_UTILS_DIR/../utils/result_filename.py" \
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
