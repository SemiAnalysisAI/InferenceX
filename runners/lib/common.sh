#!/usr/bin/env bash
# Shared utilities for InferenceX runner scripts.
# Source this file: source "$(dirname "$0")/lib/common.sh"

set -euo pipefail

# Convert a container image name to a squash file name.
# Usage: squash_name=$(image_to_squash_name "$IMAGE")
image_to_squash_name() {
    local image="$1"
    echo "$image" | sed 's/[\/:@#]/_/g'
}

# Allocate a SLURM job and return the job ID.
# Usage: JOB_ID=$(allocate_slurm_job --partition PARTITION --gres "gpu:8" --time 180 --job-name "name")
# Optional: --account, --exclude, --cpus-per-task, --exclusive
allocate_slurm_job() {
    local partition="" gres="" time_limit="" job_name="" account="" exclude="" cpus="" exclusive=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --partition)       partition="$2"; shift 2 ;;
            --gres)            gres="$2"; shift 2 ;;
            --time)            time_limit="$2"; shift 2 ;;
            --job-name)        job_name="$2"; shift 2 ;;
            --account)         account="$2"; shift 2 ;;
            --exclude)         exclude="$2"; shift 2 ;;
            --cpus-per-task)   cpus="$2"; shift 2 ;;
            --exclusive)       exclusive="true"; shift ;;
            *)                 echo "Unknown parameter: $1" >&2; return 1 ;;
        esac
    done

    local cmd="salloc --partition=$partition --gres=$gres --time=$time_limit --no-shell --job-name=$job_name"
    [[ -n "$account" ]] && cmd+=" --account=$account"
    [[ -n "$exclude" ]] && cmd+=" --exclude=$exclude"
    [[ -n "$cpus" ]] && cmd+=" --cpus-per-task=$cpus"
    [[ -n "$exclusive" ]] && cmd+=" --exclusive"

    local output
    output=$($cmd 2>&1 | tee /dev/stderr)
    local job_id
    job_id=$(echo "$output" | grep -oP 'Granted job allocation \K[0-9]+' || true)

    if [[ -z "$job_id" ]]; then
        echo "ERROR: salloc failed to allocate a job" >&2
        return 1
    fi
    echo "$job_id"
}

# Wait for a SLURM job's log file to appear and stream it until the job completes.
# Usage: wait_and_stream_job_log --job-id JOB_ID --log-file LOG_FILE
wait_and_stream_job_log() {
    local job_id="" log_file=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --job-id)   job_id="$2"; shift 2 ;;
            --log-file) log_file="$2"; shift 2 ;;
            *)          echo "Unknown parameter: $1" >&2; return 1 ;;
        esac
    done

    while ! ls "$log_file" &>/dev/null; do
        if ! squeue -j "$job_id" --noheader 2>/dev/null | grep -q "$job_id"; then
            echo "ERROR: Job $job_id failed before creating log file" >&2
            scontrol show job "$job_id" >&2
            return 1
        fi
        echo "Waiting for JOB_ID $job_id to begin and $log_file to appear..."
        sleep 5
    done

    (
        while squeue -j "$job_id" --noheader 2>/dev/null | grep -q "$job_id"; do
            sleep 10
        done
    ) &
    local poll_pid=$!

    echo "Tailing LOG_FILE: $log_file"
    tail -F -s 2 -n+1 "$log_file" --pid=$poll_pid 2>/dev/null || true

    wait $poll_pid 2>/dev/null || true
    echo "Job $job_id completed!"
}

# Import a Docker image to a squash file using enroot with flock-based locking.
# Usage: import_squash_file --image IMAGE --squash-file SQUASH_FILE --job-id JOB_ID
import_squash_file() {
    local image="" squash_file="" job_id=""
    local flock_timeout=600

    while [[ $# -gt 0 ]]; do
        case $1 in
            --image)          image="$2"; shift 2 ;;
            --squash-file)    squash_file="$2"; shift 2 ;;
            --job-id)         job_id="$2"; shift 2 ;;
            --flock-timeout)  flock_timeout="$2"; shift 2 ;;
            *)                echo "Unknown parameter: $1" >&2; return 1 ;;
        esac
    done

    local lock_file="${squash_file}.lock"

    if [[ -n "$job_id" ]]; then
        srun --jobid="$job_id" --job-name="$RUNNER_NAME" bash -c "
            exec 9>\"$lock_file\"
            flock -w $flock_timeout 9 || { echo 'Failed to acquire lock for $squash_file'; exit 1; }
            if unsquashfs -l \"$squash_file\" > /dev/null 2>&1; then
                echo 'Squash file already exists and is valid, skipping import'
            else
                rm -f \"$squash_file\"
                enroot import -o \"$squash_file\" docker://$image
            fi
        "
    else
        exec 9>"$lock_file"
        flock -w "$flock_timeout" 9 || { echo "Failed to acquire lock for $squash_file" >&2; return 1; }
        if unsquashfs -l "$squash_file" > /dev/null 2>&1; then
            echo "Squash file already exists and is valid, skipping import"
        else
            rm -f "$squash_file"
            enroot import -o "$squash_file" docker://"$image"
        fi
    fi
}

# Process result files from multinode benchmarks and copy to workspace.
# Usage: process_multinode_results --logs-dir LOGS_DIR --result-filename RESULT_FILENAME
process_multinode_results() {
    local logs_dir="" result_filename=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --logs-dir)         logs_dir="$2"; shift 2 ;;
            --result-filename)  result_filename="$2"; shift 2 ;;
            *)                  echo "Unknown parameter: $1" >&2; return 1 ;;
        esac
    done

    if [[ ! -d "$logs_dir" ]]; then
        echo "Warning: Logs directory not found at $logs_dir" >&2
        return 1
    fi

    echo "Found logs directory: $logs_dir"
    cp -r "$logs_dir" "$GITHUB_WORKSPACE/LOGS"
    tar czf "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz" -C "$logs_dir" .

    local result_subdirs
    result_subdirs=$(find "$logs_dir" -maxdepth 1 -type d -name "*isl*osl*" 2>/dev/null || true)

    if [[ -z "$result_subdirs" ]]; then
        echo "Warning: No result subdirectories found in $logs_dir"
        return 0
    fi

    for result_subdir in $result_subdirs; do
        echo "Processing result subdirectory: $result_subdir"
        local config_name
        config_name=$(basename "$result_subdir")

        local result_files
        result_files=$(find "$result_subdir" -name "results_concurrency_*.json" 2>/dev/null || true)

        for result_file in $result_files; do
            if [[ -f "$result_file" ]]; then
                local filename concurrency gpus ctx gen
                filename=$(basename "$result_file")
                concurrency=$(echo "$filename" | sed -n 's/results_concurrency_\([0-9]*\)_gpus_.*/\1/p')
                gpus=$(echo "$filename" | sed -n 's/results_concurrency_[0-9]*_gpus_\([0-9]*\)_ctx_.*/\1/p')
                ctx=$(echo "$filename" | sed -n 's/.*_ctx_\([0-9]*\)_gen_.*/\1/p')
                gen=$(echo "$filename" | sed -n 's/.*_gen_\([0-9]*\)\.json/\1/p')

                echo "Processing concurrency $concurrency with $gpus GPUs (ctx: $ctx, gen: $gen): $result_file"

                local workspace_result_file
                workspace_result_file="$GITHUB_WORKSPACE/${result_filename}_${config_name}_conc${concurrency}_gpus_${gpus}_ctx_${ctx}_gen_${gen}.json"
                cp "$result_file" "$workspace_result_file"
                echo "Copied result file to: $workspace_result_file"
            fi
        done
    done

    echo "All result files processed"
}

# Collect eval results from multinode benchmark runs.
# Usage: collect_eval_results --logs-dir LOGS_DIR
collect_eval_results() {
    local logs_dir=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --logs-dir) logs_dir="$2"; shift 2 ;;
            *)          echo "Unknown parameter: $1" >&2; return 1 ;;
        esac
    done

    local eval_dir="$logs_dir/eval_results"
    if [[ -d "$eval_dir" ]]; then
        echo "Extracting eval results from $eval_dir"
        shopt -s nullglob
        for eval_file in "$eval_dir"/*; do
            [[ -f "$eval_file" ]] || continue
            cp "$eval_file" "$GITHUB_WORKSPACE/"
            echo "Copied eval artifact: $(basename "$eval_file")"
        done
        shopt -u nullglob
    else
        echo "WARNING: No eval results found at $eval_dir"
    fi
}

# Clean up srt-slurm outputs to prevent NFS lock file issues.
# Usage: cleanup_srt_outputs
cleanup_srt_outputs() {
    echo "Cleaning up srt-slurm outputs..."
    for i in 1 2 3 4 5; do
        rm -rf outputs 2>/dev/null && break
        echo "Retry $i/5: Waiting for NFS locks to release..."
        sleep 10
    done
    find . -name '.nfs*' -delete 2>/dev/null || true
}
