#!/usr/bin/env bash

# NScale grants the runner interactive allocations but rejects equivalent
# batch submissions. Adapt the single `sbatch SCRIPT` call made by srtctl to
# the cluster's supported salloc --no-shell contract, then run the generated
# host-side orchestrator against that allocation.

set -euo pipefail

if [[ $# -ne 1 || ! -f "$1" ]]; then
    echo "Usage: sbatch SCRIPT" >&2
    exit 2
fi

script_path="$1"
if grep -q '^#SBATCH hetjob' "$script_path"; then
    echo "Error: the NScale sbatch adapter does not support heterogeneous jobs" >&2
    exit 2
fi

directive() {
    local name="$1"
    sed -n "s/^#SBATCH --${name}=//p" "$script_path" | head -n1
}

nodes="$(directive nodes)"
ntasks="$(directive ntasks)"
ntasks_per_node="$(directive ntasks-per-node)"
job_name="$(directive job-name)"
output_pattern="$(directive output)"
time_limit="$(directive time)"
account="$(directive account)"
partition="$(directive partition)"
gres="$(directive gres)"

: "${nodes:?missing #SBATCH --nodes}"
: "${ntasks:?missing #SBATCH --ntasks}"
: "${ntasks_per_node:?missing #SBATCH --ntasks-per-node}"
: "${job_name:?missing #SBATCH --job-name}"
: "${output_pattern:?missing #SBATCH --output}"
: "${time_limit:?missing #SBATCH --time}"
: "${account:?missing #SBATCH --account}"
: "${partition:?missing #SBATCH --partition}"
: "${gres:?missing #SBATCH --gres}"

allocation_output="$(salloc \
    --nodes="$nodes" \
    --ntasks="$ntasks" \
    --ntasks-per-node="$ntasks_per_node" \
    --exclusive \
    --mem=0 \
    --gres="$gres" \
    --time="$time_limit" \
    --account="$account" \
    --partition="$partition" \
    --job-name="$job_name" \
    --no-shell 2>&1)" || {
        status=$?
        printf '%s\n' "$allocation_output" >&2
        exit "$status"
    }
printf '%s\n' "$allocation_output" >&2

job_id="$(printf '%s\n' "$allocation_output" | sed -nE 's/.*job allocation ([0-9]+).*/\1/p' | tail -n1)"
if [[ -z "$job_id" ]]; then
    echo "Error: could not extract the NScale allocation ID" >&2
    exit 1
fi

cancel_allocation() {
    scancel "$job_id" >/dev/null 2>&1 || true
}

state_root="${NSCALE_SALLOC_STATE_DIR:?NSCALE_SALLOC_STATE_DIR is required}"
mkdir -p "$state_root"
persistent_script="$state_root/${job_id}.slurm"
if ! cp "$script_path" "$persistent_script"; then
    cancel_allocation
    exit 1
fi
chmod +x "$persistent_script"

node_list="$(squeue --job="$job_id" --noheader --format='%N' | head -n1)"
if [[ -z "$node_list" || "$node_list" == "(null)" ]]; then
    echo "Error: allocation $job_id has no assigned NScale nodes" >&2
    cancel_allocation
    exit 1
fi

log_file="${output_pattern//%j/$job_id}"
exit_file="${log_file}.exit-code"
ready_file="$state_root/${job_id}.ready"
mkdir -p "$(dirname "$log_file")"

# Close all inherited pipes before returning to srtctl. The workflow tails the
# generated log and remains alive until this supervisor releases the allocation.
(
    set +e
    ready=0
    for _ in $(seq 1 120); do
        if [[ -f "$ready_file" ]]; then
            ready=1
            break
        fi
        sleep 1
    done
    if [[ "$ready" != "1" ]]; then
        echo "Error: srtctl did not finish staging allocation $job_id" > "$log_file"
        printf '1\n' > "$exit_file"
        cancel_allocation
        exit 1
    fi

    export RUNNER_TRACKING_ID=""

    # The model is staged on node-local /scratch, so runtime validation and the
    # host-side orchestrator must run on a compute node. This step consumes no
    # GPU and permits the orchestrator's worker srun steps to overlap it.
    srun \
        --jobid="$job_id" \
        --nodes=1 \
        --ntasks=1 \
        --overlap \
        --gres=none \
        /usr/bin/env \
            SLURM_JOB_NUM_NODES="$nodes" \
            SLURM_NNODES="$nodes" \
            SLURM_NTASKS="$ntasks" \
            SLURM_NODELIST="$node_list" \
            bash "$persistent_script" >"$log_file" 2>&1
    exit_code=$?
    printf '%s\n' "$exit_code" > "$exit_file"
    rm -f "$persistent_script" "$ready_file"
    cancel_allocation
) </dev/null >/dev/null 2>&1 &

# Match sbatch's output contract; srtctl parses the final whitespace-delimited
# field as the job ID and then writes its normal metadata into outputs/<job_id>.
printf 'Submitted batch job %s\n' "$job_id"
