#!/bin/bash
set -euo pipefail

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        # Benchmark Configuration
        --benchmark-mode) benchmark_mode="$2"; shift 2 ;;

        # Environment and paths
        --trtllm-repo) trtllm_repo="$2"; shift 2 ;;
        --work-dir) work_dir="$2"; shift 2 ;;
        --full-logdir) full_logdir="$2"; shift 2 ;;
        --container-name) container_name="$2"; shift 2 ;;
        --container-mount) container_mount="$2"; shift 2 ;;
        --container-image) container_image="$2"; shift 2 ;;
        --build-wheel) build_wheel="$2"; shift 2 ;;
        --cuda-architectures) cuda_architectures="$2"; shift 2 ;;
        --trtllm-wheel-path) trtllm_wheel_path="$2"; shift 2 ;;
        --dataset-file) dataset_file="$2"; shift 2 ;;
        --dataset-generator) dataset_generator="$2"; shift 2 ;;
        --model-path) model_path="$2"; shift 2 ;;
        --dataset-input-length) dataset_input_length="$2"; shift 2 ;;
        --dataset-output-length) dataset_output_length="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Print all parsed arguments
echo "Parsed arguments:"
echo
echo "Benchmark Configuration:"
echo "  benchmark_mode: ${benchmark_mode}"
echo
echo "Environment Configuration:"
echo "  trtllm_repo: ${trtllm_repo}"
echo "  work_dir: ${work_dir}"
echo "  full_logdir: ${full_logdir}"
echo "  container_mount: ${container_mount}"
echo "  container_image: ${container_image}"
echo "  build_wheel: ${build_wheel}"
echo "  cuda_architectures: ${cuda_architectures}"
echo "  trtllm_wheel_path: ${trtllm_wheel_path}"
echo "  dataset_file: ${dataset_file}"
echo "  dataset_generator: ${dataset_generator}"
echo "  model_path: ${model_path}"
echo "  dataset shape: ${dataset_input_length}/${dataset_output_length}"

# Set TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1 for gen_only_no_context mode
if [ "${benchmark_mode}" = "gen_only_no_context" ]; then
    export TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1
    echo "Setting TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1 for gen_only_no_context mode"
fi

# Function to cleanup on failure
cleanup_on_failure() {
    echo "Error: $1"
    exit 1
}

cleanup_background_steps() {
    local child
    for child in $(jobs -pr); do
        kill "$child" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup_background_steps EXIT

replace_placeholder() {
    file_path="$1"
    all_nodes_str="$2"
    new_file_path="$3"
    # A compute node can get ENODATA reading a file the login node just wrote, until its
    # client-side cache revalidates (cross-client coherence lag on some parallel FS, e.g.
    # aga /scratch — can exceed a minute). Refresh metadata + retry so the job doesn't abort
    # under `set -e`.
    i=0
    until { ls -la "$file_path" >/dev/null 2>&1; cat "$file_path" > "$new_file_path" 2>/dev/null; } && [ -s "$new_file_path" ]; do
        i=$((i + 1))
        if [ "$i" -ge 120 ]; then
            echo "ERROR: could not read ${file_path} after ${i} attempts" >&2
            return 1
        fi
        [ $((i % 10)) -eq 0 ] && echo "waiting for ${file_path} to become readable (attempt ${i})..."
        sleep 3
    done
    IFS=',' read -r -a node_array <<< "$all_nodes_str"
    for i in "${!node_array[@]}"; do
        current_val="${node_array[$i]}"
        placeholder="<node${i}_placeholder>"

        # Use sed to replace the placeholder with the value in-place
        sed -i "s|$placeholder|$current_val|g" "${new_file_path}"
        echo "Replaced $placeholder with $current_val in ${new_file_path}"
    done
}

env > ${full_logdir}/environment.txt

# Start container
echo "Starting container..."
if ! srun -l --container-image=${container_image} \
        --container-name=${container_name} \
        --container-mounts=${container_mount} \
        --mpi=pmix \
        echo "Container up." &> ${full_logdir}/1_container_launch.log; then
    cleanup_on_failure "Failed to start container. Check ${full_logdir}/1_container_launch.log"
fi

# Install TensorRT-LLM
if [ -n "${trtllm_wheel_path}" ]; then
    # Install from pre-built wheel if path is provided
    echo "Installing TensorRT-LLM from wheel: ${trtllm_wheel_path}..."
    if ! srun --container-name=${container_name} \
        --container-mounts=${container_mount} --no-container-mount-home \
        --mpi=pmix --overlap -N $SLURM_NNODES --ntasks-per-node=1 \
        bash -c "pip install ${trtllm_wheel_path}[devel]" \
        &> ${full_logdir}/2_install.log; then
        cleanup_on_failure "TensorRT-LLM wheel installation failed. Check ${full_logdir}/2_install.log for details"
    fi
    echo "TensorRT-LLM wheel installation completed successfully"
elif [ -d "${trtllm_repo}" ]; then
    # Build and install from repository if no wheel path provided
    echo "Installing TensorRT-LLM from ${trtllm_repo}..."
    TRT_LLM_GIT_COMMIT=$(git -C ${trtllm_repo} rev-parse --short HEAD 2>/dev/null || echo "unknown")
    echo "TRT_LLM_GIT_COMMIT: ${TRT_LLM_GIT_COMMIT}"

    if [ "${build_wheel}" = "true" ]; then
        echo "Building TensorRT-LLM wheel on one node..."
        build_command="python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt --benchmarks --use_ccache --clean"
        if [ -n "${cuda_architectures:-}" ]; then
            build_command="${build_command} --cuda_architectures \"${cuda_architectures}\""
        fi
        if ! srun --container-name=${container_name} \
            --container-mounts=${container_mount} \
            --mpi=pmix --overlap -N 1 --ntasks-per-node=1 \
            bash -c "cd ${trtllm_repo} && ${build_command}" \
            &> ${full_logdir}/2_build.log; then
            cleanup_on_failure "TensorRT-LLM build failed. Check ${full_logdir}/2_build.log for details"
        fi
        echo "TensorRT-LLM build completed successfully"
    fi

    echo "Installing TensorRT-LLM..."
    if ! srun --container-name=${container_name} \
        --container-mounts=${container_mount} --no-container-mount-home \
        --mpi=pmix --overlap -N $SLURM_NNODES --ntasks-per-node=1 \
        bash -c "cd ${trtllm_repo} && pip install -e .[devel]" \
        &> ${full_logdir}/2_install.log; then
        cleanup_on_failure "TensorRT-LLM installation failed. Check ${full_logdir}/2_install.log for details"
    fi
    echo "TensorRT-LLM installation completed successfully"
else
    echo "trtllm_wheel_path and trtllm_repo are not provided, will use the installed TensorRT-LLM from the container"
    # get_env file is in the same directory as this script
    get_env_file=${work_dir}/get_env.py
    if ! srun --container-name=${container_name} \
        --container-mounts=${container_mount} --no-container-mount-home \
        --mpi=pmix --overlap -N 1 --ntasks-per-node=1 \
        bash -c "python ${get_env_file} -e ${full_logdir}/env_vars.json" \
        &> ${full_logdir}/2_get_env.log; then
        cleanup_on_failure "Failed to get TensorRT-LLM environment variables. Check ${full_logdir}/2_get_env.log for details"
    fi
    echo "TensorRT-LLM environment variables saved to ${full_logdir}/env_vars.json"
fi

# Build the exact fixed-length Think-Max dataset once on shared storage. The
# lock and atomic rename let concurrently dispatched matrix jobs share it.
dataset_dir=$(dirname "${dataset_file}")
dataset_lock="${dataset_file}.lock"
mkdir -p "${dataset_dir}"
if ! srun --container-name=${container_name} \
        --container-mounts=${container_mount} --no-container-mount-home \
        --mpi=pmix --overlap -N 1 -n 1 \
        bash -s -- "${dataset_file}" "${dataset_lock}" \
        "${dataset_generator}" "${model_path}" \
        "${dataset_input_length}" "${dataset_output_length}" <<'DATASET_BASH'
set -euo pipefail
dataset_file=$1
lock_file=$2
generator=$3
model_path=$4
input_length=$5
output_length=$6
expected_lines=16384
exec 9>"${lock_file}"
flock -w 7200 9
if [[ -s "${dataset_file}" ]] && [[ $(wc -l < "${dataset_file}") -eq ${expected_lines} ]]; then
    echo "Dataset cache is valid: ${dataset_file}"
    exit 0
fi
dataset_dir=$(dirname "${dataset_file}")
tmp_dir=$(mktemp -d "${dataset_dir}/.dsv4-dataset.XXXXXX")
trap 'rm -rf "${tmp_dir}"' EXIT
MODEL_PATH="${model_path}" OUT_DIR="${tmp_dir}" \
ISL="${input_length}" OSL="${output_length}" NUM_PROMPTS="${expected_lines}" \
RANDOM_RATIO=1 NUM_WORKERS=100 bash "${generator}"
generated="${tmp_dir}/DeepSeek-V4-${input_length}-${output_length}-${expected_lines}-ratio-1_for_serve.json"
[[ -s "${generated}" ]]
[[ $(wc -l < "${generated}") -eq ${expected_lines} ]]
mv "${generated}" "${dataset_file}"
echo "Published dataset cache: ${dataset_file}"
DATASET_BASH
then
    cleanup_on_failure "Dataset preparation failed: ${dataset_file}"
fi

# NOTE: the upstream lm_eval api_models.py patch step was removed — this repo runs gen-only
# SOL benchmarks (no accuracy eval), so lm_eval is never imported and the container doesn't
# ship it. The patch only added a failing srun + noisy ModuleNotFoundError to every job.

# Get node lists and replace the placeholder with the actual node names
echo "SLURM_NODELIST: ${SLURM_NODELIST}"
all_nodes=($(scontrol show hostname $SLURM_NODELIST | sort))
all_nodes_str=$(IFS=','; echo "${all_nodes[*]}")
echo "all_nodes_str: ${all_nodes_str}"

start_server_cmds_base_file=${full_logdir}/start_server_cmds_base.sh
start_server_cmds_file=${full_logdir}/start_server_cmds.sh
replace_placeholder "${start_server_cmds_base_file}" "${all_nodes_str}" "${start_server_cmds_file}"
server_config_base_file=${full_logdir}/server_config_base.yaml
server_config_file=${full_logdir}/server_config.yaml
replace_placeholder "${server_config_base_file}" "${all_nodes_str}" "${server_config_file}"
client_cmds_base_file=${full_logdir}/client_cmds_base.sh
client_cmds_file=${full_logdir}/client_cmds.sh
replace_placeholder "${client_cmds_base_file}" "${all_nodes_str}" "${client_cmds_file}"

# start the servers (skip ctx workers if TRTLLM_DISAGG_BENCHMARK_GEN_ONLY is set).
echo "Starting worker commands from ${start_server_cmds_file}..."
cat ${start_server_cmds_file} | while read cmd; do
    # Skip ctx worker commands if in gen-only mode
    # CTX appears as argument to start_worker.sh and in log filename
    if [ "${TRTLLM_DISAGG_BENCHMARK_GEN_ONLY:-0}" = "1" ] && [[ "$cmd" == *"start_worker.sh CTX"* ]]; then
        echo "Skipping ctx worker command (TRTLLM_DISAGG_BENCHMARK_GEN_ONLY is set): ${cmd}"
        continue
    fi
    echo "Executing command: ${cmd}"
    eval "${cmd}"
done
echo "Server is ready!"

# Start client commands
echo "Starting client commands from ${client_cmds_file}..."
while read -r cmd <&3; do
    echo "Starting client command: ${cmd}"
    eval "${cmd}"
    if [ $? -ne 0 ]; then
        cleanup_on_failure "Command failed: ${cmd}."
    fi
done 3< "${client_cmds_file}"

echo "Job completed successfully, total runtime: $SECONDS seconds"

# Background server steps are terminated by the EXIT trap. Exiting normally
# preserves a successful Slurm state for `sbatch --wait` and GitHub Actions.
