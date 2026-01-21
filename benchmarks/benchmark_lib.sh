#!/usr/bin/env bash

# Shared benchmarking utilities for InferenceMAX

# Global timing variables for server launch measurements
_LAUNCH_SERVER_START_TIME=""
_LAUNCH_SERVER_END_TIME=""
_WAIT_FOR_SERVER_START_TIME=""
_WAIT_FOR_SERVER_END_TIME=""

# Get current time in seconds with nanosecond precision
get_timestamp_seconds() {
    date +%s.%N
}

# Mark the start of server launch
mark_launch_server_start() {
    _LAUNCH_SERVER_START_TIME=$(get_timestamp_seconds)
}

# Mark the end of server launch (when process starts, before it's ready)
mark_launch_server_end() {
    _LAUNCH_SERVER_END_TIME=$(get_timestamp_seconds)
}

# Mark the start of wait_for_server_ready
mark_wait_for_server_start() {
    _WAIT_FOR_SERVER_START_TIME=$(get_timestamp_seconds)
}

# Mark the end of wait_for_server_ready
mark_wait_for_server_end() {
    _WAIT_FOR_SERVER_END_TIME=$(get_timestamp_seconds)
}

# Calculate duration in minutes from two timestamps
# Uses Python for calculation since bc may not be available in all containers
calculate_duration_minutes() {
    local start_time=$1
    local end_time=$2
    if [[ -n "$start_time" && -n "$end_time" ]]; then
        python3 -c "print(round(($end_time - $start_time) / 60, 4))"
    else
        echo "null"
    fi
}

# Write launch timing data to a JSON file
# Parameters:
#   --output-file: Path to the output JSON file
#   --config-name: Name of the config being benchmarked (optional)
write_launch_timing_json() {
    local output_file=""
    local config_name=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --output-file)
                output_file="$2"
                shift 2
                ;;
            --config-name)
                config_name="$2"
                shift 2
                ;;
            *)
                echo "Unknown parameter: $1"
                return 1
                ;;
        esac
    done

    if [[ -z "$output_file" ]]; then
        echo "Error: --output-file is required"
        return 1
    fi

    local launch_server_duration_min
    local wait_for_server_duration_min
    local total_startup_duration_min

    launch_server_duration_min=$(calculate_duration_minutes "$_LAUNCH_SERVER_START_TIME" "$_LAUNCH_SERVER_END_TIME")
    wait_for_server_duration_min=$(calculate_duration_minutes "$_WAIT_FOR_SERVER_START_TIME" "$_WAIT_FOR_SERVER_END_TIME")

    # Calculate total startup time (from launch start to server ready)
    if [[ -n "$_LAUNCH_SERVER_START_TIME" && -n "$_WAIT_FOR_SERVER_END_TIME" ]]; then
        total_startup_duration_min=$(calculate_duration_minutes "$_LAUNCH_SERVER_START_TIME" "$_WAIT_FOR_SERVER_END_TIME")
    else
        total_startup_duration_min="null"
    fi

    # Write JSON output
    cat > "$output_file" << EOF
{
    "config_name": "${config_name:-unknown}",
    "launch_server_duration_minutes": ${launch_server_duration_min},
    "wait_for_server_ready_duration_minutes": ${wait_for_server_duration_min},
    "total_startup_duration_minutes": ${total_startup_duration_min},
    "timestamps": {
        "launch_server_start": ${_LAUNCH_SERVER_START_TIME:-null},
        "launch_server_end": ${_LAUNCH_SERVER_END_TIME:-null},
        "wait_for_server_start": ${_WAIT_FOR_SERVER_START_TIME:-null},
        "wait_for_server_end": ${_WAIT_FOR_SERVER_END_TIME:-null}
    }
}
EOF

    echo "Launch timing data written to: $output_file"
}

# Check if required environment variables are set
# Usage: check_env_vars VAR1 VAR2 VAR3 ...
# Exits with code 1 if any variable is not set
check_env_vars() {
    local missing_vars=()

    for var_name in "$@"; do
        if [[ -z "${!var_name}" ]]; then
            missing_vars+=("$var_name")
        fi
    done

    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        echo "Error: The following required environment variables are not set:"
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
        exit 1
    fi
}

# Wait for server to be ready by polling the health endpoint
# All parameters are required
# Parameters:
#   --port: Server port
#   --server-log: Path to server log file
#   --server-pid: Server process ID (required)
#   --sleep-interval: Sleep interval between health checks (optional, default: 5)
wait_for_server_ready() {
    set +x
    local port=""
    local server_log=""
    local server_pid=""
    local sleep_interval=5

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --port)
                port="$2"
                shift 2
                ;;
            --server-log)
                server_log="$2"
                shift 2
                ;;
            --server-pid)
                server_pid="$2"
                shift 2
                ;;
            --sleep-interval)
                sleep_interval="$2"
                shift 2
                ;;
            *)
                echo "Unknown parameter: $1"
                return 1
                ;;
        esac
    done

    # Validate required parameters
    if [[ -z "$port" ]]; then
        echo "Error: --port is required"
        return 1
    fi
    if [[ -z "$server_log" ]]; then
        echo "Error: --server-log is required"
        return 1
    fi
    if [[ -z "$server_pid" ]]; then
        echo "Error: --server-pid is required"
        return 1
    fi

    # Show logs until server is ready
    tail -f -n +1 "$server_log" &
    local TAIL_PID=$!
    until curl --output /dev/null --silent --fail http://0.0.0.0:$port/health; do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "Server died before becoming healthy. Exiting."
            kill $TAIL_PID
            exit 1
        fi
        sleep "$sleep_interval"
    done
    kill $TAIL_PID
}

# Run benchmark serving with standardized parameters
# All parameters are required except --use-chat-template
# Parameters:
#   --model: Model name
#   --port: Server port
#   --backend: Backend type - e.g., 'vllm' or 'openai'
#   --input-len: Random input sequence length
#   --output-len: Random output sequence length
#   --random-range-ratio: Random range ratio
#   --num-prompts: Number of prompts
#   --max-concurrency: Max concurrency
#   --result-filename: Result filename without extension
#   --result-dir: Result directory
#   --use-chat-template: Optional flag to enable chat template
run_benchmark_serving() {
    set +x
    local model=""
    local port=""
    local backend=""
    local input_len=""
    local output_len=""
    local random_range_ratio=""
    local num_prompts=""
    local max_concurrency=""
    local result_filename=""
    local result_dir=""
    local use_chat_template=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model)
                model="$2"
                shift 2
                ;;
            --port)
                port="$2"
                shift 2
                ;;
            --backend)
                backend="$2"
                shift 2
                ;;
            --input-len)
                input_len="$2"
                shift 2
                ;;
            --output-len)
                output_len="$2"
                shift 2
                ;;
            --random-range-ratio)
                random_range_ratio="$2"
                shift 2
                ;;
            --num-prompts)
                num_prompts="$2"
                shift 2
                ;;
            --max-concurrency)
                max_concurrency="$2"
                shift 2
                ;;
            --result-filename)
                result_filename="$2"
                shift 2
                ;;
            --result-dir)
                result_dir="$2"
                shift 2
                ;;
            --use-chat-template)
                use_chat_template=true
                shift
                ;;
            *)
                echo "Unknown parameter: $1"
                return 1
                ;;
        esac
    done

    # Validate all required parameters
    if [[ -z "$model" ]]; then
        echo "Error: --model is required"
        return 1
    fi
    if [[ -z "$port" ]]; then
        echo "Error: --port is required"
        return 1
    fi
    if [[ -z "$backend" ]]; then
        echo "Error: --backend is required"
        return 1
    fi
    if [[ -z "$input_len" ]]; then
        echo "Error: --input-len is required"
        return 1
    fi
    if [[ -z "$output_len" ]]; then
        echo "Error: --output-len is required"
        return 1
    fi
    if [[ -z "$random_range_ratio" ]]; then
        echo "Error: --random-range-ratio is required"
        return 1
    fi
    if [[ -z "$num_prompts" ]]; then
        echo "Error: --num-prompts is required"
        return 1
    fi
    if [[ -z "$max_concurrency" ]]; then
        echo "Error: --max-concurrency is required"
        return 1
    fi
    if [[ -z "$result_filename" ]]; then
        echo "Error: --result-filename is required"
        return 1
    fi
    if [[ -z "$result_dir" ]]; then
        echo "Error: --result-dir is required"
        return 1
    fi
    
    # Check if git is installed, install if missing
    if ! command -v git &> /dev/null; then
        echo "git not found, installing..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y git
        else
            echo "Error: Could not install git. Package manager not found."
            return 1
        fi
    fi

    # Clone benchmark serving repo
    local BENCH_SERVING_DIR=$(mktemp -d /tmp/bmk-XXXXXX)
    git clone https://github.com/kimbochen/bench_serving.git "$BENCH_SERVING_DIR"

    # Build benchmark command
    local benchmark_cmd=(
        python3 "$BENCH_SERVING_DIR/benchmark_serving.py"
        --model "$model"
        --backend "$backend"
        --base-url "http://0.0.0.0:$port"
        --dataset-name random
        --random-input-len "$input_len"
        --random-output-len "$output_len"
        --random-range-ratio "$random_range_ratio"
        --num-prompts "$num_prompts"
        --max-concurrency "$max_concurrency"
        --request-rate inf
        --ignore-eos
        --save-result
        --percentile-metrics 'ttft,tpot,itl,e2el'
        --result-dir "$result_dir"
        --result-filename "$result_filename.json"
    )
    
    # Add --use-chat-template if requested
    if [[ "$use_chat_template" == true ]]; then
        benchmark_cmd+=(--use-chat-template)
    fi

    # Run benchmark
    set -x
    "${benchmark_cmd[@]}"
    set +x
}
