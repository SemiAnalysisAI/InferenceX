#!/usr/bin/bash

# This script sets up the environment and launches multi-node benchmarks

set -x

source "$(dirname "${BASH_SOURCE[0]}")/slurm_utils.sh"

export SLURM_PARTITION="batch"
export SLURM_ACCOUNT="benchmark"
SQUASH_DIR="/mnt/lustre01/users-public/sa-shared"

if [[ "$FRAMEWORK" == "llmd-vllm" ]]; then
    if [[ "$MODEL_PREFIX" == "dsv4" && "$PRECISION" == "fp4" ]]; then
        export MODEL_PATH="/mnt/numa1/models/DeepSeek-V4-Pro"
        export MODEL_NAME="deepseek-ai/DeepSeek-V4-Pro"
    else
        echo "Unsupported MODEL_PREFIX/PRECISION for llmd-vllm on GB200: $MODEL_PREFIX/$PRECISION" >&2
        exit 1
    fi

    SQUASH_FILE="${SQUASH_DIR}/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

    # Enroot uses '#' between the registry and repository, e.g. docker://ghcr.io#org/image:tag.
    case "$IMAGE" in
        */*)
            _registry="${IMAGE%%/*}"
            _rest="${IMAGE#*/}"
            if [[ "$_registry" == *.* || "$_registry" == *:* ]]; then
                ENROOT_URL="docker://${_registry}#${_rest}"
            else
                ENROOT_URL="docker://${IMAGE}"  # bare hub repo
            fi
            ;;
        *)  ENROOT_URL="docker://${IMAGE}" ;;
    esac
    echo "ENROOT_URL=$ENROOT_URL"

    if [[ ! -s "$SQUASH_FILE" ]]; then
        echo "enroot import -> $SQUASH_FILE"
        enroot import -o "$SQUASH_FILE" "$ENROOT_URL" || {
            echo "Error: enroot import failed for $ENROOT_URL" >&2
            exit 1
        }
    else
        echo "Reusing existing squash: $SQUASH_FILE"
    fi

    export LLMD_CONTAINER_ENGINE=pyxis
    export LLMD_SQUASH_FILE="$SQUASH_FILE"

    export BENCHMARK_LOGS_DIR="$GITHUB_WORKSPACE/benchmark_logs"
    mkdir -p "$BENCHMARK_LOGS_DIR"

    SCRIPT_NAME="${EXP_NAME%%_*}_${PRECISION}_gb200_llmd-vllm-disagg.sh"
    BENCH_SCRIPT="benchmarks/multi_node/${SCRIPT_NAME}"
    if [[ ! -f "$BENCH_SCRIPT" ]]; then
        echo "Error: llm-d wrapper not found: $BENCH_SCRIPT" >&2
        exit 1
    fi

    JOB_ID=$(bash "$BENCH_SCRIPT")
    if [[ -z "$JOB_ID" ]]; then
        echo "Error: failed to submit llm-d job" >&2
        exit 1
    fi
    echo "Submitted llm-d job: $JOB_ID"

    trap 'bundle_server_logs "$BENCHMARK_LOGS_DIR" "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz"; scancel "$JOB_ID" 2>/dev/null || true' EXIT INT TERM HUP

    LOG_FILE="${BENCHMARK_LOGS_DIR}/slurm_job-${JOB_ID}.out"
    stream_slurm_job_log "$JOB_ID" "$LOG_FILE" || exit 1

    while IFS= read -r -d '' result_file; do
        copy_to_workspace "$result_file" "$GITHUB_WORKSPACE/$(basename "$result_file")" || exit 1
    done < <(find "$BENCHMARK_LOGS_DIR" -name "${RESULT_FILENAME}*.json" -print0 2>/dev/null)

    if [[ "${RUN_EVAL:-false}" == "true" ]]; then
        EVAL_DIR=$(find "$BENCHMARK_LOGS_DIR" -type d -name eval_results -print -quit 2>/dev/null)
        if [[ -z "$EVAL_DIR" ]]; then
            EVAL_DIR="$BENCHMARK_LOGS_DIR/eval_results"
        fi
        copy_eval_artifacts "$EVAL_DIR" "$GITHUB_WORKSPACE" || exit 1
    fi

    scancel "$JOB_ID" 2>/dev/null || true
    exit 0
fi

# MODEL_PATH: Override with pre-downloaded paths on GB200 runner
# The yaml files specify HuggingFace model IDs for portability, but we use
# local paths to avoid repeated downloading on the shared GB200 cluster.
if [[ $FRAMEWORK == "dynamo-sglang" ]]; then
    export CONFIG_DIR="/mnt/lustre01/artifacts/sglang-configs/1k1k"
    if [[ $MODEL_PREFIX == "dsr1" && $PRECISION == "fp8" ]]; then
        export MODEL_PATH="/mnt/lustre01/models/deepseek-r1-0528"
        export SRT_SLURM_MODEL_PREFIX="dsr1-fp8"
    elif [[ $MODEL_PREFIX == "dsr1" && $PRECISION == "fp4" ]]; then
        export MODEL_PATH="/mnt/lustre01/models/deepseek-r1-0528-fp4-v2/"
        export SRT_SLURM_MODEL_PREFIX="dsr1-fp4"
    elif [[ $MODEL_PREFIX == "dsv4" && $PRECISION == "fp4" ]]; then
        # Lustre-resident weights staged on the GB200 external cluster.
        # SRT_SLURM_MODEL_PREFIX matches the model.path alias in our
        # DSV4 sglang recipes.
        export MODEL_PATH="/mnt/lustre01/models/deepseek-v4-pro"
        export SRT_SLURM_MODEL_PREFIX="deepseek-v4-pro"
    elif [[ $MODEL_PREFIX == "glm5.1" && $PRECISION == "fp4" ]]; then
        # SRT_SLURM_MODEL_PREFIX matches the model.path alias ("glm-5-fp4")
        # in our GLM-5.1 sglang recipes.
        export MODEL_PATH="/mnt/lustre01/models/GLM-5.1-NVFP4"
        export SRT_SLURM_MODEL_PREFIX="glm-5-fp4"
    elif [[ $MODEL_PREFIX == "qwen3.5" && $PRECISION == "fp8" ]]; then
        export MODEL_PATH="/mnt/lustre01/models/Qwen3.5-397B-A17B-FP8"
        export SRT_SLURM_MODEL_PREFIX="qwen3.5-fp8"
    elif [[ $MODEL_PREFIX == "glm5.1" && $PRECISION == "fp4" ]]; then
        # SRT_SLURM_MODEL_PREFIX matches the model.path alias ("glm-5-fp4")
        # in our GLM-5.1 sglang recipes.
        export MODEL_PATH="/mnt/lustre01/models/GLM-5.1-NVFP4"
        export SRT_SLURM_MODEL_PREFIX="glm-5-fp4"
    elif [[ $MODEL_PREFIX == "glm5.1" && $PRECISION == "fp8" ]]; then
        # SRT_SLURM_MODEL_PREFIX matches the model.path alias ("glm-5.1-fp8")
        # in our GLM-5.1 sglang recipes.
        export MODEL_PATH="/mnt/lustre01/models/GLM-5.1-FP8"
        export SRT_SLURM_MODEL_PREFIX="glm-5.1-fp8"
    else
        export MODEL_PATH=$MODEL
    fi
elif [[ $FRAMEWORK == "dynamo-trt" ]]; then
    if [[ $MODEL_PREFIX == "gptoss" ]]; then
        export MODEL_PATH="/mnt/lustre01/models/gpt-oss-120b"
        export SERVED_MODEL_NAME="gpt-oss-120b"
    elif [[ $MODEL_PREFIX == "dsr1" && $PRECISION == "fp4" ]]; then
        export MODEL_PATH="/mnt/numa1/models/DeepSeek-R1-0528-NVFP4-v2"
        export SERVED_MODEL_NAME="deepseek-r1-fp4"
        export SRT_SLURM_MODEL_PREFIX="dsr1"
    elif [[ $MODEL_PREFIX == "dsr1" && $PRECISION == "fp8" ]]; then
        export MODEL_PATH="/mnt/numa1/models/DeepSeek-R1-0528"
        export SERVED_MODEL_NAME="deepseek-r1-fp8"
        export SRT_SLURM_MODEL_PREFIX="dsr1-fp8"
    elif [[ $MODEL_PREFIX == "kimik2.5" && $PRECISION == "fp4" ]]; then
        export MODEL_PATH="/mnt/lustre01/models/kimi-k2.5-nvfp4"
        export SERVED_MODEL_NAME="kimi-k2.5-nvfp4"
        export SRT_SLURM_MODEL_PREFIX="nvidia/Kimi-K2.5-NVFP4"
    elif [[ $MODEL_PREFIX == "glm5" && $PRECISION == "fp4" ]]; then
        # SRT_SLURM_MODEL_PREFIX matches the model.path alias
        # ("nvidia/GLM-5-NVFP4") in the upstream GLM5 trtllm_dynamo recipes.
        export MODEL_PATH="/mnt/lustre01/slurm-shared/glm-model/GLM-5-NVFP4"
        export SERVED_MODEL_NAME="glm-5-nvfp4"
        export SRT_SLURM_MODEL_PREFIX="nvidia/GLM-5-NVFP4"
    else
        echo "Unsupported model prefix: $MODEL_PREFIX. Supported prefixes are: gptoss, dsr1, kimik2.5, or glm5"
        exit 1
    fi
elif [[ $FRAMEWORK == "dynamo-vllm" ]]; then
    if [[ $MODEL_PREFIX == "kimik2.5" && $PRECISION == "fp4" ]]; then
        export MODEL_PATH="/mnt/lustre01/models/kimi-k2.5-nvfp4"
        export SRT_SLURM_MODEL_PREFIX="kimi-k2.5-nvfp4"
    elif [[ $MODEL_PREFIX == "dsv4" && $PRECISION == "fp4" ]]; then
        if [[ "${IS_AGENTIC:-0}" == "1" ]]; then
            # AgentX was tuned against the canonical DeepSeek-V4-Pro
            # checkpoint, which is staged on compute-visible Lustre.
            export MODEL_PATH="/mnt/lustre01/models/DeepSeek-V4-Pro/"
        else
            # Existing fixed-sequence GB200 recipes use the NVIDIA ModelOpt
            # NVFP4 checkpoint.
            export MODEL_PATH="/mnt/lustre01/models/DeepSeek-V4-Pro-NVFP4/"
        fi
        export SRT_SLURM_MODEL_PREFIX="deepseek-v4-pro"
    elif [[ $MODEL_PREFIX == "minimaxm2.5" && $PRECISION == "fp4" ]]; then
        export MODEL_PATH="/mnt/lustre01/models/MiniMax-M2.5-NVFP4"
        export SRT_SLURM_MODEL_PREFIX="minimax-m2.5-nvfp4"
    elif [[ $MODEL_PREFIX == "minimaxm2.5" && $PRECISION == "fp8" ]]; then
        export MODEL_PATH="/mnt/lustre01/models/MiniMax-M2.5"
        export SRT_SLURM_MODEL_PREFIX="minimax-m2.5-fp8"
    elif [[ $MODEL_PREFIX == "minimaxm3" && $PRECISION == "fp8" ]]; then
        export MODEL_PATH="/mnt/lustre01/models/MiniMax-M3-MXFP8"
        export SRT_SLURM_MODEL_PREFIX="minimax-m3-mxfp8"
    else
        echo "Unsupported model prefix/precision combination: $MODEL_PREFIX/$PRECISION. Supported combinations for dynamo-vllm: kimik2.5/fp4, dsv4/fp4, minimaxm2.5/fp4, minimaxm2.5/fp8, minimaxm3/fp8"
        exit 1
    fi
else
    export MODEL_PATH=$MODEL
fi

NGINX_IMAGE="nginx:1.27.4"

uses_watchtower_shared_fs() {
    case "$MODEL_PREFIX" in
        minimaxm2.5|minimaxm3|kimik2.5) return 0 ;;
        *) return 1 ;;
    esac
}

SQUASH_FILE="${SQUASH_DIR}/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
NGINX_SQUASH_FILE="${SQUASH_DIR}/$(echo "$NGINX_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

# Concurrent matrix jobs import to the same shared-FS squash path.
# Serialize imports and atomically replace invalid images so readers never
# observe a partially written squash file.
import_squash() {
    local squash="$1" image="$2"
    local lock="${squash}.lock"
    (
        exec 9>"$lock"
        flock -w 1800 9 || { echo "Failed to acquire lock for $squash" >&2; exit 1; }
        if unsquashfs -l "$squash" > /dev/null 2>&1; then
            echo "Squash file already exists and is valid, skipping import: $squash"
        else
            rm -f "$squash" "$squash".tmp.*
            enroot import -o "${squash}.tmp.$$" "docker://$image"
            mv -f "${squash}.tmp.$$" "$squash"
        fi
    ) || exit 1
}

import_squash "$SQUASH_FILE" "$IMAGE"
import_squash "$NGINX_SQUASH_FILE" "$NGINX_IMAGE"

export EVAL_ONLY="${EVAL_ONLY:-false}"

export ISL="$ISL"
export OSL="$OSL"

# Legacy path that doesn't use srt-slurm
if [[ $FRAMEWORK == "dynamo-sglang" && -z "$CONFIG_FILE" ]]; then
    export IMAGE=$SQUASH_FILE
    export SGL_SLURM_JOBS_PATH="dynamo/examples/backends/sglang/slurm_jobs"
    SCRIPT_NAME="${EXP_NAME%%_*}_${PRECISION}_gb200_${FRAMEWORK}.sh"
    if [[ "$FRAMEWORK" == "dynamo-sglang" ]] || [[ "$FRAMEWORK" == "dynamo-trt" ]]; then
        BENCHMARK_SUBDIR="multi_node"
    else
        BENCHMARK_SUBDIR="single_node"
    fi
    bash "benchmarks/${BENCHMARK_SUBDIR}/${SCRIPT_NAME}"
    # Wait for all jobs to complete
    echo "Waiting for all jobs to complete..."
    while [ -n "$(squeue -u $USER --noheader --format='%i')" ]; do
        echo "Jobs still running..."
        squeue --steps -u $USER
        sleep 30
    done

        # Find the latest log directory that contains the data
    cat > collect_latest_results.py <<'PY'
import os, sys
sgl_job_dir, isl, osl, nexp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
for path in sorted([f"{sgl_job_dir}/logs/{name}/vllm_isl_{isl}_osl_{osl}" for name in os.listdir(f"{sgl_job_dir}/logs/") if os.path.isdir(f"{sgl_job_dir}/logs/{name}/vllm_isl_{isl}_osl_{osl}")], key=os.path.getmtime, reverse=True)[:nexp]:
    print(path)
PY

    LOGS_DIR=$(python3 collect_latest_results.py "$SGL_SLURM_JOBS_PATH" $ISL $OSL 1)
    if [ -z "$LOGS_DIR" ]; then
        echo "No logs directory found for ISL=${ISL}, OSL=${OSL}"
        exit 1
    fi

    echo "Found logs directory: $LOGS_DIR"
    ls -la $LOGS_DIR

    # Result JSON are contained within the result directory
    for result_file in $(find $LOGS_DIR -type f); do
        # result_file should directly be isl_ISL_osl_OSL_concurrency_CONC_req_rate_R_gpus_N_ctx_M_gen_N.json
        file_name=$(basename $result_file)
        if [ -f $result_file ]; then
            # Copy the result file to workspace with a unique name
            WORKSPACE_RESULT_FILE="$GITHUB_WORKSPACE/${RESULT_FILENAME}_${file_name}"
            echo "Found result file ${result_file}. Copying them to ${WORKSPACE_RESULT_FILE}"
            cp $result_file $WORKSPACE_RESULT_FILE
        fi
    done

    exit 0
fi


# srt-slurm path requires a CONFIG_FILE pointing to a recipe YAML.
# Without it, srtctl apply scans every YAML in the repo and submits hundreds of jobs.
if [[ -z "$CONFIG_FILE" ]]; then
    echo "Error: CONFIG_FILE is not set. The srt-slurm path requires a CONFIG_FILE in additional-settings." >&2
    echo "Config: MODEL_PREFIX=${MODEL_PREFIX} PRECISION=${PRECISION} FRAMEWORK=${FRAMEWORK}" >&2
    exit 1
fi

echo "Cloning srt-slurm repository..."
SRT_REPO_DIR="srt-slurm"
SRTCTL_SETUP_SCRIPT=""
if uses_watchtower_shared_fs; then
    SHARED_BASE="/mnt/lustre01/users-public/sa-shared/gha-runs"
    RUN_KEY="${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}-${RUNNER_NAME}-$$"
    SRT_REPO_DIR="${SHARED_BASE}/srt-slurm-${RUN_KEY}"
fi
if [ -d "$SRT_REPO_DIR" ]; then
    echo "Removing existing $SRT_REPO_DIR..."
    rm -rf "$SRT_REPO_DIR"
fi

# TODO(CJQ): make first class upon srt-slurm upstream refactor
if [[ "$IS_AGENTIC" == "1" ]]; then
    # v1.0.27 is the last release with the compatible mooncake_master command;
    # v1.0.28 introduced the unsupported --nof_* flag.
    # The pinned release also provides the vLLM mooncake_kv_store
    SRT_SLURM_AGENTIC_SHA="f6eb42aee4664207dcf2ec601e3bd57bd527efd6"
    git clone --branch v1.0.27 --depth 1 https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR" || exit 1
    cd "$SRT_REPO_DIR" || exit 1
    if [[ "$(git rev-parse HEAD)" != "$SRT_SLURM_AGENTIC_SHA" ]]; then
        echo "Error: NVIDIA/srt-slurm v1.0.27 did not resolve to $SRT_SLURM_AGENTIC_SHA" >&2
        exit 1
    fi

    # Backport NVIDIA/srt-slurm#90 without taking the post-v1.0.27
    # mooncake_master changes. The feature launches one vLLM process per node
    # and lets that process manage all node-local data-parallel ranks.
    SRT_SLURM_PER_NODE_DP_SHA="1a0f9e3633318ab1ee9428d2129161b583786b18"
    git fetch --depth 2 origin refs/pull/90/head || exit 1
    if [[ "$(git rev-parse FETCH_HEAD)" != "$SRT_SLURM_PER_NODE_DP_SHA" ]]; then
        echo "Error: NVIDIA/srt-slurm PR #90 commit did not resolve to $SRT_SLURM_PER_NODE_DP_SHA" >&2
        exit 1
    fi
    git cherry-pick --no-commit "$SRT_SLURM_PER_NODE_DP_SHA" || exit 1

    # Pin VLLM_PORT only for single-GPU processes. Multi-GPU worker processes
    # (multi-node tensor parallel, e.g. the TP8 aggregate spanning two 4-GPU
    # nodes) run vLLM's internal multiproc executor whose same-node subprocesses
    # otherwise all read one VLLM_PORT and race to bind the shm-broadcast port
    # (EADDRINUSE crash). Not yet in an srt-slurm release, so apply it here as a
    # patch until it lands upstream.
    git apply "$GITHUB_WORKSPACE/runners/patches/srt-slurm-vllm-port-single-gpu.patch" || exit 1

    # ai-dynamo/dynamo#11303 is merged into ai-dynamo/dynamo main, so the
    # recipe-pinned dynamo hash resolves against upstream directly -- no
    # esmeetu/dynamo fork redirect of srt-slurm's schema.py needed anymore.

    mkdir -p recipes/vllm/deepseek-v4/agentic || exit 1
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/deepseek-v4/agentic" \
        recipes/vllm/deepseek-v4/agentic || exit 1
elif [[ $FRAMEWORK == "dynamo-vllm" && $MODEL_PREFIX == "dsv4" ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout aflowers/vllm-gb200-v0.20.0
    # Use `cp -rT` so if the upstream branch ever ships a stub
    # `recipes/vllm/deepseek-v4/` directory, we overlay our recipes onto
    # it rather than nesting (`cp -r src dst` would create
    # `recipes/vllm/deepseek-v4/deepseek-v4/...` in that case).
    mkdir -p recipes/vllm/deepseek-v4
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/deepseek-v4" recipes/vllm/deepseek-v4
elif [[ $FRAMEWORK == "dynamo-sglang" && $MODEL_PREFIX == "dsv4" ]]; then
    # Stay on NVIDIA/srt-slurm:main (default) — submission branch no
    # longer needed; overlay our hand-rolled DSV4 sglang recipes onto it.
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    mkdir -p recipes/sglang/deepseek-v4
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/sglang/deepseek-v4" recipes/sglang/deepseek-v4
elif [[ $FRAMEWORK == "dynamo-sglang" && $MODEL_PREFIX == "glm5.1" ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout sa-submission-q2-2026
    mkdir -p recipes/sglang/glm5
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/sglang/glm5" recipes/sglang/glm5
elif [[ $FRAMEWORK == "dynamo-sglang" && $MODEL_PREFIX == "qwen3.5" ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    mkdir -p recipes/sglang/qwen3.5
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/sglang/qwen3.5" recipes/sglang/qwen3.5
elif [[ $FRAMEWORK == "dynamo-sglang" && $MODEL_PREFIX == "glm5.1" ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    mkdir -p recipes/sglang/glm5
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/sglang/glm5" recipes/sglang/glm5
elif [[ $FRAMEWORK == "dynamo-vllm" && $MODEL_PREFIX == "minimaxm3" && $PRECISION == "fp8" ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR" || exit 1
    cd "$SRT_REPO_DIR" || exit 1
    git checkout sa-submission-q2-2026 || exit 1
    mkdir -p recipes/vllm/minimax-m3-gb200-fp8 || exit 1
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/minimax-m3-gb200-fp8" recipes/vllm/minimax-m3-gb200-fp8 || exit 1
    SRTCTL_SETUP_SCRIPT="minimax-m3-gb200-vllm-fixes.sh"
    cp \
        "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/configs/$SRTCTL_SETUP_SCRIPT" \
        "configs/$SRTCTL_SETUP_SCRIPT" || exit 1
elif [[ $FRAMEWORK == "dynamo-vllm" && $MODEL_PREFIX == "kimik2.5" && $PRECISION == "fp4" ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR" || exit 1
    cd "$SRT_REPO_DIR" || exit 1
    git checkout main || exit 1
    mkdir -p recipes/vllm/kimi-k2.5-fp4 || exit 1
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/kimi-k2.5-fp4" recipes/vllm/kimi-k2.5-fp4 || exit 1
elif [[ $FRAMEWORK == "dynamo-vllm" ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout sa-submission-q2-2026
elif [[ $FRAMEWORK == "dynamo-trt" && $MODEL_PREFIX == "kimik2.5" ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout sa-submission-q2-2026
elif [[ $FRAMEWORK == "dynamo-trt" && $MODEL_PREFIX == "glm5" ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout v1.0.26
    mkdir -p recipes/trtllm/glm5
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/trtllm/glm5" recipes/trtllm/glm5
else
    git clone --branch cam/sa-submission-q2-2026 --single-branch https://github.com/cquil11/srt-slurm-nv.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
fi

echo "Installing srtctl..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Watchtower: the launcher runs on the head node but compute nodes
# inherit the activated .venv (via VIRTUAL_ENV) through SRT_REPO_DIR
# which is now on shared FS. If uv's default python install lives
# under a head-node-only path, .venv/bin/python3 becomes a broken
# symlink on compute. Pin the venv to /usr/bin/python3 — a system
# path that exists at the same location on both head and compute.
if uses_watchtower_shared_fs && [[ -x /usr/bin/python3 ]]; then
    uv venv --seed --python /usr/bin/python3
else
    uv venv --seed
fi
source .venv/bin/activate
uv pip install -e .

if ! command -v srtctl &> /dev/null; then
    echo "Error: Failed to install srtctl"
    exit 1
fi

echo "Configs available at: $SRT_REPO_DIR/"

# Create srtslurm.yaml for srtctl (used by both frameworks)
SRTCTL_ROOT="${GITHUB_WORKSPACE}/srt-slurm"
# Watchtower-hosted sweeps: SRT_REPO_DIR was moved to a shared-FS path
# above so srtctl's outputs/ directory (which lives under
# SRTCTL_ROOT) is visible to compute nodes.
if uses_watchtower_shared_fs; then
    SRTCTL_ROOT="$SRT_REPO_DIR"
fi

# Agentic runs bind-mount two persistent caches into every worker container
# (Lustre, shared across nodes): aiperf's content-addressed dataset mmap
# cache (~65 GB per corpus, re-tokenized from scratch without it) and the
# HF hub cache holding the trace dataset download. The container-side paths
# are referenced by the agentic recipes' benchmark.env
# (AIPERF_DATASET_MMAP_CACHE_DIR=/aiperf_mmap_cache, HF_HUB_CACHE=/hf_hub_cache).
DEFAULT_MOUNTS_BLOCK=""
if [[ "$IS_AGENTIC" == "1" ]]; then
    AIPERF_MMAP_CACHE_HOST_PATH="/mnt/lustre01/users-public/sa-shared/ai-perf-cache"
    HF_HUB_CACHE_HOST_PATH="/mnt/lustre01/users-public/sa-shared/hf-hub-cache"
    mkdir -p "$AIPERF_MMAP_CACHE_HOST_PATH" "$HF_HUB_CACHE_HOST_PATH"
    chmod 777 "$AIPERF_MMAP_CACHE_HOST_PATH" "$HF_HUB_CACHE_HOST_PATH" 2>/dev/null || true
    DEFAULT_MOUNTS_BLOCK="default_mounts:
  ${AIPERF_MMAP_CACHE_HOST_PATH}: /aiperf_mmap_cache
  ${HF_HUB_CACHE_HOST_PATH}: /hf_hub_cache"
fi

echo "Creating srtslurm.yaml configuration..."
cat > srtslurm.yaml <<EOF
# SRT SLURM Configuration for GB200

# Default SLURM settings
default_account: "${SLURM_ACCOUNT}"
default_partition: "${SLURM_PARTITION}"
default_time_limit: "6:00:00"

# Resource defaults
gpus_per_node: 4
network_interface: ""

# Path to srtctl repo root (where the configs live)
srtctl_root: "${SRTCTL_ROOT}"

# Model path aliases
model_paths:
  "${SRT_SLURM_MODEL_PREFIX}": "${MODEL_PATH}"
containers:
  dynamo-trtllm: ${SQUASH_FILE}
  dynamo-sglang: ${SQUASH_FILE}
  "${IMAGE}": ${SQUASH_FILE}
  nginx-sqsh: ${NGINX_SQUASH_FILE}
# srtctl defaults this to true, which adds #SBATCH --segment=<total_nodes>.
# On watchtower the whole batch partition (blue-cn01-18) is a single NVL72
# rack, so segment contiguity buys nothing for MNNVL — but it DOES make
# jobs unschedulable when the partition is fragmented: Slurm backfills a
# non-contiguous node set, fails segment placement at start, and the job
# dies with "CANCELLED Reason=Resources" at RunTime=0 (hit by the first
# gb200 agentic run, job 18582). Mirror launch_gb300-nv.sh and disable.
use_segment_sbatch_directive: false
${DEFAULT_MOUNTS_BLOCK}
EOF

echo "Generated srtslurm.yaml:"
cat srtslurm.yaml

echo "Running make setup..."
make setup ARCH=aarch64 || exit 1

# Export eval-related env vars for srt-slurm post-benchmark eval
export INFMAX_WORKSPACE="$GITHUB_WORKSPACE"
# Watchtower: pyxis mounts INFMAX_WORKSPACE into the container, but
# GITHUB_WORKSPACE is under /home/slurm-shared/ which compute nodes
# can't see. Stage the relevant subset to shared FS and repoint
# INFMAX_WORKSPACE there. rsync excludes the srt-slurm clone (already
# on shared FS) and .git (not needed in container) for speed.
if uses_watchtower_shared_fs; then
    SHARED_INFMAX_WORKSPACE="${SHARED_BASE}/infmax-workspace-${RUN_KEY}"
    mkdir -p "$SHARED_INFMAX_WORKSPACE" || exit 1
    rsync -a --delete \
        --exclude='.git/' \
        --exclude='srt-slurm*/' \
        --exclude='outputs/' \
        --exclude='LOGS/' \
        --exclude='*.sqsh' \
        "${GITHUB_WORKSPACE}/" "${SHARED_INFMAX_WORKSPACE}/" || exit 1
    export INFMAX_WORKSPACE="$SHARED_INFMAX_WORKSPACE"
    echo "Using shared-FS INFMAX_WORKSPACE=$INFMAX_WORKSPACE (compute-visible)"
fi

echo "Submitting job with srtctl..."

# Resolve the recipe path before editing or submitting it.
CONFIG_PATH="${CONFIG_FILE%%:*}"
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: CONFIG_FILE does not exist after srt-slurm setup: $CONFIG_PATH" >&2
    echo "Current directory: $(pwd)" >&2
    exit 1
fi

# Keep the Slurm job name aligned with the GitHub runner name.
sed -i "s/^name:.*/name: \"${RUNNER_NAME}\"/" "$CONFIG_PATH"

# Don't leak the login-node venv to the compute-node orchestrator. sbatch's
# default --export=ALL propagates VIRTUAL_ENV (set by `source
# .venv/bin/activate` above) into job_script_minimal.j2, whose
# `uv run` step then tries to inspect the *active* venv — and dies with
# "Broken symlink at .venv/bin/python3" because the login-node interpreter
# path doesn't exist on compute nodes (gb200 agentic R2, job 18587).
# srtctl itself still resolves through PATH (.venv/bin is on it).
unset VIRTUAL_ENV

# --no-preflight is only used on the agentic path, where the recipe resolves
# model.path to /mnt/numa1 (compute-node-only NVMe) that the login-node
# runner can't see. Fixed-seq-len recipes keep enforcement on.
PREFLIGHT_ARGS=()
if [[ "$IS_AGENTIC" == "1" ]]; then
    PREFLIGHT_ARGS=(--no-preflight)
fi

SRTCTL_APPLY_ARGS=(
    "${PREFLIGHT_ARGS[@]}"
    # Pass the full CONFIG_FILE (not the stripped CONFIG_PATH): srtctl needs the
    # ":zip_override_...[i]" selector to pick the recipe block. For plain-file
    # recipes CONFIG_FILE == CONFIG_PATH, so this is a no-op for them.
    -f "$CONFIG_FILE"
    --tags "gb200,${MODEL_PREFIX},${PRECISION},${ISL}x${OSL},infmax-$(date +%Y%m%d)"
)
if [[ "$FRAMEWORK" == "dynamo-sglang" ]]; then
    SRTCTL_APPLY_ARGS+=(--setup-script install-torchao.sh)
elif [[ -n "$SRTCTL_SETUP_SCRIPT" ]]; then
    SRTCTL_APPLY_ARGS+=(--setup-script "$SRTCTL_SETUP_SCRIPT")
fi
SRTCTL_OUTPUT=$(srtctl apply "${SRTCTL_APPLY_ARGS[@]}" 2>&1)
echo "$SRTCTL_OUTPUT"

JOB_ID=$(echo "$SRTCTL_OUTPUT" | grep -oP '✅ Job \K[0-9]+' || echo "$SRTCTL_OUTPUT" | grep -oP 'Job \K[0-9]+')

set +x

if [ -z "$JOB_ID" ]; then
    echo "Error: Failed to extract JOB_ID from srtctl output"
    exit 1
fi

echo "Extracted JOB_ID: $JOB_ID"

# Use the JOB_ID to find the logs directory
# srtctl creates logs in outputs/JOB_ID/logs/
LOGS_DIR="outputs/$JOB_ID/logs"
LOG_FILE="$LOGS_DIR/sweep_${JOB_ID}.log"

stream_slurm_job_log "$JOB_ID" "$LOG_FILE" || exit 1

set -x

echo "Job $JOB_ID completed!"
echo "Collecting results..."

if [ -d "$LOGS_DIR" ]; then
    echo "Found logs directory: $LOGS_DIR"
    cp -r "$LOGS_DIR" "$GITHUB_WORKSPACE/LOGS"
    bundle_server_logs "$LOGS_DIR" "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz"
else
    echo "Warning: Logs directory not found at $LOGS_DIR"
fi

if [[ "${EVAL_ONLY:-false}" != "true" ]]; then
    if [ ! -d "$LOGS_DIR" ]; then
        exit 1
    fi

    # Find all result subdirectories
    RESULT_SUBDIRS=$(find "$LOGS_DIR" -maxdepth 1 -type d -name "*isl*osl*" 2>/dev/null)

    if [ -z "$RESULT_SUBDIRS" ]; then
        echo "Warning: No result subdirectories found in $LOGS_DIR"
    else
        # Process results from all configurations
        for result_subdir in $RESULT_SUBDIRS; do
            echo "Processing result subdirectory: $result_subdir"

            # Extract configuration info from directory name
            CONFIG_NAME=$(basename "$result_subdir")

            # Find all result JSON files
            RESULT_FILES=$(find "$result_subdir" -name "results_concurrency_*.json" 2>/dev/null)

            for result_file in $RESULT_FILES; do
                if [ -f "$result_file" ]; then
                    # Extract metadata from filename
                    # Files may be "results_concurrency_N_gpus_G_ctx_C_gen_D.json" (disagg) or "results_concurrency_N_gpus_G.json" (non-disagg)
                    filename=$(basename "$result_file")
                    concurrency=$(echo "$filename" | sed -n 's/results_concurrency_\([0-9]*\)_gpus_.*/\1/p')
                    gpus=$(echo "$filename" | sed -n 's/results_concurrency_[0-9]*_gpus_\([0-9][0-9]*\).*/\1/p')
                    ctx=$(echo "$filename" | sed -n 's/.*_ctx_\([0-9]*\)_gen_.*/\1/p')
                    gen=$(echo "$filename" | sed -n 's/.*_gen_\([0-9]*\)\.json/\1/p')

                    echo "Processing concurrency $concurrency with $gpus GPUs (ctx: $ctx, gen: $gen): $result_file"

                    if [ -n "$ctx" ] && [ -n "$gen" ]; then
                        WORKSPACE_RESULT_FILE="$GITHUB_WORKSPACE/${RESULT_FILENAME}_${CONFIG_NAME}_conc${concurrency}_gpus_${gpus}_ctx_${ctx}_gen_${gen}.json"
                    else
                        WORKSPACE_RESULT_FILE="$GITHUB_WORKSPACE/${RESULT_FILENAME}_${CONFIG_NAME}_conc${concurrency}_gpus_${gpus}.json"
                    fi
                    copy_to_workspace "$result_file" "$WORKSPACE_RESULT_FILE" || exit 1
                fi
            done
        done
    fi

    echo "All result files processed"
else
    echo "EVAL_ONLY=true: Skipping benchmark result collection"
fi

# Collect eval results if eval was requested
if [[ "${RUN_EVAL:-false}" == "true" || "${EVAL_ONLY:-false}" == "true" ]]; then
    copy_eval_artifacts "$LOGS_DIR/eval_results" "$GITHUB_WORKSPACE" || exit 1
fi
