#!/usr/bin/bash

# This script sets up the environment and launches multi-node benchmarks

set -x

source "$(dirname "${BASH_SOURCE[0]}")/slurm_utils.sh"

export SLURM_PARTITION="batch"
export SLURM_ACCOUNT="benchmark"
SQUASH_DIR="/mnt/lustre01/users-public/sa-shared"

# dcgm-power producer pin — single source of truth for power lanes. Swap
# URL+PIN here (and identically in launch_gb300-nv.sh) when the upstream
# srt-slurm merge lands.
POWER_SRT_SLURM_URL="https://github.com/edwingao28/srt-slurm.git"
POWER_SRT_SLURM_PIN="6fc1bed01a0b82dae0088a105c03ce0cfb353443"

# Enroot 3.x does not parse Docker's tag@digest syntax. For digest-pinned
# images, use its explicit registry syntax and pass the digest as the
# manifest reference so the import remains immutable.
enroot_uri_for_image() {
    local image="$1"
    local image_without_digest="$image"
    local digest=""
    local first_component registry repository repository_dir repository_name

    if [[ "$image" == *@sha256:* ]]; then
        image_without_digest="${image%@*}"
        digest="${image##*@}"
    fi

    first_component="${image_without_digest%%/*}"
    if [[ "$image_without_digest" == */* && ( "$first_component" == *.* || "$first_component" == *:* || "$first_component" == "localhost" ) ]]; then
        registry="$first_component"
        repository="${image_without_digest#*/}"
    else
        registry="registry-1.docker.io"
        repository="$image_without_digest"
    fi

    if [[ -z "$digest" ]]; then
        if [[ "$registry" == "registry-1.docker.io" ]]; then
            printf 'docker://%s\n' "$image"
        else
            printf 'docker://%s#%s\n' "$registry" "$repository"
        fi
        return
    fi

    repository_dir="${repository%/*}"
    repository_name="${repository##*/}"
    repository_name="${repository_name%%:*}"
    if [[ "$repository" == */* ]]; then
        repository="${repository_dir}/${repository_name}"
    else
        repository="$repository_name"
    fi
    if [[ "$registry" == "registry-1.docker.io" && "$repository" != */* ]]; then
        repository="library/$repository"
    fi

    printf 'docker://%s#%s:%s\n' "$registry" "$repository" "$digest"
}

# Concurrent matrix jobs import to the same shared-FS squash path.
# Serialize imports and atomically replace invalid images so readers never
# observe a partially written squash file.
import_squash() {
    local squash="$1" image="$2"
    local lock="${squash}.lock"
    local tmp="${squash}.tmp.$$"
    local enroot_uri
    enroot_uri=$(enroot_uri_for_image "$image") || exit 1
    (
        exec 9>"$lock"
        flock -w 1800 9 || { echo "Failed to acquire lock for $squash" >&2; exit 1; }
        if unsquashfs -l "$squash" > /dev/null 2>&1; then
            echo "Squash file already exists and is valid, skipping import: $squash"
        else
            local enroot_runtime
            enroot_runtime=$(mktemp -d "${TMPDIR:-/tmp}/enroot-import.XXXXXX") || exit 1
            trap 'rm -rf -- "$enroot_runtime"' EXIT
            export ENROOT_RUNTIME_PATH="$enroot_runtime"

            rm -f "$squash" "$squash".tmp.*
            if ! enroot import -o "$tmp" "$enroot_uri"; then
                rm -f "$tmp"
                echo "Error: enroot import failed for $enroot_uri" >&2
                exit 1
            fi
            if ! unsquashfs -l "$tmp" > /dev/null 2>&1; then
                rm -f "$tmp"
                echo "Error: enroot import produced an invalid squash file: $tmp" >&2
                exit 1
            fi
            mv -f "$tmp" "$squash" || exit 1
        fi
    ) || exit 1
}

# MODEL_PATH: Override with pre-downloaded paths on GB200 runner
# The yaml files specify HuggingFace model IDs for portability, but we use
# local paths to avoid repeated downloading on the shared GB200 cluster.
MODEL_PATHS_EXTRA=""
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
    elif [[ $MODEL_PREFIX == "qwen3.5" && $PRECISION == "fp4" ]]; then
        export MODEL_PATH="/mnt/lustre01/models/Qwen3.5-397B-A17B-NVFP4-V2"
        export SRT_SLURM_MODEL_PREFIX="qwen3.5-fp4"
    elif [[ $MODEL_PREFIX == "glm5.2" && $PRECISION == "fp4" ]]; then
        export MODEL_PATH="/mnt/lustre01/users-public/sa-shared/models/GLM-5.2-NVFP4"
        export SRT_SLURM_MODEL_PREFIX="glm-5.2-fp4"
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
elif [[ $FRAMEWORK == "dynamo-vllm" || $FRAMEWORK == "vllm" ]]; then
    if [[ $MODEL_PREFIX == "kimik2.5" && $PRECISION == "fp4" ]]; then
        export MODEL_PATH="/mnt/lustre01/models/kimi-k2.5-nvfp4"
        export SRT_SLURM_MODEL_PREFIX="kimi-k2.5-nvfp4"
    elif [[ $MODEL_PREFIX == "kimik3" && $PRECISION == "fp4" ]]; then
        # Load Kimi K3 from node-local NVMe for faster startup. The checkpoint
        # must be pre-staged at this exact path on every allocated GB200 node.
        # This alias matches model.path in the checked-in AgentX recipes.
        export MODEL_PATH="/mnt/numa1/models/Kimi-K3"
        export SRT_SLURM_MODEL_PREFIX="kimi-k3"
    elif [[ $MODEL_PREFIX == "dsv4" && $PRECISION == "fp4" ]]; then
        # FP4 checkpoint on compute-visible Lustre (the /mnt/numa1 path is gone
        # on watchtower compute nodes). Use the base DeepSeek-V4-Pro checkpoint,
        # NOT the -NVFP4 re-quant: the recipe's served identity is plain
        # deepseek-ai/DeepSeek-V4-Pro and the pinned v0.20.1 container's
        # deepseek_v4 loader doesn't define the NVFP4 export's extra quant
        # params (e.g. ffn.experts.w13_input_scale), which KeyErrors at load.
        # The lowercase Lustre sibling is the FP8 checkpoint, so name the
        # CamelCase FP4 path explicitly (Linux is case-sensitive).
        if [[ "$FRAMEWORK" == "vllm" ]]; then
            # Preserve the checkpoint used by the historical llm-d series.
            # It is pre-staged at the same node-local path on every GB200 node.
            export MODEL_PATH="/mnt/numa1/models/DeepSeek-V4-Pro"
        else
            export MODEL_PATH="/mnt/lustre01/models/DeepSeek-V4-Pro"
        fi
        export SRT_SLURM_MODEL_PREFIX="deepseek-v4-pro"
        MODEL_PATHS_EXTRA='  "deepseek-v4-pro-mxfp4": "/mnt/lustre01/models/DeepSeek-V4-Pro"'
    elif [[ $MODEL_PREFIX == "minimaxm2.5" && $PRECISION == "fp4" ]]; then
        export MODEL_PATH="/mnt/lustre01/models/MiniMax-M2.5-NVFP4"
        export SRT_SLURM_MODEL_PREFIX="minimax-m2.5-nvfp4"
    elif [[ $MODEL_PREFIX == "minimaxm2.5" && $PRECISION == "fp8" ]]; then
        export MODEL_PATH="/mnt/lustre01/models/MiniMax-M2.5"
        export SRT_SLURM_MODEL_PREFIX="minimax-m2.5-fp8"
    elif [[ $MODEL_PREFIX == "minimaxm3" && $PRECISION == "fp8" ]]; then
        export MODEL_PATH="/mnt/lustre01/models/MiniMax-M3-MXFP8"
        export SRT_SLURM_MODEL_PREFIX="minimax-m3-mxfp8"
    elif [[ $MODEL_PREFIX == "minimaxm3" && $PRECISION == "fp4" ]]; then
        export MODEL_PATH="/mnt/lustre01/models/MiniMax-M3-NVFP4"
        export SRT_SLURM_MODEL_PREFIX="minimax-m3-nvfp4"
    else
        echo "Unsupported model prefix/precision combination: $MODEL_PREFIX/$PRECISION. Supported combinations for vllm paths: kimik2.5/fp4, kimik3/fp4, dsv4/fp4, minimaxm2.5/fp4, minimaxm2.5/fp8, minimaxm3/fp4, minimaxm3/fp8"
        exit 1
    fi
else
    export MODEL_PATH=$MODEL
fi

NGINX_IMAGE="nginx:1.27.4"

uses_watchtower_shared_fs() {
    case "$MODEL_PREFIX" in
        minimaxm2.5|minimaxm3|kimik2.5|kimik3|qwen3.5|glm5.2) return 0 ;;
    esac
    # DSV4 multinode runs under Dynamo-vLLM or direct vLLM Router on watchtower.
    # needs the srt-slurm workspace/outputs on a compute-visible shared FS
    # (the runner home is not cross-mounted to compute nodes).
    [[ ( "$FRAMEWORK" == "dynamo-vllm" || "$FRAMEWORK" == "vllm" ) && "$MODEL_PREFIX" == "dsv4" ]] && return 0
    return 1
}

SQUASH_FILE="${SQUASH_DIR}/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
NGINX_SQUASH_FILE="${SQUASH_DIR}/$(echo "$NGINX_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

import_squash "$SQUASH_FILE" "$IMAGE"
import_squash "$NGINX_SQUASH_FILE" "$NGINX_IMAGE"

# Power lane is recipe-driven: on iff the recipe this run resolves carries an
# enabled dcgm-power telemetry block. Read the workspace mirror (it overlays
# the srt-slurm clone later), since the pin decision precedes the clone.
USES_DCGM_POWER=0
_RECIPE_REL="${CONFIG_FILE%%:*}"
_RECIPE_SRC="$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/${_RECIPE_REL#recipes/}"
# Note (wenyao): a stray "enabled: true" outside the telemetry block must
# not flip the lane, so the match is scoped instead of file-wide greps.
if [[ -n "$CONFIG_FILE" && -f "$_RECIPE_SRC" ]] && awk '
    /^telemetry:/ { t = 1; next }
    t && /^[^ ]/  { t = 0 }
    t && /^  provider: dcgm-power$/ { p = 1 }
    t && /^  enabled: true$/        { e = 1 }
    END { exit !(p && e) }
' "$_RECIPE_SRC"; then
    USES_DCGM_POWER=1
fi

# Note (wenyao): the producer pin follows the srt-slurm main lineage that the
# dynamo-sglang lanes run on (fp8 validated end-to-end, fp4 recipes
# parse-verified against the pin); other frameworks clone diverging refs
# (aflowers branch, sa-submission), so fail fast for them instead.
if [[ "$USES_DCGM_POWER" == "1" && "$FRAMEWORK" != "dynamo-sglang" ]]; then
    echo "Error: dcgm-power lanes are only validated for FRAMEWORK=dynamo-sglang, got: $FRAMEWORK" >&2
    exit 1
fi

if [[ "$USES_DCGM_POWER" == "1" ]]; then
    DCGM_EXPORTER_IMAGE="nvcr.io/nvidia/k8s/dcgm-exporter:4.6.0-4.8.3-distroless"
    DCGM_EXPORTER_SQSH="${SQUASH_DIR}/$(echo "$DCGM_EXPORTER_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
    import_squash "$DCGM_EXPORTER_SQSH" "$DCGM_EXPORTER_IMAGE"
    test -r "$DCGM_EXPORTER_SQSH" || { echo "Error: DCGM exporter squash not readable: $DCGM_EXPORTER_SQSH" >&2; exit 1; }
    unsquashfs -l "$DCGM_EXPORTER_SQSH" > /dev/null || { echo "Error: DCGM exporter squash invalid: $DCGM_EXPORTER_SQSH" >&2; exit 1; }
    sha256sum "$DCGM_EXPORTER_SQSH" > "$GITHUB_WORKSPACE/exporter-image.sha256"
fi

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

# GLM-5.2 and MiniMax-M3 AgentX use v1.0.50 for complete logical-worker
# metrics discovery across aggregate, DP-attention, and disaggregated topologies.
if [[ "$IS_AGENTIC" == "1" && "$MODEL_PREFIX" == "glm5.2" && "$PRECISION" == "fp4" && "$FRAMEWORK" == "dynamo-sglang" ]]; then
    git clone --branch v1.0.50 --single-branch https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    test "$(git rev-parse HEAD)" = "e4019633c9e2bc25f38c44b81edf52bb0504d937" || {
        echo "Error: NVIDIA/srt-slurm v1.0.50 resolved to an unexpected commit" >&2
        exit 1
    }
    mkdir -p recipes/sglang/glm5.2/gb200-fp4/agentic
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/sglang/glm5.2/gb200-fp4/agentic" \
        recipes/sglang/glm5.2/gb200-fp4/agentic
elif [[ "$IS_AGENTIC" == "1" && "$MODEL_PREFIX" == "minimaxm3" && "$PRECISION" == "fp4" && "$FRAMEWORK" == "dynamo-vllm" ]]; then
    git clone --branch v1.0.50 --single-branch https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    test "$(git rev-parse HEAD)" = "e4019633c9e2bc25f38c44b81edf52bb0504d937" || {
        echo "Error: NVIDIA/srt-slurm v1.0.50 resolved to an unexpected commit" >&2
        exit 1
    }
    mkdir -p recipes/vllm/minimax-m3/gb200-fp4/agentic
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/minimax-m3/gb200-fp4/agentic" \
        recipes/vllm/minimax-m3/gb200-fp4/agentic
# These AgentX submissions use released srt-slurm custom-benchmark metrics
# discovery so AIPerf receives every logical worker endpoint.
elif [[ "$IS_AGENTIC" == "1" && (( "$MODEL_PREFIX" == "qwen3.5" && "$PRECISION" == "fp4" && "$FRAMEWORK" == "dynamo-sglang" ) || ( "$MODEL_PREFIX" == "dsv4" && "$PRECISION" == "fp4" && "$FRAMEWORK" == "dynamo-vllm" )) ]]; then
    git clone --branch v1.0.45 --single-branch https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    test "$(git rev-parse HEAD)" = "9d8d92b20c350a5d42f0709f5a0b64e30eb37d33" || {
        echo "Error: NVIDIA/srt-slurm v1.0.45 resolved to an unexpected commit" >&2
        exit 1
    }
    if [[ "$MODEL_PREFIX" == "qwen3.5" ]]; then
        mkdir -p recipes/sglang/qwen3.5/gb200-fp4/agentic
        cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/sglang/qwen3.5/gb200-fp4/agentic" \
            recipes/sglang/qwen3.5/gb200-fp4/agentic
    else
        mkdir -p recipes/vllm/deepseek-v4/agentic
        cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/deepseek-v4/agentic" \
            recipes/vllm/deepseek-v4/agentic
    fi
# Kimi-K3 requires direct multi-node vLLM frontend support from srt-slurm.
elif [[ "$IS_AGENTIC" == "1" && "$MODEL_PREFIX" == "kimik3" ]]; then
    git clone --branch v1.0.53 --single-branch https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR" || exit 1
    cd "$SRT_REPO_DIR" || exit 1
    mkdir -p recipes/vllm/kimi-k3/agentic || exit 1
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/kimi-k3/agentic" \
        recipes/vllm/kimi-k3/agentic || exit 1
# TODO(CJQ): migrate the remaining Agentic model paths to released srt-slurm.
elif [[ "$IS_AGENTIC" == "1" ]]; then
    # Agentic multi-node pins cquil11/srt-slurm-nv revisions that provide:
    #   - BenchmarkType.CUSTOM + benchmark.command + benchmark.env
    #     (the hook that hands off to benchmarks/multi_node/agentic_srt.sh)
    #   - DynamoConfig.wheel (recipes pin the ai-dynamo wheel)
    #   - srtctl apply --no-preflight (model path /mnt/numa1 is compute-node
    #     local NVMe, invisible to the login-node runner)
    #   - benchmark_stage srun_options propagation (container-remap-root
    #     must reach the agentic_srt.sh srun)
    git clone https://github.com/cquil11/srt-slurm-nv.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout de59739b172e507e15ebf145bfe305f606e82fbf
    mkdir -p recipes/vllm/deepseek-v4/agentic
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/deepseek-v4/agentic" \
        recipes/vllm/deepseek-v4/agentic
elif [[ $FRAMEWORK == "dynamo-vllm" && $MODEL_PREFIX == "dsv4" ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout v1.0.31
    # Use `cp -rT` so if the upstream branch ever ships a stub
    # `recipes/vllm/deepseek-v4/` directory, we overlay our recipes onto
    # it rather than nesting (`cp -r src dst` would create
    # `recipes/vllm/deepseek-v4/deepseek-v4/...` in that case).
    mkdir -p recipes/vllm/deepseek-v4
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/deepseek-v4" recipes/vllm/deepseek-v4
elif [[ $FRAMEWORK == "vllm" && $MODEL_PREFIX == "dsv4" && $PRECISION == "fp4" ]]; then
    SRT_SLURM_PIN="76e7d76961b2dcb27cb05c1e9e0910ceb75104ec"
    git clone https://github.com/SemiAnalysisAI/srt-slurm.git "$SRT_REPO_DIR" || exit 1
    cd "$SRT_REPO_DIR" || exit 1
    git checkout "$SRT_SLURM_PIN" || exit 1
    test "$(git rev-parse HEAD)" = "$SRT_SLURM_PIN" || {
        echo "Error: srt-slurm HEAD does not match SRT_SLURM_PIN=$SRT_SLURM_PIN" >&2
        exit 1
    }
    mkdir -p recipes/vllm/deepseek-v4-pro/GB200/8k1k configs
    cp -rT \
        "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/deepseek-v4-pro/GB200/8k1k" \
        recipes/vllm/deepseek-v4-pro/GB200/8k1k
    cp \
        "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/configs/install-vllm-router.sh" \
        configs/install-vllm-router.sh
elif [[ $FRAMEWORK == "dynamo-sglang" && $MODEL_PREFIX == "dsv4" ]]; then
    if [[ "$USES_DCGM_POWER" == "1" ]]; then
        # Note (wenyao): on this cluster the DSV4-Pro checkpoint lives on the
        # compute-node /mnt/numa1 NVMe (same staging the agentic path and the
        # llm-d sweeps load from); the lustre alias target the shared dsv4
        # block exports is not present here. Scoped to the power lane so the
        # non-power lane keeps whatever the external-cluster staging expects.
        export MODEL_PATH="/mnt/numa1/models/DeepSeek-V4-Pro"
        git clone "$POWER_SRT_SLURM_URL" "$SRT_REPO_DIR"
        cd "$SRT_REPO_DIR"
        git checkout "$POWER_SRT_SLURM_PIN" || exit 1
        # The power lane must run the exact pinned producer SHA, never a moving branch.
        test "$(git rev-parse HEAD)" = "$POWER_SRT_SLURM_PIN" || { echo "Error: srt-slurm HEAD does not match POWER_SRT_SLURM_PIN=$POWER_SRT_SLURM_PIN" >&2; exit 1; }
        git rev-parse HEAD > "$GITHUB_WORKSPACE/power-producer-sha.txt"
    else
        # Stay on NVIDIA/srt-slurm:main (default) — submission branch no
        # longer needed; overlay our hand-rolled DSV4 sglang recipes onto it.
        git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
        cd "$SRT_REPO_DIR"
    fi
    mkdir -p recipes/sglang/deepseek-v4
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/sglang/deepseek-v4" recipes/sglang/deepseek-v4
elif [[ $FRAMEWORK == "dynamo-sglang" && $MODEL_PREFIX == "glm5.1" ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout sa-submission-q2-2026
    mkdir -p recipes/sglang/glm5
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/sglang/glm5" recipes/sglang/glm5
elif [[ $FRAMEWORK == "dynamo-sglang" && $MODEL_PREFIX == "qwen3.5" ]]; then
    if [[ "$USES_DCGM_POWER" == "1" ]]; then
        # Power lanes run the exact pinned producer SHA, never a moving
        # branch; CI derives POWER_PRODUCER_SHA from the stamp file.
        git clone "$POWER_SRT_SLURM_URL" "$SRT_REPO_DIR"
        cd "$SRT_REPO_DIR"
        git checkout "$POWER_SRT_SLURM_PIN" || exit 1
        test "$(git rev-parse HEAD)" = "$POWER_SRT_SLURM_PIN" || { echo "Error: srt-slurm HEAD does not match POWER_SRT_SLURM_PIN=$POWER_SRT_SLURM_PIN" >&2; exit 1; }
        git rev-parse HEAD > "$GITHUB_WORKSPACE/power-producer-sha.txt"
    else
        git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
        cd "$SRT_REPO_DIR"
    fi
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

# Resolve a compute-visible InferenceX checkout before rendering mounts. The
# fixed-sequence custom benchmark invokes the checked-in benchmark_serving.py
# directly; it must therefore see the same checkout from every allocated node.
export INFMAX_WORKSPACE="$GITHUB_WORKSPACE"
if uses_watchtower_shared_fs; then
    WORKSPACE_FS_TYPE=$(findmnt -n -o FSTYPE -T "$GITHUB_WORKSPACE" 2>/dev/null || true)
    if [[ "$WORKSPACE_FS_TYPE" == "lustre" ]]; then
        echo "Using existing Lustre-backed INFMAX_WORKSPACE=$INFMAX_WORKSPACE"
    else
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
        echo "Staged node-local workspace to INFMAX_WORKSPACE=$INFMAX_WORKSPACE"
    fi
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
    if uses_watchtower_shared_fs && [[ "$MODEL_PREFIX" == "glm5.2" && "$PRECISION" == "fp4" && "$FRAMEWORK" == "dynamo-sglang" ]]; then
        DYNAMO_WHEELS_CACHE_HOST_PATH="${SHARED_BASE}/dynamo-wheels"
        mkdir -p "$DYNAMO_WHEELS_CACHE_HOST_PATH"
        chmod 777 "$DYNAMO_WHEELS_CACHE_HOST_PATH" 2>/dev/null || true
        DEFAULT_MOUNTS_BLOCK+="
  ${DYNAMO_WHEELS_CACHE_HOST_PATH}: /configs/dynamo-wheels"
    fi
elif [[ "$FRAMEWORK" == "vllm" && "$MODEL_PREFIX" == "dsv4" ]]; then
    DEFAULT_MOUNTS_BLOCK="default_mounts:
  ${INFMAX_WORKSPACE}: /infmax-workspace"
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
${MODEL_PATHS_EXTRA}
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

# Appended via sed so non-power lanes' generated yaml stays byte-identical.
if [[ "$USES_DCGM_POWER" == "1" ]]; then
    sed -i "/^  nginx-sqsh:/a\\  dcgm-exporter: ${DCGM_EXPORTER_SQSH}" srtslurm.yaml
    # Note (wenyao): sed's append is a silent no-op if the anchor drifts.
    grep -q "^  dcgm-exporter: " srtslurm.yaml || { echo "Error: dcgm-exporter injection failed: nginx-sqsh anchor not found in srtslurm.yaml" >&2; exit 1; }
fi

echo "Generated srtslurm.yaml:"
cat srtslurm.yaml

echo "Running make setup..."
make setup ARCH=aarch64 || exit 1

echo "Submitting job with srtctl..."

# Resolve the recipe path before editing or submitting it.
CONFIG_PATH="${CONFIG_FILE%%:*}"
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: CONFIG_FILE does not exist after srt-slurm setup: $CONFIG_PATH" >&2
    echo "Current directory: $(pwd)" >&2
    exit 1
fi

# Namespace InferenceX allocations so other repositories using the same
# physical runner names cannot cancel them with `scancel --name=gb200-nv_*`.
# Clean up any stale allocation from this InferenceX runner before submitting.
SRT_SLURM_JOB_NAME="inferencex-${RUNNER_NAME}"
if command -v squeue >/dev/null 2>&1; then
    scancel --user="$USER" --name="$SRT_SLURM_JOB_NAME" 2>/dev/null || true
    while [[ -n "$(squeue --user="$USER" --name="$SRT_SLURM_JOB_NAME" --noheader --format='%i')" ]]; do
        sleep 5
    done
fi
sed -i "s/^name:.*/name: \"${SRT_SLURM_JOB_NAME}\"/" "$CONFIG_PATH"

# Optionally inject synthetic acceptance into the recipe's speculative-config
# when SYNTHETIC_ACCEPTANCE=true (no-op otherwise). Must run after the name
# override and before srtctl apply so the rendered job picks it up.
python3 "$GITHUB_WORKSPACE/runners/inject_synthetic_acceptance.py" "$CONFIG_PATH" "$FRAMEWORK"

# Don't leak the login-node venv to the compute-node orchestrator. sbatch's
# default --export=ALL propagates VIRTUAL_ENV (set by `source
# .venv/bin/activate` above) into job_script_minimal.j2, whose
# `uv run` step then tries to inspect the *active* venv — and dies with
# "Broken symlink at .venv/bin/python3" because the login-node interpreter
# path doesn't exist on compute nodes (gb200 agentic R2, job 18587).
# srtctl itself still resolves through PATH (.venv/bin is on it).
unset VIRTUAL_ENV

# GB200 recipes resolve model.path through mounts the login-node runner
# can't reliably stat (compute-node-local NVMe, and lustre paths that
# aren't cross-mounted on the runner pod), so srtctl's login-node model-FS
# preflight fails before sbatch. Skip it on both the agentic and
# fixed-seq-len paths.
PREFLIGHT_ARGS=(--no-preflight)

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
# srtctl gives the GitHub-provided RUNNER_NAME precedence over config.name.
# Override it only for submission so the rendered #SBATCH job name retains
# the InferenceX namespace used above.
SRTCTL_OUTPUT=$(RUNNER_NAME="$SRT_SLURM_JOB_NAME" srtctl apply "${SRTCTL_APPLY_ARGS[@]}" 2>&1)
echo "$SRTCTL_OUTPUT"

JOB_ID=$(echo "$SRTCTL_OUTPUT" | grep -oP '✅ Job \K[0-9]+' || echo "$SRTCTL_OUTPUT" | grep -oP 'Job \K[0-9]+')

set +x

if [ -z "$JOB_ID" ]; then
    echo "Error: Failed to extract JOB_ID from srtctl output"
    exit 1
fi

echo "Extracted JOB_ID: $JOB_ID"

# The workflow-level cleanup keys off the physical runner name, while this
# launcher uses a repository-specific Slurm name to avoid cross-repo
# collisions. Always clean up the exact submitted allocation on exit.
cleanup_srt_job() {
    local rc=$?
    scancel "$JOB_ID" 2>/dev/null || true
    return "$rc"
}
trap cleanup_srt_job EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

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
    # Provenance markers travel inside the server-logs bundle so the offline
    # audit can tie artifacts to the exact producer SHA and exporter image.
    if [[ "$USES_DCGM_POWER" == "1" ]]; then
        mkdir -p "$LOGS_DIR/power"
        cp "$GITHUB_WORKSPACE/exporter-image.sha256" "$LOGS_DIR/power/exporter-image.sha256"
        cp "$GITHUB_WORKSPACE/power-producer-sha.txt" "$LOGS_DIR/power/power-producer-sha.txt"
    fi
    cp -r "$LOGS_DIR" "$GITHUB_WORKSPACE/LOGS"
    bundle_server_logs "$LOGS_DIR" "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz"
else
    echo "Warning: Logs directory not found at $LOGS_DIR"
fi

if [[ "${EVAL_ONLY:-false}" != "true" ]]; then
    if [ ! -d "$LOGS_DIR" ]; then
        exit 1
    fi

    if [[ "$IS_AGENTIC" == "1" ]]; then
        # The custom benchmark runs inside the compute-visible
        # INFMAX_WORKSPACE mount. Its aggregation step writes one
        # ${RESULT_FILENAME}_conc<N>.json there per point; stage those files
        # back to GITHUB_WORKSPACE for the workflow guard and artifact upload.
        copy_agentic_results \
            "$INFMAX_WORKSPACE" \
            "$GITHUB_WORKSPACE" \
            "$RESULT_FILENAME" || exit 1
    else
        # Find all fixed-sequence result subdirectories.
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
    fi

    echo "All result files processed"
else
    echo "EVAL_ONLY=true: Skipping benchmark result collection"
fi

# Collect eval results if eval was requested
if [[ "${RUN_EVAL:-false}" == "true" || "${EVAL_ONLY:-false}" == "true" ]]; then
    copy_eval_artifacts "$LOGS_DIR/eval_results" "$GITHUB_WORKSPACE" || exit 1
fi
