#!/usr/bin/bash

# This script sets up the environment and launches multi-node benchmarks

set -exo pipefail

# shellcheck source=runners/slurm_utils.sh
source "$(dirname "${BASH_SOURCE[0]}")/slurm_utils.sh"

export SLURM_PARTITION="${SLURM_PARTITION:-batch_1}"
export SBATCH_PARTITION="$SLURM_PARTITION"
export SLURM_ACCOUNT="benchmark"
export ENROOT_ROOTFS_WRITABLE=1

# Host-side directory holding aiperf's content-addressed dataset mmap cache.
# Bind-mounted into worker containers at /aiperf_mmap_cache via the
# default_mounts: block in srtslurm.yaml below; aiperf reads it via
# AIPERF_DATASET_MMAP_CACHE_DIR (set in each agentic recipe's benchmark.env).
# Without it, every run re-tokenizes and re-writes ~65 GB of mmap files
# per dataset on first use. 777 mode so all gharunner_X SLURM users can
# write to it.
export AIPERF_MMAP_CACHE_HOST_PATH="/data/home/sa-shared/gharunners/ai-perf-cache"

export HF_HUB_CACHE_HOST_PATH="/data/home/sa-shared/gharunners/hf-hub-cache"
mkdir -p "$HF_HUB_CACHE_HOST_PATH"

# Persistent dynamo source-build cache. srtctl's hash-pinned dynamo install
# (_hash_cached_source_install) caches the built wheel + src tarball at
# /configs/dynamo-wheels/<hash> with a .complete sentinel; on a warm cache the
# install is just `pip install` from the cache (no apt, no root). In CI /configs
# is the per-job srt-slurm checkout (cold every job → cold build needs apt +
# root, which the non-root server containers can't do), so persist and share
# the cache across jobs by bind-mounting this host dir at /configs/dynamo-wheels.
# Seed it once with a --container-remap-root build. 777 for multi-user runners.
export DYNAMO_WHEELS_CACHE_HOST_PATH="/data/home/sa-shared/gharunners/dynamo-wheels"
mkdir -p "$DYNAMO_WHEELS_CACHE_HOST_PATH"

export MODEL_PATH=$MODEL

if [[ $MODEL_PREFIX == "dsr1" && $PRECISION == "fp4" ]]; then
    export SERVED_MODEL_NAME="deepseek-r1-fp4"
    export MODEL_PATH=/scratch/models/DeepSeek-R1-0528-NVFP4-v2
    export SRT_SLURM_MODEL_PREFIX="dsr1"
elif [[ $MODEL_PREFIX == "dsr1" && $PRECISION == "fp8" ]]; then
    export SERVED_MODEL_NAME="deepseek-r1-fp8"
    export MODEL_PATH=/scratch/models/DeepSeek-R1-0528
    export SRT_SLURM_MODEL_PREFIX="dsr1-fp8"
elif [[ $MODEL_PREFIX == "dsv4" && $PRECISION == "fp4" ]]; then
    # Use the node-local /scratch SSD for the 806 GB DSv4-Pro
    # checkpoint. Faster than the Vast NFS path, but this dir only
    # exists on compute nodes — the GHA runner pod's view does NOT
    # have /scratch/models, so srtctl preflight (which stats the path
    # from the runner pod) may fail with "Model alias resolved to
    # /scratch/models/DeepSeek-V4-Pro, but that path is unavailable."
    # If that happens, the next step is either to (a) patch srt-slurm
    # to add a skip_model_preflight recipe field, or (b) stub a
    # symlink on the runner pod that points at the NFS copy.
    export MODEL_PATH=/scratch/models/DeepSeek-V4-Pro
    export SRT_SLURM_MODEL_PREFIX="deepseek-v4-pro"
elif [[ $MODEL_PREFIX == "glm5" && $PRECISION == "fp4" && $FRAMEWORK == "dynamo-trt" ]]; then
    export SERVED_MODEL_NAME="glm-5-nvfp4"
    export MODEL_PATH=/scratch/models/GLM-5-NVFP4
    export SRT_SLURM_MODEL_PREFIX="nvidia/GLM-5-NVFP4"
elif [[ $MODEL_PREFIX == "glm5.1" && $PRECISION == "fp4" ]]; then
    # SRT_SLURM_MODEL_PREFIX matches the model.path alias ("glm-5-fp4")
    # in our GLM-5.1 sglang recipes.
    export MODEL_PATH=/scratch/models/GLM-5.1-NVFP4
    export SRT_SLURM_MODEL_PREFIX="glm-5-fp4"
elif [[ $MODEL_PREFIX == "glm5.2" && $PRECISION == "fp4" && $FRAMEWORK == "dynamo-trt" ]]; then
    export SERVED_MODEL_NAME="GLM-5.2-NVFP4"
    export MODEL_PATH=/scratch/models/GLM-5.2-NVFP4
    export SRT_SLURM_MODEL_PREFIX="nvidia/GLM-5.2-NVFP4"
elif [[ $MODEL_PREFIX == "glm5.2" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH=/scratch/models/GLM-5.2-NVFP4
    export SRT_SLURM_MODEL_PREFIX="glm-5.2-fp4"
elif [[ $MODEL_PREFIX == "glm5" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH=/scratch/models/GLM-5-NVFP4
    export SRT_SLURM_MODEL_PREFIX="glm-5-fp4"
elif [[ $MODEL_PREFIX == "glm5" && $PRECISION == "fp8" ]]; then
    export MODEL_PATH=/scratch/models/GLM-5-FP8
    export SRT_SLURM_MODEL_PREFIX="glm-5-fp8"
elif [[ $MODEL_PREFIX == "minimaxm2.5" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH=/data/models/MiniMax-M2.5-NVFP4
    export SRT_SLURM_MODEL_PREFIX="minimax-m2.5-nvfp4"
elif [[ $MODEL_PREFIX == "minimaxm2.5" && $PRECISION == "fp8" ]]; then
    export MODEL_PATH=/data/models/MiniMax-M2.5
    export SRT_SLURM_MODEL_PREFIX="minimax-m2.5-fp8"
elif [[ $MODEL_PREFIX == "minimaxm3" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH=/scratch/models/MiniMax-M3-NVFP4
    export SRT_SLURM_MODEL_PREFIX="nvidia/MiniMax-M3-NVFP4"
elif [[ $MODEL_PREFIX == "minimaxm3" && $PRECISION == "fp8" ]]; then
    export MODEL_PATH=/data/models/MiniMax-M3-MXFP8
    export SRT_SLURM_MODEL_PREFIX="minimax-m3-mxfp8"
elif [[ $MODEL_PREFIX == "kimik2.5" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH=/scratch/models/Kimi-K2.5-NVFP4
    export SRT_SLURM_MODEL_PREFIX="nvidia/Kimi-K2.5-NVFP4"
elif [[ $MODEL_PREFIX == "kimik3" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH=/scratch/models/Kimi-K3
    export SRT_SLURM_MODEL_PREFIX="moonshotai/Kimi-K3"
elif [[ $MODEL_PREFIX == "qwen3.5" && $PRECISION == "fp4" ]]; then
    # SRT_SLURM_MODEL_PREFIX must match the model.path alias used in our
    # Qwen3.5 sglang recipes (qwen3.5-fp4).
    export MODEL_PATH=/scratch/models/Qwen3.5-397B-A17B-NVFP4-V2
    export SRT_SLURM_MODEL_PREFIX="qwen3.5-fp4"
elif [[ $MODEL_PREFIX == "qwen3.5" && $PRECISION == "fp8" ]]; then
    # SRT_SLURM_MODEL_PREFIX must match the model.path alias used in our
    # Qwen3.5 sglang recipes (qwen3.5-fp8).
    export MODEL_PATH=/scratch/models/Qwen3.5-397B-A17B-FP8
    export SRT_SLURM_MODEL_PREFIX="qwen3.5-fp8"
else
    echo "Unsupported model: $MODEL_PREFIX-$PRECISION. Supported models are: dsr1-fp4, dsr1-fp8, dsv4-fp4, glm5-fp4, glm5-fp8, glm5.2-fp4, minimaxm2.5-fp4, minimaxm2.5-fp8, minimaxm3-fp4, minimaxm3-fp8, kimik2.5-fp4, kimik3-fp4, qwen3.5-fp4, qwen3.5-fp8"
    exit 1
fi

NGINX_IMAGE="nginx:1.27.4"

# Squash files live on the Vast NFS storage; use the /data/ mount
# (not /home/sa-shared/) — both are the same backing storage but the
# /home/sa-shared/ mount has a chronic ELOOP / "Too many levels of
# symbolic links" bug from workflow worker NFS sessions on lockfiles
# AND data files. /data/ has a separate NFS client cache that isn't
# poisoned. See feedback_gb300_nfs_eloop_workaround for diagnosis.
SQUASH_FILE="/data/home/sa-shared/gharunners/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
NGINX_SQUASH_FILE="/data/home/sa-shared/gharunners/squash/$(echo "$NGINX_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

# Run the import on a compute node via srun, not on the login node:
# the login node is x86_64 while the compute nodes are aarch64, so the
# arm64 squash file has to be built on a compute node.
import_squash() {
    local squash="$1" image="$2"
    local lock="${squash}.lock"
    srun --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" --exclusive --time=180 bash -c "
        exec 9>\"$lock\"
        flock -w 600 9 || { echo 'Failed to acquire lock for $squash' >&2; exit 1; }
        if unsquashfs -l \"$squash\" > /dev/null 2>&1; then
            echo 'Squash file already exists and is valid, skipping import: $squash'
        else
            rm -f \"$squash\"
            enroot import -o \"$squash\" docker://$image
        fi
    "
}

import_squash "$SQUASH_FILE" "$IMAGE"
import_squash "$NGINX_SQUASH_FILE" "$NGINX_IMAGE"

# Power lane detection: a recipe opts in via an enabled dcgm-power telemetry
# block. CONFIG_FILE is srt-slurm-relative; resolve it against the workspace
# recipe mirror (the same tree the clone step overlays), since the checkout
# doesn't exist yet. Recipes that only exist upstream stay non-power.
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

# dcgm-power producer pin — single source of truth for power lanes. Swap
# URL+PIN here (and identically in launch_gb200-nv.sh) when the upstream
# srt-slurm merge lands.
POWER_SRT_SLURM_URL="https://github.com/edwingao28/srt-slurm.git"
POWER_SRT_SLURM_PIN="6fc1bed01a0b82dae0088a105c03ce0cfb353443"

if [[ "$USES_DCGM_POWER" == "1" ]]; then
    DCGM_EXPORTER_IMAGE="nvcr.io/nvidia/k8s/dcgm-exporter:4.6.0-4.8.3-distroless"
    DCGM_EXPORTER_SQSH="/data/home/sa-shared/gharunners/squash/$(echo "$DCGM_EXPORTER_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
    # Note (wenyao): import_squash treats an existing unsquashfs-valid file
    # as a cache hit but does not re-validate a fresh import, so check
    # explicitly — on a compute node, like the import itself (login node is
    # x86, nodes aarch64).
    import_squash "$DCGM_EXPORTER_SQSH" "$DCGM_EXPORTER_IMAGE"
    test -r "$DCGM_EXPORTER_SQSH" || { echo "Error: DCGM exporter squash not readable: $DCGM_EXPORTER_SQSH" >&2; exit 1; }
    srun --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" --exclusive --time=30 bash -c "unsquashfs -l \"$DCGM_EXPORTER_SQSH\" > /dev/null" || { echo "Error: DCGM exporter squash invalid: $DCGM_EXPORTER_SQSH" >&2; exit 1; }
    sha256sum "$DCGM_EXPORTER_SQSH" > "$GITHUB_WORKSPACE/exporter-image.sha256"
fi

export EVAL_ONLY="${EVAL_ONLY:-false}"
if [[ "$EVAL_ONLY" == "true" && -n "${EVAL_CONFIG_FILE:-}" ]]; then
    CONFIG_FILE="$EVAL_CONFIG_FILE"
    echo "EVAL_ONLY=true: selecting real-verification recipe $CONFIG_FILE"
fi

export ISL="$ISL"
export OSL="$OSL"

echo "Cloning srt-slurm repository..."
RUN_KEY=$(printf "%s" "${RESULT_FILENAME:-${RUNNER_NAME:-gb300-nv}}" | sha1sum | cut -c1-12)
SRT_REPO_DIR="${GITHUB_WORKSPACE}/srt-slurm-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-0}-${RUN_KEY}"
SRTCTL_SETUP_SCRIPT=""
rm -rf "$SRT_REPO_DIR"

if [[ "$IS_AGENTIC" == "1" && $FRAMEWORK == "dynamo-trt" && $MODEL_PREFIX == "qwen3.5" ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout v1.0.50
    TRTLLM_RECIPES_DIR="recipes/trtllm/qwen3.5/gb300-fp4/disagg/agentx"
    mkdir -p "$TRTLLM_RECIPES_DIR"
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/trtllm/qwen3.5/gb300-fp4/disagg/agentx" \
        "$TRTLLM_RECIPES_DIR"
    if [[ "${EVAL_ONLY:-false}" == "true" ]]; then
        find "$TRTLLM_RECIPES_DIR" -name "*.yaml" \
            -exec sed -i '/TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS/d' {} +
    fi
elif [[ "$IS_AGENTIC" == "1" && $FRAMEWORK == "dynamo-trt" && $MODEL_PREFIX == "glm5.2" ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout v1.0.38
    TRTLLM_RECIPES_DIR="benchmarks/multi_node/srt-slurm-recipes/trtllm/glm5.2"
    mkdir -p "$TRTLLM_RECIPES_DIR"
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/trtllm/glm5.2" \
        "$TRTLLM_RECIPES_DIR"
    if [[ "${EVAL_ONLY:-false}" == "true" ]]; then
        find "$TRTLLM_RECIPES_DIR" -name "*.yaml" \
            -exec sed -i '/TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS/d' {} +
    fi
elif [[ "$IS_AGENTIC" == "1" && $FRAMEWORK == "dynamo-sglang" && $MODEL_PREFIX == "qwen3.5" ]]; then
    # Qwen3.5 agentic uses NVIDIA/srt-slurm v1.0.38: the two features the
    # cquil11 fork was pinned for are merged upstream (present in v1.0.36) —
    #   - `srtctl apply --no-preflight` (skip the in-process model FS check):
    #     model.path resolves to /scratch/models/Qwen3.5-397B-A17B-NVFP4
    #     (compute-node-only NVMe), which the GHA runner pod can't stat, so
    #     the Path.is_dir() preflight would fail before sbatch is ever
    #     called. The engine still fails loudly at runtime if the path is
    #     genuinely missing on the compute node.
    #   - benchmark_stage propagates srun_options (container-remap-root must
    #     reach the agentic_srt.sh srun).
    # v1.0.38 additionally injects AIPERF_SERVER_METRICS_URLS for custom
    # benchmarks using each logical SGLang worker leader. This is required for
    # complete AgentX trace artifacts; the public frontend alone may expose no
    # Prometheus endpoint or only Dynamo frontend metrics.
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout v1.0.38
    mkdir -p recipes/sglang/qwen3.5
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/sglang/qwen3.5" \
        recipes/sglang/qwen3.5
elif [[ "$IS_AGENTIC" == "1" && $FRAMEWORK == "dynamo-sglang" && $MODEL_PREFIX == "dsv4" ]]; then
    # DSv4 GB300 SGLang agentic uses NVIDIA/srt-slurm v1.0.38. In addition to
    # the nginx body-size fix, session-affinity frontend, and custom benchmark
    # schema required by these recipes, this release injects every logical
    # SGLang worker leader's /metrics URL into AIPERF_SERVER_METRICS_URLS.
    # AgentX forwards that list to aiperf's --server-metrics argument so its
    # trace artifacts include backend metrics for every engine.
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout v1.0.38
    mkdir -p recipes/sglang/deepseek-v4/agentic
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/sglang/deepseek-v4/agentic" \
        recipes/sglang/deepseek-v4/agentic
elif [[ "$IS_AGENTIC" == "1" && $FRAMEWORK == "dynamo-trt" && $MODEL_PREFIX == "dsv4" ]]; then
    SRT_SLURM_MODEL_PREFIX="deepseek-ai/DeepSeek-V4-Pro"
    git clone --branch v1.0.50 --single-branch https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR" || exit 1
    cd "$SRT_REPO_DIR" || exit 1

    mkdir -p benchmarks/multi_node/srt-slurm-recipes/trtllm/deepseek-v4 || exit 1
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/trtllm/deepseek-v4" \
        benchmarks/multi_node/srt-slurm-recipes/trtllm/deepseek-v4 || exit 1
    if [[ "${EVAL_ONLY:-false}" == "true" ]]; then
        # srt-slurm v1.0.50 launches lm-eval on the allocation head and uses
        # localhost:8000. Keep AgentX frontends on first_decode for throughput,
        # but co-locate the eval-only frontend with lm-eval so loopback resolves.
        find benchmarks/multi_node/srt-slurm-recipes/trtllm/deepseek-v4 -name "*.yaml" \
            -exec sed -i \
                -e '/TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS/d' \
                -e 's/^  orchestrator_placement: first_decode$/  orchestrator_placement: head/' \
                {} +
    fi
elif [[ "$IS_AGENTIC" == "1" && $FRAMEWORK == "dynamo-sglang" && $MODEL_PREFIX == "glm5.2" ]]; then
    # GLM-5.2 GB300 sglang AgentX: srt-slurm main has the agentx-mvp scenario,
    # the zip_override sweep selectors, and the multi-frontend session-affinity
    # schema these recipes need.
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout main
    mkdir -p recipes/sglang/glm5.2/gb300-fp4
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/sglang/glm5.2/gb300-fp4" \
        recipes/sglang/glm5.2/gb300-fp4
elif [[ "$IS_AGENTIC" == "1" ]]; then
    # Agentic recipes use NVIDIA/srt-slurm v1.0.36. This is the upstream
    # version validated in InferenceX PR #2302 and includes per-node DP,
    # matching Dynamo health counts, multi-node TP port handling, and
    # Mooncake compatibility. Keep it pinned so sweeps are reproducible.
    git clone --branch v1.0.36 --single-branch https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR" || exit 1
    cd "$SRT_REPO_DIR" || exit 1

    mkdir -p recipes/vllm/deepseek-v4/agentic || exit 1
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/deepseek-v4/agentic" \
        recipes/vllm/deepseek-v4/agentic || exit 1
    mkdir -p recipes/vllm/minimax-m3/agentic || exit 1
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/minimax-m3/agentic" \
        recipes/vllm/minimax-m3/agentic || exit 1
    mkdir -p recipes/vllm/kimi-k3/agentic || exit 1
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/kimi-k3/agentic" \
        recipes/vllm/kimi-k3/agentic || exit 1
elif [[ $FRAMEWORK == "dynamo-vllm" && $MODEL_PREFIX == "dsv4" ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout aflowers/gb200-dsv4-recipes
    mkdir -p recipes/vllm/deepseek-v4
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/deepseek-v4" recipes/vllm/deepseek-v4
elif [[ $FRAMEWORK == "dynamo-sglang" && $MODEL_PREFIX == "dsv4" ]]; then
    # Fixed-length DeepSeek-V4 recipes are version-controlled in this repository;
    # overlay them onto the selected srt-slurm checkout. Power lanes use the
    # exact producer pin and stamp it for strict result provenance; non-power
    # lanes retain the v1.0.25 release that bootstraps cargo/maturin for the
    # hash-pinned Dynamo source build before launch.
    if [[ "$USES_DCGM_POWER" == "1" ]]; then
        git clone "$POWER_SRT_SLURM_URL" "$SRT_REPO_DIR" || exit 1
        cd "$SRT_REPO_DIR" || exit 1
        git checkout "$POWER_SRT_SLURM_PIN" || exit 1
        # The power lane must run the exact pinned producer SHA, never a moving branch.
        test "$(git rev-parse HEAD)" = "$POWER_SRT_SLURM_PIN" || { echo "Error: srt-slurm HEAD does not match POWER_SRT_SLURM_PIN=$POWER_SRT_SLURM_PIN" >&2; exit 1; }
        git rev-parse HEAD > "$GITHUB_WORKSPACE/power-producer-sha.txt"
    else
        git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR" || exit 1
        cd "$SRT_REPO_DIR" || exit 1
        git checkout v1.0.25 || exit 1
    fi
    mkdir -p recipes/sglang/deepseek-v4/8k1k || exit 1
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/sglang/deepseek-v4/8k1k" \
        recipes/sglang/deepseek-v4/8k1k || exit 1
elif [[ $FRAMEWORK == "dynamo-sglang" && $MODEL_PREFIX == "glm5" ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout main
    if [[ $PRECISION == "fp4" ]]; then
        mkdir -p recipes/sglang/glm5/gb300-fp4
        cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/sglang/glm5/gb300-fp4" recipes/sglang/glm5/gb300-fp4
    fi
elif [[ $FRAMEWORK == "dynamo-sglang" && $MODEL_PREFIX == "glm5.1" && $PRECISION == "fp8" ]]; then
    # GLM-5.1 FP8 (gb300) recipes are version-controlled in-repo; overlay them
    # onto the pinned submission branch.
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout sa-submission-q2-2026
    mkdir -p recipes/sglang/glm5.1
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/sglang/glm5.1" recipes/sglang/glm5.1
elif [[ $FRAMEWORK == "dynamo-sglang" && $MODEL_PREFIX == "glm5.1" ]]; then
    # GLM-5.1 MTP recipe (recipes/gb300-fp4/glm5-mtp.yaml) lives on
    # NVIDIA/srt-slurm:main — check it out; no in-repo overlay needed.
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout main
elif [[ $FRAMEWORK == "dynamo-sglang" && $MODEL_PREFIX == "qwen3.5" ]]; then
    # Overlay our version-controlled Qwen3.5 recipes onto the srt-slurm checkout.
    # fp8 recipes pin dynamo by commit hash (source install), which needs the
    # cargo/maturin bootstrap included in the srt-slurm v1.0.25 release — the
    # sa-submission-q2-2026 sglang install path assumes maturin ships in the
    # image, and the lmsysorg/sglang nightly-dev-cu13 image doesn't include it.
    # Same branch the identical gb200-fp8 recipes run on. fp4 recipes pin
    # dynamo by version (pip install) and stay on the submission branch they
    # were validated against.
    if [[ "$USES_DCGM_POWER" == "1" ]]; then
        git clone "$POWER_SRT_SLURM_URL" "$SRT_REPO_DIR"
        cd "$SRT_REPO_DIR"
        git checkout "$POWER_SRT_SLURM_PIN" || exit 1
        # The power lane must run the exact pinned producer SHA, never a moving branch.
        test "$(git rev-parse HEAD)" = "$POWER_SRT_SLURM_PIN" || { echo "Error: srt-slurm HEAD does not match POWER_SRT_SLURM_PIN=$POWER_SRT_SLURM_PIN" >&2; exit 1; }
        git rev-parse HEAD > "$GITHUB_WORKSPACE/power-producer-sha.txt"
    else
        git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
        cd "$SRT_REPO_DIR"
        if [[ $PRECISION == "fp8" ]]; then
            git checkout v1.0.25
        else
            git checkout sa-submission-q2-2026
        fi
    fi
    mkdir -p recipes/sglang/qwen3.5
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/sglang/qwen3.5" recipes/sglang/qwen3.5
elif [[ $FRAMEWORK == "dynamo-vllm" && $MODEL_PREFIX == "minimaxm3" ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    if [[ "${SPEC_DECODING:-}" == "mtp" ]]; then
        git checkout v1.0.38
    else
        git checkout sa-submission-q2-2026
    fi
    mkdir -p recipes/vllm/minimax-m3-gb300-fp8
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/minimax-m3-gb300-fp8" recipes/vllm/minimax-m3-gb300-fp8
elif [[ $FRAMEWORK == "dynamo-vllm" && $MODEL_PREFIX == "kimik2.5" && $PRECISION == "fp4" ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout main
    mkdir -p recipes/vllm/kimi-k2.5-fp4
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/kimi-k2.5-fp4" recipes/vllm/kimi-k2.5-fp4
elif [[ $FRAMEWORK == "dynamo-trt" && $MODEL_PREFIX == "dsv4" ]]; then
    # DSv4 dynamo-trt recipes use the HuggingFace model ID as model.path,
    # so override SRT_SLURM_MODEL_PREFIX to match the recipe's model path key.
    SRT_SLURM_MODEL_PREFIX="deepseek-ai/DeepSeek-V4-Pro"
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout sa-submission-q2-2026
elif [[ $FRAMEWORK == "dynamo-trt" && $MODEL_PREFIX == "qwen3.5" && $PRECISION == "fp4" ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout v1.0.29
    mkdir -p recipes/trtllm/qwen3.5/gb300-fp4/disagg
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/trtllm/qwen3.5/gb300-fp4/disagg" \
        recipes/trtllm/qwen3.5/gb300-fp4/disagg
else
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout sa-submission-q2-2026
fi

echo "Installing srtctl..."
export UV_INSTALL_DIR="$GITHUB_WORKSPACE/.local/bin"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$UV_INSTALL_DIR:$PATH"

VENV_DIR="${GITHUB_WORKSPACE}/.venv-srt-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-0}-${RUN_KEY}"
rm -rf "$VENV_DIR"
# --seed installs pip+setuptools+wheel into the venv. Without it, the
# upstream prefetch-ai-dynamo-wheel.sh script (called by srtctl when a
# recipe has dynamo.wheel set) fails with "No module named pip" because
# uv venv defaults to no-pip.
uv venv --seed "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv pip install -e .

if ! command -v srtctl &> /dev/null; then
    echo "Error: Failed to install srtctl"
    exit 1
fi

echo "Configs available at: $SRT_REPO_DIR/"

# Create srtslurm.yaml for srtctl (used by both frameworks)
SRTCTL_ROOT="${SRT_REPO_DIR}"
echo "Creating srtslurm.yaml configuration..."
SRT_DEFAULT_TIME_LIMIT="4:00:00"
if [[ "$IS_AGENTIC" == "1" && "$MODEL_PREFIX" == "dsv4" && ( "$FRAMEWORK" == "dynamo-sglang" || "$FRAMEWORK" == "dynamo-trt" ) ]]; then
    SRT_DEFAULT_TIME_LIMIT="8:00:00"
fi
cat > srtslurm.yaml <<EOF
# SRT SLURM Configuration for GB300

# Default SLURM settings
default_account: "${SLURM_ACCOUNT}"
default_partition: "${SLURM_PARTITION}"
default_time_limit: "${SRT_DEFAULT_TIME_LIMIT}"

# Resource defaults
gpus_per_node: 4
network_interface: ""

# Path to srtctl repo root (where the configs live)
srtctl_root: "${SRTCTL_ROOT}"

# Cluster-level bind mounts applied to every worker container
# (see srtctl/core/runtime.py — get_srtslurm_setting("default_mounts")).
# Used here for aiperf's persistent mmap cache so the dataset isn't
# re-tokenized + re-written every job.
default_mounts:
  "${AIPERF_MMAP_CACHE_HOST_PATH}": "/aiperf_mmap_cache"
  "${HF_HUB_CACHE_HOST_PATH}": "/hf_hub_cache"
  # Warm dynamo source-build cache (nested over the auto /configs mount) so the
  # hash-pinned install is a cache hit (pip-only, no apt/root) on every job.
  "${DYNAMO_WHEELS_CACHE_HOST_PATH}": "/configs/dynamo-wheels"

# Model path aliases
model_paths:
  "${SRT_SLURM_MODEL_PREFIX}": "${MODEL_PATH}"
containers:
  dynamo-trtllm: ${SQUASH_FILE}
  dynamo-sglang: ${SQUASH_FILE}
  v0.5.11: ${SQUASH_FILE}
  v0.5.13.post1: ${SQUASH_FILE}
  "${IMAGE}": ${SQUASH_FILE}
  nginx-sqsh: ${NGINX_SQUASH_FILE}
use_segment_sbatch_directive: false
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
make setup ARCH=aarch64

# Export eval-related env vars for srt-slurm post-benchmark eval
export INFMAX_WORKSPACE="$GITHUB_WORKSPACE"

echo "Submitting job with srtctl..."

if [[ -z "$CONFIG_FILE" ]]; then
    echo "Error: CONFIG_FILE is not set. The srt-slurm path requires a CONFIG_FILE in additional-settings." >&2
    echo "Config: MODEL_PREFIX=${MODEL_PREFIX} PRECISION=${PRECISION} FRAMEWORK=${FRAMEWORK}" >&2
    exit 1
fi

# Override the job name in the config file with the runner name.
# CONFIG_FILE may carry a ":zip_override_...[i]" selector suffix that only
# `srtctl apply -f` parses; strip it to the real path for the sed. srtctl
# below still receives the full CONFIG_FILE (with selector).
CONFIG_PATH="${CONFIG_FILE%%:*}"
sed -i "s/^name:.*/name: \"${RUNNER_NAME}\"/" "$CONFIG_PATH"

# Throughput recipes opt into synthetic acceptance through the master config.
# Eval-only jobs leave the checked-in real-MTP recipe unchanged so generated
# tokens still pass target-model verification.
inject_synthetic_acceptance "$CONFIG_PATH" "$FRAMEWORK" || exit 1

# Profiling: mutate the runtime recipe copy so workers run eager with the
# torch profiler configured (plus the ExecutionTrace shim delivered through
# the /infmax-workspace worker mount), and srtctl exports worker endpoints +
# profiler dirs to the benchmark stage (see utils/profile_recipe_inject.py).
if [[ "${PROFILE:-}" == "1" ]]; then
    PROFILE_INJECT_ARGS=("$CONFIG_PATH" --num-steps "${PROFILE_NUM_STEPS:-2}")
    if [[ "${PROFILE_DISABLE_CUDA_GRAPH:-1}" == "0" ]]; then
        PROFILE_INJECT_ARGS+=(--keep-cuda-graphs)
    fi
    # ExecutionTrace defaults off under spec decoding: libtorch's
    # execution_trace_observer segfaulted sglang schedulers on an EAGLE-MTP
    # agentic run (run 32992471735); explicit PROFILE_EXECUTION_TRACE=1
    # still forces it on.
    PROFILE_ET_DEFAULT=1
    if [[ -n "${SPEC_DECODING:-}" && "${SPEC_DECODING:-none}" != "none" ]]; then
        PROFILE_ET_DEFAULT=0
    fi
    if [[ "${PROFILE_EXECUTION_TRACE:-$PROFILE_ET_DEFAULT}" != "1" ]]; then
        PROFILE_INJECT_ARGS+=(--no-execution-trace)
    fi
    uv run --no-project --with pyyaml --python 3.12 \
        "$GITHUB_WORKSPACE/utils/profile_recipe_inject.py" "${PROFILE_INJECT_ARGS[@]}" || exit 1
fi

# --no-preflight skips srtctl's pre-submit model-path stat, which runs on
# the GHA runner host (im-gb300-login-02, an x86 login node). It's required
# whenever model.path resolves to the node-local /scratch NVMe that the login
# node can't see:
#   - the agentic path (DSv4-Pro checkpoint),
#   - glm5.1, whose GLM-5.1-NVFP4 weights are prestaged on the compute-node
#     /scratch/models, and
#   - qwen3.5 fp8, whose weights are also on the compute-node /scratch/models
#     and which runs on srt-slurm:v1.0.25 (the release that has the preflight),
#   - qwen3.5 fp4 dynamo-trt, which runs on v1.0.29 without that preflight, and
#   - the qwen3.5 fp4 and dsv4 sglang power lanes, which run the pinned
#     producer (a main-lineage fork that has the preflight) against the same
#     /scratch checkpoints.
# The engine still fails loudly at runtime if the path is genuinely missing on
# the compute node. Other fixed-seq-len recipes resolve model.path to a
# login-visible location, so keep the precheck enforced for them.
SRTCTL_APPLY_ARGS=(
    -f "$CONFIG_FILE"
    --tags "gb300,${MODEL_PREFIX},${PRECISION},${ISL}x${OSL},infmax-$(date +%Y%m%d)"
)
if [[ "$IS_AGENTIC" == "1" || "$MODEL_PREFIX" == "glm5.1" || ( "$MODEL_PREFIX" == "qwen3.5" && "$PRECISION" == "fp8" ) || ( "$MODEL_PREFIX" == "qwen3.5" && "$PRECISION" == "fp4" && ( "$FRAMEWORK" == "dynamo-trt" || "$USES_DCGM_POWER" == "1" ) ) || ( "$USES_DCGM_POWER" == "1" && "$MODEL_PREFIX" == "dsv4" && "$FRAMEWORK" == "dynamo-sglang" ) ]]; then
    SRTCTL_APPLY_ARGS+=(--no-preflight)
fi
if [[ -n "$SRTCTL_SETUP_SCRIPT" ]]; then
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

# Snapshot worker logs on any exit path — normal completion, error,
# SIGTERM (gh run cancel sends this to the launcher), even SIGKILL of
# our parent. Without this trap, the cancel-time tar lives only in the
# main flow below (after `wait $POLL_PID`), so a manual `gh run cancel`
# during the tail wait skips it entirely and the
# `Upload server logs` workflow step finds nothing to upload.
# Idempotent: the main-flow tar at the bottom of this script is now a
# no-op because the trap already produced the artifact, but it stays
# for narrative continuity in normal (non-cancel) runs.
_snapshot_server_logs() {
    if [ -n "${LOGS_DIR:-}" ] && [ -d "$LOGS_DIR" ] && [ -n "${GITHUB_WORKSPACE:-}" ]; then
        # Note (wenyao): provenance markers are copied in the trap, not the
        # main flow, so cancel paths that fire the trap early still bundle
        # them for the offline audit.
        if [[ "$USES_DCGM_POWER" == "1" ]]; then
            mkdir -p "$LOGS_DIR/power" 2>/dev/null || true
            cp "$GITHUB_WORKSPACE/exporter-image.sha256" "$LOGS_DIR/power/exporter-image.sha256" 2>/dev/null || true
            cp "$GITHUB_WORKSPACE/power-producer-sha.txt" "$LOGS_DIR/power/power-producer-sha.txt" 2>/dev/null || true
        fi
        # Copy + tar are independent best-effort; an in-flight write
        # from a worker .out file at SIGTERM time would otherwise abort
        # the whole script before either succeeds.
        cp -r "$LOGS_DIR" "$GITHUB_WORKSPACE/LOGS" 2>/dev/null || true
        tar czf "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz" -C "$LOGS_DIR" . 2>/dev/null || true
    fi
}
trap _snapshot_server_logs EXIT

# Wait for log file to appear (also check job is still alive)
while ! ls "$LOG_FILE" &>/dev/null; do
    if ! squeue -j "$JOB_ID" --noheader 2>/dev/null | grep -q "$JOB_ID"; then
        echo "ERROR: Job $JOB_ID failed before creating log file"
        scontrol show job "$JOB_ID"
        exit 1
    fi
    echo "Waiting for JOB_ID $JOB_ID to begin and $LOG_FILE to appear..."
    sleep 5
done

# Poll for job completion in background
(
    while squeue -j "$JOB_ID" --noheader 2>/dev/null | grep -q "$JOB_ID"; do
        sleep 10
    done
) &
POLL_PID=$!

echo "Tailing LOG_FILE: $LOG_FILE"

# Stream the log file until job completes (-F follows by name, polls instead of inotify for NFS)
tail -F -s 2 -n+1 "$LOG_FILE" --pid=$POLL_PID 2>/dev/null

wait $POLL_PID

set -x

echo "Job $JOB_ID completed!"
echo "Collecting results..."

if [ -d "$LOGS_DIR" ]; then
    echo "Found logs directory: $LOGS_DIR"
    # Tarball + LOGS copy + power provenance markers are produced by the EXIT
    # trap defined near JOB_ID extraction (so cancel paths also get them);
    # just log here.
    echo "multinode_server_logs.tar.gz will be (re)produced on script EXIT."
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
                    cp "$result_file" "$WORKSPACE_RESULT_FILE"

                    echo "Copied result file to: $WORKSPACE_RESULT_FILE"
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
    EVAL_DIR="$LOGS_DIR/eval_results"
    if [ -d "$EVAL_DIR" ]; then
        echo "Extracting eval results from $EVAL_DIR"
        shopt -s nullglob
        for eval_file in "$EVAL_DIR"/*; do
            [ -f "$eval_file" ] || continue
            eval_dest="$GITHUB_WORKSPACE/$(basename "$eval_file")"
            rm -f "$eval_dest"
            if cp "$eval_file" "$eval_dest"; then
                echo "Copied eval artifact: $(basename "$eval_file")"
            else
                echo "WARNING: Failed to copy eval artifact, continuing: $(basename "$eval_file")"
            fi
        done
        shopt -u nullglob
    else
        echo "WARNING: RUN_EVAL=true but no eval results found at $EVAL_DIR"
    fi

    # srt-slurm stages eval artifacts but does not write the metadata file
    # consumed by score validation. Reuse the canonical metadata writer so
    # topology and recipe identity stay aligned with the workflow inputs.
    eval_conc_value="${EVAL_CONC:-${CONC:-1}}"
    (
        export IS_MULTINODE=true
        # shellcheck source=benchmarks/benchmark_lib.sh
        source "$GITHUB_WORKSPACE/benchmarks/benchmark_lib.sh"
        _write_lm_eval_meta_json \
            "$GITHUB_WORKSPACE/meta_env.json" "" "$eval_conc_value"
    )
    echo "Wrote meta_env.json (conc=${eval_conc_value}, prefix=${MODEL_PREFIX:-unknown})"
fi

# Snapshot logs to GITHUB_WORKSPACE BEFORE cleanup, so the EXIT trap's
# `[ -d "$LOGS_DIR" ]` guard isn't already false by the time it fires
# (it runs AFTER the rm below, since EXIT traps are last-thing-before-exit).
# Without this inline call, R25 lost both 1p6d shards' logs.
_snapshot_server_logs

# Clean up srt-slurm outputs to prevent NFS silly-rename lock files
# from blocking the next job's checkout on this runner
echo "Cleaning up srt-slurm outputs..."
for i in 1 2 3 4 5; do
    rm -rf outputs 2>/dev/null && break
    echo "Retry $i/5: Waiting for NFS locks to release..."
    sleep 10
done
find . -name '.nfs*' -delete 2>/dev/null || true
