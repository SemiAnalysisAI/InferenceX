#!/usr/bin/env bash
# Contract tests for the multinode cluster profiles.
#
# Runs every profile in INFX_DRY_RUN=1 mode across the supported
# (cluster, framework, model, precision, scenario) matrix and asserts the
# resolved plan: model path + srt alias from the configs/runners.yaml
# registry, the srt-slurm pin, and cluster-specific srtslurm.yaml content.
# No network or Slurm access is needed; this runs anywhere bash + python3 +
# PyYAML exist.
#
# Usage: bash utils/multinode_contract/test_dry_run.sh

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PASS=0
FAIL=0

# run_case <launcher> <framework> <model-prefix> <precision> <model> <is-agentic> <expect-grep>...
run_case() {
    local launcher="$1" framework="$2" model_prefix="$3" precision="$4" model="$5" agentic="$6"
    shift 6
    local label="$launcher $framework $model_prefix/$precision agentic=$agentic"

    local output status
    output=$(
        cd "$REPO_ROOT" &&
        IS_MULTINODE=true \
        INFX_DRY_RUN=1 \
        GITHUB_WORKSPACE="$REPO_ROOT" \
        IMAGE="example/image:v1" \
        MODEL="$model" \
        MODEL_PREFIX="$model_prefix" \
        PRECISION="$precision" \
        FRAMEWORK="$framework" \
        ISL=1024 OSL=1024 \
        CONC_LIST="4" \
        SPEC_DECODING=none \
        CONFIG_FILE="recipes/example/recipe.yaml" \
        IS_AGENTIC="$agentic" \
        RESULT_FILENAME="contract-test" \
        RUNNER_NAME="${launcher#launch_}_0" \
        MODEL_PATH="" \
        bash "runners/${launcher}.sh" 2>&1
    )
    status=$?

    if [ $status -ne 0 ]; then
        echo "FAIL: $label — launcher exited $status"
        echo "$output" | tail -20 | sed 's/^/    /'
        FAIL=$((FAIL + 1))
        return
    fi

    local expect ok=1
    for expect in "$@"; do
        if ! echo "$output" | grep -qF -- "$expect"; then
            echo "FAIL: $label — missing: $expect"
            ok=0
        fi
    done
    if [ $ok -eq 1 ]; then
        PASS=$((PASS + 1))
    else
        echo "$output" | sed -n '/=== INFX multinode dry run ===/,/=== end dry run ===/p' | sed 's/^/    /'
        FAIL=$((FAIL + 1))
    fi
}

# expect_failure <launcher> <framework> <model-prefix> <precision>
expect_failure() {
    local launcher="$1" framework="$2" model_prefix="$3" precision="$4"
    local label="$launcher $framework $model_prefix/$precision (expected rejection)"
    if (
        cd "$REPO_ROOT" &&
        IS_MULTINODE=true INFX_DRY_RUN=1 GITHUB_WORKSPACE="$REPO_ROOT" \
        IMAGE="example/image:v1" MODEL="x/y" MODEL_PREFIX="$model_prefix" \
        PRECISION="$precision" FRAMEWORK="$framework" ISL=1024 OSL=1024 \
        CONC_LIST="4" SPEC_DECODING=none CONFIG_FILE="recipes/x.yaml" IS_AGENTIC=0 \
        RESULT_FILENAME="contract-test" RUNNER_NAME="${launcher#launch_}_0" \
        MODEL_PATH="" bash "runners/${launcher}.sh"
    ) >/dev/null 2>&1; then
        echo "FAIL: $label — launcher unexpectedly succeeded"
        FAIL=$((FAIL + 1))
    else
        PASS=$((PASS + 1))
    fi
}

STANDARD_PIN="srt-slurm ref:  main"

# --- gb300 ------------------------------------------------------------
run_case launch_gb300-nv dynamo-vllm dsv4 fp4 deepseek-ai/DeepSeek-V4-Pro 0 \
    "$STANDARD_PIN" \
    'export MODEL_PATH=/scratch/models/DeepSeek-V4-Pro' \
    '"deepseek-v4-pro": "/scratch/models/DeepSeek-V4-Pro"' \
    'use_segment_sbatch_directive: false' \
    '"/data/home/sa-shared/gharunners/ai-perf-cache": "/aiperf_mmap_cache"' \
    'gpus_per_node: 4' \
    'no-preflight:   no' \
    'tags:           gb300,dsv4,fp4,1024x1024'

run_case launch_gb300-nv dynamo-trt dsv4 fp4 deepseek-ai/DeepSeek-V4-Pro 0 \
    '"deepseek-ai/DeepSeek-V4-Pro": "/scratch/models/DeepSeek-V4-Pro"'

run_case launch_gb300-nv dynamo-vllm dsv4 fp4 deepseek-ai/DeepSeek-V4-Pro 1 \
    'no-preflight:   yes'

run_case launch_gb300-nv dynamo-sglang glm5 fp4 zai-org/GLM-5 0 \
    'export MODEL_PATH=/scratch/models/GLM-5-NVFP4' \
    '"glm-5-fp4": "/scratch/models/GLM-5-NVFP4"'

run_case launch_gb300-nv dynamo-trt glm5 fp4 zai-org/GLM-5 0 \
    '"nvidia/GLM-5-NVFP4": "/scratch/models/GLM-5-NVFP4"' \
    'export SERVED_MODEL_NAME=glm-5-nvfp4'

expect_failure launch_gb300-nv dynamo-vllm notamodel fp4

# --- gb200 ------------------------------------------------------------
run_case launch_gb200-nv dynamo-sglang dsr1 fp8 deepseek-ai/DeepSeek-R1-0528 0 \
    "$STANDARD_PIN" \
    'export MODEL_PATH=/mnt/lustre01/models/deepseek-r1-0528' \
    '"dsr1-fp8": "/mnt/lustre01/models/deepseek-r1-0528"' \
    'setup script:   install-torchao.sh' \
    'use_segment_sbatch_directive: false' \
    'tags:           gb200,'

run_case launch_gb200-nv dynamo-trt dsr1 fp8 deepseek-ai/DeepSeek-R1-0528 0 \
    'export MODEL_PATH=/mnt/numa1/groups/sa-shared/models/deepseek-r1-0528' \
    'export SERVED_MODEL_NAME=deepseek-r1-fp8'

# Unregistered model on gb200 falls back to the HF id (Hub download).
run_case launch_gb200-nv dynamo-sglang qwen3.5 fp4 Qwen/Qwen3.5-397B-A17B-NVFP4 0 \
    'export MODEL_PATH=Qwen/Qwen3.5-397B-A17B-NVFP4' \
    '"qwen3.5": "Qwen/Qwen3.5-397B-A17B-NVFP4"'

run_case launch_gb200-nv dynamo-vllm dsv4 fp4 deepseek-ai/DeepSeek-V4-Pro 0 \
    'export MODEL_PATH=/mnt/lustre01/models/DeepSeek-V4-Pro-NVFP4' \
    '"deepseek-v4-pro": "/mnt/lustre01/models/DeepSeek-V4-Pro-NVFP4"'

# --- b300 -------------------------------------------------------------
run_case launch_b300-nv dynamo-vllm minimaxm3 fp8 MiniMaxAI/MiniMax-M3-MXFP8 0 \
    "$STANDARD_PIN" \
    'export MODEL_PATH=/data/models/MiniMax-M3-MXFP8' \
    '"MiniMaxAI/MiniMax-M3-MXFP8": "/data/models/MiniMax-M3-MXFP8"' \
    'use_exclusive_sbatch_directive: true' \
    '"/opt/ucx-no-ud": "/usr/local/ucx"' \
    'gpus_per_node: 8' \
    'tags:           b300,'

# path-candidates fall back to the first candidate when none exist here.
run_case launch_b300-nv dynamo-vllm dsv4 fp4 deepseek-ai/DeepSeek-V4-Pro 0 \
    'export MODEL_PATH=/data/models/dsv4-pro'

expect_failure launch_b300-nv dynamo-vllm gptoss fp4

# --- b200 (both launcher names) ----------------------------------------
run_case launch_b200-dgxc dynamo-vllm dsv4 fp4 deepseek-ai/DeepSeek-V4-Pro 0 \
    "$STANDARD_PIN" \
    'export MODEL_PATH=/lustre/fsw/models/deepseek-v4-pro' \
    'use_exclusive_sbatch_directive: true' \
    'tags:           b200,'

run_case launch_b200-dgxc-slurm dynamo-vllm minimaxm3 fp8 MiniMaxAI/MiniMax-M3-MXFP8 0 \
    'export MODEL_PATH=/lustre/fsw/gharunners/models/MiniMax-M3-MXFP8' \
    '"minimax-m3-mxfp8": "/lustre/fsw/gharunners/models/MiniMax-M3-MXFP8"'

run_case launch_b200-dgxc dynamo-sglang dsr1 fp4 nvidia/DeepSeek-R1-0528-NVFP4 0 \
    'export MODEL_PATH=/scratch/fsw/models/DeepSeek-R1-0528-NVFP4-v2' \
    '"dsr1": "/scratch/fsw/models/DeepSeek-R1-0528-NVFP4-v2"'

# --- h200 / h100 --------------------------------------------------------
run_case launch_h200-dgxc-slurm dynamo-trt dsr1 fp8 deepseek-ai/DeepSeek-R1-0528 0 \
    "$STANDARD_PIN" \
    'export MODEL_PATH=/models/DeepSeek-R1-0528' \
    '"DeepSeek-R1-0528": "/models/DeepSeek-R1-0528"' \
    'use_gpus_per_node_directive: true' \
    'tags:           h200,'

# TRT recipes reference nvcr images in pyxis format (nvcr.io#...).
run_case_env() { :; }
(
    cd "$REPO_ROOT" &&
    IS_MULTINODE=true INFX_DRY_RUN=1 GITHUB_WORKSPACE="$REPO_ROOT" \
    IMAGE="nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.8.1" \
    MODEL="deepseek-ai/DeepSeek-R1-0528" MODEL_PREFIX=dsr1 PRECISION=fp8 \
    FRAMEWORK=dynamo-trt ISL=1024 OSL=1024 CONC_LIST="4" SPEC_DECODING=none \
    CONFIG_FILE="recipes/x.yaml" IS_AGENTIC=0 RESULT_FILENAME="contract-test" \
    RUNNER_NAME="h200-dgxc-slurm_0" MODEL_PATH="" \
    bash runners/launch_h200-dgxc-slurm.sh 2>&1
) | grep -qF '"nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:0.8.1"' \
    && PASS=$((PASS + 1)) \
    || { echo "FAIL: h200 pyxis container-key mapping"; FAIL=$((FAIL + 1)); }

run_case launch_h200-dgxc-slurm dynamo-sglang dsr1 fp8 deepseek-ai/DeepSeek-R1-0528 0 \
    '"dsr1-fp8": "/models/DeepSeek-R1-0528"'

expect_failure launch_h200-dgxc-slurm dynamo-vllm dsr1 fp8

run_case launch_h100-dgxc-slurm dynamo-sglang dsr1 fp8 deepseek-ai/DeepSeek-R1-0528 0 \
    "$STANDARD_PIN" \
    'export MODEL_PATH=/mnt/nfs/lustre/models/dsr1-fp8' \
    '"lmsysorg/sglang:v0.5.8-cu130"' \
    'tags:           h100,'

run_case launch_h100-dgxc-slurm dynamo-trt dsr1 fp8 deepseek-ai/DeepSeek-R1-0528 0 \
    'export SERVED_MODEL_NAME=DeepSeek-R1-0528'

expect_failure launch_h100-dgxc-slurm dynamo-vllm dsr1 fp8

# --- pin override -------------------------------------------------------
override_output=$(
    cd "$REPO_ROOT" &&
    IS_MULTINODE=true INFX_DRY_RUN=1 GITHUB_WORKSPACE="$REPO_ROOT" \
    IMAGE="example/image:v1" MODEL="deepseek-ai/DeepSeek-V4-Pro" MODEL_PREFIX=dsv4 \
    PRECISION=fp4 FRAMEWORK=dynamo-vllm ISL=8192 OSL=1024 CONC_LIST="4" \
    SPEC_DECODING=none CONFIG_FILE="recipes/x.yaml" IS_AGENTIC=0 \
    RESULT_FILENAME="contract-test" RUNNER_NAME="gb300-nv_0" MODEL_PATH="" \
    SRT_SLURM_REF="my-test-branch" \
    bash runners/launch_gb300-nv.sh 2>&1
)
if echo "$override_output" | grep -qF 'srt-slurm ref:  my-test-branch'; then
    PASS=$((PASS + 1))
else
    echo "FAIL: SRT_SLURM_REF override not honored"
    FAIL=$((FAIL + 1))
fi

echo
echo "multinode contract tests: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
