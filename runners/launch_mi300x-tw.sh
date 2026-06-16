#!/usr/bin/env bash

# TensorWave MI300X (tw018, tw032) — docker runners, NOT slurm.
# Unlike the amds/325x/355x fleets (salloc + enroot squash on a Slurm worker),
# these are standalone docker nodes: the benchmark image runs directly via
# `docker run` on the same host the GitHub Actions runner lives on. Storage is
# node-local (single ~14T root, no shared/NFS mount), so the HF cache lives
# under the runner user's home and weights download on first use.

# The runner user (cam) is not in the docker group, but has passwordless sudo.
# Prefer rootless `docker` when available, fall back to `sudo docker`.
DOCKER="docker"
if ! docker ps >/dev/null 2>&1; then
    DOCKER="sudo docker"
fi

export HF_HUB_CACHE_MOUNT="${HF_HUB_CACHE_MOUNT:-$HOME/hf_hub_cache/}"
export PORT=8888

server_name="bmk-server"

# Route spec-decoding=mtp configs to the _mtp benchmark script (e.g.
# minimaxm3_fp8_mi300x_mtp.sh), matching the h100/h200 docker launchers.
SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" ]] && printf '_mtp' || printf '')

# Variables the benchmark scripts (and benchmark_lib.sh) read. The slurm
# launchers get these for free via `srun --export=ALL`; docker needs each named
# explicitly so its value is pulled from the job environment.
PASS_ENV=(
    HF_TOKEN HF_HUB_CACHE
    MODEL TP CONC MAX_MODEL_LEN ISL OSL RANDOM_RANGE_RATIO
    CONTEXT_LENGTH ATTN_BACKEND DP_ATTENTION EP EP_SIZE
    DRAFT_MODEL NUM_SPEC_TOKENS SPEC_DECODING
    OFFLOADING OFFLOAD_ARGS PARALLEL_ARGS MAX_PREFILL_TOKENS
    FUSE_ROPE_KVCACHE NF TOTAL_CPU_DRAM_GB
    FRAMEWORK PRECISION EXP_NAME DISAGG
    RESULT_DIR RESULT_FILENAME RUNNER_TYPE RUN_EVAL EVAL_ONLY
    EVAL_CONTEXT_ARGS EVAL_MAX_MODEL_LEN
    PROFILE SGLANG_TORCH_PROFILER_DIR VLLM_TORCH_PROFILER_DIR VLLM_RPC_TIMEOUT
)
ENV_FLAGS=()
for v in "${PASS_ENV[@]}"; do
    ENV_FLAGS+=(-e "$v")
done
ENV_FLAGS+=(-e PORT="$PORT" -e PYTHONPYCACHEPREFIX=/tmp/pycache/)

mkdir -p "$HF_HUB_CACHE_MOUNT"

set -x

# numa_balancing hurts MI300X throughput; disable it like the older AMD docker
# launchers did (passwordless sudo is available on these nodes).
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing' || true

$DOCKER run --rm --ipc=host --shm-size=16g --network=host --name=$server_name \
--privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
--cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
-v "$HF_HUB_CACHE_MOUNT":"$HF_HUB_CACHE" \
-v "$GITHUB_WORKSPACE":/workspace/ -w /workspace/ \
"${ENV_FLAGS[@]}" \
--entrypoint=/bin/bash \
"$IMAGE" \
benchmarks/single_node/${SCENARIO_SUBDIR}"${EXP_NAME%%_*}_${PRECISION}_mi300x${SPEC_SUFFIX}.sh"
