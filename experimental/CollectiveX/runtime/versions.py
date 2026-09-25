"""Pinned backend sources and build recipes; changing a pin changes its cache identity."""

# Upstream DeepEP main carries #630 (single-node V2 init), #642 (the Blackwell LL combine
# fence, issue #700), #715 (release before GIN barrier), #688 (NCCL Device API), and
# #640/#627 (pip-wheel SO-name resolution). Cache identity includes the exact commit.
DEEPEP_REPO = "https://github.com/deepseek-ai/DeepEP"
DEEPEP_COMMIT = "01dc3aaac82068020353dce2c302e38153c0bfaa"

# The cu12 NVSHMEM wheel on cu130 images poisons sm103's MNNVL heap initialization. Torch
# 2.10.0+cu130 also poisoned that context on driver 580.159.03; 2.11.0 matches the image.
# Both pins belong in the venv cache key, not just in its installation command.
DEEPEP_NVSHMEM = "nvidia-nvshmem-cu13==3.4.5"
DEEPEP_TORCH = "torch==2.11.0"
# Bump when build flags change without a pin change, preventing stale .ready cache reuse.
DEEPEP_BUILD_GEN = "dlarch1"

UCCL_REPO = "https://github.com/uccl-project/uccl"
UCCL_COMMIT = "fc1b582031221645ea9fce58aeb57187713145e3"

# nccl-extensions owns nccl.ep since nccl4py 0.4 stopped bundling it. Its combine-recv
# fence releases the LL ladder clamp. nccl4py is pinned alongside it; both key the cache.
NCCL_EP_SPECS = ("nccl-extensions[cu13]==0.1.0", "nccl4py[cu13]==0.5.0")

# Fresh rank tasks receive only these backend-created settings. Network configuration is
# applied separately at the rank boundary, preserving Slurm's exact HCA selector handling.
RANK_ENV_VARS = (
    "PATH", "VIRTUAL_ENV", "LD_LIBRARY_PATH", "PYTHONPATH", "CUDA_HOME", "CPATH",
    "NVCC_PREPEND_FLAGS", "NVSHMEM_DIR", "EP_NCCL_ROOT_DIR", "EP_NVSHMEM_ROOT_DIR",
    "EP_JIT_CACHE_DIR", "EP_REUSE_NCCL_COMM", "NCCL_CUMEM_ENABLE", "UCCL_EP_ENABLE_AGGRESSIVE_ATOMIC",
)
DEEPEP_UNSETS = ("EP_SUPPRESS_NCCL_CHECK",)
