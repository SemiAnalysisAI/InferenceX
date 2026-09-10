#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"

python3 - "$ROOT/configs/amd-master.yaml" "$HERE/models_vllm.yaml" <<'PY'
import json
import re
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
models = yaml.safe_load(open(sys.argv[2], encoding="utf-8"))
models_path = Path(sys.argv[2])

recipe = config["kimik3-fp4-mi355x-vllm-disagg-agentic"]
point = recipe["scenarios"]["agentic-coding"][0]
arm = point["search-space"][0]
assert len(point["search-space"]) == 1
assert recipe["image"] == (
    "vllm/vllm-openai-rocm:nightly@"
    "sha256:91e381f072d6a44e1e4c97c82dce06e50e5189905cb3999a11471c5a8fc6a563"
)
assert recipe["framework"] == "vllm-disagg"
assert recipe["kv-p2p-transfer"] == "moriio"
assert arm["prefill"]["tp"] == 8
assert arm["prefill"]["dcp-size"] == 8
assert arm["decode"]["tp"] == 8
assert arm["decode"]["dcp-size"] == 8
assert point["dram-utilization"] == 0.60
assert arm["spec-decoding"] == "none"
assert arm["conc-list"] == [1, 40, 48, 70]
assert arm["kv-offloading"] == "dram"
assert arm["kv-offload-backend"]["name"] == "lmcache-k3"
assert arm["kv-offload-backend"]["version"] == "nightly-rocm"
settings = arm["prefill"]["additional-settings"] + arm["decode"]["additional-settings"]
assert "DECODE_CP_KV_CACHE_INTERLEAVE_SIZE=1536" in settings
assert "PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE=1536" in settings
assert "TOTAL_CPU_DRAM_GB=1799" in settings
assert "LMCACHE_CHUNK_SIZE=12288" in settings
assert "LMCACHE_L1_SIZE_GB=1799" in settings
assert "LMCACHE_L1_READ_TTL_SECONDS=1800" in settings
assert "LMCACHE_MAX_GPU_WORKERS=8" in settings
assert "LMCACHE_VERSION=latest-rocm" in settings
assert "GPU_MEMORY_UTILIZATION=0.88" in settings
assert "SERVER_UP_TIMEOUT=900" in settings
assert "VLLM_K3_FORK_REF=k3-pd-recovery-integration" in settings
assert any(item.startswith("VLLM_K3_FORK_SHA=710a6cbef") for item in settings)
assert "mooncake" not in repr(recipe).lower()

prefill_scale_recipe = config[
    "kimik3-fp4-mi355x-vllm-disagg-agentic-prefill-scale"
]
prefill_scale_point = prefill_scale_recipe["scenarios"]["agentic-coding"][0]
assert len(prefill_scale_point["search-space"]) == 1
prefill_scale_arm = prefill_scale_point["search-space"][0]
assert prefill_scale_arm["conc-list"] == [48]
assert prefill_scale_arm["spec-decoding"] == "none"
assert prefill_scale_arm["prefill"]["num-worker"] == 2
assert prefill_scale_arm["decode"]["num-worker"] == 1
assert prefill_scale_arm["prefill"]["dcp-size"] == 8
assert prefill_scale_arm["decode"]["dcp-size"] == 8
prefill_scale_settings = (
    prefill_scale_arm["prefill"]["additional-settings"]
    + prefill_scale_arm["decode"]["additional-settings"]
)
for expected in (
    "PREFILL_NODES=2",
    "DECODE_NODES=1",
    "PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE=1536",
    "DECODE_CP_KV_CACHE_INTERLEAVE_SIZE=1536",
    "TOTAL_CPU_DRAM_GB=1799",
    "LMCACHE_L1_SIZE_GB=1799",
    "LMCACHE_MAX_GPU_WORKERS=8",
    "LMCACHE_VERSION=latest-rocm",
    "VLLM_K3_FORK_REF=k3-pd-recovery-integration",
    "VLLM_K3_FORK_SHA=710a6cbef37aa7b7fb88255e09795729483943ad",
):
    assert expected in prefill_scale_settings
assert "LMCACHE_ON_DECODE=true" not in repr(prefill_scale_arm)
assert "mooncake" not in repr(prefill_scale_recipe).lower()

dspark_recipe = config["kimik3-fp4-mi355x-vllm-disagg-agentic-dspark"]
dspark_point = dspark_recipe["scenarios"]["agentic-coding"][0]
assert len(dspark_point["search-space"]) == 1
dspark_arm = dspark_point["search-space"][0]
assert dspark_arm["conc-list"] == [48]
assert dspark_arm["spec-decoding"] == "mtp"
assert dspark_arm["prefill"]["num-worker"] == 1
assert dspark_arm["decode"]["num-worker"] == 1
assert dspark_arm["prefill"]["dcp-size"] == 8
assert dspark_arm["decode"]["dcp-size"] == 8
assert dspark_arm["kv-offload-backend"] == {
    "name": "lmcache-k3",
    "version": "nightly-rocm",
}
dspark_settings = (
    dspark_arm["prefill"]["additional-settings"]
    + dspark_arm["decode"]["additional-settings"]
)
for expected in (
    "PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE=1",
    "DECODE_CP_KV_CACHE_INTERLEAVE_SIZE=1",
    "SPEC_NUM_TOKENS=4",
    "SPEC_ATTN_BACKEND=TRITON_MLA",
    "SPEC_REJECTION_SAMPLE_METHOD=synthetic",
    "SPEC_SYNTHETIC_ACCEPTANCE_LENGTH=3.36",
    "SPEC_MAX_NUM_SEQS=16",
    "SPEC_MAX_NUM_BATCHED_TOKENS=4096",
    "SPEC_PREFILL_CUDAGRAPH_MODE=FULL_AND_PIECEWISE",
    "SPEC_DECODE_CUDAGRAPH_MODE=FULL_DECODE_ONLY",
    "LMCACHE_CHUNK_SIZE=24576",
    "LMCACHE_VERSION=latest-rocm",
    "LMCACHE_WORKER_REGISTRATION_GRACE_SECONDS=7200",
    "VLLM_ROCM_PAGE_ALIGN_KV=1",
):
    assert expected in dspark_settings
assert all(
    "VLLM_K3_FORK_SHA=4510af190bb5e7b840df837e691a040a01033021"
    in settings
    for settings in (
        dspark_arm["prefill"]["additional-settings"],
        dspark_arm["decode"]["additional-settings"],
    )
)

scale_recipe = config["kimik3-fp4-mi355x-vllm-disagg-agentic-decode-scale"]
scale_arms = scale_recipe["scenarios"]["agentic-coding"][0]["search-space"]
assert len(scale_arms) == 2
for scale_arm, num_decode_workers, concurrencies in zip(
    scale_arms, (2, 3), ([40, 48, 70], [40]), strict=True
):
    assert scale_arm["conc-list"] == concurrencies
    assert scale_arm["spec-decoding"] == "none"
    assert scale_arm["prefill"]["num-worker"] == 1
    assert scale_arm["prefill"]["dcp-size"] == 8
    assert scale_arm["decode"]["num-worker"] == num_decode_workers
    assert scale_arm["decode"]["dcp-size"] == 8
    assert f"PREFILL_NODES=1" in scale_arm["prefill"]["additional-settings"]
    assert (
        f"DECODE_NODES={num_decode_workers}"
        in scale_arm["decode"]["additional-settings"]
    )
    assert "LMCACHE_ON_DECODE=true" not in repr(scale_arm)
    scale_settings = (
        scale_arm["prefill"]["additional-settings"]
        + scale_arm["decode"]["additional-settings"]
    )
    assert "VLLM_K3_FORK_REF=k3-pd-recovery-integration" in scale_settings
    assert any(
        item.startswith("VLLM_K3_FORK_SHA=710a6cbef") for item in scale_settings
    )
    assert "LMCACHE_L1_READ_TTL_SECONDS=1800" in scale_settings
    assert "LMCACHE_VERSION=latest-rocm" in scale_settings
    assert scale_arm["kv-offload-backend"] == {
        "name": "lmcache-k3",
        "version": "nightly-rocm",
    }
assert "mooncake" not in repr(scale_recipe).lower()

balanced_recipe = config[
    "kimik3-fp4-mi355x-vllm-disagg-agentic-balanced-scale"
]
balanced_point = balanced_recipe["scenarios"]["agentic-coding"][0]
assert balanced_point["dram-utilization"] == 0.60
assert len(balanced_point["search-space"]) == 1
balanced_arm = balanced_point["search-space"][0]
assert balanced_arm["conc-list"] == [70]
assert balanced_arm["spec-decoding"] == "none"
assert balanced_arm["prefill"]["num-worker"] == 2
assert balanced_arm["prefill"]["dcp-size"] == 8
assert balanced_arm["decode"]["num-worker"] == 2
assert balanced_arm["decode"]["dcp-size"] == 8
balanced_settings = (
    balanced_arm["prefill"]["additional-settings"]
    + balanced_arm["decode"]["additional-settings"]
)
assert "PREFILL_NODES=2" in balanced_settings
assert "DECODE_NODES=2" in balanced_settings
assert "LMCACHE_L1_READ_TTL_SECONDS=1800" in balanced_settings
assert "LMCACHE_VERSION=latest-rocm" in balanced_settings
assert "LMCACHE_ON_DECODE=true" not in repr(balanced_arm)
assert balanced_arm["kv-offload-backend"] == {
    "name": "lmcache-k3",
    "version": "nightly-rocm",
}

k3 = models["Kimi-K3"]
env = k3["env"]
assert "VLLM_SSM_CONV_STATE_LAYOUT=DS" in env
assert "VLLM_USE_BREAKABLE_CUDAGRAPH" not in env
assert "VLLM_ALLOW_DCP_FULL_CUDAGRAPH=1" in env
assert "PREFIX_CACHING_HASH_ALGO=sha256" in env
assert "VLLM_USE_BREAKABLE_CUDAGRAPH=1" in k3["prefill_env"]
assert "TORCH_NCCL_BLOCKING_WAIT=0" in k3["prefill_env"]
assert "VLLM_USE_BREAKABLE_CUDAGRAPH=0" in k3["decode_env"]
assert "TORCH_NCCL_BLOCKING_WAIT=0" in k3["decode_env"]

server_vllm = models_path.with_name("server_vllm.sh").read_text(encoding="utf-8-sig")
job_slurm = models_path.with_name("job.slurm").read_text(encoding="utf-8-sig")
setup_deps = models_path.with_name("setup_deps.sh").read_text(encoding="utf-8-sig")
assert "apply_vllm_gpu_memory_utilization" in server_vllm
assert "-e GPU_MEMORY_UTILIZATION=" in job_slurm
assert "-e LMCACHE_L1_READ_TTL_SECONDS=" in job_slurm
assert "-e LMCACHE_WORKER_REGISTRATION_GRACE_SECONDS=" in job_slurm
assert "-e LMCACHE_VERSION=" in job_slurm
assert "-e AIPERF_EXPERIMENTAL_FAST=" in job_slurm
for expected in (
    "SPEC_NUM_TOKENS",
    "SPEC_MODEL",
    "SPEC_ATTN_BACKEND",
    "SPEC_DRAFT_SAMPLE_METHOD",
    "SPEC_REJECTION_SAMPLE_METHOD",
    "SPEC_SYNTHETIC_ACCEPTANCE_LENGTH",
    "SPEC_MAX_NUM_SEQS",
    "SPEC_MAX_NUM_BATCHED_TOKENS",
    "SPEC_CUDAGRAPH_MODE",
    "SPEC_PREFILL_CUDAGRAPH_MODE",
    "SPEC_DECODE_CUDAGRAPH_MODE",
):
    assert f"-e {expected}=" in job_slurm
assert '"${MODEL_NAME:-}" == "Kimi-K3"' in server_vllm
assert '"${SPEC_DECODING:-}" == "mtp"' in server_vllm
assert "--speculative-config '${spec_config}'" in server_vllm
assert "spec_capture_size=" in server_vllm
assert "SPEC_PREFILL_CUDAGRAPH_MODE" in server_vllm
assert "SPEC_DECODE_CUDAGRAPH_MODE" in server_vllm
assert "role_cudagraph_mode=$spec_prefill_cudagraph_mode" in server_vllm
assert "role_cudagraph_mode=$spec_decode_cudagraph_mode" in server_vllm
assert '"v1/attention/backends/mla/triton_mla.py"' in setup_deps
assert "VLLM_ROCM_PAGE_ALIGN_KV" in setup_deps
assert "alignment_offset = (-allocation.data_ptr()) % page_size" in setup_deps
assert "-e VLLM_ROCM_PAGE_ALIGN_KV=" in job_slurm
lmcache_mp = models_path.with_name("lmcache_mp.sh").read_text(encoding="utf-8-sig")
assert "latest-rocm" in lmcache_mp
assert "LMCACHE_RESOLVED_VERSION" in lmcache_mp
assert "--worker-registration-grace-seconds" in lmcache_mp
assert '"kv_cache_config" in inspect.signature(get_dcp_decorated_model_name).parameters' in lmcache_mp

flags = k3["prefill_flags"]
for expected in (
    "--gpu-memory-utilization 0.90",
    "--max-num-seqs 80",
    "--max-model-len 1048576",
    "--kv-cache-dtype fp8",
    "--block-size 128",
    "--max-num-batched-tokens 16384",
    "--enable-prefix-caching",
    "--prefix-match-unit 128",
    "--attention-backend ROCM_AITER_MLA",
    "use_prefill_query_quantization\":true",
):
    assert expected in flags, expected
assert "--speculative-config" not in flags

comp = json.loads(re.search(r"--compilation-config '(\{.*\})'", flags).group(1))
assert comp["cudagraph_mode"] == "PIECEWISE"
assert comp["max_cudagraph_capture_size"] == 512
assert comp["cudagraph_capture_sizes"] == (
    list(range(1, 17)) + [24, 32, 48, 64, 96, 128, 256, 512]
)

decode_comp = json.loads(
    re.search(r"--compilation-config '(\{.*\})'", k3["decode_flags"]).group(1)
)
assert decode_comp["cudagraph_mode"] == "FULL_DECODE_ONLY"
assert decode_comp["max_cudagraph_capture_size"] == 4096
assert decode_comp["cudagraph_capture_sizes"] == list(range(1, 81)) + [
    128,
    256,
    512,
    1024,
    2048,
    4096,
]
PY

echo "Kimi-K3 PD recipe tests passed"
