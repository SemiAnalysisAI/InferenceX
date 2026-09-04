#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"

python3 - "$ROOT/configs/amd-master.yaml" "$HERE/models_vllm.yaml" <<'PY'
import json
import re
import sys

import yaml

config = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
models = yaml.safe_load(open(sys.argv[2], encoding="utf-8"))

recipe = config["kimik3-fp4-mi355x-vllm-disagg-agentic"]
point = recipe["scenarios"]["agentic-coding"][0]
arm = point["search-space"][0]
arm_2p1d = point["search-space"][1]
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
assert arm["conc-list"] == [40]
assert arm["kv-offloading"] == "dram"
assert arm["kv-offload-backend"]["name"] == "lmcache-k3"
assert arm["kv-offload-backend"]["version"].startswith("git-d131cecf")
assert arm_2p1d["prefill"]["num-worker"] == 2
assert arm_2p1d["prefill"]["dcp-size"] == 8
assert arm_2p1d["decode"]["num-worker"] == 1
assert arm_2p1d["decode"]["dcp-size"] == 8
assert arm_2p1d["conc-list"] == [70]
settings = arm["prefill"]["additional-settings"] + arm["decode"]["additional-settings"]
assert "DECODE_CP_KV_CACHE_INTERLEAVE_SIZE=1536" in settings
assert "PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE=1536" in settings
assert "TOTAL_CPU_DRAM_GB=1799" in settings
assert "LMCACHE_CHUNK_SIZE=12288" in settings
assert "LMCACHE_L1_SIZE_GB=1799" in settings
assert "LMCACHE_MAX_GPU_WORKERS=8" in settings
assert "SERVER_UP_TIMEOUT=900" in settings
assert "VLLM_K3_FORK_REF=moriio-k3" in settings
assert "VLLM_K3_FORK_SHA=e91a9cd3f" in settings
assert "mooncake" not in repr(recipe).lower()

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
assert comp["cudagraph_mode"] == "FULL_AND_PIECEWISE"
assert comp["max_cudagraph_capture_size"] == 4096
assert comp["cudagraph_capture_sizes"] == list(range(1, 81)) + [128, 256, 512, 1024, 2048, 4096]

decode_comp = json.loads(
    re.search(r"--compilation-config '(\{.*\})'", k3["decode_flags"]).group(1)
)
assert decode_comp["cudagraph_mode"] == "FULL_DECODE_ONLY"
assert decode_comp["max_cudagraph_capture_size"] == 4096
assert decode_comp["cudagraph_capture_sizes"] == comp["cudagraph_capture_sizes"]
PY

echo "Kimi-K3 PD recipe tests passed"
