#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"

python3 - "$ROOT/configs/amd-master.yaml" "$HERE/models_vllm.yaml" <<'PY'
import sys
import yaml

config = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
models = yaml.safe_load(open(sys.argv[2], encoding="utf-8"))

recipe = config["kimik3-fp4-mi355x-vllm-disagg-agentic"]
point = recipe["scenarios"]["agentic-coding"][0]
arm = point["search-space"][0]
assert recipe["image"].endswith("nightly-1dc464d42681d22f38caf1fdc1eb632dc4421c45")
assert recipe["framework"] == "vllm-disagg"
assert recipe["kv-p2p-transfer"] == "moriio"
assert point["dram-utilization"] == 0.60
assert arm["spec-decoding"] == "none"
assert arm["conc-list"] == [40]
assert arm["kv-offload-backend"]["name"] == "lmcache-k3"
assert arm["kv-offload-backend"]["version"].startswith("git-140819c9")
assert arm["prefill"]["tp"] == 8
assert arm["decode"]["tp"] == 8
assert arm["decode"]["dcp-size"] == 8
settings = arm["prefill"]["additional-settings"] + arm["decode"]["additional-settings"]
assert "DECODE_CP_KV_CACHE_INTERLEAVE_SIZE=1" in settings
assert any(item.startswith("VLLM_K3_FORK_SHA=f1870840") for item in settings)
assert "mooncake" not in repr(recipe).lower()

k3 = models["Kimi-K3"]
for role in ("prefill_flags", "decode_flags"):
    flags = k3[role]
    for expected in (
        "--gpu-memory-utilization 0.90",
        "--max-num-seqs 80",
        "--max-model-len 1048576",
        "--kv-cache-dtype fp8",
        "--max-num-batched-tokens 16384",
        "--attention-backend ROCM_AITER_MLA",
        '"cudagraph_mode":"PIECEWISE"',
    ):
        assert expected in flags
    assert "FULL_AND_PIECEWISE" not in flags
PY

echo "Kimi-K3 PD recipe tests passed"
