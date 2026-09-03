#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

! grep -Eqi 'mooncake|MooncakeStoreConnector|INFERENCEX_MOONCAKE|MC_WORKERS_PER_CTX' \
    "$HERE/server_vllm.sh" \
    "$HERE/lmcache_mp.sh" \
    "$HERE/job.slurm"

for removed in \
    apply_vllm_46240_scheduler_patch.py \
    apply_vllm_mooncake_transfer_batches.py \
    apply_aiter_mla_block_n_fallback.py \
    apply_vllm_aiter_mla_head_pad.py \
    apply_vllm_aiter_mla_persistent_mtp.py \
    apply_vllm_kv_group_debug.py
do
    [[ ! -e "$HERE/patches/$removed" ]]
    ! grep -Rqs "$removed" "$HERE"
done

python3 - "$HERE/setup_deps.sh" <<'PY'
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()
match = re.search(r"files = \[(.*?)\]\nfor rel in files:", text, re.S)
assert match, "overlay allowlist not found"
paths = re.findall(r'"([^"]+\.py)"', match.group(1))
assert paths == [
    "distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py",
    "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py",
    "distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py",
    "distributed/kv_transfer/kv_connector/v1/moriio/moriio_layout.py",
    "model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py",
    "model_executor/models/qwen3_dflash.py",
]
assert "os.walk(" not in text
assert "VLLM_K3_FORK_SHA" in text
assert "from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector" in text
assert "MoRIIOConnector" in text
PY

grep -q 'export SERVER_FLUSH_URLS_CSV' "$HERE/server_vllm.sh"
grep -qx 'sentencepiece' "$HERE/../../../utils/agentic-benchmark/requirements.txt"

python3 - "$HERE/job.slurm" <<'PY'
import sys

job = open(sys.argv[1], encoding="utf-8").read()
for removed in (
    "INFERENCEX_MOONCAKE",
    "MOONCAKE_LOAD_ASYNC",
    "MOONCAKE_LOOKUP_ASYNC",
    "MOONCAKE_DECODE_STORE",
    "VLLM_PATCH_46240",
    "BENCH_MIN_NUM_PROMPTS",
    "KEEP_SERVER_ALIVE",
):
    assert removed not in job
PY

echo "vLLM disagg framework static tests passed"
