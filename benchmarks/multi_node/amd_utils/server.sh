#!/bin/bash
# SGLang Disaggregated Server Launcher with Model-Specific Configurations
# =============================================================================

# =============================================================================
# Environment Configuration
# =============================================================================

NODE0_ADDR="${NODE0_ADDR:-localhost}"
NODE_RANK="${NODE_RANK:-0}"
MODEL_DIR="${MODEL_DIR:-}"
MODEL_NAME="${MODEL_NAME:-}"

xP="${xP:-1}" #-> Number of Prefill Workers
yD="${yD:-1}" #-> Number of Decode Workers

IPADDRS="${IPADDRS:-localhost}"
HEADNODE_PORT="${HEADNODE_PORT:-20000}"
# Parallelism Configuration
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-8}"
PREFILL_ENABLE_EP="${PREFILL_ENABLE_EP:-true}"
PREFILL_ENABLE_DP="${PREFILL_ENABLE_DP:-true}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-8}"
DECODE_ENABLE_EP="${DECODE_ENABLE_EP:-true}"
DECODE_ENABLE_DP="${DECODE_ENABLE_DP:-true}"
DECODE_MTP_SIZE="${DECODE_MTP_SIZE:-0}"

# Benchmark Configuration
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-1024}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-1024}"
BENCH_RANDOM_RANGE_RATIO="${BENCH_RANDOM_RANGE_RATIO:-1}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-inf}"
BENCH_NUM_PROMPTS_MULTIPLIER="${BENCH_NUM_PROMPTS_MULTIPLIER:-10}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-512}"

# Dry Run for debugging purpose
DRY_RUN="${DRY_RUN:-0}"

# GPU count (expandable for different hardware)
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"


# =============================================================================
# DeepSeek-V4 model_type compat patch (NODE_RANK==0 only)
# =============================================================================
# DSv4-Pro-FP8 ships config.json with `"model_type": "deepseek_v4"`, which
# HuggingFace Transformers does not yet recognize. PR #23608's fallback in
# sglang/srt/hf_transformers_utils.get_config writes a patched copy under
# /tmp at import time, but the benchmark client bypasses that path: every
# downstream caller of AutoConfig.from_pretrained on the prefill node
# (`bench_serving.py`, `lm-eval`, the OpenAI tokenizer endpoint, etc.)
# crashes with `'PreTrainedConfig' object has no attribute
# 'max_position_embeddings'` because they get the stub config back.
#
# Patching the cached config.json directly with `model_type=deepseek_v3`
# fixes it WITHOUT changing the SGLang model dispatch, because dispatch
# uses the `architectures: ["DeepseekV4ForCausalLM"]` field. Idempotent.
#
# We only patch on NODE_RANK==0 to avoid NFS write contention; the
# cross-node barrier on port 5000 below ensures the other nodes only
# read the file after it's been patched.
if [[ "$NODE_RANK" == "0" && "$MODEL_NAME" == *DeepSeek-V4* ]]; then
    DSV4_CONFIG_PATH="${MODEL_DIR}/${MODEL_NAME}/config.json"
    if [[ -f "$DSV4_CONFIG_PATH" ]]; then
        if grep -q '"model_type": "deepseek_v4"' "$DSV4_CONFIG_PATH"; then
            echo "[DSv4-PATCH] $DSV4_CONFIG_PATH model_type deepseek_v4 -> deepseek_v3"
            python3 - "$DSV4_CONFIG_PATH" <<'PYEOF'
import json, sys
path = sys.argv[1]
with open(path) as f:
    cfg = json.load(f)
if cfg.get("model_type") == "deepseek_v4":
    cfg["model_type"] = "deepseek_v3"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[DSv4-PATCH] Wrote patched config.json to {path}")
else:
    print(f"[DSv4-PATCH] Skipped: model_type is now {cfg.get('model_type')!r}")
PYEOF
        else
            echo "[DSv4-PATCH] No patch needed at $DSV4_CONFIG_PATH (model_type already non-v4)"
        fi
    else
        echo "[DSv4-PATCH][WARN] $DSV4_CONFIG_PATH not found; skipping (server will likely fail to load)"
    fi
fi

# =============================================================================
# Dependencies and Environment Setup
# =============================================================================
source $SGLANG_WS_PATH/env.sh

# =============================================================================
# DeepSeekV4 disagg compat patch — is_mla_backend recognizes DeepSeekV4TokenToKVPool
# =============================================================================
# Upstream bug: sglang's disagg/utils.py::is_mla_backend() only matches
# isinstance(MLATokenToKVPool). DSv4's KV pool (DeepSeekV4TokenToKVPool) is a
# *sibling* class of MLATokenToKVPool (both extend KVCache) and is missed by
# the isinstance check, so the whole disagg path treats DSv4 as non-MLA. Then
# disagg/prefill.py:163-164 falls through to `kv_args.kv_head_num =
# self.token_to_kv_pool.head_num`, which crashes with AttributeError because
# the DSv4 pool doesn't expose `head_num` (it has qk_nope_head_dim,
# qk_rope_head_dim, indexer_head_dim instead — the MLA-flavored fields).
#
# Fix: in-place edit of the cached pyc/source inside the (--rm) container,
# adding DeepSeekV4TokenToKVPool to the isinstance tuple. Idempotent (skips
# if already patched). Gated on MODEL_NAME so non-DSv4 disagg configs are
# unaffected. Runs on EVERY rank (each container has its own filesystem).
#
# This downstream-bug log lives at:
#   ~/chun/scripts/dsv4/dsv4_disagg_enablement/bugs/04_upstream_dsv4_disagg_kv_pool_missing_head_num.md
# Permanent fix should be a 3-line PR upstream; remove this block once the
# next sglang DSv4 daily image (post a8410de-20260502) includes it.
if [[ "$MODEL_NAME" == *DeepSeek-V4* ]]; then
    # ----- Patch 1: is_mla_backend recognizes DSv4 KV pool (Bug 4) -----
    python3 - <<'PYEOF'
import sys
try:
    import sglang.srt.disaggregation.utils as u
except Exception as e:
    print(f"[DSv4-DISAGG-PATCH] import failed: {e}", file=sys.stderr)
    sys.exit(0)

path = u.__file__
src = open(path).read()

if 'DeepSeekV4TokenToKVPool' in src:
    print(f"[DSv4-DISAGG-PATCH] {path} already patched (DeepSeekV4TokenToKVPool present), skipping")
    sys.exit(0)

old = (
    "def is_mla_backend(target_kv_pool) -> bool:\n"
    "    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool\n"
    "\n"
    "    return isinstance(target_kv_pool, MLATokenToKVPool)"
)
new = (
    "def is_mla_backend(target_kv_pool) -> bool:\n"
    "    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool\n"
    "    from sglang.srt.mem_cache.deepseekv4_memory_pool import (\n"
    "        DeepSeekV4TokenToKVPool,\n"
    "    )\n"
    "\n"
    "    return isinstance(\n"
    "        target_kv_pool, (MLATokenToKVPool, DeepSeekV4TokenToKVPool)\n"
    "    )"
)

if old not in src:
    print(f"[DSv4-DISAGG-PATCH] WARN: pattern not found in {path}; "
          "is_mla_backend may have changed upstream — leaving unchanged",
          file=sys.stderr)
    sys.exit(0)

new_src = src.replace(old, new, 1)
open(path, "w").write(new_src)
print(f"[DSv4-DISAGG-PATCH] Patched is_mla_backend in {path} (added DeepSeekV4TokenToKVPool)")
PYEOF

    # ----- Patch 2: ForwardMode.is_prefill() accepts include_draft_extend_v2 kwarg (Bug 6b) -----
    # disaggregation/deepseek_v4_backend_radix.py:687 calls
    #   forward_batch.forward_mode.is_prefill(include_draft_extend_v2=True)
    # but forward_batch_info.py:108 defines `def is_prefill(self):` (no kwargs)
    # and the body just delegates to `self.is_extend()` (no kwarg either).
    # `is_extend()` already accepts include_draft_extend_v2 (lines 111-120) so
    # we just need to plumb it through.
    python3 - <<'PYEOF'
import sys
try:
    import sglang.srt.model_executor.forward_batch_info as fbi
except Exception as e:
    print(f"[DSv4-DISAGG-PATCH-6b] import failed: {e}", file=sys.stderr)
    sys.exit(0)

path = fbi.__file__
src = open(path).read()

# Idempotency marker that's unique to OUR patch (not used elsewhere in the file)
PATCH_MARKER = "[PATCH-6b]"
if PATCH_MARKER in src:
    print(f"[DSv4-DISAGG-PATCH-6b] {path} already patched (marker found), skipping")
    sys.exit(0)

old = (
    "    def is_prefill(self):\n"
    "        return self.is_extend()\n"
)
new = (
    "    def is_prefill(self, include_draft_extend_v2: bool = False):\n"
    "        # [PATCH-6b] forward the kwarg to is_extend() so the DSv4 radix\n"
    "        # attention backend's is_prefill(include_draft_extend_v2=True)\n"
    "        # call (deepseek_v4_backend_radix.py:687) doesn't TypeError.\n"
    "        return self.is_extend(include_draft_extend_v2=include_draft_extend_v2)\n"
)

if old not in src:
    print(f"[DSv4-DISAGG-PATCH-6b] WARN: pattern not found in {path}", file=sys.stderr)
    sys.exit(0)

new_src = src.replace(old, new, 1)
open(path, "w").write(new_src)
print(f"[DSv4-DISAGG-PATCH-6b] Patched is_prefill in {path} (kwarg-tolerant + forwarded)")
PYEOF

    # ----- Patch 5: relax loc.dtype assertion in _set_k_and_s_triton (Bug 9) -----
    # nsa/index_buf_accessor.py:378 has:
    #   assert loc.dtype == torch.int64, f"{loc.dtype=}"  # can be int32
    # The author's own comment "# can be int32" tells us int32 is supposed to
    # work, but the assert is overly strict. In disagg's prefill forward,
    # loc arrives as int32 (it's an alloc indices tensor — see allocator.py).
    # Relax to accept both int32 and int64; if downstream really needs int64,
    # cast on the spot. Simplest is to broaden the allowed set.
    python3 - <<'PYEOF'
import sys, glob

candidates = glob.glob("/sgl-workspace/**/nsa/index_buf_accessor.py", recursive=True)
candidates += glob.glob("/opt/venv/lib/**/nsa/index_buf_accessor.py", recursive=True)
if not candidates:
    print("[DSv4-DISAGG-PATCH-9] WARN: index_buf_accessor.py not found", file=sys.stderr)
    sys.exit(0)
path = candidates[0]
src = open(path).read()

PATCH_MARKER = "[PATCH-9] loc.dtype int32-tolerant"
if PATCH_MARKER in src:
    print(f"[DSv4-DISAGG-PATCH-9] {path} already patched, skipping")
    sys.exit(0)

old = '    assert loc.dtype == torch.int64, f"{loc.dtype=}"  # can be int32\n'
new = (
    '    # [PATCH-9] loc.dtype int32-tolerant — original author already noted\n'
    '    # "# can be int32" but kept the strict assertion. Relaxed for disagg.\n'
    '    assert loc.dtype in (torch.int32, torch.int64), f"{loc.dtype=}"\n'
)

if old not in src:
    print(f"[DSv4-DISAGG-PATCH-9] WARN: pattern not found in {path}", file=sys.stderr)
    sys.exit(0)

new_src = src.replace(old, new, 1)
open(path, "w").write(new_src)
print(f"[DSv4-DISAGG-PATCH-9] Patched _set_k_and_s_triton in {path}")
PYEOF

    # ----- Patch 4: _create_flashmla_metadata graceful when flash_mla is absent (Bug 7) -----
    # `_create_flashmla_metadata` (deepseek_v4_backend_radix.py:130-133) does:
    #     import flash_mla
    #     return flash_mla.get_mla_metadata()[0]
    # but the rocm/sgl-dev DSv4 image doesn't ship the `flash_mla` PyPI package
    # (it's CUDA-only; HIP doesn't need it). The actual attention call routes
    # through `flash_mla_with_kvcache_entrypoint` which on HIP overrides backend
    # to "tilelang" and calls `dpsk_v4_fp8_attention_fwd` — completely bypassing
    # flash_mla. The returned metadata is consumed by that path but treated as
    # opaque (`tile_scheduler_metadata: Any`), and one existing call site already
    # sets c4/c128 metadata to None (line 1220-1221). So returning None here is
    # safe for the tilelang backend on HIP. Patch swaps in a try/except.
    python3 - <<'PYEOF'
import sys, glob

candidates = glob.glob("/sgl-workspace/**/deepseek_v4_backend_radix.py", recursive=True)
candidates += glob.glob("/opt/venv/lib/**/deepseek_v4_backend_radix.py", recursive=True)
if not candidates:
    print("[DSv4-DISAGG-PATCH-7] WARN: deepseek_v4_backend_radix.py not found", file=sys.stderr)
    sys.exit(0)
path = candidates[0]
src = open(path).read()

PATCH_MARKER = "[PATCH-7] flash_mla import-tolerant"
if PATCH_MARKER in src:
    print(f"[DSv4-DISAGG-PATCH-7] {path} already patched, skipping")
    sys.exit(0)

old = (
    "def _create_flashmla_metadata():\n"
    "    import flash_mla\n"
    "\n"
    "    return flash_mla.get_mla_metadata()[0]\n"
)
new = (
    "def _create_flashmla_metadata():\n"
    "    # [PATCH-7] flash_mla import-tolerant. The rocm/sgl-dev DSv4 image\n"
    "    # doesn't ship flash_mla (CUDA-only). On HIP, the attention compute\n"
    "    # routes to dpsk_v4_fp8_attention_fwd (tilelang) and ignores this\n"
    "    # metadata. None is already a valid value at line ~1220-1221 below.\n"
    "    try:\n"
    "        import flash_mla\n"
    "    except ImportError:\n"
    "        return None\n"
    "    return flash_mla.get_mla_metadata()[0]\n"
)

if old not in src:
    print(f"[DSv4-DISAGG-PATCH-7] WARN: pattern not found in {path}", file=sys.stderr)
    sys.exit(0)

new_src = src.replace(old, new, 1)
open(path, "w").write(new_src)
print(f"[DSv4-DISAGG-PATCH-7] Patched _create_flashmla_metadata in {path}")
PYEOF

    # ----- Patch 3: DeepSeekV4TokenToKVPool seeds full_to_swa_index_mapping (Bug 6a) -----
    # model_runner_kv_cache_mixin.py:663-671 constructs PagedTokenToKVPoolAllocator
    # for v4 models (NOT SWATokenToKVPoolAllocator), so the SWA allocator's
    # register_mapping(...) is never called and self.full_to_swa_index_mapping
    # stays unset. The radix attention backend then asserts on it during cuda
    # graph capture (deepseek_v4_backend_radix.py reaches the
    # translate_loc_from_full_to_swa assert at deepseekv4_memory_pool.py:537).
    #
    # SWATokenToKVPoolAllocator's mapping is just `torch.zeros(size + page_size,
    # dtype=int64) + tensor([-1])` (swa_memory_pool.py:290-299) — a placeholder
    # populated later by alloc_extend. We seed the same tensor at the END of
    # `_init_paged_compress_states` (the routine that was already running for
    # DPSK_V4_RADIX=True, which is what disagg requires).
    #
    # IMPORTANT: this MUST be a source-file edit (not a class wrapper) because
    # `python3 -m sglang.launch_server` spawns its scheduler subprocesses with
    # the "spawn" multiprocessing start method, which re-imports modules from
    # source — class objects modified in the parent process don't propagate.
    # Patches 1+2 (above) work the same way.
    #
    # For correctness against real SWA allocations a non-trivial per-step
    # mapping update would be needed, but for an enabling smoke test the
    # placeholder lets the radix backend's translate_loc_from_full_to_swa run
    # (returning 0 for every slot — i.e. "no SWA translation"). Track at
    # bugs/06_*.md.
    python3 - <<'PYEOF'
import sys, os, glob

# We CAN'T `import sglang.srt.mem_cache.deepseekv4_memory_pool` at this stage:
# importing the deepseekv4 KV pool transitively imports fp8_kernel.is_fp8_fnuz()
# which calls torch.cuda.get_device_properties(0) — fine inside the SLURM-launched
# container with --device=/dev/kfd, but the launch_server forks happen later.
# We're guaranteed to be running inside the docker container at server.sh time
# but not guaranteed to have CUDA inited yet. Read the file path via filesystem
# search instead.
candidates = glob.glob("/sgl-workspace/**/deepseekv4_memory_pool.py", recursive=True)
candidates += glob.glob("/opt/venv/lib/**/deepseekv4_memory_pool.py", recursive=True)
candidates = [p for p in candidates if "/srt/mem_cache/" in p]
if not candidates:
    print("[DSv4-DISAGG-PATCH-6a] WARN: deepseekv4_memory_pool.py not found anywhere", file=sys.stderr)
    sys.exit(0)
path = candidates[0]
src = open(path).read()

PATCH_MARKER = "[PATCH-6a] seed full_to_swa_index_mapping"
if PATCH_MARKER in src:
    print(f"[DSv4-DISAGG-PATCH-6a] {path} already patched (marker found), skipping")
    sys.exit(0)

# Anchor: the end of _init_paged_compress_states. We replace the last 2 lines
# of that method (the appends) plus the blank line + the next def line, then
# put back everything except adding our seed in the middle.
old = (
    "            self.compress_state_pools.append(compress_state_pool)\n"
    "            self.indexer_compress_state_pools.append(indexer_compress_state_pool)\n"
    "\n"
    "    def _init_compressed_layer_mapping(self):\n"
)
new = (
    "            self.compress_state_pools.append(compress_state_pool)\n"
    "            self.indexer_compress_state_pools.append(indexer_compress_state_pool)\n"
    "\n"
    "        # [PATCH-6a] seed full_to_swa_index_mapping with SWA allocator's default\n"
    "        # placeholder tensor. See comment in benchmarks/multi_node/amd_utils/server.sh.\n"
    "        if getattr(self, 'full_to_swa_index_mapping', None) is None:\n"
    "            import torch as _torch_patch\n"
    "            _patch_size = int(getattr(self, 'swa_size', 0) or 0)\n"
    "            _patch_page = int(getattr(self, 'swa_page_size', 0) or 0) or int(getattr(self, 'page_size', 1))\n"
    "            _patch_dev = getattr(self, 'device', 'cpu')\n"
    "            self.full_to_swa_index_mapping = _torch_patch.cat([\n"
    "                _torch_patch.zeros(_patch_size + _patch_page, dtype=_torch_patch.int64, device=_patch_dev),\n"
    "                _torch_patch.tensor([-1], dtype=_torch_patch.int64, device=_patch_dev),\n"
    "            ])\n"
    "            import logging as _logging_patch\n"
    "            _logging_patch.getLogger(__name__).warning(\n"
    "                '[DSv4-DISAGG-PATCH-6a] Seeded full_to_swa_index_mapping '\n"
    "                'on %s (numel=%d, device=%s)',\n"
    "                self.__class__.__name__,\n"
    "                self.full_to_swa_index_mapping.numel(),\n"
    "                str(_patch_dev),\n"
    "            )\n"
    "\n"
    "    def _init_compressed_layer_mapping(self):\n"
)

if old not in src:
    print(f"[DSv4-DISAGG-PATCH-6a] WARN: pattern not found in {path}; "
          "_init_paged_compress_states may have changed upstream", file=sys.stderr)
    sys.exit(0)

new_src = src.replace(old, new, 1)
open(path, "w").write(new_src)
print(f"[DSv4-DISAGG-PATCH-6a] Patched _init_paged_compress_states in {path}")
PYEOF
fi

host_ip=$(ip route get 1.1.1.1 | awk '/src/ {print $7}')
host_name=$(hostname)

# MORI_RDMA_TC configuration (optional)
# If set by runner, use it for RDMA traffic class configuration
# If not set, RDMA operations will proceed without QoS/traffic class settings
if [[ -n "${MORI_RDMA_TC}" ]]; then
    echo "[INFO] Using MORI_RDMA_TC=$MORI_RDMA_TC for RDMA traffic class configuration"
    echo "[INFO] Host '$host_name' configured with MORI_RDMA_TC=$MORI_RDMA_TC"
else
    echo "[INFO] MORI_RDMA_TC not set. Skipping RDMA traffic class configuration."
    echo "[INFO] This is normal for clusters without QoS requirements."
fi

# =============================================================================
# Model-Specific Configuration from YAML
# =============================================================================
MODELS_YAML="${SGLANG_WS_PATH}/models.yaml"

if [[ ! -f "$MODELS_YAML" ]]; then
    echo "ERROR: models.yaml not found at $MODELS_YAML"
    exit 1
fi

# Load model config via inline Python (PyYAML is available in SGLang containers)
# Formula evaluation (e.g. "SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK * TP * xP")
# is done here in Python to avoid bash glob-expanding the * characters.
eval "$(python3 -c "
import yaml, sys, os

config_path = '${MODELS_YAML}'
model_name = '${MODEL_NAME}'

with open(config_path) as f:
    models = yaml.safe_load(f)

if model_name not in models:
    print(f'echo \"ERROR: Model {model_name} not in models.yaml\"; exit 1')
    sys.exit(0)

m = models[model_name]

def eval_formula(val):
    \"\"\"Evaluate chunked_prefill_size: if string, resolve variable names from env and compute.\"\"\"
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val)
    # Build a namespace from env vars (convert numeric values to int)
    ns = {}
    for k, v in os.environ.items():
        try:
            ns[k] = int(v)
        except (ValueError, TypeError):
            pass
    try:
        return int(eval(s, {'__builtins__': {}}, ns))
    except Exception as e:
        print(f'echo \"WARNING: Cannot evaluate formula: {s} ({e})\"', file=sys.stderr)
        return val

def parse_range(cuda_range, default_start, default_end):
    if '-' in str(cuda_range):
        s, e = str(cuda_range).split('-')
        return s, e
    return str(default_start), str(default_end)

# Output shell variables
print(f'MODEL_BASE_FLAGS=\"{m.get(\"base_flags\", \"\")}\"')
print(f'MODEL_MTP_FLAGS=\"{m.get(\"mtp_flags\", \"\")}\"')
print(f'MODEL_DP_FLAGS=\"{m.get(\"dp_flags\", \"\")}\"')

prefill = m.get('prefill', {})
decode = m.get('decode', {})

print(f'PREFILL_MEM_FRACTION_STATIC=\"{prefill.get(\"mem_fraction_static\", 0.8)}\"')
print(f'PREFILL_DISABLE_RADIX_CACHE=\"{prefill.get(\"disable_radix_cache\", True)}\"')

dp = prefill.get('dp', {})
no_dp = prefill.get('no_dp', {})
print(f'PREFILL_MAX_RUNNING_REQUESTS_DP=\"{dp.get(\"max_running_requests\", 24)}\"')
print(f'PREFILL_CHUNKED_PREFILL_SIZE_DP=\"{eval_formula(dp.get(\"chunked_prefill_size\", 262144))}\"')
print(f'PREFILL_CUDA_GRAPH_BS_DP=\"{dp.get(\"cuda_graph_bs\", \"1 2 3\")}\"')
print(f'PREFILL_MAX_RUNNING_REQUESTS_NO_DP=\"{no_dp.get(\"max_running_requests\", 128)}\"')
print(f'PREFILL_CHUNKED_PREFILL_SIZE_NO_DP=\"{eval_formula(no_dp.get(\"chunked_prefill_size\", 262144))}\"')
s, e = parse_range(no_dp.get('cuda_graph_bs_range', '1-128'), 1, 128)
print(f'PREFILL_CUDA_GRAPH_BS_NO_DP_START=\"{s}\"')
print(f'PREFILL_CUDA_GRAPH_BS_NO_DP_END=\"{e}\"')

print(f'DECODE_MEM_FRACTION_STATIC=\"{decode.get(\"mem_fraction_static\", 0.85)}\"')
print(f'DECODE_PREFILL_ROUND_ROBIN_BALANCE=\"{decode.get(\"prefill_round_robin_balance\", True)}\"')

dp = decode.get('dp', {})
ep_only = decode.get('ep_only', {})
no_dp = decode.get('no_dp', {})

# Decode DP config
print(f'DECODE_MAX_RUNNING_REQUESTS_DP=\"{dp.get(\"max_running_requests\", 4096)}\"')
print(f'DECODE_CHUNKED_PREFILL_SIZE_DP=\"{eval_formula(dp.get(\"chunked_prefill_size\", 262144))}\"')
s, e = parse_range(dp.get('cuda_graph_bs_range', '1-160'), 1, 160)
print(f'DECODE_CUDA_GRAPH_BS_DP_START=\"{s}\"')
print(f'DECODE_CUDA_GRAPH_BS_DP_END=\"{e}\"')

# Decode EP-only config (EP enabled but DP disabled)
print(f'DECODE_MAX_RUNNING_REQUESTS_EP_ONLY=\"{ep_only.get(\"max_running_requests\", 256)}\"')
print(f'DECODE_CHUNKED_PREFILL_SIZE_EP_ONLY=\"{eval_formula(ep_only.get(\"chunked_prefill_size\", 262144))}\"')
s, e = parse_range(ep_only.get('cuda_graph_bs_range', '1-256'), 1, 256)
print(f'DECODE_CUDA_GRAPH_BS_EP_ONLY_START=\"{s}\"')
print(f'DECODE_CUDA_GRAPH_BS_EP_ONLY_END=\"{e}\"')

# Decode no-DP config
print(f'DECODE_MAX_RUNNING_REQUESTS_NO_DP=\"{no_dp.get(\"max_running_requests\", 128)}\"')
print(f'DECODE_CHUNKED_PREFILL_SIZE_NO_DP=\"{eval_formula(no_dp.get(\"chunked_prefill_size\", 262144))}\"')
s, e = parse_range(no_dp.get('cuda_graph_bs_range', '1-128'), 1, 128)
print(f'DECODE_CUDA_GRAPH_BS_NO_DP_START=\"{s}\"')
print(f'DECODE_CUDA_GRAPH_BS_NO_DP_END=\"{e}\"')
")"

echo "Loaded model configuration for: $MODEL_NAME"

# Compute DP-dependent prefill parameters
if [[ "$PREFILL_ENABLE_DP" == "true" ]]; then
    prefill_cuda_graph_bs=($PREFILL_CUDA_GRAPH_BS_DP)
    prefill_max_running_requests=$PREFILL_MAX_RUNNING_REQUESTS_DP
    prefill_chunked_prefill_size=$PREFILL_CHUNKED_PREFILL_SIZE_DP
else
    prefill_cuda_graph_bs=($(seq $PREFILL_CUDA_GRAPH_BS_NO_DP_START $PREFILL_CUDA_GRAPH_BS_NO_DP_END))
    prefill_max_running_requests=$PREFILL_MAX_RUNNING_REQUESTS_NO_DP
    prefill_chunked_prefill_size=$PREFILL_CHUNKED_PREFILL_SIZE_NO_DP
fi

# Compute DP-dependent decode parameters (3-way: DP > EP-only > no_dp)
if [[ "$DECODE_ENABLE_DP" == "true" ]]; then
    decode_cuda_graph_bs=($(seq $DECODE_CUDA_GRAPH_BS_DP_START $DECODE_CUDA_GRAPH_BS_DP_END))
    decode_max_running_requests=$((DECODE_CUDA_GRAPH_BS_DP_END * DECODE_TP_SIZE))
elif [[ "$DECODE_ENABLE_EP" == "true" ]]; then
    decode_cuda_graph_bs=($(seq $DECODE_CUDA_GRAPH_BS_EP_ONLY_START $DECODE_CUDA_GRAPH_BS_EP_ONLY_END))
    decode_max_running_requests=$DECODE_MAX_RUNNING_REQUESTS_EP_ONLY
else
    decode_cuda_graph_bs=($(seq $DECODE_CUDA_GRAPH_BS_NO_DP_START $DECODE_CUDA_GRAPH_BS_NO_DP_END))
    decode_max_running_requests=$DECODE_MAX_RUNNING_REQUESTS_NO_DP
fi

# Use Decode configuration to configure different TP/DP size between P and D
PREFILL_DECODE_DIFFERENT_TP=""
if [[ "$PREFILL_ENABLE_DP" != "$DECODE_ENABLE_DP" ]]; then
    if [[ "$DECODE_ENABLE_DP" == "true" ]]; then
        PREFILL_DECODE_DIFFERENT_TP="--disaggregation-decode-tp ${DECODE_TP_SIZE} --disaggregation-decode-dp ${DECODE_TP_SIZE}"
    else
        PREFILL_DECODE_DIFFERENT_TP="--disaggregation-decode-tp ${DECODE_TP_SIZE} --disaggregation-decode-dp 1"
    fi
fi

# Build the composed config strings (equivalent to the old MODEL_PREFILL_CONFIGS / MODEL_DECODE_CONFIGS)
PREFILL_MODE_FLAGS="--mem-fraction-static ${PREFILL_MEM_FRACTION_STATIC} --max-running-requests ${prefill_max_running_requests} --chunked-prefill-size ${prefill_chunked_prefill_size} --cuda-graph-bs ${prefill_cuda_graph_bs[*]} ${PREFILL_DECODE_DIFFERENT_TP}"
if [[ "$PREFILL_DISABLE_RADIX_CACHE" == "True" ]] || [[ "$PREFILL_DISABLE_RADIX_CACHE" == "true" ]]; then
    PREFILL_MODE_FLAGS="$PREFILL_MODE_FLAGS --disable-radix-cache"
fi

DECODE_MODE_FLAGS="--mem-fraction-static ${DECODE_MEM_FRACTION_STATIC} --max-running-requests ${decode_max_running_requests} --cuda-graph-bs ${decode_cuda_graph_bs[*]}"
if [[ "$DECODE_PREFILL_ROUND_ROBIN_BALANCE" == "True" ]] || [[ "$DECODE_PREFILL_ROUND_ROBIN_BALANCE" == "true" ]]; then
    DECODE_MODE_FLAGS="$DECODE_MODE_FLAGS --prefill-round-robin-balance"
fi

if [[ "$DECODE_MTP_SIZE" -gt 0 ]]; then
    MORI_MAX_DISPATCH_TOKENS_DECODE=$((MORI_MAX_DISPATCH_TOKENS_DECODE * (DECODE_MTP_SIZE + 1)))
fi

# =============================================================================
# Cluster Topology Configuration
# =============================================================================
IFS=',' read -ra IP_ARRAY <<< "$IPADDRS"

# Ceiling division by GPUS_PER_NODE for nodes-per-worker
PREFILL_NODES_PER_WORKER=$(((PREFILL_TP_SIZE + 7) / GPUS_PER_NODE))
DECODE_NODES_PER_WORKER=$(((DECODE_TP_SIZE + 7) / GPUS_PER_NODE))
NODE_OFFSET=$((PREFILL_NODES_PER_WORKER * xP))

# Build prefill arguments dynamically based on xP
PREFILL_HEADNODE_URLS=()
PREFILL_ARGS=""
for i in $(seq 0 $((xP - 1))); do
    prefill_idx=$((i * PREFILL_NODES_PER_WORKER))
    PREFILL_HEADNODE_URLS[$i]="${IP_ARRAY[$prefill_idx]}:${HEADNODE_PORT}"
    PREFILL_ARGS="$PREFILL_ARGS --prefill http://${IP_ARRAY[$prefill_idx]}:8000"
done

# Build decode arguments dynamically based on yD
DECODE_HEADNODE_URLS=()
DECODE_ARGS=""
for i in $(seq 0 $((yD - 1))); do
    decode_idx=$((i * DECODE_NODES_PER_WORKER + NODE_OFFSET))
    DECODE_HEADNODE_URLS[$i]="${IP_ARRAY[$decode_idx]}:${HEADNODE_PORT}"
    DECODE_ARGS="$DECODE_ARGS --decode http://${IP_ARRAY[$decode_idx]}:8000"
done

echo "Prefill worker headnode list: ${PREFILL_HEADNODE_URLS[@]}"
echo "Decode  worker headnode list: ${DECODE_HEADNODE_URLS[@]}"

# =============================================================================
# Configuration Builder Functions
# =============================================================================

build_server_config() {
    local mode="$1"
    local model_name="$2"
    local tp_size="$3"
    local enable_ep="$4"
    local enable_dp="$5"
    local decode_mtp_size="$6"

    # Calculate EP and DP sizes based on enable flags
    local ep_size=1
    local dp_size=1

    if [[ "$enable_ep" == "true" ]]; then
        ep_size=$tp_size
    fi

    if [[ "$enable_dp" == "true" ]]; then
        dp_size=$tp_size
    fi

    # Build parallelism arguments
    local parallel_args="--tp-size ${tp_size}"

    if [[ "$enable_ep" == "true" ]]; then
        parallel_args="$parallel_args --ep-size ${ep_size}"
    fi

    if [[ "$enable_dp" == "true" ]]; then
        parallel_args="$parallel_args --dp-size ${dp_size}"
    fi

    # Get model-specific configuration from YAML-loaded variables
    local base_config="$MODEL_BASE_FLAGS"
    local mtp_config=""
    local dp_config=""
    local specific_config=""

    # MTP config (only if MTP is enabled and mode is decode)
    if [ "$decode_mtp_size" -gt 0 ]; then
        mtp_config="${MODEL_MTP_FLAGS} --speculative-num-steps ${decode_mtp_size} --speculative-num-draft-tokens $((decode_mtp_size + 1))"
    fi

    # DP config (only if DP is enabled)
    if [[ "$enable_dp" == "true" ]]; then
        dp_config="$MODEL_DP_FLAGS"
    fi

    # Mode-specific config
    if [[ "$mode" == "prefill" ]]; then
        specific_config="$PREFILL_MODE_FLAGS"
    elif [[ "$mode" == "decode" ]]; then
        specific_config="$DECODE_MODE_FLAGS"
    fi

    # Combine: parallel args + base config + mtp config (decode only) + dp config + specific config
    local full_config="$parallel_args"
    if [[ -n "$base_config" ]]; then
        full_config="$full_config $base_config"
    fi
    if [[ -n "$mtp_config" ]] && [[ "$mode" == "decode" ]]; then
        full_config="$full_config $mtp_config"
    fi
    if [[ -n "$dp_config" ]]; then
        full_config="$full_config $dp_config"
    fi
    if [[ -n "$specific_config" ]]; then
        full_config="$full_config $specific_config"
    fi

    echo "$full_config"
}

# Build complete server configurations
PREFILL_SERVER_CONFIG=$(build_server_config "prefill" "$MODEL_NAME" "$PREFILL_TP_SIZE" "$PREFILL_ENABLE_EP" "$PREFILL_ENABLE_DP" "$DECODE_MTP_SIZE")
DECODE_SERVER_CONFIG=$(build_server_config "decode" "$MODEL_NAME" "$DECODE_TP_SIZE" "$DECODE_ENABLE_EP" "$DECODE_ENABLE_DP" "$DECODE_MTP_SIZE")

if [[ -n "$MODEL_NAME" ]]; then
    echo "Using model-specific configuration for: $MODEL_NAME"
fi

# =============================================================================
# Container Synchronization
# =============================================================================

echo "Waiting at the container creation barrier on $host_name"
python3 $SGLANG_WS_PATH/sync.py barrier \
    --local-ip ${host_ip} \
    --local-port 5000 \
    --enable-port \
    --node-ips ${IPADDRS} \
    --node-ports 5000 \
    --wait-for-all-ports \
    --timeout 300


# =============================================================================
# Node Role Assignment and Server Launch
# =============================================================================

if [ "$NODE_RANK" -eq 0 ]; then
    echo "NODE INFO ======================================="
    echo "================================================"
    echo "Node List : ${SLURM_JOB_NODELIST}"
    echo "Node IPs : ${IPADDRS}"
    echo "Model Name : ${MODEL_NAME:-'Not specified'}"
    echo "================================================"

    echo "CLUSTER INFO ===================================="
    echo "================================================"
    echo "${host_name}:${host_ip} is Proxy Node and Prefill Node"
    echo "Using prefill config: $PREFILL_SERVER_CONFIG"
    echo "Prefill parallelism: TP=${PREFILL_TP_SIZE}, EP enabled: ${PREFILL_ENABLE_EP}, DP enabled: ${PREFILL_ENABLE_DP}, MTP size=${DECODE_MTP_SIZE}"
    echo "Decode  parallelism: TP=${DECODE_TP_SIZE},  EP enabled: ${DECODE_ENABLE_EP},  DP enabled: ${DECODE_ENABLE_DP},  MTP size=${DECODE_MTP_SIZE}"
    echo "Prefill servers ($((PREFILL_TP_SIZE/GPUS_PER_NODE)) nodes): ${PREFILL_ARGS}"
    echo "Decode servers  ($((DECODE_TP_SIZE/GPUS_PER_NODE))  nodes): ${DECODE_ARGS}"
    echo "Prefill env: SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK: ${MORI_MAX_DISPATCH_TOKENS_PREFILL}"
    echo "Decode env: SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${MORI_MAX_DISPATCH_TOKENS_DECODE}"
    echo "================================================"

    # start the head prefill server
    PREFILL_CMD="SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${MORI_MAX_DISPATCH_TOKENS_PREFILL} python3 -m sglang.launch_server \
        --model-path $MODEL_DIR/$MODEL_NAME \
        --disaggregation-mode prefill \
        --disaggregation-ib-device ${IBDEVICES} \
        --host 0.0.0.0 \
        --port 8000 \
        --trust-remote-code \
        ${PREFILL_SERVER_CONFIG} \
        --log-level-http warning"

    if [ "$PREFILL_NODES_PER_WORKER" -gt 1 ]; then
        PREFILL_CMD="$PREFILL_CMD --dist-init-addr ${PREFILL_HEADNODE_URLS[0]} --nnodes ${PREFILL_NODES_PER_WORKER} --node-rank 0"
    fi


    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $PREFILL_CMD"
    else
        set -x
        eval "$PREFILL_CMD" \
            2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/prefill_${host_name}.log &
        set +x
        prefill0_pid=$!
    fi


    echo "Waiting for all prefill and decode servers to be up . . ."


    BARRIER_CMD="python3 $SGLANG_WS_PATH/sync.py barrier \
        --node-ips ${IPADDRS} \
        --node-ports 8000 \
        --wait-for-all-ports \
        --timeout 1800"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $BARRIER_CMD"
    else
        eval "$BARRIER_CMD"
    fi
    echo "Congratulations!!! All prefill and decode servers are up . . ."

    ROUTER_CMD="python -m sglang_router.launch_router \
        --pd-disaggregation \
        --port 30000 \
        --policy random \
        --prefill-policy random \
        --decode-policy random \
        ${PREFILL_ARGS} \
        ${DECODE_ARGS}"


    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $ROUTER_CMD"
    else
        ROUTER_LOG_FILE="/tmp/slurm_job-${SLURM_JOB_ID}_proxy_${host_name}.log"
        set -x
        if [[ "${SGLANG_ROUTER_STDOUT_LOGS:-0}" == "1" ]]; then
            eval "$ROUTER_CMD" 2>&1 | tee "$ROUTER_LOG_FILE" &
        else
            eval "$ROUTER_CMD" >"$ROUTER_LOG_FILE" 2>&1 &
        fi
        set +x
        proxy_pid=$!

        # Wait for router to be ready via health endpoint
        HEALTH_BARRIER_CMD="python3 $SGLANG_WS_PATH/sync.py barrier \
            --node-ips ${NODE0_ADDR} \
            --node-ports 30000 \
            --wait-for-all-health \
            --health-endpoint /readiness \
            --timeout 1800"

        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "DRY RUN: $HEALTH_BARRIER_CMD"
        else
            eval "$HEALTH_BARRIER_CMD"
        fi

        echo "Router is ready for benchmarking"
    fi


    echo "Ready for benchmarking on ${host_name}:${host_ip}"

    echo "Benchmarking on ${host_name}:${host_ip}"
    cd $SGLANG_WS_PATH

    # Export IS_MTP based on whether MTP is enabled
    if [ "$DECODE_MTP_SIZE" -gt 0 ]; then
        export IS_MTP=true
    else
        export IS_MTP=false
    fi

    # n_prefill n_decode prefill_gpus decode_gpus model_dir model_name log_path isl osl concurrency_list req_rate random_range_ratio num_prompts_multiplier
    BENCH_CMD="bash $SGLANG_WS_PATH/bench.sh ${xP} ${yD} $((PREFILL_TP_SIZE*xP)) $((DECODE_TP_SIZE*yD)) \
        $MODEL_DIR $MODEL_NAME /run_logs/slurm_job-${SLURM_JOB_ID} ${BENCH_INPUT_LEN} \
        ${BENCH_OUTPUT_LEN} "${BENCH_MAX_CONCURRENCY}" ${BENCH_REQUEST_RATE} \
        ${BENCH_RANDOM_RANGE_RATIO} ${BENCH_NUM_PROMPTS_MULTIPLIER}"

    if [[ "${EVAL_ONLY:-false}" == "true" ]]; then
        echo "EVAL_ONLY mode: skipping throughput benchmark"
    elif [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $BENCH_CMD"
    else
        set -x
        eval "$BENCH_CMD"
        set +x
    fi

    # Run evaluation if requested (before killing router)
    if [[ "${RUN_EVAL:-false}" == "true" ]]; then
        echo "Running lm-eval evaluation on Node 0..."

        # Health check: verify the router is still serving before running eval.
        # The throughput benchmark may have crashed/exhausted decode workers.
        EVAL_HEALTH_OK=false
        for _attempt in 1 2 3; do
            if curl -sf --max-time 10 "http://0.0.0.0:30000/readiness" >/dev/null 2>&1; then
                EVAL_HEALTH_OK=true
                break
            fi
            echo "Eval health check attempt $_attempt failed, retrying in 10s..."
            sleep 10
        done

        if [[ "$EVAL_HEALTH_OK" != "true" ]]; then
            echo "WARNING: Router health check failed after 3 attempts. Skipping eval."
        else
            # Must run from repo root so utils/evals/${task}.yaml resolves
            pushd /workspace

            # Source eval functions from benchmark_lib.sh
            source /workspace/benchmarks/benchmark_lib.sh

            # Use EVAL_CONC from workflow if set, otherwise fall back to max of conc list
            if [[ -n "${EVAL_CONC:-}" ]]; then
                export EVAL_CONCURRENT_REQUESTS="${EVAL_CONC}"
            else
                export EVAL_CONCURRENT_REQUESTS=$(echo "$BENCH_MAX_CONCURRENCY" | tr 'x' '\n' | sort -n | tail -1)
            fi

            if [[ "$DRY_RUN" -eq 1 ]]; then
                echo "DRY RUN: run_eval --framework lm-eval --port 30000 (conc=${EVAL_CONCURRENT_REQUESTS})"
            else
                # Run lm-eval against the router on port 30000
                run_eval --framework lm-eval --port 30000
                eval_rc=$?

                if [[ $eval_rc -ne 0 ]]; then
                    echo "ERROR: run_eval exited rc=$eval_rc; skipping metadata write and eval artifact staging" >&2
                    EVAL_FAILED=1
                else
                    # Set metadata env vars for append_lm_eval_summary
                    export TP="${PREFILL_TP_SIZE}"
                    export CONC="${EVAL_CONCURRENT_REQUESTS}"
                    export EP_SIZE=1
                    [[ "${PREFILL_ENABLE_EP}" == "true" ]] && EP_SIZE="${PREFILL_TP_SIZE}"
                    export PREFILL_TP="${PREFILL_TP_SIZE}"
                    export PREFILL_EP=1
                    [[ "${PREFILL_ENABLE_EP}" == "true" ]] && PREFILL_EP="${PREFILL_TP_SIZE}"
                    export PREFILL_NUM_WORKERS="${xP}"
                    export DECODE_TP="${DECODE_TP_SIZE}"
                    export DECODE_EP=1
                    [[ "${DECODE_ENABLE_EP}" == "true" ]] && DECODE_EP="${DECODE_TP_SIZE}"
                    export DECODE_NUM_WORKERS="${yD}"
                    export DP_ATTENTION="${PREFILL_ENABLE_DP}"
                    export PREFILL_DP_ATTENTION="${PREFILL_ENABLE_DP}"
                    export DECODE_DP_ATTENTION="${DECODE_ENABLE_DP}"
                    export ISL="${BENCH_INPUT_LEN}"
                    export OSL="${BENCH_OUTPUT_LEN}"
                    # IS_MULTINODE, FRAMEWORK, PRECISION, MODEL_PREFIX, RUNNER_TYPE,
                    # RESULT_FILENAME are already set via Docker -e flags from job.slurm

                    append_lm_eval_summary
                    # Files (meta_env.json, results*.json, sample*.jsonl) are now in /workspace

                    # Copy eval artifacts to run_logs for NFS extraction by runner
                    EVAL_COPY_DIR="/run_logs/slurm_job-${SLURM_JOB_ID}/eval_results"
                    mkdir -p "$EVAL_COPY_DIR"
                    for f in meta_env.json; do
                        [ -e "/workspace/$f" ] && cp -f "/workspace/$f" "$EVAL_COPY_DIR/"
                    done
                    # Use find for glob patterns to avoid "no match" errors
                    find /workspace -maxdepth 1 -name 'results*.json' -exec cp -f {} "$EVAL_COPY_DIR/" \;
                    find /workspace -maxdepth 1 -name 'sample*.jsonl' -exec cp -f {} "$EVAL_COPY_DIR/" \;

                    echo "Eval completed. Artifacts staged in $EVAL_COPY_DIR"
                fi
            fi

            popd
        fi
    fi

    # Copy benchmark results to BENCHMARK_LOGS_DIR (mounted from host)
    LOGS_OUTPUT="${BENCHMARK_LOGS_DIR:-/run_logs}/logs"
    mkdir -p "$LOGS_OUTPUT"

    if [[ "$DRY_RUN" -eq 0 ]]; then
        cp -r /run_logs/slurm_job-${SLURM_JOB_ID} "$LOGS_OUTPUT/"
        echo "Copied results to $LOGS_OUTPUT/slurm_job-${SLURM_JOB_ID}"
    fi

    echo "Killing the proxy server and prefill server"

    if [[ "$DRY_RUN" -eq 0 ]]; then
        kill $proxy_pid
        kill $prefill0_pid
    fi

    if [[ "${EVAL_FAILED:-0}" -eq 1 ]]; then
        echo "ERROR: eval failed; exiting node-0 with rc=1"
        exit 1
    fi

elif [ "$NODE_RANK" -gt 0 ] && [ "$NODE_RANK" -lt "$NODE_OFFSET" ]; then
    echo "${host_name}:${host_ip} is Prefill Node (Model: ${MODEL_NAME:-'default'})"
    echo "Using prefill config: $PREFILL_SERVER_CONFIG"
    echo "Prefill parallelism: TP=${PREFILL_TP_SIZE}, EP enabled: ${PREFILL_ENABLE_EP}, DP enabled: ${PREFILL_ENABLE_DP}"

    PREFILL_CMD="SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${MORI_MAX_DISPATCH_TOKENS_PREFILL} python3 -m sglang.launch_server \
        --model-path $MODEL_DIR/${MODEL_NAME} \
        --disaggregation-mode prefill \
        --disaggregation-ib-device ${IBDEVICES} \
        --host 0.0.0.0 \
        --port 8000 \
        --trust-remote-code \
        ${PREFILL_SERVER_CONFIG} \
        --log-level-http warning"

    if [ "$PREFILL_NODES_PER_WORKER" -gt 1 ]; then
        rank=$((NODE_RANK % PREFILL_NODES_PER_WORKER))
        prefill_idx=$((NODE_RANK / PREFILL_NODES_PER_WORKER))
        PREFILL_CMD="$PREFILL_CMD --dist-init-addr ${PREFILL_HEADNODE_URLS[$prefill_idx]} --nnodes ${PREFILL_NODES_PER_WORKER} --node-rank $rank"
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $PREFILL_CMD"
    else
        set -x
        eval "$PREFILL_CMD" \
            2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/prefill_${host_name}.log &
        set +x
        prefill_pid=$!
    fi

    echo "Waiting for proxy server to be up..."
    BARRIER_CMD="python3 $SGLANG_WS_PATH/sync.py barrier \
        --node-ips ${NODE0_ADDR} \
        --node-ports 30000 \
        --wait-for-all-ports \
        --timeout 1800"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $BARRIER_CMD"
    else
        eval "$BARRIER_CMD"
    fi

    echo "Waiting until proxy server closes..."
    WAIT_CMD="python3 $SGLANG_WS_PATH/sync.py wait \
        --remote-ip ${NODE0_ADDR} \
        --remote-port 30000"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $WAIT_CMD"
    else
        eval "$WAIT_CMD"
    fi

    echo "Killing the rank $NODE_RANK prefill server"

    if [[ "$DRY_RUN" -eq 0 ]]; then
        kill $prefill_pid
    fi

else
    RANK=$((NODE_RANK - xP * PREFILL_NODES_PER_WORKER))
    echo "${host_name}:${host_ip} is Decode Node (Model: ${MODEL_NAME:-'default'})"
    echo "Using decode config: $DECODE_SERVER_CONFIG"
    echo "Decode node rank: $RANK"
    echo "Decode parallelism: TP=${DECODE_TP_SIZE}, EP enabled: ${DECODE_ENABLE_EP}, DP enabled: ${DECODE_ENABLE_DP}"

    DECODE_CMD="SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${MORI_MAX_DISPATCH_TOKENS_DECODE} python3 -m sglang.launch_server \
        --model-path ${MODEL_DIR}/${MODEL_NAME} \
        --disaggregation-mode decode \
        --disaggregation-ib-device ${IBDEVICES} \
        --host 0.0.0.0 \
        --port 8000 \
        --trust-remote-code \
        ${DECODE_SERVER_CONFIG} \
        --log-level-http warning"

    if [ "$DECODE_NODES_PER_WORKER" -gt 1 ]; then
        rank=$((RANK % DECODE_NODES_PER_WORKER))
        decode_idx=$((RANK / DECODE_NODES_PER_WORKER))
        DECODE_CMD="$DECODE_CMD --dist-init-addr ${DECODE_HEADNODE_URLS[$decode_idx]} --nnodes ${DECODE_NODES_PER_WORKER} --node-rank $rank"
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $DECODE_CMD"
    else
        set -x
        eval "$DECODE_CMD" \
            2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/decode_${host_name}.log &

        set +x
        decode_pid=$!
    fi


    echo "Waiting for proxy server to be up..."
    BARRIER_CMD="python3 $SGLANG_WS_PATH/sync.py barrier \
        --node-ips ${NODE0_ADDR} \
        --node-ports 30000 \
        --wait-for-all-ports \
        --timeout 1800"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $BARRIER_CMD"
    else
        eval "$BARRIER_CMD"
    fi


    echo "Waiting until proxy server closes..."
    WAIT_CMD="python3 $SGLANG_WS_PATH/sync.py wait \
        --remote-ip ${NODE0_ADDR} \
        --remote-port 30000"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $WAIT_CMD"
    else
        eval "$WAIT_CMD"
    fi

    echo "Killing the rank $RANK decode server"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        kill $decode_pid
    fi

fi

echo "Script completed successfully"
exit 0
