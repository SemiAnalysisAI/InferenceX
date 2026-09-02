#!/bin/bash
# vLLM Disaggregated Server Launcher with Model-Specific Configurations
# =============================================================================
#
# Node role assignment (by NODE_RANK):
#   0           -> Proxy/Router + first Prefill node  (kv_producer)
#   1..xP-1     -> Additional Prefill nodes            (kv_producer)
#   xP..xP+yD-1 -> Decode nodes                        (kv_consumer)
#
# Total nodes = xP + yD (router co-located with first prefill, like SGLang).

# =============================================================================
# Dependency Setup (idempotent; required when using base vLLM image)
# =============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/setup_deps.sh"

# =============================================================================
# Environment Configuration
# =============================================================================

NODE0_ADDR="${NODE0_ADDR:-localhost}"
NODE_RANK="${NODE_RANK:-0}"
MODEL_DIR="${MODEL_DIR:-}"
MODEL_NAME="${MODEL_NAME:-}"

xP="${xP:-1}"
yD="${yD:-1}"

IPADDRS="${IPADDRS:-localhost}"

# Benchmark Configuration
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-1024}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-1024}"
BENCH_RANDOM_RANGE_RATIO="${BENCH_RANDOM_RANGE_RATIO:-1}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-inf}"
BENCH_NUM_PROMPTS_MULTIPLIER="${BENCH_NUM_PROMPTS_MULTIPLIER:-10}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-512}"

DRY_RUN="${DRY_RUN:-0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-$GPUS_PER_NODE}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-$GPUS_PER_NODE}"

ROUTER_PORT="${ROUTER_PORT:-30000}"
SERVER_PORT="${SERVER_PORT:-2584}"
ENGINE_ID="${ENGINE_ID:-${MODEL_NAME}-pd-run}"

# Prefer MODEL_PATH from job.slurm (handles HF cache snapshot resolution)
MODEL_PATH="${MODEL_PATH:-${MODEL_DIR}/${MODEL_NAME}}"

# =============================================================================
# Dependencies and Environment Setup
# =============================================================================
source $WS_PATH/env.sh

host_ip=$(ip route get 1.1.1.1 2>/dev/null | awk '/src/ {print $7}')
# RDMA IP for Nixl KV transfer (prefer 192.168.x.x subnet if available)
rdma_ip=$(hostname -I | tr ' ' '\n' | grep '^192\.168\.' | head -1)
rdma_ip="${rdma_ip:-$host_ip}"
host_name=$(hostname)

echo "[INFO] Management IP (barriers/proxy): $host_ip"
echo "[INFO] RDMA IP (Nixl KV transfer): $rdma_ip"

# =============================================================================
# RDMA / Nixl Workarounds
# =============================================================================

setup_rdma_env() {
    # Pensando ionic (RoCEv2) point-to-point /31 route fix.
    # Each benic interface has a /31 to the TOR switch. Without explicit routes,
    # traffic to other nodes' RDMA IPs falls through to the management network.
    if [[ "$rdma_ip" =~ ^192\.168\.([0-9]+)\.([0-9]+)$ ]]; then
        local rdma_subnet="${BASH_REMATCH[1]}"
        local rdma_host="${BASH_REMATCH[2]}"
        local rdma_gw="192.168.${rdma_subnet}.$(( rdma_host | 1 ))"
        local rdma_iface
        rdma_iface=$(ip -o addr show | awk -v ip="$rdma_ip" '$4 ~ ip {print $2}' | head -1)
        if [[ -n "$rdma_iface" ]]; then
            ip route replace "192.168.${rdma_subnet}.0/24" via "$rdma_gw" dev "$rdma_iface" 2>/dev/null && \
                echo "[RDMA-ROUTE] Added 192.168.${rdma_subnet}.0/24 via $rdma_gw dev $rdma_iface" || \
                echo "[RDMA-ROUTE] Route add failed for 192.168.${rdma_subnet}.0/24"
        fi
    fi

    # Patch Nixl UCX backend: set ucx_error_handling_mode=none.
    # Required for ALL NIC types under high concurrency (C512+). Without this,
    # UCX's default UCP_ERR_HANDLING_MODE_PEER triggers transport-level error
    # recovery on ibv_post_send failures, preventing RIXL RDMA READ retries from
    # recovering gracefully. This causes the prefill KV cache to fill to 100%
    # and deadlock the pipeline. On ionic NICs this was already applied (rdmacm
    # incompatibility); on mlx5 NICs it was incorrectly skipped.
    local nixl_api
    nixl_api=$(python3 -c "import rixl._api; print(rixl._api.__file__)" 2>/dev/null)
    if [[ -n "$nixl_api" ]]; then
        if ! grep -q 'ucx_error_handling_mode' "$nixl_api"; then
            sed -i '/self\.create_backend(bknd, init)/i\                init["ucx_error_handling_mode"] = "none"' "$nixl_api"
            echo "[PATCH] Added ucx_error_handling_mode=none to $nixl_api (IBDEVICES=${IBDEVICES:-unset})"
        else
            echo "[PATCH] ucx_error_handling_mode already set in $nixl_api"
        fi
    fi
}

setup_rdma_env

if [[ -z "$UCX_NET_DEVICES" ]]; then
    echo "Error: UCX_NET_DEVICES is empty after env.sh detection" >&2
    exit 1
fi

# =============================================================================
# Model-Specific Configuration from YAML
# =============================================================================
MODELS_YAML="${WS_PATH}/models_vllm.yaml"

if [[ ! -f "$MODELS_YAML" ]]; then
    echo "ERROR: models.yaml not found at $MODELS_YAML"
    exit 1
fi

if [[ -z "$MODEL_NAME" ]]; then
    echo "ERROR: MODEL_NAME is not set"; exit 1
fi

eval "$(python3 -c "
import yaml, sys

with open('${MODELS_YAML}') as f:
    models = yaml.safe_load(f)

model_name = '${MODEL_NAME}'
if model_name not in models:
    print(f'echo \"ERROR: Model {model_name} not in models.yaml\"; exit 1')
    sys.exit(0)

m = models[model_name]

def bash_escape(s):
    \"\"\"Escape a value for safe embedding in a bash double-quoted assignment.\"\"\"
    return s.replace('\\\\', '\\\\\\\\').replace('\"', '\\\\\"').replace('\$', '\\\\\$').replace('\`', '\\\\\`')

pf = bash_escape(m.get('prefill_flags', '--tensor-parallel-size 8'))
df = bash_escape(m.get('decode_flags', '--tensor-parallel-size 8'))
ev = bash_escape(m.get('env', ''))
dev = bash_escape(m.get('decode_env', ''))
pev = bash_escape(m.get('prefill_env', ''))
print(f'PREFILL_SERVER_CONFIG=\"{pf}\"')
print(f'DECODE_SERVER_CONFIG=\"{df}\"')
print(f'MODEL_ENVS=\"{ev}\"')
print(f'DECODE_MODEL_ENVS=\"{dev}\"')
print(f'PREFILL_MODEL_ENVS=\"{pev}\"')
")"

echo "Loaded model configuration for: $MODEL_NAME"

# Apply tensor-parallel size and EP/DP flags from submit pipeline.
if [[ -n "${PREFILL_TP_SIZE:-}" ]]; then
    if echo "$PREFILL_SERVER_CONFIG" | grep -q -- '--tensor-parallel-size'; then
        PREFILL_SERVER_CONFIG=$(echo "$PREFILL_SERVER_CONFIG" | sed -E "s/--tensor-parallel-size[[:space:]]+[0-9]+/--tensor-parallel-size ${PREFILL_TP_SIZE}/g")
    else
        PREFILL_SERVER_CONFIG+=" --tensor-parallel-size ${PREFILL_TP_SIZE}"
    fi
fi
if [[ -n "${DECODE_TP_SIZE:-}" ]]; then
    if echo "$DECODE_SERVER_CONFIG" | grep -q -- '--tensor-parallel-size'; then
        DECODE_SERVER_CONFIG=$(echo "$DECODE_SERVER_CONFIG" | sed -E "s/--tensor-parallel-size[[:space:]]+[0-9]+/--tensor-parallel-size ${DECODE_TP_SIZE}/g")
    else
        DECODE_SERVER_CONFIG+=" --tensor-parallel-size ${DECODE_TP_SIZE}"
    fi
fi
if [[ "${PREFILL_ENABLE_EP:-false}" == "true" ]] && ! echo "$PREFILL_SERVER_CONFIG" | grep -q -- '--enable-expert-parallel'; then
    PREFILL_SERVER_CONFIG+=" --enable-expert-parallel"
fi
if [[ "${DECODE_ENABLE_EP:-false}" == "true" ]] && ! echo "$DECODE_SERVER_CONFIG" | grep -q -- '--enable-expert-parallel'; then
    DECODE_SERVER_CONFIG+=" --enable-expert-parallel"
fi

# DEP8 on ROCm vLLM (mori-0625): TP1 + data-parallel-size + EP, not --enable-dp-attention
# (same as benchmarks/single_node/fixed_seq_len/minimaxm3_fp4_mi355x_vllm.sh).
apply_vllm_dp_config() {
    local cfg="$1"
    local tp_size="$2"
    local enable_dp="${3:-false}"

    cfg=$(echo "$cfg" | sed -E 's/[[:space:]]*--enable-dp-attention//g')
    cfg=$(echo "$cfg" | sed -E 's/[[:space:]]*--data-parallel-size[[:space:]]+[0-9]+//g')

    if [[ "$enable_dp" != "true" ]]; then
        echo "$cfg"
        return
    fi

    if echo "$cfg" | grep -q -- '--tensor-parallel-size'; then
        echo "$cfg" | sed -E "s/--tensor-parallel-size[[:space:]]+[0-9]+/--tensor-parallel-size 1 --data-parallel-size ${tp_size}/"
    else
        echo "$cfg --tensor-parallel-size 1 --data-parallel-size ${tp_size}"
    fi
}

PREFILL_SERVER_CONFIG="$(apply_vllm_dp_config "$PREFILL_SERVER_CONFIG" "${PREFILL_TP_SIZE:-8}" "${PREFILL_ENABLE_DP:-false}")"
DECODE_SERVER_CONFIG="$(apply_vllm_dp_config "$DECODE_SERVER_CONFIG" "${DECODE_TP_SIZE:-8}" "${DECODE_ENABLE_DP:-false}")"

apply_vllm_dcp_config() {
    local cfg="$1"
    local dcp_size="${2:-1}"
    local dcp_comm="${3:-a2a}"
    local interleave="${4:-1}"

    cfg=$(echo "$cfg" | sed -E 's/[[:space:]]*--decode-context-parallel-size[[:space:]]+[0-9]+//g')
    cfg=$(echo "$cfg" | sed -E 's/[[:space:]]*--dcp-comm-backend[[:space:]]+[^[:space:]]+//g')
    cfg=$(echo "$cfg" | sed -E 's/[[:space:]]*--cp-kv-cache-interleave-size[[:space:]]+[0-9]+//g')

    if [[ "$dcp_size" != "1" ]]; then
        cfg+=" --decode-context-parallel-size ${dcp_size}"
        cfg+=" --dcp-comm-backend ${dcp_comm}"
        cfg+=" --cp-kv-cache-interleave-size ${interleave}"
    fi
    echo "$cfg"
}

PREFILL_SERVER_CONFIG="$(apply_vllm_dcp_config "$PREFILL_SERVER_CONFIG" "${PREFILL_DCP_SIZE:-1}" "${PREFILL_DCP_COMM:-a2a}" "${PREFILL_CP_KV_CACHE_INTERLEAVE_SIZE:-1}")"
DECODE_SERVER_CONFIG="$(apply_vllm_dcp_config "$DECODE_SERVER_CONFIG" "${DECODE_DCP_SIZE:-1}" "${DECODE_DCP_COMM:-a2a}" "${DECODE_CP_KV_CACHE_INTERLEAVE_SIZE:-1}")"

apply_gpu_memory_utilization() {
    local cfg="$1"
    local gmu="${GPU_MEMORY_UTILIZATION:-}"
    if [[ -z "$gmu" ]]; then
        echo "$cfg"
        return
    fi
    if echo "$cfg" | grep -q -- '--gpu-memory-utilization'; then
        echo "$cfg" | sed -E "s/--gpu-memory-utilization[[:space:]]+[0-9.]+/--gpu-memory-utilization ${gmu}/g"
    else
        echo "$cfg --gpu-memory-utilization ${gmu}"
    fi
}

if [[ -n "${GPU_MEMORY_UTILIZATION:-}" ]]; then
    PREFILL_SERVER_CONFIG="$(apply_gpu_memory_utilization "$PREFILL_SERVER_CONFIG")"
    DECODE_SERVER_CONFIG="$(apply_gpu_memory_utilization "$DECODE_SERVER_CONFIG")"
    echo "Applied GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
fi

echo "PREFILL_SERVER_CONFIG (after TP/EP/DP): $PREFILL_SERVER_CONFIG"
echo "DECODE_SERVER_CONFIG (after TP/EP/DP): $DECODE_SERVER_CONFIG"

# MAX_MODEL_LEN_OVERRIDE: bring-up-only escape hatch, deliberately separate from
# MAX_MODEL_LEN. The agentic entrypoints unset MAX_MODEL_LEN before applying the
# model's native window, so a recipe cannot quietly shrink the context and
# flatter its numbers -- that guard must stay. But a plumbing smoke sometimes has
# to fit a smaller window to reach the code under test at all (ROCM_AITER_MLA
# wants 54.56 GiB for a 1M request against a 52.59 GiB pool, so engine init dies
# before graph capture). This knob is loud, separately named, and never set by
# any recipe or by CI, so it cannot be mistaken for a scoring configuration.
if [[ -n "${MAX_MODEL_LEN_OVERRIDE:-}" ]]; then
    echo "WARNING: MAX_MODEL_LEN_OVERRIDE=${MAX_MODEL_LEN_OVERRIDE} replaces MAX_MODEL_LEN=${MAX_MODEL_LEN:-<unset>}."
    echo "WARNING: bring-up only -- results from this run are NOT comparable to a native-context run."
    MAX_MODEL_LEN="${MAX_MODEL_LEN_OVERRIDE}"
fi

apply_max_model_len() {
    local cfg="$1"
    if [[ -n "${MAX_MODEL_LEN:-}" && "${MAX_MODEL_LEN}" != "0" ]]; then
        if echo "$cfg" | grep -q -- '--max-model-len'; then
            echo "$cfg" | sed -E "s/--max-model-len[[:space:]]+[0-9]+/--max-model-len ${MAX_MODEL_LEN}/g"
        else
            echo "$cfg --max-model-len ${MAX_MODEL_LEN}"
        fi
    else
        echo "$cfg"
    fi
}

enable_prefix_caching=false
if [[ "${IS_AGENTIC:-0}" == "1" || "${IS_AGENTIC:-}" == "true" ]]; then
    enable_prefix_caching=true
fi
if [[ "${ENABLE_PREFIX_CACHING:-0}" == "1" || "${ENABLE_PREFIX_CACHING:-}" == "true" ]]; then
    enable_prefix_caching=true
fi
if [[ "$enable_prefix_caching" == "true" ]]; then
    PREFILL_SERVER_CONFIG="${PREFILL_SERVER_CONFIG//--no-enable-prefix-caching/--enable-prefix-caching}"
    DECODE_SERVER_CONFIG="${DECODE_SERVER_CONFIG//--no-enable-prefix-caching/--enable-prefix-caching}"
fi
if [[ -n "${MAX_MODEL_LEN:-}" && "${MAX_MODEL_LEN}" != "0" ]]; then
    PREFILL_SERVER_CONFIG="$(apply_max_model_len "$PREFILL_SERVER_CONFIG")"
    DECODE_SERVER_CONFIG="$(apply_max_model_len "$DECODE_SERVER_CONFIG")"
    echo "Applied MAX_MODEL_LEN=${MAX_MODEL_LEN}"
fi
if [[ "$enable_prefix_caching" == "true" || -n "${MAX_MODEL_LEN:-}" ]]; then
    echo "PREFILL_SERVER_CONFIG (overrides): $PREFILL_SERVER_CONFIG"
    echo "DECODE_SERVER_CONFIG (overrides): $DECODE_SERVER_CONFIG"
fi

# Debug: LOAD_FORMAT override (e.g. dummy) — model-less launcher/plumbing smoke
# on clusters without the checkpoint staged. Rewrites --load-format <x> or appends.
apply_load_format() {
    local cfg="$1"
    if echo "$cfg" | grep -q -- '--load-format'; then
        echo "$cfg" | sed -E "s/--load-format[[:space:]]+[A-Za-z0-9_.-]+/--load-format ${LOAD_FORMAT}/g"
    else
        echo "$cfg --load-format ${LOAD_FORMAT}"
    fi
}
if [[ -n "${LOAD_FORMAT:-}" && "${LOAD_FORMAT}" != "auto" ]]; then
    PREFILL_SERVER_CONFIG="$(apply_load_format "$PREFILL_SERVER_CONFIG")"
    DECODE_SERVER_CONFIG="$(apply_load_format "$DECODE_SERVER_CONFIG")"
    echo "Applied LOAD_FORMAT=${LOAD_FORMAT}"
fi

# Debug: DISABLE_SPECULATIVE strips the single-quoted --speculative-config block.
# Optional escape hatch only; dummy-weight smokes keep DSpark on (symmetric P+D)
# and only skip the main checkpoint via LOAD_FORMAT=dummy.
if [[ "${DISABLE_SPECULATIVE:-0}" == "1" || "${DISABLE_SPECULATIVE:-}" == "true" ]]; then
    PREFILL_SERVER_CONFIG="$(echo "$PREFILL_SERVER_CONFIG" | sed -E "s/[[:space:]]*--speculative-config[[:space:]]+'[^']*'//g")"
    DECODE_SERVER_CONFIG="$(echo "$DECODE_SERVER_CONFIG" | sed -E "s/[[:space:]]*--speculative-config[[:space:]]+'[^']*'//g")"
    echo "Applied DISABLE_SPECULATIVE=1 (stripped --speculative-config)"
fi

# KV_CACHE_DTYPE: recipe-level --kv-cache-dtype (e.g. fp8). Halves MLA KV
# bytes/token, which is what buys context length on a checkpoint whose weights
# already take ~195 GB/GPU at TP8. AiterMLABackend.supported_kv_cache_dtypes
# accepts auto/float16/bfloat16/fp8/fp8_e4m3/fp8_e5m2, so ROCM_AITER_MLA honors
# this; the separate FP8 *ASM prefill* fast path additionally needs
# num_heads % 16 == 0 per rank and stays off for K3 at TP8 (96/8 = 12 heads).
apply_kv_cache_dtype() {
    local cfg="$1"
    if echo "$cfg" | grep -q -- '--kv-cache-dtype'; then
        echo "$cfg" | sed -E "s/--kv-cache-dtype[[:space:]]+[A-Za-z0-9_]+/--kv-cache-dtype ${KV_CACHE_DTYPE}/g"
    else
        echo "$cfg --kv-cache-dtype ${KV_CACHE_DTYPE}"
    fi
}
if [[ -n "${KV_CACHE_DTYPE:-}" && "${KV_CACHE_DTYPE}" != "auto" ]]; then
    PREFILL_SERVER_CONFIG="$(apply_kv_cache_dtype "$PREFILL_SERVER_CONFIG")"
    DECODE_SERVER_CONFIG="$(apply_kv_cache_dtype "$DECODE_SERVER_CONFIG")"
    echo "Applied KV_CACHE_DTYPE=${KV_CACHE_DTYPE}"
fi

# ATTENTION_BACKEND: override the target model's --attention-backend.
# Default (unset) keeps models_vllm.yaml's ROCM_AITER_MLA, which is correct for
# K3 at TP8 -- see the KV-dtype note below before changing it.
#
# K3 TP8 gives 96/8 = 12 MLA heads/rank, i.e. nhead <= 16, so aiter serves decode
# from mla_gluon. That kernel has three regimes and picks by KV dtype:
#   bh16bn64  : bf16 Q + bf16 KV, nhead <= 16, batch_size >= 1   <-- what we use
#   bh16bn128 : bf16 Q + fp8  KV, nhead <= 16, batch_size == 1
#   bh64      : nhead in {64,128}, batch_size in {64,128,256}
# So batched decode on 12 heads is fine on bf16 KV; it is *fp8 KV* that pins the
# batch to 1 and aborts with
#   AssertionError: mla_gluon[bh16bn128] requires batch_size=1, got <N>
# This is why the validated real-weight run (GSM8K 44/50) served fine on
# ROCM_AITER_MLA: it never set --kv-cache-dtype, so it landed on bh16bn64.
apply_attention_backend() {
    local cfg="$1"
    if echo "$cfg" | grep -q -- '--attention-backend'; then
        echo "$cfg" | sed -E "s/--attention-backend[[:space:]]+[A-Za-z0-9_]+/--attention-backend ${ATTENTION_BACKEND}/g"
    else
        echo "$cfg --attention-backend ${ATTENTION_BACKEND}"
    fi
}
if [[ -n "${ATTENTION_BACKEND:-}" ]]; then
    PREFILL_SERVER_CONFIG="$(apply_attention_backend "$PREFILL_SERVER_CONFIG")"
    DECODE_SERVER_CONFIG="$(apply_attention_backend "$DECODE_SERVER_CONFIG")"
    echo "Applied ATTENTION_BACKEND=${ATTENTION_BACKEND}"
fi

# SPEC_ATTN_BACKEND: override the DRAFT model's attention backend, i.e. the
# "attention_backend" key inside --speculative-config's JSON. Independent axis
# from ATTENTION_BACKEND above, which only moves the target model.
#
# The draft runs its own MLA over the same KV pages, one token per step
# (qo_len == 1), so it never hits the qo_len > 4 persistent-mode gate that the
# target's MTP verify step does -- which makes it safe to A/B on its own.
# models_vllm.yaml pins TRITON_MLA there (PR #2403); ROCM_AITER_MLA is the arm
# worth measuring, since 7 of every 8 forward passes in a DSpark n=7 step are
# draft passes.
# "attention_backend" is unique to the speculative-config JSON (the target uses
# the --attention-backend CLI flag), so a global substitution is unambiguous.
apply_spec_attn_backend() {
    local cfg="$1"
    if ! echo "$cfg" | grep -q -- '--speculative-config'; then
        echo "$cfg"
    elif echo "$cfg" | grep -q '"attention_backend"'; then
        echo "$cfg" | sed -E "s/(\"attention_backend\"[[:space:]]*:[[:space:]]*)\"[A-Za-z0-9_]+\"/\1\"${SPEC_ATTN_BACKEND}\"/g"
    else
        echo "$cfg" | sed -E "s/(--speculative-config[[:space:]]+'\\{)/\\1\"attention_backend\":\"${SPEC_ATTN_BACKEND}\",/"
    fi
}
if [[ -n "${SPEC_ATTN_BACKEND:-}" ]]; then
    PREFILL_SERVER_CONFIG="$(apply_spec_attn_backend "$PREFILL_SERVER_CONFIG")"
    DECODE_SERVER_CONFIG="$(apply_spec_attn_backend "$DECODE_SERVER_CONFIG")"
    echo "Applied SPEC_ATTN_BACKEND=${SPEC_ATTN_BACKEND} (draft/speculative-config)"
fi

# SPEC_DRAFT_SAMPLE_METHOD / SPEC_REJECTION_SAMPLE_METHOD: override the two DSpark
# sampling keys inside --speculative-config. models_vllm.yaml pins the non-default
# pair "probabilistic" + "block" (PR #2403); vLLM's defaults are "greedy" +
# "standard".
#
# Worth being able to move, because the block rejection sampler is where the run
# dies. The five Triton kernels that JIT-compile immediately before the GPU queue
# aborts with HSA_STATUS_ERROR_EXCEPTION 0x1016 all live in
# v1/worker/gpu/spec_decode/rejection_sampler_utils.py:
#   _compute_local_logits_stats_kernel, _compute_cumulative_log_p_kernel,
#   _compute_local_residual_mass_kernel, _rejection_kernel, _resample_kernel
# and the fault only appears once the Mooncake tier starts serving hits, i.e. once
# prefill arrives with almost every token already cached -- a shape these kernels
# were never warmed up for.
apply_spec_sample_method() {
    local cfg="$1" key="$2" val="$3"
    if ! echo "$cfg" | grep -q -- '--speculative-config'; then
        echo "$cfg"
    elif echo "$cfg" | grep -q "\"${key}\""; then
        echo "$cfg" | sed -E "s/(\"${key}\"[[:space:]]*:[[:space:]]*)\"[A-Za-z0-9_]+\"/\1\"${val}\"/g"
    else
        echo "$cfg" | sed -E "s/(--speculative-config[[:space:]]+'\\{)/\\1\"${key}\":\"${val}\",/"
    fi
}
if [[ -n "${SPEC_DRAFT_SAMPLE_METHOD:-}" ]]; then
    PREFILL_SERVER_CONFIG="$(apply_spec_sample_method "$PREFILL_SERVER_CONFIG" draft_sample_method "$SPEC_DRAFT_SAMPLE_METHOD")"
    DECODE_SERVER_CONFIG="$(apply_spec_sample_method "$DECODE_SERVER_CONFIG" draft_sample_method "$SPEC_DRAFT_SAMPLE_METHOD")"
    echo "Applied SPEC_DRAFT_SAMPLE_METHOD=${SPEC_DRAFT_SAMPLE_METHOD}"
fi
if [[ -n "${SPEC_REJECTION_SAMPLE_METHOD:-}" ]]; then
    PREFILL_SERVER_CONFIG="$(apply_spec_sample_method "$PREFILL_SERVER_CONFIG" rejection_sample_method "$SPEC_REJECTION_SAMPLE_METHOD")"
    DECODE_SERVER_CONFIG="$(apply_spec_sample_method "$DECODE_SERVER_CONFIG" rejection_sample_method "$SPEC_REJECTION_SAMPLE_METHOD")"
    echo "Applied SPEC_REJECTION_SAMPLE_METHOD=${SPEC_REJECTION_SAMPLE_METHOD}"
fi

# SPEC_NUM_TOKENS: override "num_speculative_tokens" (DSpark's n). The recipe pins 7 and
# every fault so far was measured at 7, so n has never been varied -- yet it sets the
# verify-step qo_len, the draft-loop trip count and the sampler's per-request logit count
# all at once, which makes it the cheapest axis for bounding the fault.
apply_spec_num_tokens() {
    local cfg="$1"
    if ! echo "$cfg" | grep -q -- '--speculative-config'; then
        echo "$cfg"
    elif echo "$cfg" | grep -q '"num_speculative_tokens"'; then
        echo "$cfg" | sed -E "s/(\"num_speculative_tokens\"[[:space:]]*:[[:space:]]*)[0-9]+/\1${SPEC_NUM_TOKENS}/g"
    else
        echo "$cfg" | sed -E "s/(--speculative-config[[:space:]]+'\\{)/\\1\"num_speculative_tokens\":${SPEC_NUM_TOKENS},/"
    fi
}
if [[ -n "${SPEC_NUM_TOKENS:-}" ]]; then
    PREFILL_SERVER_CONFIG="$(apply_spec_num_tokens "$PREFILL_SERVER_CONFIG")"
    DECODE_SERVER_CONFIG="$(apply_spec_num_tokens "$DECODE_SERVER_CONFIG")"
    echo "Applied SPEC_NUM_TOKENS=${SPEC_NUM_TOKENS} (DSpark n on P and D)"
fi

# SPEC_MODEL: override the draft checkpoint. The recipe names the hub id
# "Inferact/Kimi-K3-DSpark", which vLLM resolves over the network; a toy draft has to be
# pointed at a path inside the container instead. The value may contain '/', so substitute
# with a delimiter that cannot appear in a path.
apply_spec_model() {
    local cfg="$1"
    if ! echo "$cfg" | grep -q -- '--speculative-config'; then
        echo "$cfg"
    else
        echo "$cfg" | sed -E "s|(\"model\"[[:space:]]*:[[:space:]]*)\"[^\"]*\"|\1\"${SPEC_MODEL}\"|g"
    fi
}
if [[ -n "${SPEC_MODEL:-}" ]]; then
    PREFILL_SERVER_CONFIG="$(apply_spec_model "$PREFILL_SERVER_CONFIG")"
    DECODE_SERVER_CONFIG="$(apply_spec_model "$DECODE_SERVER_CONFIG")"
    echo "Applied SPEC_MODEL=${SPEC_MODEL} (draft checkpoint on P and D)"
fi

# MAX_NUM_SEQS: override --max-num-seqs. models_vllm.yaml pins 16 and warns not to raise it
# without re-checking the aiter MLA decode path; lowering it is the safe direction and it
# bounds how ragged a decode batch can get.
apply_max_num_seqs() {
    local cfg="$1"
    if echo "$cfg" | grep -q -- '--max-num-seqs'; then
        echo "$cfg" | sed -E "s/(--max-num-seqs[[:space:]]+)[0-9]+/\1${MAX_NUM_SEQS}/g"
    else
        echo "$cfg --max-num-seqs ${MAX_NUM_SEQS}"
    fi
}
if [[ -n "${MAX_NUM_SEQS:-}" ]]; then
    PREFILL_SERVER_CONFIG="$(apply_max_num_seqs "$PREFILL_SERVER_CONFIG")"
    DECODE_SERVER_CONFIG="$(apply_max_num_seqs "$DECODE_SERVER_CONFIG")"
    echo "Applied MAX_NUM_SEQS=${MAX_NUM_SEQS} (P and D)"
fi

# ENFORCE_EAGER: disable CUDA graphs. Escape hatch, not a default -- AiterMLA
# declares AttentionCGSupport.UNIFORM_BATCH and the K3 fork adds
# _uniform_padded_mtp_qo_len specifically so full-CG padded MTP decode works, so
# graphs are the intended mode. Idempotent: never appended twice.
if [[ "${ENFORCE_EAGER:-0}" == "1" || "${ENFORCE_EAGER:-}" == "true" ]]; then
    echo "$PREFILL_SERVER_CONFIG" | grep -q -- '--enforce-eager' \
        || PREFILL_SERVER_CONFIG="$PREFILL_SERVER_CONFIG --enforce-eager"
    echo "$DECODE_SERVER_CONFIG" | grep -q -- '--enforce-eager' \
        || DECODE_SERVER_CONFIG="$DECODE_SERVER_CONFIG --enforce-eager"
    echo "Applied ENFORCE_EAGER=1 (--enforce-eager on P and D)"
fi

install_mooncake_rocm() {
    local mooncake_tag="v0.3.11.post1"
    local mooncake_src="/tmp/Mooncake-$mooncake_tag"
    local mooncake_stage="/tmp/mooncake-stage-$mooncake_tag"
    local build_jobs cache_root cache_key cache_archive cache_tmp engine_path
    local os_version python_abi rocm_version

    # Already-installed fast path. Everything below (apt-get update + ~20 build
    # deps, then a source build or a cache untar) is pure setup cost, so skip it
    # when the image already ships a HIP-linked mooncake plus the master binary
    # -- e.g. vllm-openai-rocm:kimi-k3-mc. Matches the idempotency contract the
    # other installers in setup_deps.sh follow, and removes the only
    # unconditional apt-get in the vllm-disagg path (which stalls whenever
    # repo.radeon.com is slow, killing the container mid-setup).
    if command -v mooncake_master >/dev/null 2>&1 \
       && engine_path=$(python3 -c 'import mooncake.engine; print(mooncake.engine.__file__)' 2>/dev/null) \
       && [[ -n "$engine_path" ]] \
       && ldd "$engine_path" 2>/dev/null | grep -q 'libamdhip64.so'; then
        echo "[Mooncake] Already present and HIP-linked ($engine_path); skipping build"
        return 0
    fi

    build_jobs=$(nproc)
    if ((build_jobs > 32)); then
        build_jobs=32
    fi

    os_version=$(. /etc/os-release && printf '%s-%s' "$ID" "$VERSION_ID")
    python_abi=$(python3 -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
    rocm_version=$(sed -n '1p' /opt/rocm/.info/version 2>/dev/null || true)
    if [[ -z "$rocm_version" ]]; then
        rocm_version=$(hipconfig --version)
    fi
    rocm_version=${rocm_version//[^[:alnum:]._-]/_}
    local hf_hub_cache="${HF_HUB_CACHE:-${MODEL_DIR}/.cache/huggingface/hub}"
    cache_root="${hf_hub_cache}/inferencex/mooncake"
    cache_key="${mooncake_tag}-${os_version}-${python_abi}-${rocm_version}-$(uname -m)-hip"
    cache_archive="$cache_root/$cache_key.tar.gz"
    mkdir -p "$cache_root"

    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential cmake git libasio-dev libboost-dev libcurl4-openssl-dev \
        libgflags-dev libgoogle-glog-dev libibverbs-dev libjsoncpp-dev \
        libnuma-dev libpython3-dev libssl-dev libunwind-dev liburing-dev \
        libxxhash-dev libyaml-cpp-dev libzstd-dev ninja-build pybind11-dev

    exec 9>"$cache_archive.lock"
    flock -w 1800 9
    if [[ -f "$cache_archive" ]] && ! tar -tzf "$cache_archive" >/dev/null 2>&1; then
        rm -f "$cache_archive"
    fi
    if [[ ! -f "$cache_archive" ]]; then
        echo "[Mooncake] Building HIP cache artifact: $cache_archive"
        rm -rf "$mooncake_src" "$mooncake_stage"
        git clone --depth 1 --branch "$mooncake_tag" --recurse-submodules \
            --shallow-submodules https://github.com/kvcache-ai/Mooncake.git "$mooncake_src"
        cmake -S "$mooncake_src/extern/yalantinglibs" \
            -B "$mooncake_src/extern/yalantinglibs/build" \
            -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF
        cmake --build "$mooncake_src/extern/yalantinglibs/build" -j "$build_jobs"
        cmake --install "$mooncake_src/extern/yalantinglibs/build"
        cmake -S "$mooncake_src" -B "$mooncake_src/build" -G Ninja \
            -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF -DUSE_HIP=ON \
            -DWITH_EP=OFF -DWITH_STORE=ON -DWITH_STORE_RUST=OFF \
            -DWITH_RUST_EXAMPLE=OFF -DBUILD_EXAMPLES=OFF -DBUILD_UNIT_TESTS=OFF
        cmake --build "$mooncake_src/build" -j "$build_jobs"
        mkdir -p "$mooncake_stage"
        DESTDIR="$mooncake_stage" cmake --install "$mooncake_src/build"
        cache_tmp=$(mktemp "$cache_root/$cache_key.tmp.XXXXXX")
        tar -C "$mooncake_stage" -czf "$cache_tmp" .
        mv -f "$cache_tmp" "$cache_archive"
    else
        echo "[Mooncake] Using HIP cache artifact: $cache_archive"
    fi
    tar -C / -xzf "$cache_archive"
    engine_path=$(python3 -c 'import mooncake.engine; print(mooncake.engine.__file__)')
    ldd "$engine_path" | grep -q 'libamdhip64.so'
    exec 9>&-
}

# MiniMax-M3 agentic DRAM offload: per-node mooncake_master + MooncakeStoreConnector.
# MoRIIOConnector still handles P/D transfer via vLLM MultiConnector.
ensure_mooncake_kv_offload() {
    local tp_size="$1"
    if [[ "${KV_OFFLOADING:-none}" != "dram" || "${KV_OFFLOAD_BACKEND:-}" != "mooncake" ]]; then
        return 0
    fi
    if [[ -n "${MOONCAKE_SETUP_DONE:-}" ]]; then
        return 0
    fi
    if [[ ! "${TOTAL_CPU_DRAM_GB:-}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: KV_OFFLOADING=dram with KV_OFFLOAD_BACKEND=mooncake requires positive TOTAL_CPU_DRAM_GB" >&2
        exit 1
    fi
    if ! python3 -c "from mooncake.store import MooncakeDistributedStore" >/dev/null 2>&1; then
        install_mooncake_rocm
    fi
    python3 -c "from mooncake.store import MooncakeDistributedStore" >/dev/null

    local per_rank_gb=$((TOTAL_CPU_DRAM_GB / tp_size))
    MOONCAKE_MASTER_PORT=$((SERVER_PORT + 12000))
    MOONCAKE_CONFIG_PATH="/run_logs/slurm_job-${SLURM_JOB_ID}/mooncake_config_${host_name}.json"
    mkdir -p "$(dirname "$MOONCAKE_CONFIG_PATH")"
    cat > "$MOONCAKE_CONFIG_PATH" <<EOF
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:${MOONCAKE_MASTER_PORT}",
  "global_segment_size": "${per_rank_gb}GB",
  "local_buffer_size": "4GB",
  "protocol": "tcp",
  "device_name": "",
  "enable_offload": false
}
EOF
    # MC_SLICE_SIZE only governs the RDMA transport. The TCP transport -- which is what
    # "protocol": "tcp" above actually selects -- has its own knob, MC_TCP_SLICE_SIZE,
    # defaulting to 65536. Setting only MC_SLICE_SIZE therefore left every Mooncake
    # transfer sliced at 64KB, so a single ~650MB BatchPut needed thousands of queue
    # entries and the session queue overflowed ("SQ full ... requested=4672 max=16384"),
    # after which the completion path segfaulted in getTransferStatus and killed the
    # decode worker. Keep both in step, and make them tunable.
    export MOONCAKE_CONFIG_PATH PYTHONHASHSEED=0
    export MC_SLICE_SIZE="${MC_SLICE_SIZE:-1048576}"
    export MC_TCP_SLICE_SIZE="${MC_TCP_SLICE_SIZE:-1048576}"
    export MC_TCP_ENABLE_CONNECTION_POOL=1

    local transfer_batch_keys_log="off"
    local mc_workers_log="default"
    if [[ -n "${INFERENCEX_MOONCAKE_MAX_TRANSFER_BATCH_KEYS:-}" ]]; then
        export MC_WORKERS_PER_CTX="${MC_WORKERS_PER_CTX:-4}"
        transfer_batch_keys_log="${INFERENCEX_MOONCAKE_MAX_TRANSFER_BATCH_KEYS}"
        mc_workers_log="${MC_WORKERS_PER_CTX}"

        MOONCAKE_BATCH_PATCH_SCRIPT="$(dirname "${BASH_SOURCE[0]}")/patches/apply_vllm_mooncake_transfer_batches.py"
        if [[ ! -f "$MOONCAKE_BATCH_PATCH_SCRIPT" ]]; then
            echo "ERROR: Mooncake transfer batch patch missing: $MOONCAKE_BATCH_PATCH_SCRIPT" >&2
            exit 1
        fi
        python3 "$MOONCAKE_BATCH_PATCH_SCRIPT"
    fi

    local mooncake_master_cmd="mooncake_master --port ${MOONCAKE_MASTER_PORT} --default_kv_lease_ttl=120s --eviction_high_watermark_ratio=0.80 --eviction_ratio=0.10"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $mooncake_master_cmd"
    else
        MOONCAKE_MASTER_LOG="/run_logs/slurm_job-${SLURM_JOB_ID}/mooncake_master_${host_name}.log"
        $mooncake_master_cmd > "$MOONCAKE_MASTER_LOG" 2>&1 &
        MOONCAKE_MASTER_PID=$!
        sleep 2
        kill -0 "$MOONCAKE_MASTER_PID"
    fi

    echo "Applied Mooncake DRAM KV offload on ${host_name}: TOTAL_CPU_DRAM_GB=${TOTAL_CPU_DRAM_GB} tp=${tp_size} per_rank=${per_rank_gb}GB master_port=${MOONCAKE_MASTER_PORT} transfer_batch_keys=${transfer_batch_keys_log} mc_workers_per_ctx=${mc_workers_log}"
    MOONCAKE_SETUP_DONE=1
}

# Kimi-K3 agentic DRAM offload, LMCache MP instead of Mooncake. Node-local `lmcache server` plus
# LMCacheMPConnector as the second child of the MultiConnector; MoRIIOConnector still owns P/D.
#
# Prefill-only by default. Decode's KV arrives over MoRIIO already computed, so a decode-side tier
# has almost nothing to load -- measured on the Mooncake tier as load_get=0 with saves only, where
# the "100% external hit rate" decode reported came from the MoRIIO transfer, not the tier.
# LMCACHE_ON_DECODE=true attaches it on both sides for an A/B.
#
# The numeric couplings (N, chunk % N, [N,2N), fp8) live in lmcache_mp.sh and are validated by
# test_lmcache_mp_geometry.sh; every one of them has a run that died at engine init, i.e. after the
# weight load, so they are checked before launch rather than discovered afterwards.
ensure_lmcache_kv_offload() {
    local tp_size="$1"
    local mori_role="${2:-}"
    if [[ "${KV_OFFLOADING:-none}" != "dram" || "${KV_OFFLOAD_BACKEND:-}" != "lmcache-k3" ]]; then
        return 0
    fi
    if [[ -n "${LMCACHE_SETUP_DONE:-}" ]]; then
        return 0
    fi

    # kv_consumer is the decode role. Skip the tier there unless explicitly asked for it, and skip it
    # without failing: the connector chain builder makes the same decision, so the two stay in step.
    if [[ "$mori_role" == "kv_consumer" && "${LMCACHE_ON_DECODE:-false}" != "true" ]]; then
        echo "[lmcache] decode role: tier not attached (LMCACHE_ON_DECODE=${LMCACHE_ON_DECODE:-false})"
        LMCACHE_SETUP_DONE=1
        return 0
    fi

    # shellcheck source=/dev/null
    . "$(dirname "${BASH_SOURCE[0]}")/lmcache_mp.sh"

    # fp8 is not a tuning choice on this path. Under bf16 the unified block N is 768, `align` pins
    # max_num_batched_tokens into [768, 1536), and the resulting 1464-token prefill chunk collided
    # with the spec-decode clamp and serialised the engine to Running:1. fp8 lifts N to 1536 and the
    # chunk to ~3000. Set KV_CACHE_DTYPE=auto deliberately for a bf16 A/B, knowing that.
    KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
    export KV_CACHE_DTYPE

    lmcache_mp_export_env
    lmcache_mp_derive_geometry "$tp_size" || exit 1
    lmcache_mp_size_l1 || exit 1
    lmcache_mp_install || exit 1
    lmcache_mp_assert_hybrid_ok || true
    lmcache_mp_start "$tp_size" "/run_logs/slurm_job-${SLURM_JOB_ID}" "${host_name}" || exit 1

    echo "Applied LMCache DRAM KV offload on ${host_name}: role=${mori_role:-<none>} tp=${tp_size}" \
         "L1=${LMCACHE_L1_SIZE_GB}GB N=${LMCACHE_UNIFIED_BLOCK} chunk=${LMCACHE_CHUNK_SIZE}" \
         "mnbt=${LMCACHE_MNBT} port=${LMCACHE_PORT}"
    LMCACHE_SETUP_DONE=1
}

build_kv_transfer_config_json() {
    local mori_role="$1"
    # Both tier backends take the same shape: MoRIIO first, the tier second. MoRIIO leads because on
    # decode it is the only thing that can produce the KDA recurrent state -- that state is not
    # prefix-cacheable and cannot come out of a content-addressed DRAM tier -- and MultiConnector
    # loads from the first child that reports matched tokens. The reverse order would let the tier
    # claim MLA blocks for a request whose KDA state MoRIIO is still fetching, which is exactly the
    # half-restored state the Mooncake fault points at.
    local _tier_backend="${KV_OFFLOAD_BACKEND:-}"
    if [[ "${KV_OFFLOADING:-none}" == "dram" && ( "$_tier_backend" == "mooncake" || "$_tier_backend" == "lmcache-k3" ) ]]; then
        local _lmcache_child=""
        if [[ "$_tier_backend" == "lmcache-k3" ]]; then
            if [[ "$mori_role" != "kv_consumer" || "${LMCACHE_ON_DECODE:-false}" == "true" ]]; then
                # shellcheck source=/dev/null
                . "$(dirname "${BASH_SOURCE[0]}")/lmcache_mp.sh"
                _lmcache_child="$(lmcache_mp_connector_json)"
            fi
        fi
        LMCACHE_CHILD_JSON="$_lmcache_child" KV_TIER_BACKEND="$_tier_backend" \
            NODE0_ADDR="$NODE0_ADDR" PROXY_PING_PORT="$PROXY_PING_PORT" SERVER_PORT="$SERVER_PORT" \
            MORI_KV_ROLE="$mori_role" python3 -c '
import json, os, sys

mori_extra = {
    "proxy_ip": os.environ["NODE0_ADDR"],
    "proxy_ping_port": os.environ["PROXY_PING_PORT"],
    "http_port": os.environ["SERVER_PORT"],
    # Kimi-K3 MI355X validated run pins the MoRIIO backend to rdma explicitly
    # (k3-agentx/gen_k3_mc.sh); IBDEVICES/MORI_RDMA_TC come from the harness env.
    "backend": os.environ.get("MORIIO_BACKEND", "rdma"),
    "read_mode": True,
}
# [mc-role] MOONCAKE_DECODE_STORE=0 drops the Mooncake store from the decode side only.
# Decode never loads from the tier -- measured load_get count 0 with saves only, and the external
# hit rate it reports comes from the MoRIIO transfer -- so its store is writes and RDMA traffic
# without reuse.
_mooncake_entry = {
    "kv_connector": "MooncakeStoreConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "load_async": (os.environ.get("MOONCAKE_LOAD_ASYNC") or "1") == "1",
        "lookup_async": (os.environ.get("MOONCAKE_LOOKUP_ASYNC") or "1") == "1",
    },
}
_connectors = [
    {
        "kv_connector": "MoRIIOConnector",
        "kv_role": os.environ["MORI_KV_ROLE"],
        "kv_connector_extra_config": {
            **mori_extra,
            "read_mode": (os.environ.get("VLLM_MORIIO_WRITE_MODE") or "0") != "1",
        },
    }
]
# The tier is the second child, whichever tier it is. Selecting it here rather than by editing the
# connector list per backend keeps one code path for the ordering guarantee that matters: MoRIIO
# first, tier second.
_backend = os.environ.get("KV_TIER_BACKEND") or "mooncake"
_role = os.environ["MORI_KV_ROLE"]
if _backend == "lmcache-k3":
    # ensure_lmcache_kv_offload made the same prefill-only decision, and hands the child dict in
    # already-rendered so its shape lives in exactly one place (lmcache_mp.sh). Empty means the tier
    # is deliberately absent on this role.
    _child = os.environ.get("LMCACHE_CHILD_JSON") or ""
    if _child:
        _connectors.append(json.loads(_child))
    sys.stderr.write(
        "[tier] backend=lmcache-k3 role=%s attached=%s\n" % (_role, bool(_child))
    )
else:
    if _role != "kv_consumer" or (os.environ.get("MOONCAKE_DECODE_STORE") or "1") == "1":
        _connectors.append(_mooncake_entry)
    sys.stderr.write(
        "[mc-role] role=%s mooncake_store=%s\n" % (_role, len(_connectors) > 1)
    )
print(json.dumps({
    "kv_connector": "MultiConnector",
    "kv_role": "kv_both",
    # KVTransferConfig.kv_load_failure_policy is read by the SCHEDULER from the
    # TOP level (v1/core/sched/scheduler.py: kv_transfer_config.kv_load_failure_policy).
    # It was previously set inside the MoRIIOConnector entry, where nothing reads it,
    # so the effective policy was the "fail" default: any KV load that did not land
    # killed the request outright. On the agentic corpus that shows up as
    #   scheduler.py: Failing 1 request(s) due to KV load failure
    #     (failure_policy=fail, 394752 tokens affected)
    # on the largest traces, and aiperf then aborts the whole concurrency point
    # because a root warmup request failed.
    # "recompute" reschedules the request and recomputes the failed blocks instead.
    # That path is _update_requests_with_invalid_blocks(), which the pinned fork
    # already taught to handle the multiple KV-cache groups of a hybrid model
    # (commit eed3a092) -- the recipe simply never switched the policy over to use it.
    # NOTE: no apostrophes anywhere in this block. It lives inside python3 -c '...',
    # so a single quote closes the shell string and the rest is parsed as shell.
    # "or", not a get() default -- see the MooncakeStoreConnector note below:
    # job.slurm forwards this as -e VAR=${VAR:-}, so leaving it unset still binds
    # the name to "" inside the container and get(k, "recompute") would hand vLLM
    # an empty policy.
    "kv_load_failure_policy": (os.environ.get("KV_LOAD_FAILURE_POLICY") or "recompute"),
    "kv_connector_extra_config": {
        "connectors": _connectors,
    },
}))
'
        return
    fi

    cat <<EOF
{"kv_connector": "MoRIIOConnector", "kv_role": "${mori_role}", "kv_load_failure_policy": "${KV_LOAD_FAILURE_POLICY:-recompute}", "kv_connector_extra_config": {"proxy_ip": "${NODE0_ADDR}", "proxy_ping_port": "${PROXY_PING_PORT}", "http_port": "${SERVER_PORT}", "read_mode": true}}
EOF
}

if [[ "${KV_OFFLOADING:-none}" == "dram" ]]; then
    case "${KV_OFFLOAD_BACKEND:-}" in
        mooncake|lmcache-k3) ;;
        native)
            echo "ERROR: KV_OFFLOAD_BACKEND=native is not supported for vLLM disagg" >&2
            exit 1
            ;;
        *)
            # Fail loudly on an unknown backend rather than silently running with no tier: an arm that
            # was meant to exercise the tier and quietly did not is worse than one that refuses to
            # start, and several rounds were lost to exactly that (clean results with load_get=0).
            echo "ERROR: KV_OFFLOADING=dram needs KV_OFFLOAD_BACKEND=mooncake or lmcache-k3," \
                 "got '${KV_OFFLOAD_BACKEND:-<unset>}'" >&2
            exit 1
            ;;
    esac
fi

# Append the tier's serve-arg overrides to a role's config string. Scoped to the LMCache arm so the
# Mooncake and no-offload curves are untouched: the values (align, [N,2N) budget, halved max_num_seqs,
# lower gpu_memory_utilization) are only correct with LMCacheMPConnector attached.
apply_lmcache_serve_args() {
    local cfg="$1"
    [[ "${KV_OFFLOADING:-none}" == "dram" && "${KV_OFFLOAD_BACKEND:-}" == "lmcache-k3" ]] || {
        printf '%s' "$cfg"; return 0
    }
    [[ -n "${LMCACHE_MNBT:-}" ]] || { printf '%s' "$cfg"; return 0; }
    local extra
    extra="$(lmcache_mp_serve_args | paste -sd' ')"
    # --kv-cache-dtype is not optional here; see ensure_lmcache_kv_offload for why bf16 serialises
    # the engine. Only add it if the recipe has not already pinned one.
    if [[ "$cfg" != *"--kv-cache-dtype"* ]]; then
        extra="$extra --kv-cache-dtype ${KV_CACHE_DTYPE:-fp8}"
    fi
    # A recipe-supplied value wins over ours for anything already present, so strip our duplicate
    # rather than passing the flag twice and relying on argparse's last-wins.
    local flag
    for flag in --max-num-batched-tokens --mamba-cache-mode --max-num-seqs --gpu-memory-utilization; do
        if [[ "$cfg" == *"$flag"* ]]; then
            extra="$(printf '%s' "$extra" | sed -E "s/(^| )${flag} [^ ]+//g")"
            echo "[lmcache] recipe already sets $flag; keeping the recipe value" >&2
        fi
    done
    printf '%s %s' "$cfg" "$extra"
}

# Kimi-K3 runtime optimisation patches (four upstream PRs plus the K3 enablement scripts), carried
# separately from the fork branch because they are NOT in cb8104839c -- the nightly our branch is
# rebased onto and the prepatched image is built from. Rebasing gave us that image's base, not its
# patches. Opt-in via K3_OPT_PATCHES=1, and each step is individually skippable so they can be A/B'd
# one at a time against the numbers already measured on the unpatched base.
if [[ -f "$(dirname "${BASH_SOURCE[0]}")/k3_opt_patches.sh" ]]; then
    # shellcheck source=/dev/null
    . "$(dirname "${BASH_SOURCE[0]}")/k3_opt_patches.sh"
    k3_opt_patches_apply || exit 1
    [[ "${K3_OPT_PATCHES:-0}" == "1" ]] && k3_opt_patches_expected_markers
fi

# vLLM #46240: skip stale KV xfer completions instead of assert-killing EngineCore.
# https://github.com/vllm-project/vllm/issues/46240
if [[ "${VLLM_PATCH_46240:-${KV_OFFLOADING:-none}}" == "dram" || "${VLLM_PATCH_46240:-}" == "1" ]]; then
    PATCH_SCRIPT="$(dirname "${BASH_SOURCE[0]}")/patches/apply_vllm_46240_scheduler_patch.py"
    if [[ ! -f "$PATCH_SCRIPT" ]]; then
        echo "ERROR: VLLM_PATCH_46240 enabled but missing $PATCH_SCRIPT" >&2
        exit 1
    fi
    python3 "$PATCH_SCRIPT"
fi

# Two more hybrid-KV bugs found during the K3 1P1D bring-up are NOT patched here
# any more -- they are fixed in the pinned vLLM fork
# (VLLM_K3_FORK_REF=yichaozhu/moriio-k3-dspark), which is where they belong:
#   eed3a092  scheduler._update_requests_with_invalid_blocks unpacked
#             get_block_ids() as one KV-cache group, so a failed Mooncake load
#             ValueError'd and killed EngineCore instead of recomputing.
#   1755c10c  MLAAttentionSpec.merge required indexes_kv_by_block_stride to
#             match, which split the DSpark draft's 5 MLA layers into their own
#             KV group padded to 24 -- the whole 1.65x KV bytes/token penalty
#             ROCM_AITER_MLA paid over TRITON_MLA, and what put the native 1M
#             context out of reach on aiter.

# Log-only: dump the KV-cache layer bucketing. Explains "Add N padding layers"
# warnings, and specifically which bucket the DSpark draft's MLA layers land in.
if [[ "${VLLM_PATCH_KV_GROUP_DEBUG:-0}" == "1" || "${VLLM_PATCH_KV_GROUP_DEBUG:-}" == "true" ]]; then
    KV_GROUP_DEBUG_PATCH_SCRIPT="$(dirname "${BASH_SOURCE[0]}")/patches/apply_vllm_kv_group_debug.py"
    if [[ ! -f "$KV_GROUP_DEBUG_PATCH_SCRIPT" ]]; then
        echo "ERROR: missing $KV_GROUP_DEBUG_PATCH_SCRIPT" >&2
        exit 1
    fi
    python3 "$KV_GROUP_DEBUG_PATCH_SCRIPT"
fi

# aiter MLA head padding: lets the ASM decode kernel serve head counts that do
# not divide 16 (Kimi-K3 TP8 -> 12 heads/rank). Without it those decodes are
# routed to mla_gluon, whose fp8 regime (bh16bn128) is batch_size=1 only, so
# fp8 KV and concurrency > 1 become mutually exclusive. Opt-in.
if [[ "${VLLM_PATCH_MLA_HEAD_PAD:-0}" == "1" || "${VLLM_PATCH_MLA_HEAD_PAD:-}" == "true" ]]; then
    MLA_PAD_PATCH_SCRIPT="$(dirname "${BASH_SOURCE[0]}")/patches/apply_vllm_aiter_mla_head_pad.py"
    if [[ ! -f "$MLA_PAD_PATCH_SCRIPT" ]]; then
        echo "ERROR: VLLM_PATCH_MLA_HEAD_PAD enabled but missing $MLA_PAD_PATCH_SCRIPT" >&2
        exit 1
    fi
    python3 "$MLA_PAD_PATCH_SCRIPT"

    # Reaching the ASM path exposes aiter's fp8 split-heuristic table, which is
    # keyed on nhead*max_seqlen_q and KeyErrors on untabulated products (16*15
    # = 240 during spec-decode warmup). Ship the fallback with the pad patch.
    MLA_BLOCKN_PATCH_SCRIPT="$(dirname "${BASH_SOURCE[0]}")/patches/apply_aiter_mla_block_n_fallback.py"
    if [[ ! -f "$MLA_BLOCKN_PATCH_SCRIPT" ]]; then
        echo "ERROR: missing $MLA_BLOCKN_PATCH_SCRIPT" >&2
        exit 1
    fi
    python3 "$MLA_BLOCKN_PATCH_SCRIPT"

    # gfx950 fp8 ASM decode needs persistent mode once qo_len > 4 (DSpark n=7
    # verifies 8), but vLLM only builds persistent metadata for qo_len == 1.
    # Off by default: setting work_meta_data on the metadata object is not enough
    # for the kernel to accept qo_len=8 -- the flag is not reaching the launch
    # site -- so the patch changes metadata construction without lifting the gate,
    # which only muddies any other experiment sharing the run. Set
    # VLLM_PATCH_MLA_PERSISTENT_MTP=1 to resume work on it.
    if [[ "${VLLM_PATCH_MLA_PERSISTENT_MTP:-0}" == "1" || "${VLLM_PATCH_MLA_PERSISTENT_MTP:-}" == "true" ]]; then
        MLA_PERSIST_PATCH_SCRIPT="$(dirname "${BASH_SOURCE[0]}")/patches/apply_vllm_aiter_mla_persistent_mtp.py"
        if [[ ! -f "$MLA_PERSIST_PATCH_SCRIPT" ]]; then
            echo "ERROR: missing $MLA_PERSIST_PATCH_SCRIPT" >&2
            exit 1
        fi
        python3 "$MLA_PERSIST_PATCH_SCRIPT"
    fi
fi

# =============================================================================
# Container Synchronization
# =============================================================================

echo "Waiting at the container creation barrier on $host_name"
python3 $WS_PATH/sync.py barrier \
    --local-ip ${host_ip} \
    --local-port 5000 \
    --enable-port \
    --node-ips ${IPADDRS} \
    --node-ports 5000 \
    --wait-for-all-ports \
    --timeout 600

# =============================================================================
# Cluster Topology Configuration
# =============================================================================
IFS=',' read -ra IP_ARRAY <<< "$IPADDRS"

PREFILL_ARGS=""
DECODE_ARGS=""

for ((i=0; i<xP && i<${#IP_ARRAY[@]}; i++)); do
    PREFILL_ARGS+="${IP_ARRAY[$i]} "
done

for ((i=xP; i<${#IP_ARRAY[@]}; i++)); do
    DECODE_ARGS+="${IP_ARRAY[$i]} "
done

echo "Prefill node IPs: ${PREFILL_ARGS}"
echo "Decode  node IPs: ${DECODE_ARGS}"

# Per-worker Prometheus /metrics and cache-flush base URLs for agentic replay.
# vLLM workers listen on SERVER_PORT; the vllm-router on ROUTER_PORT does not
# expose Prometheus or fan out cache resets.
SERVER_METRICS_URLS=()
SERVER_FLUSH_URLS=()
for ((i=0; i<xP && i<${#IP_ARRAY[@]}; i++)); do
    SERVER_METRICS_URLS+=("http://${IP_ARRAY[$i]}:${SERVER_PORT}/metrics")
    SERVER_FLUSH_URLS+=("http://${IP_ARRAY[$i]}:${SERVER_PORT}")
done
for ((i=0; i<yD; i++)); do
    idx=$((xP + i))
    if (( idx < ${#IP_ARRAY[@]} )); then
        SERVER_METRICS_URLS+=("http://${IP_ARRAY[$idx]}:${SERVER_PORT}/metrics")
        SERVER_FLUSH_URLS+=("http://${IP_ARRAY[$idx]}:${SERVER_PORT}")
    fi
done

# MoRI-IO proxy ZMQ registration port (must match vllm-router --vllm-discovery-address)
PROXY_PING_PORT="${PROXY_PING_PORT:-36367}"

# vLLM runtime environment (static vars moved to env.sh; these depend on per-node state)
setup_vllm_env() {
    export VLLM_NIXL_SIDE_CHANNEL_HOST=${rdma_ip}
    export VLLM_NIXL_SIDE_CHANNEL_PORT=5600
    for env_pair in ${MODEL_ENVS}; do
        export "$env_pair"
    done
}

# =============================================================================
# Node Role Assignment and Server Launch
# =============================================================================

if [ "$NODE_RANK" -eq 0 ]; then
    echo "NODE INFO ======================================="
    echo "================================================"
    echo "Node List : ${SLURM_JOB_NODELIST}"
    echo "Node IPs  : ${IPADDRS}"
    echo "Model     : ${MODEL_NAME:-'Not specified'}"
    echo "================================================"

    echo "CLUSTER INFO ===================================="
    echo "================================================"
    echo "${host_name}:${host_ip} is Proxy Node and Prefill Node"
    echo "Using prefill config: $PREFILL_SERVER_CONFIG"
    echo "Prefill servers: ${PREFILL_ARGS}"
    echo "Decode  servers: ${DECODE_ARGS}"
    echo "================================================"

    setup_vllm_env
    ensure_mooncake_kv_offload "$PREFILL_TP_SIZE"
    ensure_lmcache_kv_offload "$PREFILL_TP_SIZE" kv_producer
    PREFILL_SERVER_CONFIG="$(apply_lmcache_serve_args "$PREFILL_SERVER_CONFIG")"
    KV_TRANSFER_JSON=$(build_kv_transfer_config_json kv_producer)

    for env_pair in ${PREFILL_MODEL_ENVS}; do
        export "$env_pair"
        echo "[PREFILL_ENV] $env_pair"
    done

    # Router is started as an external container by job.slurm (VLLM_ROUTER_IMAGE)
    echo "Using external vllm-router container (started by job.slurm on this node)"

    SERVED_MODEL="${MODEL_NAME}"
    PREFILL_CMD="vllm serve ${MODEL_PATH} \
        --served-model-name ${SERVED_MODEL} \
        --port $SERVER_PORT \
        --trust-remote-code \
        --kv-transfer-config '${KV_TRANSFER_JSON}' \
        ${PREFILL_SERVER_CONFIG}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $PREFILL_CMD"
    else
        PREFILL_LOG_FILE="/run_logs/slurm_job-${SLURM_JOB_ID}/prefill_${host_name}.log"
        set -x
        eval "$PREFILL_CMD" > "$PREFILL_LOG_FILE" 2>&1 &
        set +x
        prefill_pid=$!
    fi

    # SERVER_UP_TIMEOUT: how long to wait for every worker to bind its port, i.e.
    # essentially how long weight loading may take. 1800 s is fine for a 300-600 GB
    # checkpoint but not for Kimi-K3: 1.7 TB over 96 shards off wekafs, read by both
    # nodes at once, measured ~23 s/shard = ~37 min, so the old fixed 30 min expired
    # ~10 min before the servers were ready. server_vllm.sh has no `set -e`, so the
    # failure was survivable (the router /health barrier that follows granted another
    # 1800 s) -- but it logged a spurious "Timeout ... waiting for ports to open"
    # followed by "Congratulations!!! All prefill and decode servers are up", which
    # is a confusing pair to debug from.
    echo "Waiting for all prefill and decode servers to be up . . ."
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: skipping barrier (wait-for-all-ports)"
    else
        python3 $WS_PATH/sync.py barrier \
            --node-ips ${IPADDRS} \
            --node-ports $SERVER_PORT \
            --wait-for-all-ports \
            --timeout "${SERVER_UP_TIMEOUT:-1800}"
    fi

    echo "Congratulations!!! All prefill and decode servers are up . . ."

    # Wait for proxy /health to confirm it is accepting requests
    HEALTH_BARRIER_CMD="python3 $WS_PATH/sync.py barrier \
        --node-ips ${NODE0_ADDR} \
        --node-ports ${ROUTER_PORT} \
        --wait-for-all-health \
        --health-endpoint /health \
        --timeout 1800"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $HEALTH_BARRIER_CMD"
    else
        eval "$HEALTH_BARRIER_CMD"
        echo "MoRI-IO proxy is ready for benchmarking"
    fi

    echo "Ready for benchmarking on ${host_name}:${host_ip}"
    echo "Benchmarking on ${host_name}:${host_ip}"
    cd $WS_PATH

    export ROUTER_PORT=$ROUTER_PORT

    # IS_AGENTIC=1/true  → agentic trace replay (trace_replay.sh)
    # IS_AGENTIC unset/0 → fixed-seq-len throughput benchmark (bench.sh)
    if [[ "${IS_AGENTIC:-0}" == "1" || "${IS_AGENTIC:-}" == "true" ]]; then
        if [[ "${ENABLE_METRICS:-0}" == "1" && "${#SERVER_METRICS_URLS[@]}" -gt 0 ]]; then
            AIPERF_SERVER_METRICS_URLS=$(IFS=,; echo "${SERVER_METRICS_URLS[*]}")
            export AIPERF_SERVER_METRICS_URLS
            echo "AIPERF_SERVER_METRICS_URLS=${AIPERF_SERVER_METRICS_URLS}"
        fi
        if [[ "${#SERVER_FLUSH_URLS[@]}" -gt 0 ]]; then
            SERVER_FLUSH_URLS_CSV=$(IFS=,; echo "${SERVER_FLUSH_URLS[*]}")
            export SERVER_FLUSH_URLS_CSV
            echo "SERVER_FLUSH_URLS_CSV=${SERVER_FLUSH_URLS_CSV}"
        fi
        export ENGINE="${FRAMEWORK:-vllm-disagg}"
        BENCH_CMD="bash $WS_PATH/trace_replay.sh \
            $MODEL_DIR $MODEL_NAME $BENCH_MAX_CONCURRENCY /run_logs/slurm_job-${SLURM_JOB_ID}"
        echo "Benchmark runner: trace_replay.sh (agentic, KV_OFFLOADING=${KV_OFFLOADING:-none}, CONC=${BENCH_MAX_CONCURRENCY})"
    else
        BENCH_CMD="bash $WS_PATH/bench.sh ${xP} ${yD} $((PREFILL_TP_SIZE*xP)) $((DECODE_TP_SIZE*yD)) \
            $MODEL_DIR $MODEL_NAME /run_logs/slurm_job-${SLURM_JOB_ID} ${BENCH_INPUT_LEN} \
            ${BENCH_OUTPUT_LEN} \"${BENCH_MAX_CONCURRENCY}\" ${BENCH_REQUEST_RATE} \
            ${BENCH_RANDOM_RANGE_RATIO} ${BENCH_NUM_PROMPTS_MULTIPLIER}"
        echo "Benchmark runner: bench.sh (fixed-seq-len)"
    fi

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

        EVAL_HEALTH_OK=false
        for _attempt in 1 2 3; do
            if curl -sf --max-time 10 "http://0.0.0.0:${ROUTER_PORT}/health" >/dev/null 2>&1; then
                EVAL_HEALTH_OK=true
                break
            fi
            echo "Eval health check attempt $_attempt failed, retrying in 10s..."
            sleep 10
        done

        if [[ "$EVAL_HEALTH_OK" != "true" ]]; then
            echo "WARNING: Router health check failed after 3 attempts. Skipping eval."
        else
            pushd /workspace

            source /workspace/benchmarks/benchmark_lib.sh

            if [[ -n "${EVAL_CONC:-}" ]]; then
                export EVAL_CONCURRENT_REQUESTS="${EVAL_CONC}"
            else
                export EVAL_CONCURRENT_REQUESTS=$(echo "$BENCH_MAX_CONCURRENCY" | tr 'x' '\n' | sort -n | tail -1)
            fi

            if [[ "$DRY_RUN" -eq 1 ]]; then
                echo "DRY RUN: run_eval --framework lm-eval --port $ROUTER_PORT (conc=${EVAL_CONCURRENT_REQUESTS}, ctx=${EVAL_MAX_MODEL_LEN:-auto})"
            else
                run_eval --framework lm-eval --port "$ROUTER_PORT"
                eval_rc=$?

                if [[ $eval_rc -ne 0 ]]; then
                    echo "ERROR: run_eval exited rc=$eval_rc; skipping metadata write and eval artifact staging" >&2
                    EVAL_FAILED=1
                else
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

                    append_lm_eval_summary

                    EVAL_COPY_DIR="/run_logs/slurm_job-${SLURM_JOB_ID}/eval_results"
                    mkdir -p "$EVAL_COPY_DIR"
                    for f in meta_env.json; do
                        [ -e "/workspace/$f" ] && cp -f "/workspace/$f" "$EVAL_COPY_DIR/"
                    done
                    find /workspace -maxdepth 1 -name 'results*.json' -exec cp -f {} "$EVAL_COPY_DIR/" \;
                    find /workspace -maxdepth 1 -name 'sample*.jsonl' -exec cp -f {} "$EVAL_COPY_DIR/" \;

                    echo "Eval completed. Artifacts staged in $EVAL_COPY_DIR"
                fi
            fi

            popd
        fi
    fi

    # Copy benchmark/eval results to BENCHMARK_LOGS_DIR (mounted from host)
    LOGS_OUTPUT="${BENCHMARK_LOGS_DIR:-/run_logs}/logs"
    mkdir -p "$LOGS_OUTPUT"

    if [[ "$DRY_RUN" -eq 0 ]]; then
        cp -r /run_logs/slurm_job-${SLURM_JOB_ID} "$LOGS_OUTPUT/"
        echo "Copied results to $LOGS_OUTPUT/slurm_job-${SLURM_JOB_ID}"
    fi

    # KEEP_SERVER_ALIVE=1 holds the whole stack (router + prefill + decode) up after
    # the benchmark instead of tearing it down, so a second load run costs only the
    # load itself. Bringing 1P1D up is ~10 min even with --load-format dummy (weights,
    # KV alloc, Mooncake DRAM registration, graph capture), which dominated every
    # crash-repro iteration. Debug-only; the sentinel is deleted to release the job.
    if [[ "${KEEP_SERVER_ALIVE:-0}" == "1" && "$DRY_RUN" -eq 0 ]]; then
        KEEP_ALIVE_SENTINEL="/run_logs/slurm_job-${SLURM_JOB_ID}/KEEP_SERVER_ALIVE"
        : > "$KEEP_ALIVE_SENTINEL"
        echo "KEEP_SERVER_ALIVE=1: stack stays up; 'rm ${KEEP_ALIVE_SENTINEL}' to let it exit"
        while [[ -f "$KEEP_ALIVE_SENTINEL" ]]; do sleep 15; done
        echo "KEEP_SERVER_ALIVE: sentinel removed, proceeding to shutdown"
    fi

    echo "Killing the prefill server"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        [[ -n "${prefill_pid:-}" ]] && kill $prefill_pid 2>/dev/null || true
        sleep 2
        pkill -f "vllm serve" 2>/dev/null || true
    fi

    if [[ "${EVAL_FAILED:-0}" -eq 1 ]]; then
        echo "ERROR: eval failed; exiting node-0 with rc=1"
        exit 1
    fi

elif [ "$NODE_RANK" -gt 0 ] && [ "$NODE_RANK" -lt "$xP" ]; then
    echo "${host_name}:${host_ip} is Additional Prefill Node (Model: ${MODEL_NAME})"
    echo "Using prefill config: $PREFILL_SERVER_CONFIG"

    setup_vllm_env
    ensure_mooncake_kv_offload "$PREFILL_TP_SIZE"
    ensure_lmcache_kv_offload "$PREFILL_TP_SIZE" kv_producer
    PREFILL_SERVER_CONFIG="$(apply_lmcache_serve_args "$PREFILL_SERVER_CONFIG")"
    KV_TRANSFER_JSON=$(build_kv_transfer_config_json kv_producer)

    for env_pair in ${PREFILL_MODEL_ENVS}; do
        export "$env_pair"
        echo "[PREFILL_ENV] $env_pair"
    done

    SERVED_MODEL="${MODEL_NAME}"
    PREFILL_CMD="vllm serve ${MODEL_PATH} \
        --served-model-name ${SERVED_MODEL} \
        --port $SERVER_PORT \
        --trust-remote-code \
        --kv-transfer-config '${KV_TRANSFER_JSON}' \
        ${PREFILL_SERVER_CONFIG}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $PREFILL_CMD"
    else
        PREFILL_LOG_FILE="/run_logs/slurm_job-${SLURM_JOB_ID}/prefill_${host_name}.log"
        set -x
        eval "$PREFILL_CMD" > "$PREFILL_LOG_FILE" 2>&1 &
        set +x
        prefill_pid=$!
    fi

    echo "Waiting for proxy server to be up..."
    BARRIER_CMD="python3 $WS_PATH/sync.py barrier \
        --node-ips ${NODE0_ADDR} \
        --node-ports ${ROUTER_PORT} \
        --wait-for-all-ports \
        --timeout 1800"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $BARRIER_CMD"
    else
        eval "$BARRIER_CMD"
    fi

    echo "Waiting until proxy server closes..."
    WAIT_CMD="python3 $WS_PATH/sync.py wait \
        --remote-ip ${NODE0_ADDR} \
        --remote-port ${ROUTER_PORT}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $WAIT_CMD"
    else
        eval "$WAIT_CMD"
    fi

    echo "Killing the prefill server"
    [[ "$DRY_RUN" -eq 0 ]] && kill $prefill_pid 2>/dev/null || true

else
    echo "${host_name}:${host_ip} is Decode Node (Model: ${MODEL_NAME})"
    echo "Using decode config: $DECODE_SERVER_CONFIG"

    setup_vllm_env
    ensure_mooncake_kv_offload "$DECODE_TP_SIZE"
    ensure_lmcache_kv_offload "$DECODE_TP_SIZE" kv_consumer
    DECODE_SERVER_CONFIG="$(apply_lmcache_serve_args "$DECODE_SERVER_CONFIG")"
    KV_TRANSFER_JSON=$(build_kv_transfer_config_json kv_consumer)

    for env_pair in ${DECODE_MODEL_ENVS}; do
        export "$env_pair"
        echo "[DECODE_ENV] $env_pair"
    done

    SERVED_MODEL="${MODEL_NAME}"
    DECODE_CMD="vllm serve ${MODEL_PATH} \
        --served-model-name ${SERVED_MODEL} \
        --port $SERVER_PORT \
        --trust-remote-code \
        --kv-transfer-config '${KV_TRANSFER_JSON}' \
        ${DECODE_SERVER_CONFIG}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $DECODE_CMD"
    else
        DECODE_LOG_FILE="/run_logs/slurm_job-${SLURM_JOB_ID}/decode_${host_name}.log"
        set -x
        eval "$DECODE_CMD" > "$DECODE_LOG_FILE" 2>&1 &
        set +x
        decode_pid=$!
    fi

    echo "Waiting for proxy server to be up..."
    BARRIER_CMD="python3 $WS_PATH/sync.py barrier \
        --node-ips ${NODE0_ADDR} \
        --node-ports ${ROUTER_PORT} \
        --wait-for-all-ports \
        --timeout 1800"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $BARRIER_CMD"
    else
        eval "$BARRIER_CMD"
    fi

    echo "Waiting until proxy server closes..."
    WAIT_CMD="python3 $WS_PATH/sync.py wait \
        --remote-ip ${NODE0_ADDR} \
        --remote-port ${ROUTER_PORT}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $WAIT_CMD"
    else
        eval "$WAIT_CMD"
    fi

    echo "Killing the decode server"
    [[ "$DRY_RUN" -eq 0 ]] && kill $decode_pid 2>/dev/null || true
fi

# echo "Killing the etcd server"
# kill $etcd_pid 2>/dev/null || true
# pkill -f etcd 2>/dev/null || true

echo "Script completed successfully"
exit 0
