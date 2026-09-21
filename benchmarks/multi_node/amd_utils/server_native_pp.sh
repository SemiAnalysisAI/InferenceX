#!/usr/bin/env bash
set -euo pipefail

if [[ -f /config/hicache_mc.env ]]; then
    set -a
    source /config/hicache_mc.env
    set +a
fi
source "$(dirname "${BASH_SOURCE[0]}")/../../benchmark_lib.sh"
check_env_vars NODE_RANK NODE0_ADDR MODEL_PATH MODEL_DIR MODEL_NAME ROUTER_PORT \
    BENCH_MAX_CONCURRENCY DURATION RESULT_FILENAME SLURM_JOB_ID

run_root="${NATIVE_PP_SHARED_LOG_ROOT:-/shared/data/R7N/InferenceX_CI/live}"
run_dir="${run_root}/slurm_job-${SLURM_JOB_ID}"
shared_dir="/benchmark_logs/logs/slurm_job-${SLURM_JOB_ID}"
done_file="$run_dir/native_pp_done"
mkdir -p "$run_dir" "$shared_dir"
echo "native PP live logs: $run_dir"

max_num_seqs="${K3_MAX_NUM_SEQS:-80}"
max_capture_size="${K3_MAX_CUDAGRAPH_CAPTURE_SIZE:-96}"
capture_sizes="${K3_CUDAGRAPH_CAPTURE_SIZES:-1,2,4,8,16,24,32,40,48,56,64,72,80,88,96}"
compile_config=$(printf '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","max_cudagraph_capture_size":%s,"custom_ops":["+fused_rms_norm_gated"],"cudagraph_capture_sizes":[%s]}' "$max_capture_size" "$capture_sizes")
spec_config=$(printf '{"model":"%s","num_speculative_tokens":%s,"method":"dspark","attention_backend":"ROCM_AITER_MLA","kv_cache_dtype":"fp8","draft_sample_method":"probabilistic","rejection_sample_method":"synthetic","synthetic_acceptance_length":%s,"disable_eagle_block_drop":true,"draft_load_config":{"load_format":"safetensors"}}' "${SPEC_MODEL:-/models/models/Kimi-K3-DSpark}" "${SPEC_NUM_TOKENS:-4}" "${SPEC_SYNTHETIC_ACCEPTANCE_LENGTH:-3.36}")

export VLLM_PP_LAYER_PARTITION="${K3_PP_LAYER_PARTITION:-48,45}"
export K3_HYBRID_PP_FIX=1 K3_PP_ASYNC_ACTIVATION=1
export K3_PP_RING_SIZE="${K3_PP_RING_SIZE:-2}"
export K3_PP_PREPOST_RECV=1 K3_PP_BATCH_CAP="${K3_PP_BATCH_CAP:-104}"
export K3_DSPARK_FUSION_SKIP_MISSING=1 K3_DSPARK_FUSION_STAGE=0
export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MLA=1
export VLLM_ROCM_USE_AITER_RMSNORM=1 VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1
export AITER_SITUV2_A8W4=1 AITER_BF16_FP8_MOE_BOUND=0
export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm VLLM_ROCM_AITER_NATIVE_DCP_VERIFY=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 VLLM_KV_CACHE_LAYOUT=HND
export VLLM_USE_BREAKABLE_CUDAGRAPH=1 HSA_ENABLE_IPC_MODE_LEGACY=1
export HIP_FORCE_DEV_KERNARG=1 TORCH_NCCL_BLOCKING_WAIT=0 NCCL_BLOCKING_WAIT=0
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export ROCR_VISIBLE_DEVICES="${K3_VISIBLE_DEVICES:-0,1,2,3}"
export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0

rank_addr=$(printf '%s' "${IPADDRS:-$NODE0_ADDR}" | cut -d, -f$((NODE_RANK + 1)))
export VLLM_HOST_IP="$rank_addr" HOST_IP="$rank_addr"
peer_field=$((NODE_RANK == 0 ? 2 : 1))
peer_addr=$(printf '%s' "${IPADDRS:-$NODE0_ADDR}" | cut -d, -f"$peer_field")
fabric_if=$(PEER_ADDR="$peer_addr" python3 - <<'PY'
import fcntl
import os
import socket
import struct

peer = os.environ["PEER_ADDR"]
route = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
route.connect((peer, 1))
source_ip = route.getsockname()[0]
route.close()

probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
for name in os.listdir("/sys/class/net"):
    try:
        packed = struct.pack("256s", name.encode()[:15])
        addr = socket.inet_ntoa(
            fcntl.ioctl(probe.fileno(), 0x8915, packed)[20:24]
        )
    except OSError:
        continue
    if addr == source_ip:
        print(name)
        break
else:
    raise SystemExit(f"no interface found for private source IP {source_ip}")
PY
)
export NCCL_SOCKET_IFNAME="$fabric_if" GLOO_SOCKET_IFNAME="$fabric_if"
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,ENV
echo "native PP fabric: rank=$NODE_RANK host_ip=$rank_addr peer=$peer_addr interface=$fabric_if"
if [[ -z "${NCCL_IB_HCA:-}" && -d /sys/class/infiniband ]]; then
    NCCL_IB_HCA=$(find /sys/class/infiniband -mindepth 1 -maxdepth 1 -printf '%f\n' | paste -sd,)
    export NCCL_IB_HCA
fi
export NCCL_IB_DISABLE=0 NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-1}"

dist=(--nnodes 2 --node-rank "$NODE_RANK" --master-addr "$NODE0_ADDR" --master-port 29500)
[[ "$NODE_RANK" == 1 ]] && dist+=(--headless)

vllm serve "$MODEL_PATH" --served-model-name "$MODEL_NAME" \
    --host 0.0.0.0 --port "$ROUTER_PORT" --trust-remote-code \
    --tensor-parallel-size "${K3_TP_SIZE:-4}" --pipeline-parallel-size 2 \
    --decode-context-parallel-size "${K3_DCP_SIZE:-4}" --dcp-comm-backend a2a "${dist[@]}" \
    --load-format safetensors --gpu-memory-utilization "${K3_GPU_MEMORY_UTILIZATION:-0.90}" \
    --language-model-only --max-num-seqs "$max_num_seqs" --max-num-batched-tokens "${K3_MBT:-16384}" \
    --max-model-len 1048576 --kv-cache-dtype fp8 --enable-prefix-caching \
    --distributed-timeout-seconds 21600 --enable-auto-tool-choice \
    --tool-call-parser kimi_k3 --reasoning-parser kimi_k3 \
    --compilation-config "$compile_config" --async-scheduling \
    --speculative-config "$spec_config" --moe-backend auto \
    --kv-transfer-config "$(printf '{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"cuda\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use_per_rank\":%s,\"lazy_offload\":false,\"kv_offload_backend\":\"cpu\"}}' "${K3_SIMPLECPU_BYTES_PER_RANK:-224875000000}")" \
    --mamba-ssm-cache-dtype "${K3_MAMBA_SSM_CACHE_DTYPE:-bfloat16}" --mamba-cache-mode "${K3_MAMBA_CACHE_MODE:-all}" \
    --enable-mamba-fine-grained-prefix-cache --prefix-match-unit 128 --block-size 128 \
    >"$run_dir/native_pp_rank${NODE_RANK}.log" 2>&1 &
server_pid=$!

cleanup() { kill "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

if [[ "$NODE_RANK" == 0 ]]; then
    health_waits=0
    health_started=$(date +%s)
    until curl -sf --max-time 5 "http://127.0.0.1:${ROUTER_PORT}/health" >/dev/null; do
        kill -0 "$server_pid" 2>/dev/null || { tail -200 "$run_dir/native_pp_rank0.log"; exit 1; }
        if (( $(date +%s) - health_started >= ${SERVER_UP_TIMEOUT:-1800} )); then
            echo "native PP health timed out after ${SERVER_UP_TIMEOUT:-1800}s" >&2
            tail -300 "$run_dir/native_pp_rank0.log" >&2
            exit 1
        fi
        health_waits=$((health_waits + 1))
        if (( health_waits % 2 == 0 )); then
            echo "[$(date -Is)] waiting for native PP health; rank0 log bytes=$(stat -c %s "$run_dir/native_pp_rank0.log" 2>/dev/null || echo 0)"
            tail -5 "$run_dir/native_pp_rank0.log" 2>/dev/null || true
        fi
        sleep 30
    done
    echo "[$(date -Is)] native PP health ready"
    export ENGINE=vllm-disagg SERVER_FLUSH_URLS_CSV="http://127.0.0.1:${ROUTER_PORT}"
    bash "$(dirname "${BASH_SOURCE[0]}")/trace_replay.sh" \
        "$MODEL_DIR" "$MODEL_NAME" "$BENCH_MAX_CONCURRENCY" "$run_dir"
    cp -r "$run_dir/." "$shared_dir/"
    touch "$done_file"
else
    rank1_waits=0
    while [[ ! -e "$done_file" ]]; do
        kill -0 "$server_pid" 2>/dev/null || { tail -200 "$run_dir/native_pp_rank1.log"; exit 1; }
        rank1_waits=$((rank1_waits + 1))
        if (( rank1_waits % 6 == 0 )); then
            echo "[$(date -Is)] rank1 active; log bytes=$(stat -c %s "$run_dir/native_pp_rank1.log" 2>/dev/null || echo 0)"
        fi
        sleep 10
    done
fi
