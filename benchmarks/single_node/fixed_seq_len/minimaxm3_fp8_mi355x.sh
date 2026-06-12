#!/usr/bin/env bash

# MiniMax-M3 MXFP8 MI355X (gfx950) single-node vLLM recipe
# (https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3). MXFP8 uses native MXFP8
# matrix cores on CDNA4 and runs from TP=4 per the recipe; main-model
# attention on AMD is the Triton backend (TRITON_ATTN). --block-size 128 is
# mandatory (MSA sparse_block_size; default 16 fails with "No common block
# size for 16" on AMD).
#
# Day-zero caveat: no public ROCm image carries M3 support yet
# (vllm-project/vllm#45381 unmerged; the recipe's AMD image is a placeholder).
# The M3 AMD path is pure Python/Triton (vllm/models/minimax_m3/{amd,common}),
# so we overlay the m3_release python tree onto the installed nightly wheel —
# the image's base commit (6fbfdd18) is ~6 commits behind the PR merge-base
# (0cd9b7af), keeping drift minimal. Compiled .so artifacts from the wheel are
# preserved; the new csrc kernels in the PR are NVIDIA-only.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    EP_SIZE \
    DP_ATTENTION \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

# Overlay the m3_release python tree if this vllm build doesn't know M3 yet.
# The M3 AMD path is Python + Triton (vllm/models/minimax_m3/{amd,common}/ops
# are all @triton.jit), with ONE exception: the horizontally-fused QK-norm +
# partial-NeoX-RoPE (+ bf16 KV/index-cache scatter) helper is a compiled csrc
# kernel (fused_minimax_m3_qknorm_rope_kv_insert) that no public ROCm wheel
# ships yet. We overlay the python tree and then register a pure-torch fallback
# for that single op (bit-parity vs the PR's reference, verified on gfx950).
if ! python3 -c "import vllm.models.minimax_m3" 2>/dev/null; then
    echo "Overlaying vLLM m3_release python tree onto installed package"
    rm -rf /tmp/vllm-m3
    git clone --depth 1 --branch m3_release https://github.com/vllm-project/vllm.git /tmp/vllm-m3
    VLLM_PKG_DIR=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
    cp -r /tmp/vllm-m3/vllm/* "$VLLM_PKG_DIR/"
    python3 -c "import vllm.models.minimax_m3; print('m3 overlay OK')"

    # Day-zero ROCm shim: the m3_release fused_minimax_m3_qknorm_rope_kv_insert
    # csrc kernel is absent from the nightly ROCm wheel (torch.ops._C has no
    # such symbol), so vllm/_custom_ops.py's python wrapper would dispatch into
    # a non-existent op. Append a torch fallback that takes over when the
    # compiled symbol is missing (a no-op once an M3-aware ROCm image lands).
    # Semantics mirror tests/kernels/test_fused_minimax_m3_qknorm_rope_kv_insert.py.
    cat >> "$VLLM_PKG_DIR/_custom_ops.py" <<'M3_SHIM_EOF'


# ─── Day-zero MiniMax-M3 ROCm fallback (auto-removed once csrc lands) ─────────
if not hasattr(torch.ops._C, "fused_minimax_m3_qknorm_rope_kv_insert"):

    def fused_minimax_m3_qknorm_rope_kv_insert(  # type: ignore[no-redef]
        qkv, q_norm_weight, k_norm_weight, cos_sin_cache, positions,
        num_heads, num_kv_heads, rotary_dim, eps,
        index_q_norm_weight=None, index_k_norm_weight=None, num_index_heads=0,
        slot_mapping=None, index_slot_mapping=None, kv_cache=None,
        index_cache=None, block_size=0, q_out=None, index_q_out=None,
    ):
        HEAD_DIM = 128
        half = rotary_dim // 2
        cs = cos_sin_cache[positions].float()
        cos = cs[..., :half].unsqueeze(1)
        sin = cs[..., half:].unsqueeze(1)

        def _norm_rope(x, w):  # x: [nt, nh, 128]
            xf = x.float()
            var = xf.pow(2).mean(-1, keepdim=True)
            normed = xf * torch.rsqrt(var + eps) * (1.0 + w.float())
            rot = normed[..., :rotary_dim]
            x1 = rot[..., :half]
            x2 = rot[..., half:]
            out = normed.clone()
            out[..., :half] = x1 * cos - x2 * sin
            out[..., half:rotary_dim] = x2 * cos + x1 * sin
            return out.to(x.dtype)

        nt = qkv.shape[0]
        qsz = num_heads * HEAD_DIM
        kvsz = num_kv_heads * HEAD_DIM
        has_index = bool(num_index_heads and num_index_heads > 0)
        if has_index:
            iqsz = num_index_heads * HEAD_DIM
            q_in, k_in, v_in, iq_in, ik_in = qkv.split(
                [qsz, kvsz, kvsz, iqsz, HEAD_DIM], dim=-1)
        else:
            q_in, k_in, v_in = qkv.split([qsz, kvsz, kvsz], dim=-1)

        q_r = _norm_rope(q_in.view(nt, num_heads, HEAD_DIM),
                         q_norm_weight).view(nt, qsz)
        k_r = _norm_rope(k_in.view(nt, num_kv_heads, HEAD_DIM),
                         k_norm_weight).view(nt, kvsz)
        (q_out if q_out is not None else q_in).copy_(q_r)
        k_in.copy_(k_r)

        if has_index:
            iq_r = _norm_rope(iq_in.view(nt, num_index_heads, HEAD_DIM),
                              index_q_norm_weight).view(nt, iqsz)
            ik_r = _norm_rope(ik_in.view(nt, 1, HEAD_DIM),
                              index_k_norm_weight).view(nt, HEAD_DIM)
            (index_q_out if index_q_out is not None else iq_in).copy_(iq_r)
            ik_in.copy_(ik_r)

        if kv_cache is not None and slot_mapping is not None:
            blk = slot_mapping // block_size
            pos = slot_mapping % block_size
            kv_cache[blk, 0, pos] = k_r.view(
                nt, num_kv_heads, HEAD_DIM).to(kv_cache.dtype)
            kv_cache[blk, 1, pos] = v_in.reshape(
                nt, num_kv_heads, HEAD_DIM).to(kv_cache.dtype)
        if has_index and index_cache is not None:
            ism = (index_slot_mapping
                   if index_slot_mapping is not None else slot_mapping)
            index_cache.view(-1, HEAD_DIM)[ism] = ik_r.to(index_cache.dtype)
M3_SHIM_EOF
    python3 -c "from vllm import _custom_ops as ops; assert callable(ops.fused_minimax_m3_qknorm_rope_kv_insert); print('m3 fused-op fallback registered')"
fi

# Weights live on the NFS hub cache (/it-share/hf-hub-cache mounted as
# HF_HUB_CACHE by launch_mi355x-amds.sh) — pre-downloaded; this is a no-op
# when the snapshot is complete.
if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

# Set HIP_VISIBLE_DEVICES to match ROCR_VISIBLE_DEVICES for Ray compatibility in vLLM 0.14+
if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

SERVER_LOG=/workspace/server.log

# 444 GB of MXFP8 weights off NFS; engine startup can exceed the default
# 600s readiness window.
export VLLM_ENGINE_READY_TIMEOUT_S=3600

if [ "${DP_ATTENTION}" = "true" ]; then
  PARALLEL_ARGS="--tensor-parallel-size=1 --data-parallel-size=$TP --enable-expert-parallel"
elif [ "$EP_SIZE" -gt 1 ]; then
  PARALLEL_ARGS="--tensor-parallel-size=$TP --enable-expert-parallel"
else
  PARALLEL_ARGS="--tensor-parallel-size=$TP"
fi

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
vllm serve $MODEL --port $PORT \
$PARALLEL_ARGS \
--gpu-memory-utilization 0.92 \
--max-model-len $MAX_MODEL_LEN \
--block-size 128 \
--language-model-only \
--attention-backend TRITON_ATTN \
--max-num-batched-tokens "$((ISL * 2 ))" \
--no-enable-prefix-caching \
--trust-remote-code > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --trust-remote-code

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
