FROM vllm/vllm-openai-rocm:nightly-09663abde0f50944a8d5ea30120666024b503faa

ENV SITE=/usr/local/lib/python3.12/dist-packages

COPY patches/ /opt/patches/
COPY vllm/     ${SITE}/vllm/

RUN python3 -c "import py_compile; \
  [py_compile.compile('${SITE}/vllm/'+f, doraise=True) for f in [ \
    '_aiter_ops.py', \
    'model_executor/kernels/linear/scaled_mm/aiter.py', \
    'model_executor/kernels/mhc/aiter.py', \
    'model_executor/layers/mhc.py', \
    'model_executor/layers/fused_moe/router/fused_topk_bias_router.py', \
    'models/deepseek_v4/amd/model.py', \
    'v1/attention/ops/rocm_aiter_mla_sparse.py', \
    'v1/worker/block_table.py', \
  ]]; print('OK')"

# Verify fixed-P4 (bpreshuffle) markers are present in the overlay.
RUN grep -q 'As.transpose(0, 1).contiguous().view(As.shape)' ${SITE}/vllm/_aiter_ops.py \
 && grep -q '_weight_preshuffled = False' ${SITE}/vllm/model_executor/kernels/linear/scaled_mm/aiter.py \
 && grep -q 'gemm_a8w8_blockscale_bpreshuffle' ${SITE}/vllm/model_executor/kernels/linear/scaled_mm/aiter.py \
 && grep -q 'is_bmm' ${SITE}/vllm/model_executor/kernels/linear/scaled_mm/aiter.py \
 && echo 'Fixed-P4 markers verified OK'
