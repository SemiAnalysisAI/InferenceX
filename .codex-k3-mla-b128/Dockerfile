FROM vllm/vllm-openai-rocm:kimi-k3

COPY mla_gluon.py \
  /usr/local/lib/python3.12/dist-packages/aiter/ops/triton/gluon/mla_gluon.py

COPY mla_attention.py \
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/mla_attention.py

COPY rocm_aiter_mla.patch /tmp/rocm_aiter_mla.patch

RUN cd /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla \
    && patch -p0 < /tmp/rocm_aiter_mla.patch \
    && rm /tmp/rocm_aiter_mla.patch

RUN python3 -m py_compile \
      /usr/local/lib/python3.12/dist-packages/aiter/ops/triton/gluon/mla_gluon.py \
      /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/mla_attention.py \
      /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/rocm_aiter_mla.py \
    && ! grep -q 'requires batch_size=1' \
      /usr/local/lib/python3.12/dist-packages/aiter/ops/triton/gluon/mla_gluon.py

LABEL org.opencontainers.image.description="Kimi-K3 ROCm image with batched FP8 MLA decode and bounded long-context FP8 prefill chunks"
