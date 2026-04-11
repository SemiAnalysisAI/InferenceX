# MI300X SGLang v0.5.10 + MoRI + Broadcom IBGDA image
#
# Builds on the official SGLang v0.5.10 ROCm MI300X image and adds:
#   - MoRI (Memory over RDMA Infrastructure) disaggregation backend
#   - Broadcom bnxt_re userspace RDMA libraries (Thor 2 NICs / IBGDA)
#   - GLM-5 transformers support (glm_moe_dsa model type)
#
# Components are copied from the v0.5.9-bnxt image which was built
# from the jordannanos/sglang fork with MoRI + bnxt patches.

FROM docker.io/semianalysiswork/sgl-bnxt-cdna3:v0.5.9-bnxt AS bnxt-donor

FROM lmsysorg/sglang:v0.5.10-rocm720-mi30x

LABEL org.opencontainers.image.description="SGLang v0.5.10 MI300X + MoRI + Broadcom IBGDA (gfx942)"

# 1. Copy MoRI build artifacts and Python package
COPY --from=bnxt-donor /sgl-workspace/mori /sgl-workspace/mori

# 2. Copy Broadcom bnxt_re userspace RDMA libraries
COPY --from=bnxt-donor /usr/local/lib/libbnxt_re-231.0.162.0.so /usr/local/lib/
COPY --from=bnxt-donor /usr/local/lib/libbnxt_re-rdmav34.so /usr/local/lib/
COPY --from=bnxt-donor /usr/local/lib/libbnxt_re.a /usr/local/lib/
COPY --from=bnxt-donor /usr/local/lib/libbnxt_re.la /usr/local/lib/
COPY --from=bnxt-donor /usr/local/lib/libbnxt_re.so /usr/local/lib/
COPY --from=bnxt-donor /usr/local/etc/libibverbs.d/bnxt_re.driver /usr/local/etc/libibverbs.d/

# 3. Copy SGLang MoRI disaggregation backend
COPY --from=bnxt-donor /sgl-workspace/sglang/python/sglang/srt/disaggregation/mori \
     /sgl-workspace/sglang/python/sglang/srt/disaggregation/mori

# 4. Set up MoRI Python package (egg-link install — no setup.py, path-based)
#    Create the egg-link so pip recognizes it, and add to PYTHONPATH
RUN echo "/sgl-workspace/mori/python" > /opt/venv/lib/python3.10/site-packages/amd-mori.egg-link && \
    echo "/sgl-workspace/mori/python" >> /opt/venv/lib/python3.10/site-packages/easy-install.pth && \
    ldconfig

ENV PYTHONPATH="/sgl-workspace/mori/python:${PYTHONPATH}"

# 5. Fix ROCm SMI library version mismatch
#    MoRI (built against ROCm 7.0) expects librocm_smi64.so.7 but
#    ROCm 7.2 ships librocm_smi64.so.1. Add compat symlink.
RUN ln -sf /opt/rocm-7.2.0/lib/librocm_smi64.so.1 /opt/rocm-7.2.0/lib/librocm_smi64.so.7 && ldconfig

# 6. Install transformers with GLM-5 (glm_moe_dsa) model type support
RUN pip install --no-cache-dir \
    "git+https://github.com/huggingface/transformers.git@6ed9ee36f608fd145168377345bfc4a5de12e1e2"

CMD ["/bin/bash"]
