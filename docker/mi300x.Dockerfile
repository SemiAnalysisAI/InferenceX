# MI300X disaggregated inference image
# Reuses the MI325X MORI + Broadcom IBGDA image since both are gfx942 (CDNA3)
# and the same Vultr/CPE cluster topology with Broadcom Thor 2 bnxt_re NICs.
#
# To build and push:
#   bash docker/build-sglang-bnxt-mi300x.sh
#
# Source image built with:
#   GPU_ARCH=gfx942, ENABLE_MORI=1, NIC_BACKEND=ibgda (bnxt_rocelib)
FROM ghcr.io/jordannanos/sgl-mi325x-mori:v0.5.9-bnxt-good

LABEL org.opencontainers.image.description="SGLang MI300X MORI + Broadcom IBGDA (gfx942, bnxt_re NICs)"
LABEL org.opencontainers.image.source="https://github.com/JordanNanos/sglang"

CMD ["/bin/bash"]
