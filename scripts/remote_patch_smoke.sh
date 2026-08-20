#!/usr/bin/env bash
set -euo pipefail
scp -o BatchMode=yes /tmp/ix_patch_smoke.tgz aim-head:/tmp/ix_patch_smoke.tgz
ssh -o BatchMode=yes aim-head 'bash -s' <<'EOF'
set -euo pipefail
rm -rf /tmp/ix_patch_smoke && mkdir -p /tmp/ix_patch_smoke
tar xzf /tmp/ix_patch_smoke.tgz -C /tmp/ix_patch_smoke
sed -i 's/\r$//' /tmp/ix_patch_smoke/benchmarks/multi_node/amd_utils/apply_k3_moriio_patches.sh
bash /tmp/ix_patch_smoke/scripts/check_k3_moriio_patch.sh /tmp/ix_patch_smoke
IMG=vllm/vllm-openai-rocm:nightly-ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9
docker run --rm --entrypoint bash \
  -v /tmp/ix_patch_smoke:/workspace \
  -e K3_MORIIO_PATCH=/workspace/benchmarks/multi_node/amd_utils/patches/k3_moriio_51052.patch \
  "$IMG" -lc 'bash /workspace/benchmarks/multi_node/amd_utils/apply_k3_moriio_patches.sh && python3 -c "from vllm.distributed.kv_transfer.kv_connector.v1.moriio import moriio_common as c; assert hasattr(c, \"as_attn_mamba\"); print(\"PATCH_SMOKE_OK\")"'
EOF
