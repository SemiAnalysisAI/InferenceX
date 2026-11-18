#!/usr/bin/env bash
set -eo pipefail

if [[ -z ${HEAD_NODE_IP:-} ]]; then
    exit 0
fi
source /infmax-workspace/benchmarks/check_env.sh
check_env_vars K31_RUNTIME_ROLE
[[ $K31_RUNTIME_ROLE == P || $K31_RUNTIME_ROLE == D ]]
export PATH=/opt/runtime-venv/bin:$PATH
/opt/runtime-venv/bin/python /opt/k31-unified/verify-image.py
overlay_b64=/infmax-workspace/benchmarks/multi_node/srt-slurm-recipes/configs/k31-umbp-de441-compatible.tar.gz.b64
overlay_archive=$(mktemp /tmp/k31-umbp-overlay-XXXXXX.tar.gz)
base64 --decode "$overlay_b64" > "$overlay_archive"
echo 'efea3aee31ec36af0aca87fc5d7473e2a14ff99110bfdd772bc89d7abb33a6fa  '"$overlay_archive" | sha256sum -c -
site_packages=$(/opt/runtime-venv/bin/python -c 'import site; print(site.getsitepackages()[0])')
tar -xzf "$overlay_archive" -C "$site_packages"
rm -f "$overlay_archive"
/opt/runtime-venv/bin/python - <<'PY'
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime import UMBPRuntimeFactory

if set(UMBPRuntimeFactory._builders) != {"embedded", "standalone"}:
    raise RuntimeError("Compatible UMBP overlay did not register both runtime modes")
print({"umbp_runtime_modes": sorted(UMBPRuntimeFactory._builders)})
PY
mkdir -p "$HOME"
source /infmax-workspace/benchmarks/multi_node/srt-slurm-recipes/configs/k31-unified-umbp-pool.sh
