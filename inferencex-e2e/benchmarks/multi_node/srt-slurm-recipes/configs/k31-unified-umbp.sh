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
overlay_b64=/infmax-workspace/benchmarks/multi_node/srt-slurm-recipes/configs/k31-umbp-released-034694a-compatible.tar.gz.b64
overlay_archive=$(mktemp /tmp/k31-umbp-overlay-XXXXXX.tar.gz)
base64 --decode "$overlay_b64" > "$overlay_archive"
echo 'afdb596f82a17b4c0da7446355adf15f83e1044c05b39e55574cede5ca2357f6  '"$overlay_archive" | sha256sum -c -
site_packages=$(/opt/runtime-venv/bin/python -c 'import site; print(site.getsitepackages()[0])')
tar -xzf "$overlay_archive" -C "$site_packages"
rm -f "$overlay_archive"
echo '24ef494d8baa9b749e8fea893fb3dd5beae97ecf8a6a5c517c62a2a24be74571  '"$site_packages/vllm/distributed/kv_transfer/kv_connector/v1/umbp/scheduler.py" | sha256sum -c -
echo 'e42f2146273c4ab8c3d549b087008f7b03a0daaad9f2e647ada696d5154ac7ef  '"$site_packages/vllm/v1/core/kv_cache_planning.py" | sha256sum -c -
/opt/runtime-venv/bin/python - <<'PY'
from pathlib import Path
import site

from vllm.distributed.kv_transfer.kv_connector.v1.umbp.runtime import UMBPRuntimeFactory

if set(UMBPRuntimeFactory._builders) != {"embedded", "standalone"}:
    raise RuntimeError("Compatible UMBP overlay did not register both runtime modes")
planner = (
    Path(site.getsitepackages()[0]) / "vllm/v1/core/kv_cache_planning.py"
).read_text()
if "group.enable_kv_transfer = False" not in planner:
    raise RuntimeError("Released-draft KV-transfer exclusion is missing")
print({"umbp_runtime_modes": sorted(UMBPRuntimeFactory._builders)})
PY
mkdir -p "$HOME"
source /infmax-workspace/benchmarks/multi_node/srt-slurm-recipes/configs/k31-unified-umbp-pool.sh
