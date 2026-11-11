#!/usr/bin/env bash
set -eo pipefail
source /infmax-workspace/benchmarks/check_env.sh
check_env_vars DYNAMO_PYTHON DYNAMO_VERSION HOME
test -f /opt/k31-dynamo-provenance.json
"$DYNAMO_PYTHON" -c 'import os; from importlib.metadata import version; from dynamo.vllm.kv_connector_protocols import KV_CONNECTOR_PROTOCOLS; assert version("ai-dynamo") == os.environ["DYNAMO_VERSION"]; assert "MoRIIOConnector" in KV_CONNECTOR_PROTOCOLS; print("Baked Dynamo MoRIIO protocol available")'
mkdir -p "$HOME"
bash /configs/k31-umbp-highconc-pool.sh
