#!/usr/bin/env bash
# X1-LR startup diagnostic v3.1 (experiment branch only, not a benchmark): install the recipe's Mooncake, then run
# x1lr_rail_probe.py: per-rank EFA registration at the recipe's aggregate size, the InfiniBand HCAs and TCP, each
# with a cross-process store round trip from host and GPU memory. Stops before the engine starts. User-space verbs
# and Mooncake store calls only; no driver, firmware or system setting is changed.
# Writes /logs/x1lr_rail_probe.txt and /logs/x1lr_rail_probe.json.
set -o pipefail
here="$(dirname "${BASH_SOURCE[0]}")"
out=/logs/x1lr_rail_probe.txt
expected='{"vllm_3696c77_store_worker_sha256":"5e4f932c53045bbd871c8ec9024234183fd4d677a0bec4ba4ea96298c8910676","mooncake_0.3.11.post1_cp312_store_so_sha256":"5f4be8d7e080b7f4578e3223b2c5ea43635d9584683dd9d41c1b89665a080d13","mooncake_0.3.11.post1_cp312_engine_so_sha256":"65c74ffd50b0c9774a4ac397c9e1d475236e9d56edc97b028e58cc553d17008b"}'
{
    echo "== host"; hostname; date -u +%FT%TZ
    # The binder exports the master config's kv-offload-backend version only to setup scripts it knows by name;
    # this renamed probe sets the same value (configs/nvidia-master.yaml, this key) so the probe runs the recipe's
    # Mooncake and not the image's preinstalled build.
    export KV_OFFLOAD_BACKEND_VERSION="${KV_OFFLOAD_BACKEND_VERSION:-0.3.11.post1}"
    if bash "$here/vllm-mooncake.sh"; then echo "mooncake install: ok"; else echo "mooncake install: FAILED rc=$?"; fi
    python3 -m pip show mooncake-transfer-engine-cuda13 2>/dev/null | head -2
    python3 "$here/x1lr_rail_probe.py" --out /logs/x1lr_rail_probe.json --expected "$expected" --phases "${X1LR_RAIL_PHASES:-ib_one,efa_small_per_rank,efa_small_one}"
    echo "probe rc=$?"
    date -u +%FT%TZ
} > "$out" 2>&1
cat "$out"
echo "== x1lr_rail_probe.json"
cat /logs/x1lr_rail_probe.json 2>/dev/null
echo
echo "x1lr diagnostic: stopping before the engine starts (startup diagnostic, not a benchmark result)" >&2
exit 97
