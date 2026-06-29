#!/usr/bin/env bash
# v4 full re-run for one (single-node) SKU under one allocation: the headline matrix
# (_v3_rerun.sh: bf16/fp8 x normal{layout,cached}/LL, decode+prefill) followed by the routing
# sweep (_routing_rerun.sh: balanced/zipf/zipf+eplb). Both invoke the CURRENT v4 harness, so
# every JSON carries publication_status/validity/measured-roundtrip — overwriting the legacy v3
# files of the same name. Env (RUNNER/TOPO/TRANSPORT/DEC/PRE/DO_LL/DO_EPLB/ITERS/TRIALS/WARMUP)
# is provided by _singlenode_orchestrate.sh.
set -uo pipefail
echo "=== V4 HEADLINE (_v3_rerun.sh) ==="
bash /cx/launchers/_v3_rerun.sh || echo "WARN headline returned nonzero"
echo "=== V4 ROUTING (_routing_rerun.sh) ==="
bash /cx/launchers/_routing_rerun.sh || echo "WARN routing returned nonzero"
echo "=== V4 ALL DONE ==="
