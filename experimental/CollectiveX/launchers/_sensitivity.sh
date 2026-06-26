#!/usr/bin/env bash
# Distribution-sensitivity driver (single-node torchrun). Runs the headline (uniform) + the
# stressor distributions at ANCHOR tokens only (not the full ladder), so tests/sensitivity.py can
# form distribution_sensitivity_ratio = p99_worst / p99_uniform per (sku,backend,phase). One
# torchrun per (phase, routing). BF16 / normal / layout-and-dispatch-v1 (the cross-vendor contract).
# Reusable across NVIDIA (deepep) + AMD (mori) via env, mirroring _routing_rerun.sh:
#   BACKEND(deepep|mori) NG RUNNER TOPO TRANSPORT ITERS/TRIALS/WARMUP  ADEC/APRE anchor ladders
#   ROUTINGS (override the distribution set)  PHASES (decode prefill)
set -uo pipefail
cd /cx 2>/dev/null || cd /ix/experimental/CollectiveX 2>/dev/null || { echo "no cx dir"; exit 2; }
mkdir -p results
NG="${NG:-8}"; RUNNER="${RUNNER:-x-8x}"; TOPO="${TOPO:-x}"; TRANSPORT="${TRANSPORT:-nvlink}"
BACKEND="${BACKEND:-deepep}"; WARMUP="${WARMUP:-32}"; ITERS="${ITERS:-200}"; TRIALS="${TRIALS:-3}"
ADEC="${ADEC:-1 8 32 128}"; APRE="${APRE:-128 512 2048}"; PHASES="${PHASES:-decode prefill}"
# headline=uniform; balanced-rank-local = min-comm best case; zipf-heavy/hotspot-single = worst.
# All are backend-agnostic (routing.py), so the same set applies to deepep + mori.
ROUTINGS="${ROUTINGS:-uniform balanced balanced-rank-local zipf zipf-heavy hotspot-single}"

run(){  # phase routing ladder
  local phase="$1" routing="$2" ladder="$3"
  # sens-<routing> tag so these anchor runs never overwrite the full-ladder headline/routing files;
  # sensitivity.py groups by config (reads shape.routing), not filename, and MERGES T points.
  local out="results/${RUNNER}_${BACKEND}_${phase}_bf16_normal_layout-and-dispatch-v1_sens-${routing}.json"
  echo "### sens $phase routing=$routing -> $out"
  # shellcheck disable=SC2086
  timeout -k 30 "${CX_RUN_TIMEOUT:-900}" torchrun --nproc_per_node="$NG" tests/run_ep.py --backend "$BACKEND" \
    --phase "$phase" --dispatch-dtype bf16 --mode normal --measurement-contract layout-and-dispatch-v1 \
    --routing "$routing" --resource-mode tuned --tokens-ladder "$ladder" \
    --warmup "$WARMUP" --iters "$ITERS" --trials "$TRIALS" \
    --runner "$RUNNER" --topology-class "$TOPO" --transport "$TRANSPORT" --out "$out" 2>&1 | tail -7
  echo "### rc=${PIPESTATUS[0]} -> $out"
}

for ph in $PHASES; do
  L="$ADEC"; [ "$ph" = prefill ] && L="$APRE"
  for r in $ROUTINGS; do run "$ph" "$r" "$L"; done
done

echo "=== SENSITIVITY RUNS DONE — summarize: python3 tests/sensitivity.py --results-dir results ==="
