#!/usr/bin/env bash
# Fire the canonical v4 comparison matrix for ONE SKU via `gh workflow run`, so every
# point carries GHA provenance (validity.provenance_complete=true ->
# publication_status=comparable-experimental) instead of ad-hoc SSH provenance.
#
# 9 dispatches -> 16 phase-split JSON results (phase=both fans out decode+prefill):
#   A both  bf16 normal layout-and-dispatch-v1      uniform
#   B both  fp8  normal layout-and-dispatch-v1      uniform
#   C both  bf16 normal cached-layout-comm-only-v1  uniform
#   D both  fp8  normal cached-layout-comm-only-v1  uniform
#   E decode bf16 ll    layout-and-dispatch-v1      uniform   (Hopper only; --ll)
#   F decode fp8  ll    layout-and-dispatch-v1      uniform   (Hopper only; --ll)
#   G both  bf16 normal layout-and-dispatch-v1      balanced
#   H both  bf16 normal layout-and-dispatch-v1      zipf
#   I both  bf16 normal layout-and-dispatch-v1      zipf  +eplb
# resource_mode + tokens_ladder are LEFT AT THE WORKFLOW DEFAULTS (normalized / phase
# default) to match the already-published H100 GHA set exactly. LL is decode-only and is
# fired ONLY with --ll (Hopper: H100/H200); Blackwell fabrics (B300/GB300) abort LL at
# runtime, so it is omitted there to keep the matrix free of expected-red runs.
#
# Usage:
#   _gha_matrix.sh --sku h200 --ll                 # Hopper: all 9
#   _gha_matrix.sh --sku b300                       # Blackwell: 7 (no LL)
#   _gha_matrix.sh --sku gb300 --nodes 1            # GB300 EP4 single tray: 7 (no LL)
#   _gha_matrix.sh --sku h200 --ll --dry            # print dispatches, fire nothing
set -euo pipefail
WF="collectivex-experimental.yml"
SKU=""; NODES=""; LL=0; REF="collectivex"; DRY=0; CANON=0; OFFICIAL=0
BENCH="deepep"; SLEEP="${CX_DISPATCH_SLEEP:-8}"
while [ $# -gt 0 ]; do
  case "$1" in
    --sku)     SKU="$2";   shift 2 ;;
    --nodes)   NODES="$2"; shift 2 ;;
    --ll)      LL=1;       shift   ;;
    --ref)     REF="$2";   shift 2 ;;
    --dry)     DRY=1;      shift   ;;
    --canonical) CANON=1;  shift   ;;   # thread canonical=true to every dispatch (official-grade)
    --official)  OFFICIAL=1; CANON=1; shift ;;  # fire ONLY the headline canonical config (the cohort)
    --bench)   BENCH="$2"; shift 2 ;;   # deepep (NVIDIA) | mori (AMD MI355X)
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[ -n "$SKU" ] || { echo "need --sku <gb200|b200-dgxc|mi355x|h100-dgxc|h200|b300|gb300>" >&2; exit 2; }
# MI355X is AMD -> mori; everything else here is NVIDIA -> deepep (unless --bench overrides).
[ "$SKU" = mi355x ] && BENCH="${BENCH/deepep/mori}"

N=0
fire() {  # phase dtype mode contract routing eplb(true|false)
  local args=( -f sku="$SKU" -f benchmark="$BENCH" -f phase="$1" -f dispatch_dtype="$2"
               -f mode="$3" -f contract="$4" -f routing="$5" )
  [ "$6" = true ]  && args+=( -f eplb=true )      # else omit -> workflow default false
  [ "$CANON" = 1 ] && args+=( -f canonical=true ) # official-grade canonical workload identity
  [ -n "$NODES" ]  && args+=( -f nodes="$NODES" )
  N=$((N+1))
  printf '[%d] sku=%s bench=%s phase=%-7s dtype=%-4s mode=%-6s contract=%-26s routing=%-9s eplb=%s canon=%s nodes=%s\n' \
    "$N" "$SKU" "$BENCH" "$1" "$2" "$3" "$4" "$5" "$6" "$CANON" "${NODES:-default}"
  [ "$DRY" = 1 ] && return 0
  gh workflow run "$WF" --ref "$REF" "${args[@]}"
  sleep "$SLEEP"   # stagger: ease the API and let each run claim a runner before the next
}

# --official: fire ONLY the cross-SKU/cross-vendor headline cohort config (canonical bf16 normal
# layout-and-dispatch uniform). This is the publication-'official' point (goal P1 DoD).
if [ "$OFFICIAL" = 1 ]; then
  fire both bf16 normal layout-and-dispatch-v1 uniform false
else
  # Headline (A-D)
  fire both   bf16 normal layout-and-dispatch-v1      uniform false
  fire both   fp8  normal layout-and-dispatch-v1      uniform false
  fire both   bf16 normal cached-layout-comm-only-v1  uniform false
  fire both   fp8  normal cached-layout-comm-only-v1  uniform false
  # Low-latency (E-F), decode-only, Hopper only
  if [ "$LL" = 1 ]; then
    fire decode bf16 ll layout-and-dispatch-v1 uniform false
    fire decode fp8  ll layout-and-dispatch-v1 uniform false
  fi
  # Routing (G-I)
  fire both bf16 normal layout-and-dispatch-v1 balanced false
  fire both bf16 normal layout-and-dispatch-v1 zipf     false
  fire both bf16 normal layout-and-dispatch-v1 zipf     true
fi

# NB: do NOT use ${DRY:+...} here — DRY=0 is a NON-EMPTY string, so :+ would expand
# on real dispatches too. Branch on the value explicitly.
verb="dispatched"; tail=""
if [ "$DRY" = 1 ]; then verb="would dispatch"; tail=" — DRY-RUN, nothing fired"; fi
echo "=== $verb $N runs for sku=$SKU (ref=$REF${NODES:+, nodes=$NODES})$tail ==="
