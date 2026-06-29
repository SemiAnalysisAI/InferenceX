#!/usr/bin/env bash
# Dispatch EVERY resolved case of a named suite via GitHub Actions (so all runs are GHA, not SSH).
# Resolves the suite with generate_matrix.py, DROPS gb300 (compute unavailable — capacity-queued),
# maps each case to a `gh workflow run` with the right -f flags (model dims from workloads.yaml,
# canonical=true, all distribution/contract/resource axes), and dedups identical dispatches.
#
# SKU guards: mi355x/MoRI is bf16/normal/layout-only + wedges at T>=32 (validated envelope), so its
# cases are capped to decode, ladder "1 2 4 8 16", resource_mode=tuned (official, not floored).
#
#   _gha_suite.sh --suite ep-nightly-v1            # fire all non-gb300 cases
#   _gha_suite.sh --suite ep-nightly-v1 --dry      # print the dispatch plan, fire nothing
#   _gha_suite.sh --all --dry                      # plan for every suite
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; CXDIR="$(cd "$HERE/.." && pwd)"
WF="collectivex-experimental.yml"; REF="${CX_REF:-collectivex}"; DRY=0; SUITE=""; ALL=0; ONLYSKU=""
V2=0; BACKEND_OVERRIDE=""   # full-parity knobs (see below)
SLEEP="${CX_DISPATCH_SLEEP:-6}"
# --deepep-v2     : add -f deepep_v2=true to every deepep dispatch (kernel_gen=v2 from-source build).
# --backend NAME  : remap the suite's `deepep` cases onto NAME (uccl|flashinfer|deepep-hybrid|nccl-ep)
#                   so the full V1 matrix runs for that library too; capability-invalid cases are
#                   pre-filtered (so we never fire a dispatch the Validate-capability step would reject).
while [ $# -gt 0 ]; do case "$1" in
  --suite) SUITE="$2"; shift 2;; --all) ALL=1; shift;; --dry) DRY=1; shift;;
  --only-sku) ONLYSKU="$2"; shift 2;;   # dispatch only this SKU's cases (e.g. backfill one chip)
  --deepep-v2) V2=1; shift;;
  --backend) BACKEND_OVERRIDE="$2"; shift 2;;
  --ref) REF="$2"; shift 2;; *) echo "unknown arg: $1" >&2; exit 2;; esac; done

suites_list() { python3 -c "import yaml;print(' '.join(yaml.safe_load(open('$CXDIR/configs/suites.yaml'))['suites']))"; }
[ "$ALL" = 1 ] && SUITES="$(suites_list)" || SUITES="$SUITE"
[ -n "$SUITES" ] || { echo "need --suite <name> or --all" >&2; exit 2; }

# Resolve one suite -> pipe-separated dispatch tuples (one per UNIQUE workflow_dispatch input set).
emit_tuples() {  # suite
  CX_ONLYSKU="$ONLYSKU" CX_BACKEND_OVERRIDE="$BACKEND_OVERRIDE" python3 - "$1" "$CXDIR" <<'PY'
import sys, os, json, subprocess
suite, cxdir = sys.argv[1], sys.argv[2]
import yaml
wl_cfg = yaml.safe_load(open(os.path.join(cxdir, "configs", "workloads.yaml")))
suites = yaml.safe_load(open(os.path.join(cxdir, "configs", "suites.yaml")))["suites"]
s = suites[suite]
# workload name -> (hidden, topk, experts); ds-like-ref/synthetic -> defaults (blank).
def dims(name):
    for sec in ("synthetic", "model_derived"):
        m = (wl_cfg.get(sec) or {}).get(name)
        if m:
            e = m.get("experts", m.get("routed_experts"))
            return m.get("hidden"), m.get("topk"), e
    return None, None, None
# resolve the matrix (stdlib + the repo's generate_matrix)
sys.path.insert(0, cxdir)
import generate_matrix as gm
m = gm.generate(suite)
SKU = {"h100": "h100-dgxc", "h200": "h200", "b300": "b300", "mi355x": "mi355x", "gb300": "gb300"}
def ladder(phase):
    if phase == "decode" and s.get("token_points_decode"): return " ".join(map(str, s["token_points_decode"]))
    if phase == "prefill" and s.get("token_points_prefill"): return " ".join(map(str, s["token_points_prefill"]))
    if s.get("token_points"): return " ".join(map(str, s["token_points"]))
    return ""
seen = set(); out = []
for c in m["cases"]:
    plat = c["platform"]
    if plat == "gb300":      # compute unavailable (capacity) — skipped per directive
        continue
    beng = c["backend"]
    if beng not in ("deepep", "mori"):   # collectives aren't EP suites
        continue
    # --backend override: remap the deepep matrix onto another NVIDIA EP library (mori stays AMD).
    ov = os.environ.get("CX_BACKEND_OVERRIDE", "")
    if ov and beng == "deepep":
        beng = ov
    # capability pre-filter: skip cases the target backend can't run (e.g. flashinfer has no LL,
    # deepep-hybrid is bf16/normal/layout only) so we never fire a doomed dispatch.
    try:
        if os.path.join(cxdir, "tests") not in sys.path:
            sys.path.insert(0, os.path.join(cxdir, "tests"))
        import capability as _cap
        _ok, _r = _cap.resolve(plat, beng, mode=c["mode"], dtype=c["dtype"], contract=c["contract"],
                               routing=c["routing"], eplb=bool(c.get("eplb")),
                               activation_profile=c.get("activation_profile", "normal"))
        if not _ok:
            continue
    except Exception:
        pass
    sku = SKU.get(plat, plat)
    only = os.environ.get("CX_ONLYSKU", "")
    if only and sku != only:
        continue                     # --only-sku: backfill just one chip
    h, t, e = dims(c["workload"])
    hidden = "" if (h in (None, 7168)) else str(h)
    topk = "" if (t in (None, 8)) else str(t)
    experts = "" if (e in (None, 256)) else str(e)
    phase = c["phase"]; rmode = c["resource_mode"]; lad = ladder(phase)
    # MoRI envelope guard: bf16/normal/layout only, decode-safe, wedges T>=32, tuned=official.
    if sku == "mi355x":
        if phase == "prefill":      # MoRI wedges on the prefill ladder — skip
            continue
        lad = "1 2 4 8 16"; rmode = "tuned"
    tup = (sku, beng, phase, c["dtype"], c["mode"], c["contract"], c["routing"],
           "true" if c.get("eplb") else "", rmode, c.get("activation_profile", "normal"),
           c.get("placement", "packed"), str(c.get("routing_step", 0)),
           c.get("uneven_tokens", "none"), hidden, topk, experts, lad)
    if tup in seen:
        continue
    seen.add(tup)
    out.append("|".join(tup))
print("\n".join(out))
PY
}

N=0
fire_tuple() {  # pipe-separated tuple
  IFS='|' read -r sku beng phase dtype mode contract routing eplb rmode act placement rstep uneven hidden topk experts lad <<<"$1"
  local a=( -f sku="$sku" -f benchmark="$beng" -f phase="$phase" -f dispatch_dtype="$dtype"
            -f mode="$mode" -f contract="$contract" -f routing="$routing" -f resource_mode="$rmode"
            -f activation_profile="$act" -f placement="$placement" -f uneven_tokens="$uneven" )
  # canonical workload requires a fixed serialized trace: incompatible with uneven allocation
  # (variable per-rank gt) AND with routing_step != 0 (make_workloads has no step-specific trace).
  # Those diagnostic suites run seeded-runtime (comparable-experimental).
  [ "$uneven" = none ] && [ "$rstep" = 0 ] && a+=( -f canonical=true )
  [ "$V2" = 1 ] && a+=( -f deepep_v2=true )   # DeepEP V2 from-source build (kernel_gen=v2)
  [ "$eplb" = true ] && a+=( -f eplb=true )
  [ "$rstep" != 0 ] && a+=( -f routing_step="$rstep" )
  [ -n "$hidden" ]   && a+=( -f hidden="$hidden" )
  [ -n "$topk" ]     && a+=( -f topk="$topk" )
  [ -n "$experts" ]  && a+=( -f experts="$experts" )
  [ -n "$lad" ]      && a+=( -f tokens_ladder="$lad" )
  N=$((N+1))
  printf '[%d] %s/%s %s %s/%s/%s rt=%s eplb=%s rmode=%s act=%s plc=%s step=%s un=%s dims=%s/%s/%s lad=[%s]\n' \
    "$N" "$sku" "$beng" "$phase" "$dtype" "$mode" "${contract/-v1/}" "$routing" "${eplb:-f}" "$rmode" \
    "$act" "$placement" "$rstep" "$uneven" "${hidden:-d}" "${topk:-d}" "${experts:-d}" "$lad"
  [ "$DRY" = 1 ] && return 0
  gh workflow run "$WF" --ref "$REF" "${a[@]}" >/dev/null 2>&1 || echo "    WARN: dispatch failed"
  sleep "$SLEEP"
}

# Gather every suite's tuples, then DEDUP GLOBALLY (a config shared by several suites fires once —
# still covers every suite, without wasteful exact-duplicate dispatches). Preserves first-seen order.
allf="$(mktemp)"; trap 'rm -f "$allf"' EXIT
for suite in $SUITES; do
  t="$(emit_tuples "$suite")"
  cnt=0; [ -n "$t" ] && cnt=$(printf '%s\n' "$t" | grep -c .)
  echo "=== suite $suite: $cnt case(s) ==="
  [ -n "$t" ] && printf '%s\n' "$t" >> "$allf"
done
# dedup, keep first-seen order (portable; macOS bash 3.2 has no mapfile)
uniqf="$(mktemp)"; trap 'rm -f "$allf" "$uniqf"' EXIT
awk 'NF && !seen[$0]++' "$allf" > "$uniqf"
echo "=== $(grep -c . "$uniqf") unique config(s) after cross-suite dedup ==="
while IFS= read -r tup; do [ -n "$tup" ] && fire_tuple "$tup"; done < "$uniqf"
verb="dispatched"; [ "$DRY" = 1 ] && verb="WOULD dispatch (dry-run)"
echo "=== $verb $N unique GHA run(s) across suites: $SUITES ==="
