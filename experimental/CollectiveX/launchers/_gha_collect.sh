#!/usr/bin/env bash
# Collect CollectiveX GHA result artifacts into results/ so the plot is built from
# provenance-complete (GHA) JSONs. Optionally archive the superseded SSH-provenance
# NVIDIA results aside, since plot_ep.py does NOT dedup: two files for the same
# SKU+config (one SSH runner name, one GHA) would draw as colliding series.
#
# Usage:
#   _gha_collect.sh --since 2026-06-26T06:00:00Z         # all successful dispatch runs since ts
#   _gha_collect.sh --runs "281.. 282.."                 # explicit run ids
#   _gha_collect.sh --since <ts> --archive-ssh            # also move {h100,h200,b300,gb300}-8x_*
#                                                          # SSH results -> results/_ssh_v4_archive/
# Keeps mi355x-8x_* (the SSH AMD cross-vendor point, no GHA runner this round).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; CXDIR="$(cd "$HERE/.." && pwd)"
WF="collectivex-experimental.yml"; RESULTS="$CXDIR/results"
SINCE=""; RUNS=""; ARCHIVE=0
while [ $# -gt 0 ]; do case "$1" in
  --since) SINCE="$2"; shift 2;;
  --runs)  RUNS="$2";  shift 2;;
  --archive-ssh) ARCHIVE=1; shift;;
  *) echo "unknown arg: $1" >&2; exit 2;;
esac; done

if [ -z "$RUNS" ]; then
  [ -n "$SINCE" ] || { echo "need --since <ISO8601> or --runs <ids>" >&2; exit 2; }
  RUNS="$(gh run list --workflow="$WF" -L "${CX_COLLECT_LIMIT:-500}" \
            --json databaseId,event,conclusion,createdAt \
            --jq "[.[] | select(.event==\"workflow_dispatch\" and .conclusion==\"success\" and .createdAt>=\"$SINCE\")] | .[].databaseId" )"
fi
[ -n "$RUNS" ] || { echo "no successful runs matched" >&2; exit 1; }

if [ "$ARCHIVE" = 1 ]; then
  arch="$RESULTS/_ssh_v4_archive"; mkdir -p "$arch"
  n=0; for f in "$RESULTS"/h100-8x_*.json "$RESULTS"/h200-8x_*.json \
                "$RESULTS"/b300-8x_*.json "$RESULTS"/gb300-8x_*.json; do
    [ -e "$f" ] || continue; mv "$f" "$arch/"; n=$((n+1))
  done
  echo "archived $n SSH-provenance NVIDIA result(s) -> $arch (mi355x-8x kept)"
fi

tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
got=0
for rid in $RUNS; do
  if gh run download "$rid" --dir "$tmp/$rid" >/dev/null 2>&1; then
    # copy the EP result + env JSONs + the NCCL collective op results (family=nccl,
    # named <runner>_<op>_<ts>.json); artifact dirs may nest per phase
    while IFS= read -r f; do cp -f "$f" "$RESULTS/" && got=$((got+1)); done \
      < <(find "$tmp/$rid" \( -name '*deepep*.json' -o -name '*mori*.json' -o -name '*uccl*.json' \
            -o -name '*flashinfer*.json' -o -name 'env_*.json' \
            -o -name '*_all_reduce_*.json' -o -name '*_all_gather_*.json' \
            -o -name '*_reduce_scatter_*.json' -o -name '*_alltoall_*.json' \) -print)
  else
    echo "WARN: download failed for run $rid" >&2
  fi
done
echo "copied $got JSON file(s) from $(echo "$RUNS" | wc -w | tr -d ' ') run(s) -> $RESULTS"

# Per-SKU/provenance tally of what's now in results/ (deepep+mori only).
python3 - "$RESULTS" <<'PY'
import json,glob,os,sys,collections
rd=sys.argv[1]; t=collections.Counter()
for f in glob.glob(os.path.join(rd,"*.json")):
    b=os.path.basename(f)
    if "deepep" not in b and "mori" not in b: continue
    try: d=json.load(open(f))
    except Exception: continue
    sku=(d.get("runner") or "?").split("_")[0].split("-")[0]
    prov="prov-complete" if (d.get("validity") or {}).get("provenance_complete") else "ssh"
    t[(sku,prov,d.get("publication_status","?"))]+=1
for k in sorted(t): print(f"  {k[0]:8s} {k[1]:14s} {k[2]:24s} x{t[k]}")
PY
