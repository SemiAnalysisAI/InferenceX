#!/usr/bin/env bash
# Run one pool's CollectiveX cases by hand, from that pool's Slurm login node.
#
# The pool's cases resolve into shards (one backend, mode, precision and node count each; see
# sweep_matrix.py), and each shard gets its own allocation through the pool's launcher. Results
# land in collectivex/results/<sku>/<case_id>.json, replacing older results for the same cases;
# review them with `git diff` and commit them in a PR. Run it detached on long sweeps:
#   setsid nohup collectivex/run.sh h200-dgxc > run-h200.log 2>&1 < /dev/null &
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

usage() {
  cat <<'EOF'
usage: collectivex/run.sh <sku> [options]
  --suite LIST         ep, swap-blocks or both (default: ep,swap-blocks)
  --backend NAME       one EP backend (default: all)
  --ep LIST            EP sizes, e.g. 8,16 (default: all)
  --modes LIST         normal, low-latency (default: all)
  --precisions LIST    bf16, fp8 (default: all the backend declares)
  --swap-profile NAME  configs/swap_sweep.json profile (default: smoke)
  --shard ID           run only this shard (see --list-shards); repeatable
  --list-shards        print the shards and exit
  --dry-run            print what each shard would run and exit
  --allow-dirty        run despite uncommitted collectivex/ edits (results are marked
                       and cannot be published)
EOF
}

SKU="${1:-}"
case "$SKU" in ""|-*) usage >&2; exit 2 ;; esac
shift
matrix_args=(--sku "$SKU" --suites ep,swap-blocks)
shards=(); list=0; dry=0; allow_dirty=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --suite) matrix_args+=(--suites "$2"); shift 2 ;;
    --backend) matrix_args+=(--backend "$2"); shift 2 ;;
    --ep) matrix_args+=(--ep-sizes "$2"); shift 2 ;;
    --modes) matrix_args+=(--modes "$2"); shift 2 ;;
    --precisions) matrix_args+=(--precisions "$2"); shift 2 ;;
    --swap-profile) matrix_args+=(--swap-profile "$2"); shift 2 ;;
    --shard) shards+=("$2"); shift 2 ;;
    --list-shards) list=1; shift ;;
    --dry-run) dry=1; shift ;;
    --allow-dirty) allow_dirty=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# Results record the commit they were measured at; uncommitted edits to the benchmark (anything
# under collectivex/ except results/) would make that a lie, so they block the run by default.
source_sha="$(git -C "$REPO_ROOT" rev-parse HEAD)"
dirty="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no -- \
  collectivex ':(exclude)collectivex/results')"
source_dirty=0
if [ -n "$dirty" ]; then
  source_dirty=1
  if [ "$allow_dirty" = 0 ] && [ "$list" = 0 ] && [ "$dry" = 0 ]; then
    echo "collectivex/ has uncommitted edits; commit them or pass --allow-dirty:" >&2
    echo "$dirty" >&2
    exit 1
  fi
fi

mkdir -p "$HERE/.shards"
matrix="$HERE/.shards/matrix-$SKU.json"
python3 "$HERE/sweep_matrix.py" "${matrix_args[@]}" --out "$matrix" >/dev/null

# One tab-separated line per selected shard:
# id launcher backend nodes gpus_per_node scale_up_domain cases image staged_image_dir minutes
rows=()
while IFS= read -r row; do rows+=("$row"); done < <(python3 - "$matrix" "${shards[@]}" <<'EOF'
import json, sys
include = json.load(open(sys.argv[1]))["include"]
wanted = set(sys.argv[2:])
unknown = wanted - {shard["id"] for shard in include}
if unknown:
    sys.exit(f"unknown shard(s): {', '.join(sorted(unknown))}")
for shard in include:
    if wanted and shard["id"] not in wanted:
        continue
    print("\t".join(str(field) for field in (
        shard["id"], shard["launcher"], shard["backend"], shard["nodes"],
        shard["gpus_per_node"], shard["scale_up_domain"], len(shard["cases"]),
        shard.get("image") or "-", shard.get("staged_image_dir") or "-",
        shard.get("allocation_minutes") or "-")))
EOF
)
[ "${#rows[@]}" -gt 0 ] || { echo "no runnable shards for $SKU" >&2; exit 1; }

if [ "$list" = 1 ]; then
  printf '%-34s %-14s %-8s %s\n' SHARD BACKEND NODES CASES
  for row in "${rows[@]}"; do
    IFS=$'\t' read -r id _ backend nodes gpn _ ncases _ <<< "$row"
    printf '%-34s %-14s %-8s %s\n' "$id" "$backend" "${nodes}x${gpn}" "$ncases"
  done
  exit 0
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
rcs=()
for row in "${rows[@]}"; do
  IFS=$'\t' read -r id launcher backend nodes gpn domain ncases image staged minutes <<< "$row"
  control="$HERE/.shards/$id.json"
  python3 "$HERE/sweep_matrix.py" --extract-from "$matrix" --shard-id "$id" --out "$control" \
    >/dev/null 2>&1
  # The pid keeps two runs of one shard started in the same second (graphed and eager, say) out
  # of each other's stage directory.
  execution="manual-$stamp-$$-$id"
  echo "=== shard $id: $backend, ${nodes}x${gpn} GPUs, $ncases case(s)"
  # A subshell keeps one shard's image and placement from leaking into the next.
  (
    unset COLLX_NGPUS
    export COLLX_SHARD_FILE="$control" COLLX_SHARD_SKU="$SKU" COLLX_BENCH="$backend"
    export COLLX_NODES="$nodes" COLLX_GPUS_PER_NODE="$gpn" COLLX_SCALE_UP_DOMAIN="$domain"
    export COLLECTIVEX_EXECUTION_ID="$execution"
    export COLLX_LOG_DIR="/tmp/inferencex-collectivex-$(id -u)/logs/$execution"
    export COLLECTIVEX_SOURCE_SHA="$source_sha" COLLECTIVEX_SOURCE_DIRTY="$source_dirty"
    export CX_FP8_CONSUME="${CX_FP8_CONSUME:-native}"
    export COLLX_IMAGE_REFRESH="${COLLX_IMAGE_REFRESH:-0}"
    # A suite that ships its own image (swap-blocks) names it on the shard; otherwise the pool's.
    unset COLLX_IMAGE_OVERRIDE
    [ "$image" = - ] || export COLLX_IMAGE_OVERRIDE="$image"
    [ "$staged" = - ] || export COLLX_STAGED_IMAGE_DIR="$staged"
    # The caller's COLLX_TIME wins: a first-time image import can outlast a short shard's ask.
    [ -n "${COLLX_TIME:-}" ] || { [ "$minutes" = - ] && COLLX_TIME=300 || COLLX_TIME="$minutes"; }
    export COLLX_TIME
    if [ "$dry" = 1 ]; then
      env | grep -E '^(COLLX_|COLLECTIVEX_|CX_)' | sort
      echo "would run: bash collectivex/launchers/launch_$launcher.sh"
      exit 0
    fi
    echo "logs: $COLLX_LOG_DIR"
    bash "$HERE/launchers/launch_$launcher.sh"
  ) && rc=0 || rc=$?
  rcs+=("$rc")
done
[ "$dry" = 0 ] || exit 0

echo "=== summary for $SKU (source $source_sha, dirty=$source_dirty)"
failed=0
for i in "${!rows[@]}"; do
  printf '%-34s rc=%s\n' "${rows[$i]%%$'\t'*}" "${rcs[$i]}"
  [ "${rcs[$i]}" = 0 ] || failed=$((failed + 1))
done
python3 "$HERE/summarize.py" --results-dir "$HERE/results/$SKU" || true
git -C "$REPO_ROOT" status --short -- "collectivex/results/$SKU"
[ "$failed" = 0 ] || { echo "$failed shard(s) failed; rerun them with --shard" >&2; exit 1; }
