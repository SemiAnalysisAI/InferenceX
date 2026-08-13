#!/usr/bin/env bash
#SBATCH --job-name=img-diff
#SBATCH --account=amd-aifw-aim
#SBATCH --qos=amd-aifw-aim-qos
#SBATCH --partition=amd-spur
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --time=01:30:00
#SBATCH --output=/home/jiacao/InferenceX/imgdiff-%j.out
#SBATCH --exclude=crsuse2-m2m-203,crsuse2-m2m-074,crsuse2-m2m-240,crsuse2-m2m-029,crsuse2-m2m-061,crsuse2-m2m-071
#
# Establish the provenance of the DSv4 measurement image against the intended
# CI base, so every delta can be attributed to an upstream PR (or flagged as
# having none) in a docs/waiver/<PR>.md table.
#
# BASE is the image the CI config would pin. NEW is the image every measured
# number in this campaign was produced on (see sbatch_bake_pr51473_image.sh:
# it is vendor base f2fbead + the two-file #51473 back-port).
#
# No GPU is requested: this only untars layers and runs diff. That also lets
# the job schedule immediately alongside the measurement jobs rather than
# queueing behind them.
#
# Run 6309 extracted 0 bytes from BASE: its tar wildcards assumed a layout
# ('*/site-packages/vllm/*.py', '*/vllm/vllm/*.py') that matches the vendor
# image's editable /src/vllm checkout but not the nightly's. Rather than guess
# again, this version extracts the whole filesystem and then locates each
# package by finding its __init__.py. Costs disk and a few minutes; removes the
# failure mode entirely.
set -uo pipefail

BASE="${BASE:-vllm/vllm-openai-rocm:nightly-b22afe45ac797ae58e67a7a3ad79ee5714024420}"
NEW="${NEW:-jiahcao/vllm-dsv4@sha256:5e44cbd690811bdf9129bd2f552f22241177c6ad0c51bd057afb817cca35e1e9}"
OUT="${OUT:-/home/jiacao/imgdiff-$(date -u +%Y%m%d)-full}"

hostname; date -u
echo "BASE : $BASE"
echo "NEW  : $NEW"
echo "OUT  : $OUT"
echo

fail() { echo "FATAL: $*" >&2; exit 1; }
docker info >/dev/null 2>&1 || fail "no docker access on $(hostname)"
rm -rf "$OUT"; mkdir -p "$OUT" || fail "cannot create $OUT"

# ---------------------------------------------------------------- pull
for img in "$BASE" "$NEW"; do
    for _try in 1 2 3; do
        docker image inspect "$img" >/dev/null 2>&1 && break
        echo "pulling $img (attempt $_try/3) ... $(date -u +%H:%M:%S)Z"
        docker pull "$img" || true
    done
    docker image inspect "$img" >/dev/null 2>&1 || fail "pull failed: $img"
done
echo "PULL OK $(date -u +%H:%M:%S)Z"

# ---------------------------------------------------------------- locate
# Ask each image where it actually imports vllm/aiter from, instead of
# assuming. --entrypoint /bin/bash is safe here because nothing imports the
# package (importing aiter runs rocminfo, which aborts without a GPU); this
# only reads importlib's spec origin.
locate_in() { # $1=image $2=pkg -> prints the in-container package dir
    docker run --rm --entrypoint /bin/bash "$1" -c \
        "python3 -c 'import importlib.util as u,os,sys
s=u.find_spec(\"$2\")
print(os.path.dirname(s.origin) if s and s.origin else \"\")' 2>/dev/null" 2>/dev/null | tr -d '\r' | tail -1
}

declare -A PKGDIR
for side in base new; do
    img="$BASE"; [ "$side" = "new" ] && img="$NEW"
    for pkg in vllm aiter; do
        d=$(locate_in "$img" "$pkg")
        PKGDIR[$side:$pkg]="$d"
        echo "  $side/$pkg -> ${d:-<not importable>}"
    done
done
echo

# ---------------------------------------------------------------- export
# Extract the located package dirs only. `docker export` streams the flattened
# container filesystem; the tar member paths are relative (no leading /), so
# strip the leading slash off the located path to build the match pattern.
extract() { # $1=image  $2=destdir  $3..=in-container abs dirs
    local img="$1" dst="$2"; shift 2
    local cid pats=()
    mkdir -p "$dst"
    for p in "$@"; do
        [ -n "$p" ] && pats+=("${p#/}/*") && pats+=("${p#/}")
    done
    [ ${#pats[@]} -eq 0 ] && { echo "  no patterns for $img"; return 1; }
    cid=$(docker create "$img") || fail "docker create failed for $img"
    docker export "$cid" | tar -x -C "$dst" --wildcards "${pats[@]}" 2>/dev/null
    docker rm -f "$cid" >/dev/null 2>&1
    return 0
}

echo "extracting BASE ... $(date -u +%H:%M:%S)Z"
extract "$BASE" "$OUT/base" "${PKGDIR[base:vllm]}" "${PKGDIR[base:aiter]}"
echo "  base tree: $(du -sh "$OUT/base" 2>/dev/null | cut -f1)"
echo "extracting NEW  ... $(date -u +%H:%M:%S)Z"
extract "$NEW" "$OUT/new" "${PKGDIR[new:vllm]}" "${PKGDIR[new:aiter]}"
echo "  new tree:  $(du -sh "$OUT/new" 2>/dev/null | cut -f1)"

# Resolve back to the extracted-on-disk package root. The in-container path is
# reproduced verbatim under $dst, so join them -- but fall back to a search in
# case the layout surprises us again.
onbox() { # $1=side $2=pkg
    local d="$OUT/$1/${PKGDIR[$1:$2]#/}"
    [ -d "$d" ] && { echo "$d"; return; }
    find "$OUT/$1" -type f -path "*/$2/__init__.py" 2>/dev/null \
        | awk '{print length"\t"$0}' | sort -n | head -1 | cut -f2- | xargs -r dirname
}

# ---------------------------------------------------------------- diff
for pkg in vllm aiter; do
    b=$(onbox base "$pkg"); n=$(onbox new "$pkg")
    echo
    echo "=============== $pkg ==============="
    echo "base: ${b:-<absent>}"
    echo "new : ${n:-<absent>}"
    if [ -z "$b" ] || [ -z "$n" ] || [ ! -d "$b" ] || [ ! -d "$n" ]; then
        echo "SKIP: $pkg missing on one side"
        continue
    fi
    # Compare .py only: the compiled extensions and .co kernels are opaque to
    # diff, and __pycache__ differs on mtime alone.
    diff -ruN -x '__pycache__' -x '*.pyc' -x '*.so' "$b" "$n" > "$OUT/$pkg.diff" 2>/dev/null
    nfiles=$(grep -c '^diff -ruN' "$OUT/$pkg.diff" 2>/dev/null || echo 0)
    echo "changed files: $nfiles"
    echo "diff bytes   : $(stat -c%s "$OUT/$pkg.diff" 2>/dev/null || echo 0)"
    echo
    echo "--- per-file churn (+added -removed) ---"
    awk -v base="$b" -v new="$n" '
        /^diff -ruN/ {
            if (f != "") printf "  %+7d %-7d %s\n", a, -r, f
            f=$NF; sub(new"/","",f); a=0; r=0; next
        }
        /^\+/ && !/^\+\+\+/ { a++ }
        /^-/  && !/^---/    { r++ }
        END { if (f != "") printf "  %+7d %-7d %s\n", a, -r, f }
    ' "$OUT/$pkg.diff" | sort -k3
done

# ---------------------------------------------------------------- surfaces
# The knobs the CI cannot reach are engine-side features, not recipe flags.
# Whether BASE honours them at all is the question the grid depends on.
echo
echo "=============== feature surface: BASE vs NEW ==============="
probe() { # $1=label $2=pattern
    echo "$1:"
    for side in base new; do
        hits=$(grep -rl -- "$2" "$OUT/$side" --include='*.py' 2>/dev/null | wc -l)
        printf "  %-5s %s file(s)\n" "$side" "$hits"
    done
}
probe "VLLM_ROCM_DSV4_SPARSE_GLUON"            "VLLM_ROCM_DSV4_SPARSE_GLUON"
probe "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS" "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS"
probe "flydsl_mega_moe (literal)"              "flydsl_mega_moe"
probe "mega_moe (any)"                         "mega_moe"

echo
echo "=============== accepted --moe-backend literals ==============="
for side in base new; do
    echo "-- $side"
    grep -rhoE '"[a-z0-9_]*mega_moe[a-z0-9_]*"' "$OUT/$side" --include='*.py' 2>/dev/null \
        | sort -u | sed 's/^/     /'
done

# The #51473 marker: the one line the whole back-port reduces to.
echo
echo "=============== vLLM #51473 marker (384-wide TP8 shard) ==============="
for side in base new; do
    if grep -rq 'AITER_MXFP4_BF16 and activation == MoEActivation.SILU' "$OUT/$side" 2>/dev/null; then
        echo "  $side: PRESENT"
    else
        echo "  $side: ABSENT"
    fi
done

echo
echo "DONE $(date -u +%H:%M:%S)Z"
echo "Diffs written under: $OUT"
