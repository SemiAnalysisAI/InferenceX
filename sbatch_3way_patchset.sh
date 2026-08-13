#!/usr/bin/env bash
#SBATCH --job-name=3way-patch
#SBATCH --account=amd-aifw-aim
#SBATCH --qos=amd-aifw-aim-qos
#SBATCH --partition=amd-spur
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --time=02:00:00
#SBATCH --output=/home/jiacao/InferenceX/3way-%j.out
#SBATCH --exclude=crsuse2-m2m-203,crsuse2-m2m-074,crsuse2-m2m-240,crsuse2-m2m-029,crsuse2-m2m-061,crsuse2-m2m-071
#
# Extract the *vendor patch set* out of the DSv4 measurement image and port it
# onto the intended CI baseline.
#
# A straight VENDOR-vs-TARGET diff (job 6316) is useless for this: it returned
# 769 vllm files / 4.8 MB, because TARGET is 454 upstream commits AHEAD of the
# vendor image's base, so the diff is dominated by upstream drift running the
# wrong direction. The vendor's own changes are buried in it.
#
# Three-way subtraction fixes that. REF is a public ROCm nightly built from
# 6f91edf9 (2026-07-29), only 11 commits behind the vendor image's own base
# 02e63f2e4 (2026-07-30). So:
#
#     REF ---- 11 commits ----> vendor base 02e63f2e4  --(vendor patches)--> VENDOR
#     REF ---- 465 commits ---> TARGET b22afe45
#
#   diff(REF, VENDOR)  = 11 commits of drift + THE VENDOR PATCH SET   <-- small
#   diff(REF, TARGET)  = 465 commits of upstream                      <-- large
#
# The first is the deliverable. Anything it touches that the second also
# touches is a conflict that needs a human decision; anything it touches that
# the second does not is a clean port onto TARGET.
#
# Deliverables under $OUT:
#   vendor_patchset.{vllm,aiter}.diff  - REF -> VENDOR, the thing to port
#   upstream_drift.vllm.diff           - REF -> TARGET, for conflict detection
#   CONFLICTS.txt                      - files both sides touch
#   CLEAN.txt                          - files only the vendor touches
#   applied/                           - TARGET tree with the patch set applied
#   APPLY_LOG.txt                      - per-file apply result
set -uo pipefail

REF="${REF:-vllm/vllm-openai-rocm:nightly-6f91edf96d3f3272945809c04702380053bff4de}"
VENDOR="${VENDOR:-jiahcao/vllm-dsv4:dsv4-pr51473-20260812}"
TARGET="${TARGET:-vllm/vllm-openai-rocm:nightly-b22afe45ac797ae58e67a7a3ad79ee5714024420}"
OUT="${OUT:-/home/jiacao/3way-$(date -u +%Y%m%d-%H%M)}"

hostname; date -u
echo "REF    : $REF      (6f91edf9, 2026-07-29)"
echo "VENDOR : $VENDOR   (base 02e63f2e4, 2026-07-30)"
echo "TARGET : $TARGET   (b22afe45,  2026-08-10)"
echo "OUT    : $OUT"
echo

fail() { echo "FATAL: $*" >&2; exit 1; }
docker info >/dev/null 2>&1 || fail "no docker access on $(hostname)"
rm -rf "$OUT"; mkdir -p "$OUT" || fail "cannot create $OUT"

# ---------------------------------------------------------------- pull
for img in "$REF" "$VENDOR" "$TARGET"; do
    for _try in 1 2 3; do
        docker image inspect "$img" >/dev/null 2>&1 && break
        echo "pulling $img (attempt $_try/3) ... $(date -u +%H:%M:%S)Z"
        docker pull "$img" || true
    done
    docker image inspect "$img" >/dev/null 2>&1 || fail "pull failed: $img"
done
echo "PULL OK $(date -u +%H:%M:%S)Z"

# ---------------------------------------------------------------- locate
# Layouts differ between images: the nightlies install into dist-packages, the
# vendor image carries an editable /src/vllm checkout. Ask each image where it
# imports from rather than assuming (job 6309 extracted 0 bytes by assuming).
locate_in() { # $1=image $2=pkg
    docker run --rm --entrypoint /bin/bash "$1" -c \
        "python3 -c 'import importlib.util as u,os
s=u.find_spec(\"$2\")
print(os.path.dirname(s.origin) if s and s.origin else \"\")' 2>/dev/null" 2>/dev/null | tr -d '\r' | tail -1
}

declare -A P
for side in ref vendor target; do
    case $side in ref) img="$REF";; vendor) img="$VENDOR";; target) img="$TARGET";; esac
    for pkg in vllm aiter; do
        P[$side:$pkg]=$(locate_in "$img" "$pkg")
        echo "  $side/$pkg -> ${P[$side:$pkg]:-<not importable>}"
    done
done
echo

# ---------------------------------------------------------------- extract
extract() { # $1=image $2=destdir $3..=in-container dirs
    local img="$1" dst="$2"; shift 2
    local cid pats=()
    mkdir -p "$dst"
    for p in "$@"; do [ -n "$p" ] && pats+=("${p#/}/*"); done
    [ ${#pats[@]} -eq 0 ] && return 1
    cid=$(docker create "$img") || fail "docker create failed: $img"
    docker export "$cid" | tar -x -C "$dst" --wildcards "${pats[@]}" 2>/dev/null
    docker rm -f "$cid" >/dev/null 2>&1
}

for side in ref vendor target; do
    case $side in ref) img="$REF";; vendor) img="$VENDOR";; target) img="$TARGET";; esac
    echo "extracting $side ... $(date -u +%H:%M:%S)Z"
    extract "$img" "$OUT/$side" "${P[$side:vllm]}" "${P[$side:aiter]}"
    echo "  $(du -sh "$OUT/$side" 2>/dev/null | cut -f1)"
done

root() { # $1=side $2=pkg -> on-disk package root
    local d="$OUT/$1/${P[$1:$2]#/}"
    [ -d "$d" ] && { echo "$d"; return; }
    find "$OUT/$1" -type f -path "*/$2/__init__.py" 2>/dev/null \
        | awk '{print length"\t"$0}' | sort -n | head -1 | cut -f2- | xargs -r dirname
}

# ---------------------------------------------------------------- diff
# Normalise away the layout difference: diff the package dirs directly and
# rewrite the labels to bare a/<pkg> and b/<pkg> so the resulting patch applies
# with -p1 regardless of where each image kept the package.
DIFFOPTS=(-ruN -x '__pycache__' -x '*.pyc' -x '*.so' -x '*.pyd' -x '_version.py')

echo
for pkg in vllm aiter; do
    r=$(root ref "$pkg"); v=$(root vendor "$pkg"); t=$(root target "$pkg")
    echo "=============== $pkg ==============="
    echo "  ref   : ${r:-<absent>}"
    echo "  vendor: ${v:-<absent>}"
    echo "  target: ${t:-<absent>}"
    if [ -z "$r" ] || [ -z "$v" ] || [ -z "$t" ]; then
        echo "  SKIP: missing on one side"; continue
    fi

    # (1) THE DELIVERABLE: ref -> vendor. 11 commits of drift + vendor patches.
    ( cd "$(dirname "$r")" && diff "${DIFFOPTS[@]}" \
        --label "a/$pkg" --label "b/$pkg" "$pkg" "$v" ) \
        > "$OUT/vendor_patchset.$pkg.diff" 2>/dev/null
    # (2) ref -> target: 465 upstream commits, used only to find conflicts.
    ( cd "$(dirname "$r")" && diff "${DIFFOPTS[@]}" \
        --label "a/$pkg" --label "b/$pkg" "$pkg" "$t" ) \
        > "$OUT/upstream_drift.$pkg.diff" 2>/dev/null

    vs=$(stat -c%s "$OUT/vendor_patchset.$pkg.diff" 2>/dev/null || echo 0)
    us=$(stat -c%s "$OUT/upstream_drift.$pkg.diff" 2>/dev/null || echo 0)
    vf=$(grep -c '^diff ' "$OUT/vendor_patchset.$pkg.diff" 2>/dev/null || echo 0)
    uf=$(grep -c '^diff ' "$OUT/upstream_drift.$pkg.diff" 2>/dev/null || echo 0)
    echo "  vendor patch set : $vf files, $vs bytes   <-- to port"
    echo "  upstream drift   : $uf files, $us bytes"

    # Real line counts. (Job 6316's per-file awk under-reported removals; total
    # +/- over the whole file is the number to trust.)
    printf "  vendor patch set : +%s / -%s lines\n" \
        "$(grep -c '^+[^+]' "$OUT/vendor_patchset.$pkg.diff" 2>/dev/null || echo 0)" \
        "$(grep -c '^-[^-]' "$OUT/vendor_patchset.$pkg.diff" 2>/dev/null || echo 0)"

    # ---- conflict analysis -------------------------------------------------
    grep '^diff ' "$OUT/vendor_patchset.$pkg.diff" 2>/dev/null \
        | awk '{print $NF}' | sed "s|^$v/||;s|^$pkg/||" | sort -u > "$OUT/.v.$pkg"
    grep '^diff ' "$OUT/upstream_drift.$pkg.diff" 2>/dev/null \
        | awk '{print $NF}' | sed "s|^$t/||;s|^$pkg/||" | sort -u > "$OUT/.u.$pkg"
    comm -12 "$OUT/.v.$pkg" "$OUT/.u.$pkg" | sed "s|^|$pkg/|" >> "$OUT/CONFLICTS.txt"
    comm -23 "$OUT/.v.$pkg" "$OUT/.u.$pkg" | sed "s|^|$pkg/|" >> "$OUT/CLEAN.txt"
    echo "  files vendor-only (clean port): $(comm -23 "$OUT/.v.$pkg" "$OUT/.u.$pkg" | wc -l)"
    echo "  files touched by BOTH (conflict): $(comm -12 "$OUT/.v.$pkg" "$OUT/.u.$pkg" | wc -l)"
done

# ---------------------------------------------------------------- apply
# Apply the vendor patch set onto a copy of TARGET, file by file, so a failure
# names the file instead of aborting the whole port. git apply is tried first
# (exact), then patch --fuzz=3 (tolerant of the 11-commit drift), and finally
# a whole-file copy for files that exist only in the vendor image -- those are
# additions, not modifications, and are the bulk of the MegaMoE kernel tree.
echo
echo "=============== APPLY onto TARGET ==============="
: > "$OUT/APPLY_LOG.txt"
for pkg in vllm aiter; do
    v=$(root vendor "$pkg"); t=$(root target "$pkg")
    [ -z "$v" ] || [ -z "$t" ] && continue
    dst="$OUT/applied/$pkg"
    mkdir -p "$(dirname "$dst")"
    cp -a "$t" "$dst" || fail "cannot stage TARGET copy for $pkg"

    ok=0; fuzz=0; added=0; bad=0
    # Walk the vendor tree: every .py that differs from the staged target.
    while IFS= read -r f; do
        rel="${f#$v/}"
        if [ ! -e "$dst/$rel" ]; then
            mkdir -p "$(dirname "$dst/$rel")"
            cp -a "$f" "$dst/$rel" && { added=$((added+1)); \
                echo "ADD   $pkg/$rel" >> "$OUT/APPLY_LOG.txt"; } \
                || { bad=$((bad+1)); echo "FAIL-ADD $pkg/$rel" >> "$OUT/APPLY_LOG.txt"; }
            continue
        fi
        cmp -s "$f" "$dst/$rel" && continue
        # Modified on both sides: produce a patch from ref->vendor for just this
        # file and try to apply it to the target copy.
        r=$(root ref "$pkg")
        if [ -e "$r/$rel" ]; then
            diff -u --label "a/$rel" --label "b/$rel" "$r/$rel" "$f" > "$OUT/.one.diff" 2>/dev/null
            if patch -s -p1 -d "$dst" --fuzz=3 --forward --no-backup-if-mismatch \
                     -i "$OUT/.one.diff" >/dev/null 2>&1; then
                fuzz=$((fuzz+1)); echo "PATCH $pkg/$rel" >> "$OUT/APPLY_LOG.txt"
            else
                bad=$((bad+1)); echo "CONFLICT $pkg/$rel" >> "$OUT/APPLY_LOG.txt"
            fi
        else
            # No ref version: vendor rewrote a file that upstream also changed.
            bad=$((bad+1)); echo "CONFLICT-NOREF $pkg/$rel" >> "$OUT/APPLY_LOG.txt"
        fi
    done < <(find "$v" -type f -name '*.py' ! -path '*__pycache__*' 2>/dev/null)

    echo "  $pkg: added=$added patched=$fuzz conflicts=$bad"
done
rm -f "$OUT/.one.diff" "$OUT"/.v.* "$OUT"/.u.*

echo
echo "--- conflicts needing a decision ---"
grep -E '^(CONFLICT|FAIL)' "$OUT/APPLY_LOG.txt" 2>/dev/null | head -60
echo "  total: $(grep -cE '^(CONFLICT|FAIL)' "$OUT/APPLY_LOG.txt" 2>/dev/null || echo 0)"

# ---------------------------------------------------------------- verify
echo
echo "=============== feature surface after port ==============="
probe() { # $1=label $2=pattern
    printf "%-45s" "$1:"
    for side in "$OUT/target" "$OUT/applied"; do
        n=$(grep -rl -- "$2" "$side" --include='*.py' 2>/dev/null | wc -l)
        printf " %-4s" "$n"
    done
    echo "   (target -> applied)"
}
probe "VLLM_ROCM_DSV4_SPARSE_GLUON"  "VLLM_ROCM_DSV4_SPARSE_GLUON"
probe "flydsl_mega_moe"              "flydsl_mega_moe"
probe "mega_moe (any)"               "mega_moe"
probe "#51473 marker"                "AITER_MXFP4_BF16 and activation == MoEActivation.SILU"

echo
echo "=============== python syntax check on applied tree ==============="
# A port that lands syntactically-broken files would only surface 20 minutes
# into a measured cell. Parse every file the port touched.
bad=0
while IFS= read -r line; do
    rel=$(echo "$line" | awk '{print $2}')
    f="$OUT/applied/$rel"
    [ -f "$f" ] || continue
    python3 -c "import ast,sys; ast.parse(open(sys.argv[1],encoding='utf-8').read())" "$f" 2>/dev/null \
        || { echo "  SYNTAX FAIL: $rel"; bad=$((bad+1)); }
done < <(grep -E '^(ADD|PATCH)' "$OUT/APPLY_LOG.txt" 2>/dev/null)
echo "  syntax failures: $bad"

echo
echo "DONE $(date -u +%H:%M:%S)Z"
echo "Artifacts: $OUT"
echo "  vendor_patchset.{vllm,aiter}.diff  <- the port"
echo "  CONFLICTS.txt / CLEAN.txt / APPLY_LOG.txt"
echo "  applied/  <- TARGET + vendor patch set"
