#!/usr/bin/env bash
# The measurement image is 85.8% v0.1.19.post2, not main. If post2 already
# carries gluon / MegaMoE / FSE, then the whole "aiter is 107 commits behind"
# framing is wrong: the fix is a TAG BUMP in Dockerfile.rocm_base, not a patch.
set -u
W=/tmp/aitermain
cd "$W" || exit 1
git checkout -q v0.1.19.post2

echo "=== v0.1.19.post2 = $(git rev-parse --short HEAD) ==="
echo "  date: $(git log -1 --format=%ci)"
echo "  is v0.1.19 an ancestor? $(git merge-base --is-ancestor v0.1.19 HEAD && echo yes || echo no)"
echo "  commits v0.1.19..post2: $(git rev-list --count v0.1.19..HEAD)"

echo
echo "=== feature presence in v0.1.19.post2 ==="
p() { if [ -e "$W/$1" ]; then echo "  PRESENT  $2"; else echo "  MISSING  $2"; fi; }
m() { if grep -qF -- "$1" "$W/$2" 2>/dev/null; then echo "  PRESENT  $3"; else echo "  MISSING  $3"; fi; }

p "aiter/ops/triton/_gluon_kernels/gfx950/attention/pa_decode_sparse.py" "#4382 gluon kernel"
m "_pa_decode_sparse_gfx950_gluon" "aiter/ops/triton/attention/pa_decode_sparse.py" "#4382 gfx950 routing"
m "max_addressable_bytes" "aiter/ops/triton/utils/common_utils.py"        "#4673 overflow fix"
p "aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py" "#4439 MegaMoE"
p "aiter/fhmoe.py"                                    "#4269 FSE / FHMoE"
m "shared_expert_id" "aiter/fused_moe.py"             "#4269 fused_moe shared_expert_id"
p "aiter/configs/model_configs/dsv4_a8w8_blockscale_tuned_gemm.csv" "tuned-gemm csv"

echo
echo "=== which merged PRs are in post2 but not v0.1.19? ==="
for sha in b3c13c932207bef03aa3a8123bf34acad02f40d3 97d0c6e4cb7a0919c12291c7c7d560ad412f15c1 00cbe979f20cf380548d7d1d9d73136aa359c276; do
    case "$sha" in
        b3c13c*) n="#4382 gluon" ;;
        97d0c6*) n="#4439 MegaMoE" ;;
        00cbe9*) n="#4269 FSE" ;;
    esac
    if git merge-base --is-ancestor "$sha" HEAD 2>/dev/null; then
        echo "  IN post2      $n"
    else
        echo "  NOT in post2  $n"
    fi
done

echo
echo "=== and in v0.1.19 (the base image's pin)? ==="
for sha in b3c13c932207bef03aa3a8123bf34acad02f40d3 97d0c6e4cb7a0919c12291c7c7d560ad412f15c1 00cbe979f20cf380548d7d1d9d73136aa359c276; do
    case "$sha" in
        b3c13c*) n="#4382 gluon" ;;
        97d0c6*) n="#4439 MegaMoE" ;;
        00cbe9*) n="#4269 FSE" ;;
    esac
    if git merge-base --is-ancestor "$sha" v0.1.19 2>/dev/null; then
        echo "  IN v0.1.19      $n"
    else
        echo "  NOT in v0.1.19  $n"
    fi
done
