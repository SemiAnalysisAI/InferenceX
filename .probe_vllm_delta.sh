#!/usr/bin/env bash
# The vendor_patchset.vllm.diff was generated per-file with a fixed --label, so
# the diff headers all read "a/vllm b/vllm" and the filenames are lost. Recover
# them by re-diffing the ref and vendor trees directly, and for each changed
# file report its churn plus whether the target already has the same change.
#
# ref    = the vendor's upstream base (vLLM 02e63f2e4, 2026-07-30)
# vendor = the measurement image (ref + the vendor patch set)
# target = the nightly we want to land on (b22afe45, 465 commits past ref)
set -u
D=/home/jiacao/3way-20260812-2214
R=$(find "$D/ref"    -maxdepth 9 -type d -name vllm -path '*packages*' | head -1)
# The vendor image installs vLLM editable from /src/vllm, so its package root is
# /src/vllm/vllm rather than a dist-packages path like the other two sides.
V="$D/vendor/src/vllm/vllm"
T=$(find "$D/target" -maxdepth 9 -type d -name vllm -path '*packages*' | head -1)
echo "ref=$R"; echo "vendor=$V"; echo "target=$T"; echo

printf "%-6s %6s %6s  %-9s %s\n" STATUS +LINES -LINES "IN-TARGET" FILE
( cd "$V" && find . -name '*.py' -not -path '*__pycache__*' | sort ) | while read -r rel; do
    rel=${rel#./}
    [ "$rel" = "_version.py" ] && continue
    vf="$V/$rel"; rf="$R/$rel"; tf="$T/$rel"
    if [ ! -f "$rf" ]; then
        st=NEW; add=$(wc -l < "$vf"); del=0
    else
        cmp -s "$rf" "$vf" && continue
        st=MOD
        add=$(diff "$rf" "$vf" | grep -c '^>')
        del=$(diff "$rf" "$vf" | grep -c '^<')
    fi
    # Does the target already carry this file, and does it already match vendor?
    if   [ ! -f "$tf" ];        then int="MISSING"
    elif cmp -s "$tf" "$vf";    then int="SAME"
    else                             int="DIFFERS"
    fi
    printf "%-6s %6s %6s  %-9s %s\n" "$st" "$add" "$del" "$int" "$rel"
done
