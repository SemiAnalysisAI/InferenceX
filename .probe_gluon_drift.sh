#!/usr/bin/env bash
# arch_info is a submodule, not a symbol -- the earlier MODULE-ABSENT was a
# false positive. So the gluon surface resolves. Remaining question: how much of
# the base->main delta on the two shared files is #4382/#4673, and how much is
# unrelated drift we'd be dragging in by copying main's version wholesale?
set -u
W=/tmp/aitermain/aiter
T=/home/jiacao/3way-20260812-2214/target/usr/local/lib/python3.12/dist-packages/aiter

echo "=== base(v0.1.19) -> main(97d0c6e4) : ops/triton/attention/pa_decode_sparse.py"
diff -u "$T/ops/triton/attention/pa_decode_sparse.py" "$W/ops/triton/attention/pa_decode_sparse.py" | diffstat 2>/dev/null \
  || diff -u "$T/ops/triton/attention/pa_decode_sparse.py" "$W/ops/triton/attention/pa_decode_sparse.py" | grep -c '^[+-]'

echo
echo "=== the same file, but versus the vendor (measurement) image ==="
V=/home/jiacao/3way-20260812-2214/vendor/usr/local/lib/python3.12/dist-packages/aiter
if cmp -s "$V/ops/triton/attention/pa_decode_sparse.py" "$W/ops/triton/attention/pa_decode_sparse.py"; then
    echo "  vendor == main : the measurement image runs stock upstream here"
else
    echo "  vendor != main, delta:"
    diff -u "$V/ops/triton/attention/pa_decode_sparse.py" "$W/ops/triton/attention/pa_decode_sparse.py" | head -40
fi

echo
echo "=== gluon kernel file: vendor vs main ==="
if cmp -s "$V/ops/triton/_gluon_kernels/gfx950/attention/pa_decode_sparse.py" \
          "$W/ops/triton/_gluon_kernels/gfx950/attention/pa_decode_sparse.py"; then
    echo "  vendor == main"
else
    diff -u "$V/ops/triton/_gluon_kernels/gfx950/attention/pa_decode_sparse.py" \
            "$W/ops/triton/_gluon_kernels/gfx950/attention/pa_decode_sparse.py" | grep -c '^[+-]'
fi

echo
echo "=== common_utils.py: does main already carry #4673? ==="
grep -n "max_addressable_bytes" "$W/ops/triton/utils/common_utils.py" | head
echo "  (empty above = #4673 not merged yet, must come from the PR)"
