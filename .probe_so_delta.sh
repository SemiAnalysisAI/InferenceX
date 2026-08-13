#!/usr/bin/env bash
# Do the vendor and base images ship the same set of prebuilt aiter modules?
# If the vendor has extra .so files, the container patch cannot be pure-Python.
set -u
D=/home/jiacao/3way-20260812-2214/
T="$D/target/usr/local/lib/python3.12/dist-packages/aiter"
V="$D/vendor/usr/local/lib/python3.12/dist-packages/aiter"
cd "$T" && find . -name '*.so' | sort > /tmp/so_target.txt
cd "$V" && find . -name '*.so' | sort > /tmp/so_vendor.txt
echo "vendor-only .so:"
comm -13 /tmp/so_target.txt /tmp/so_vendor.txt | sed 's/^/  /'
echo "target-only .so:"
comm -23 /tmp/so_target.txt /tmp/so_vendor.txt | sed 's/^/  /'
echo
echo "hsaco/co asm blobs:"
cd "$T" && find . \( -name '*.co' -o -name '*.hsaco' \) | sort > /tmp/co_target.txt
cd "$V" && find . \( -name '*.co' -o -name '*.hsaco' \) | sort > /tmp/co_vendor.txt
echo "  vendor-only: $(comm -13 /tmp/co_target.txt /tmp/co_vendor.txt | wc -l)"
comm -13 /tmp/co_target.txt /tmp/co_vendor.txt | head -20 | sed 's/^/    /'
