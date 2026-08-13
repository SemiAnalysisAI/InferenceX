#!/usr/bin/env bash
# MegaMoE needs `flydsl` and `mori` as top-level packages. Are they in the base
# image at all, and at what version relative to the measurement image? If they
# differ these are compiled deps and no Python patch can bridge them.
set -u
D=/home/jiacao/3way-20260812-2214
for img in target vendor; do
    echo "=== $img"
    for pkg in flydsl mori; do
        found=$(find "$D/$img" -maxdepth 8 -type d -name "$pkg" -not -path '*/__pycache__/*' 2>/dev/null | head -3)
        if [ -z "$found" ]; then echo "  $pkg: ABSENT"; continue; fi
        echo "  $pkg: $found"
        for d in $found; do
            v=$(find "$(dirname "$d")" -maxdepth 1 -name "$pkg*dist-info" -o -maxdepth 1 -name "$pkg*egg-info" 2>/dev/null | head -1)
            [ -n "$v" ] && echo "      dist: $(basename "$v")"
            [ -f "$d/_version.py" ] && echo "      _version: $(head -3 "$d/_version.py" | tr '\n' ' ')"
            [ -f "$d/version.py" ] && echo "      version: $(head -3 "$d/version.py" | tr '\n' ' ')"
            echo "      .so count: $(find "$d" -name '*.so' | wc -l)"
        done
    done
done

echo
echo "=== which files in the BASE image import the 5 drifted flydsl kernels? ==="
T="$D/target/usr/local/lib/python3.12/dist-packages/aiter"
for m in flydsl_dispatch_combine_intranode_op flydsl_dispatch_combine_intranode_kernel tensor_shim mxfp4_gemm_common communication_ops_utils; do
    n=$(grep -rl "$m" "$T" --include='*.py' 2>/dev/null | wc -l)
    echo "  $m: $n importer(s)"
    grep -rl "$m" "$T" --include='*.py' 2>/dev/null | sed "s|$T/|      |"
done
