#!/usr/bin/env bash
set -eo pipefail

patch_file="$(dirname "$(readlink -f "$0")")/kimik3-optimizations/62069-cute-context-prefill.patch"
package_root="$(python3 -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("vllm").origin).resolve().parent.parent)')"
cd "$package_root"
exec 9> .kimik3-optimizations.lock
flock 9

if patch --dry-run --reverse --silent --force --fuzz=0 -p1 < "$patch_file" >/dev/null 2>&1; then
    echo "CuTe context prefill patch already applied"
else
    patch --dry-run --forward --batch --fuzz=0 -p1 < "$patch_file"
    patch --forward --batch --fuzz=0 -p1 < "$patch_file"
fi
