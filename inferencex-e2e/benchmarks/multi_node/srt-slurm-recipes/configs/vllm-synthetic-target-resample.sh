#!/usr/bin/env bash
# Test-only patch (vllm-project/vllm#65849): under synthetic acceptance, resample the
# forced-rejection slot from the target distribution instead of max(p - q, 0).
set -euo pipefail
python3 - <<'EOF'
import importlib.util
import pathlib

root = pathlib.Path(importlib.util.find_spec("vllm").submodule_search_locations[0])
path = root / "v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
src = path.read_text()
tag = "[synthetic-target-resample]"
if "or not is_valid_rejected_draft or SYNTHETIC_MODE:" in src:
    print(f"{tag} already applied: {path}")
    raise SystemExit(0)

kernel = src.index("def _resample_kernel(")
sig_end = src.index("\n):", kernel)
src = src[:sig_end] + "\n    SYNTHETIC_MODE: tl.constexpr," + src[sig_end:]

cond = "    if is_bonus or not is_valid_rejected_draft:\n"
assert src.count(cond) == 1, "resample condition not found exactly once"
src = src.replace(cond, "    if is_bonus or not is_valid_rejected_draft or SYNTHETIC_MODE:\n")

call = src.index("_resample_kernel[(")
call_end = src.index("\n    )\n", call)
assert "synthetic_conditional_rates" in src[:call], "synthetic_conditional_rates not in scope"
src = src[:call_end] + "\n        SYNTHETIC_MODE=synthetic_conditional_rates is not None," + src[call_end:]

compile(src, str(path), "exec")
path.write_text(src)
print(f"{tag} applied: {path}")
EOF
