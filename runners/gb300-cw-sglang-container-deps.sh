#!/bin/bash
# Custom container-deps installer for gb300-cw + sglang. pip-installs
# dynamo from a wheel + source archive that launch_gb300-cw.sh pre-built
# on /mnt/vast BEFORE submitting sbatch.
#
# Why the prebuild design (mirrors the vllm sibling at
# gb300-cw-vllm-container-deps.sh from PR #1150):
#   srt-slurm's per-rank install path runs `maturin build` inside every
#   container srtctl srun's. The lmsysorg/sglang:deepseek-v4-grace-
#   blackwell_arm64 image lacks rust pre-installed, so the per-rank
#   build path can't run; pinning a published dev wheel (1.2.0.dev*)
#   trips API drift against the bundled sglang 0.5.9 (compat shim
#   warning + disagg startup warmup hang — see runs ending 2026-04-27).
#   Building dynamo ONCE from hash 6a159fed (the same commit the gb200
#   vllm recipe pins, known to be sglang-API-stable) on a single-node
#   srun in launch_gb300-cw.sh sidesteps both: every rank pip-installs
#   from the cache here (~30 s, no contention).
#
#   Used in tandem with `dynamo.install: false` in the gb300-cw sglang
#   recipes so srt-slurm's hardcoded install path is skipped and this
#   script is the sole installer.

set -e

DYNAMO_HASH="${DYNAMO_INSTALL_HASH:-6a159fedd8e4a1563aa647c31f622aedbf254b5b}"
CACHE_DIR="/mnt/vast/dynamo_cache/$DYNAMO_HASH"
DONE_MARKER="$CACHE_DIR/.done"

if [ ! -f "$DONE_MARKER" ]; then
    echo "[dynamo-cache] ERROR: prebuilt cache missing at $CACHE_DIR" >&2
    echo "[dynamo-cache] launch_gb300-cw.sh should have prebuilt this. Did the prebuild srun fail?" >&2
    exit 1
fi

echo "[dynamo-cache] installing prebuilt wheel + source from $CACHE_DIR"
pip install --break-system-packages "$CACHE_DIR"/ai_dynamo_runtime*.whl --force-reinstall

rm -rf /tmp/dynamo_build
mkdir -p /tmp/dynamo_build/dynamo
tar xzf "$CACHE_DIR/dynamo-source.tar.gz" -C /tmp/dynamo_build/dynamo
cd /tmp/dynamo_build/dynamo
pip install --break-system-packages -e .

echo "Dynamo installed from prebuilt cache ($DYNAMO_HASH)"

# --- NIXL DSv4 state-buffer patch: sglang PR #23773 --------------------------
# The disagg recipes use NIXL KV transfer. Without this patch, NIXL
# silently drops auxiliary state buffers (SWA / NSA / Mamba), causing
# decode-side accuracy to collapse on DSv4-Pro. The patch mirrors what
# the Mooncake backend already does. See NVIDIA/srt-slurm PR #85 README.
SGLANG_DIR="${SGLANG_DIR:-/sgl-workspace/sglang}"
SGLANG_REMOTE="https://github.com/sgl-project/sglang.git"
SGLANG_PR_NUMBER="23773"
SGLANG_PR_REF="refs/pull/${SGLANG_PR_NUMBER}/head"
SGLANG_LOCAL_BRANCH="nixl-dsv4-pr-${SGLANG_PR_NUMBER}"

echo "=== Installing SGLang NIXL DSV4 fix from PR #${SGLANG_PR_NUMBER} ==="

if [ -d "$SGLANG_DIR/.git" ]; then
    cd "$SGLANG_DIR"
    git config --global --add safe.directory "$SGLANG_DIR" 2>/dev/null || true
    if git remote get-url origin >/dev/null 2>&1; then
        git remote set-url origin "$SGLANG_REMOTE"
    else
        git remote add origin "$SGLANG_REMOTE"
    fi
    git fetch --depth 1 origin "$SGLANG_PR_REF"
    git checkout -f -B "$SGLANG_LOCAL_BRANCH" FETCH_HEAD
    echo "Checked out SGLang PR #${SGLANG_PR_NUMBER} at $(git rev-parse HEAD)"
else
    echo "WARNING: $SGLANG_DIR/.git not found; skipping NIXL patch (container may already include fix)"
fi

# --- API-drift patch: dynamo 1.1.0 vs sglang 0.5.9 --------------------------
# ai-dynamo at hash 6a159fed (1.1.0-equivalent) calls
# `engine.async_generate(return_routed_experts=...)`, but the sglang 0.5.9
# bundled in lmsysorg/sglang:deepseek-v4-grace-blackwell_arm64 has an
# Engine.async_generate signature that doesn't accept that kwarg, so every
# request 500s with:
#   TypeError: Engine.async_generate() got an unexpected keyword argument
#       'return_routed_experts'
# (See run 24973148979 → mooncake unblocked the disagg warmup; this is the
# next failure layer.) Strip the kwarg from every call site in the
# extracted dynamo source. `pip install -e .` above is editable, so the
# patch propagates immediately at next `python3 -m dynamo.sglang ...`.
DYNAMO_SRC=/tmp/dynamo_build/dynamo
patch_targets=$(grep -rl 'return_routed_experts' "$DYNAMO_SRC" --include='*.py' 2>/dev/null || true)
if [ -n "$patch_targets" ]; then
    # Match WHOLE LINES that are just a kwarg pass:
    #     return_routed_experts=<simple-identifier-or-attr>,?
    # The value is constrained to a simple identifier ([A-Za-z_][\w.]*),
    # which deliberately excludes function calls (no `(` allowed). This
    # leaves the multi-line assignment statement at decode_handler.py:275
    # intact:
    #     return_routed_experts = getattr(
    #         self.config.server_args, "enable_return_routed_experts", False
    #     )
    # That assignment is dead code after we strip the kwarg passes, but
    # leaving it costs nothing and avoids the syntax-error trap from the
    # earlier (over-greedy) version of this patch.
    for f in $patch_targets; do
        echo "[dynamo-patch] stripping return_routed_experts kwarg lines in $f"
        python3 - "$f" <<'PYEOF'
import re, sys
path = sys.argv[1]
with open(path) as fh:
    src = fh.read()
# Whole-line kwarg pass: indented `return_routed_experts=<simple>,?` then EOL.
# `[A-Za-z_][\w.]*` matches identifiers, attribute access, True/False/None — but NOT calls.
new = re.sub(
    r'^[ \t]+return_routed_experts\s*=\s*[A-Za-z_][\w.]*\s*,?[ \t]*\n',
    '',
    src,
    flags=re.MULTILINE,
)
if new != src:
    with open(path, 'w') as fh:
        fh.write(new)
    print(f'[dynamo-patch]   patched: {path}')
else:
    print(f'[dynamo-patch]   no kwarg-pass lines matched in: {path}')
PYEOF
    done
    # Sanity: any remaining occurrence is fine if it's the assignment;
    # log it so the next person knows what's left.
    echo "[dynamo-patch] residual occurrences (expected: only the dead assignment in decode_handler.py):"
    grep -rn 'return_routed_experts' "$DYNAMO_SRC" --include='*.py' 2>/dev/null || echo "  (none)"
else
    echo "[dynamo-patch] no occurrences of return_routed_experts found in $DYNAMO_SRC (already patched or moved upstream)"
fi
