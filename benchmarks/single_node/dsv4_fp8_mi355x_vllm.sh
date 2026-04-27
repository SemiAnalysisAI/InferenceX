#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4-Pro FP8 on MI355X via vLLM with AITER MLA decode.
# Based on vllm-project/vllm#40889 (AITER-accelerated sparse MLA decode,
# stacked on #40871 which adds base DSv4 ROCm support).
#
# Uses the ATOM MI355X image as the base (ROCm 7.2.2, PyTorch 2.10,
# aiter with MLA decode, MI355X GPU detection). vLLM is rebuilt from
# the PR branch on top. Once both PRs merge into a release, switch to
# a vLLM ROCm MI355X image and remove the build.

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

hf download "$MODEL"

if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

export VLLM_ROCM_USE_AITER=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_PLUGINS=""

# Build vLLM from PR #40889 branch (includes #40871 base). The ATOM
# image provides ROCm 7.2.2 toolchain (hipcc, cmake, ninja, torch,
# aiter with MLA decode); we rebuild vLLM in-place.
# Bump VLLM_PR_SHA when the PR moves.
VLLM_PR_SHA="b3a4a44f01e565219dd353611712d0ea2e8d11ee"
VLLM_PR_DIR="/tmp/vllm-pr40889"

sanitize_stale_triton_test_metadata() {
    # The ATOM image was built with local /triton-test packages and the final
    # layer removed that directory. Pip's resolver follows those metadata refs
    # when installing unrelated deps, so remove only the stale metadata lines.
    python3 - <<'PY'
import importlib.metadata
import site
import sys
from pathlib import Path

STALE = "/triton-test"
metadata_files = ("direct_url.json", "METADATA", "requires.txt")
changed = False

for dist in importlib.metadata.distributions():
    dist_path = Path(str(dist._path))
    name = dist.metadata.get("Name") or dist_path.name
    for relpath in metadata_files:
        path = dist_path / relpath
        if not path.exists():
            continue
        text = path.read_text(errors="replace")
        if STALE not in text:
            continue
        changed = True
        if relpath == "direct_url.json":
            path.unlink()
            print(f"Removed stale editable metadata for {name}: {path}")
            continue
        lines = text.splitlines(keepends=True)
        kept = [line for line in lines if STALE not in line]
        path.write_text("".join(kept))
        print(
            f"Removed {len(lines) - len(kept)} stale {STALE} metadata "
            f"line(s) for {name}: {path}"
        )

for dist in importlib.metadata.distributions():
    dist_path = Path(str(dist._path))
    name = (dist.metadata.get("Name") or dist_path.name).lower().replace("_", "-")
    if name != "torch":
        continue
    for relpath in ("METADATA", "requires.txt"):
        path = dist_path / relpath
        if not path.exists():
            continue
        lines = path.read_text(errors="replace").splitlines(keepends=True)
        kept = []
        for line in lines:
            normalized = line.strip().lower()
            is_triton_req = (
                relpath == "METADATA"
                and normalized.startswith("requires-dist: triton")
            ) or (
                relpath == "requires.txt"
                and normalized.startswith("triton")
            )
            if not is_triton_req:
                kept.append(line)
        if len(kept) == len(lines):
            continue
        changed = True
        path.write_text("".join(kept))
        print(
            f"Removed {len(lines) - len(kept)} torch triton dependency "
            f"metadata line(s): {path}"
        )

roots = set()
for getter in (site.getsitepackages,):
    try:
        roots.update(Path(p) for p in getter())
    except Exception:
        pass
try:
    roots.add(Path(site.getusersitepackages()))
except Exception:
    pass
roots.update(Path(p) for p in sys.path if "site-packages" in p or "dist-packages" in p)

for root in roots:
    if not root.exists():
        continue
    for pattern in ("*.egg-link", "*.pth"):
        for path in root.glob(pattern):
            text = path.read_text(errors="replace")
            if STALE not in text:
                continue
            changed = True
            kept = [line for line in text.splitlines(keepends=True) if STALE not in line]
            if kept:
                path.write_text("".join(kept))
                print(f"Removed stale {STALE} line(s): {path}")
            else:
                path.unlink()
                print(f"Removed stale {STALE} link file: {path}")

remaining = []
for dist in importlib.metadata.distributions():
    dist_path = Path(str(dist._path))
    for relpath in metadata_files:
        path = dist_path / relpath
        if path.exists() and STALE in path.read_text(errors="replace"):
            remaining.append(str(path))
for root in roots:
    if root.exists():
        for pattern in ("*.egg-link", "*.pth"):
            for path in root.glob(pattern):
                if STALE in path.read_text(errors="replace"):
                    remaining.append(str(path))

if remaining:
    print("Stale /triton-test metadata remains:")
    for path in remaining:
        print(f"  {path}")
    raise SystemExit(1)
if not changed:
    print("No stale /triton-test package metadata found.")
PY
}

if [ ! -d "$VLLM_PR_DIR/.git" ]; then
    git clone --filter=blob:none https://github.com/ChuanLi1101/vllm.git "$VLLM_PR_DIR"
fi
(
    cd "$VLLM_PR_DIR"
    git fetch --depth=1 origin "$VLLM_PR_SHA" 2>/dev/null \
        || git fetch --depth=1 origin rocm/aiter-mla-dsv4-decode
    git checkout --force "$VLLM_PR_SHA"
    test "$(git rev-parse HEAD)" = "$VLLM_PR_SHA"

    sanitize_stale_triton_test_metadata

    # Pin ROCm packages so pip's resolver can't replace them with
    # CUDA builds from PyPI (torch, torchvision, aiter, triton, etc.).
    pip freeze | grep -iE '^(torch|aiter|triton|mori)' > /tmp/rocm-pins.txt
    if grep -n "/triton-test" /tmp/rocm-pins.txt; then
        echo "Stale /triton-test reference found in ROCm constraints"
        exit 1
    fi

    pip install setuptools-scm
    # Install vLLM code + build C++ extensions (no deps to avoid touching ROCm)
    pip install --no-build-isolation --no-deps --force-reinstall -e .
    # Install runtime deps separately, constrained to keep ROCm packages intact.
    pip install -c /tmp/rocm-pins.txt -r requirements/rocm.txt
)

python3 -c "import vllm; print(f'vLLM {vllm.__version__} from {vllm.__path__[0]}')"

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

start_gpu_monitor

set -x
vllm serve $MODEL --port $PORT \
    --tensor-parallel-size $TP \
    --gpu-memory-utilization 0.95 \
    --max-model-len $MAX_MODEL_LEN \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --enforce-eager \
    --moe-backend "triton_unfused" \
    --no-enable-prefix-caching \
    --max-num-seqs 256 \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 > $SERVER_LOG 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --trust-remote-code

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
