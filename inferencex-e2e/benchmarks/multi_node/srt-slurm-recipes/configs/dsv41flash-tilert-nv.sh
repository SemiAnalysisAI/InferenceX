#!/usr/bin/env bash
# Per-role setup for the DeepSeek-V4.1-Flash TileRT recipe on GB200 / GB300.
# The images carry no tilert; every role installs the pinned public release from PyPI here.
set -eo pipefail

source /infmax-workspace/benchmarks/check_env.sh
# The tilert release ships the tilert-pd-router, so the master's router version is its version.
check_env_vars ROUTER_VERSION TILERT_ROLE
case "$TILERT_ROLE" in
    prefill|decode|router) ;;
    *) echo "Unknown TileRT role: $TILERT_ROLE" >&2; exit 1 ;;
esac

# BOOTSTRAP_TORCH=<torch spec>: the decode container is the plain CUDA runtime image (no Python
# stack); install Python, pip and the pinned official Torch wheel before tilert.
if [[ "$TILERT_ROLE" == decode && -n "${BOOTSTRAP_TORCH:-}" ]]; then
    if ! command -v python3 >/dev/null 2>&1 || ! python3 -m pip --version >/dev/null 2>&1; then
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq
        apt-get install -y -qq --no-install-recommends python3 python3-pip python3-dev >/dev/null
    fi
    export PIP_BREAK_SYSTEM_PACKAGES=1  # the container's own Python is the serving Python
    python3 -m pip install --quiet --no-cache-dir --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu132}" "$BOOTSTRAP_TORCH"
fi

if [[ "$TILERT_ROLE" != decode ]]; then
    # The prefill (KV connector) and the router run in the prefill image and load only
    # tilert/pd_vllm, as a namespace package, without tilert/__init__.py, which would load the
    # decode library built for another Torch. The prefill image's vLLM / Torch / serving stack
    # stays as it is; the router uses vLLM's tokenizer from it.
    pd_root=/tmp/tilert-pd-vllm
    python3 -m pip download --quiet --no-cache-dir --no-deps -d "$pd_root/wheel" "tilert==$ROUTER_VERSION"
    python3 - "$pd_root" <<'PY'
import pathlib, sys, zipfile
root = pathlib.Path(sys.argv[1])
(wheel,) = (root / "wheel").glob("tilert-*.whl")
with zipfile.ZipFile(wheel) as z:
    z.extractall(root / "pkg", [n for n in z.namelist() if n.startswith("tilert/pd_vllm/")])
PY
    site_dir=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    echo "$pd_root/pkg" > "$site_dir/tilert-pd-vllm.pth"
    python3 -c 'import importlib.util as u; assert u.find_spec("tilert.pd_vllm.prefill_connector"), "no tilert.pd_vllm.prefill_connector"'
    if [[ "$TILERT_ROLE" == router ]]; then
        python3 -c 'import tilert.pd_vllm.pd_router, vllm.tokenizers.deepseek_v41'
        # srt-slurm starts the router as `python -m tilert.pd_vllm.pd_router`.
        if ! command -v python >/dev/null 2>&1; then
            ln -sf "$(command -v python3)" /usr/local/bin/python
        fi
    fi
else
    # The tilert release carries the serving layer (PD code and its engine API).
    python3 -m pip install --quiet --no-cache-dir "tilert==$ROUTER_VERSION"
fi

# The prompt KV moves over the NVLink fabric inside one NVL72; no Mooncake is needed.
if [[ "$TILERT_ROLE" == decode ]]; then
    if [[ -n "${SERVE_DEPS:-}" ]]; then
        # The recipe pins the serving stack the release was gated with.
        # shellcheck disable=SC2086
        python3 -m pip install --quiet --no-cache-dir $SERVE_DEPS
    elif ! python3 -c 'import uvicorn' >/dev/null 2>&1; then
        python3 -m pip install --quiet --no-cache-dir fastapi uvicorn httpx
    fi
    if ! python3 -c 'import transformers' >/dev/null 2>&1; then
        python3 -m pip install --quiet --no-cache-dir 'transformers>=4.56'
    fi
fi

# The decode library is built against one Torch / CUDA pair; fail early on a mismatch instead of
# at the first kernel launch.
if [[ "$TILERT_ROLE" == decode ]]; then
    python3 - <<'PY'
import torch
want = "2.13.0+cu132"  # the official Torch wheel the release libtilert.so is built against
if torch.__version__ != want:
    raise SystemExit(f"decode image has torch {torch.__version__}; the tilert decode needs {want}")
PY
    # The release MoE kernels are built for one SM count (152 on GB300); a different part would
    # silently size its persistent grids wrong.
    if [[ -n "${DECODE_EXPECTED_SMS:-}" ]]; then
        python3 - <<'PY'
import os, torch
want = int(os.environ["DECODE_EXPECTED_SMS"])
got = torch.cuda.get_device_properties(0).multi_processor_count
if got != want:
    raise SystemExit(f"decode GPU has {got} SMs; this tilert build is for {want}")
PY
    fi
fi
