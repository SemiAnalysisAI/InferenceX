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
export VLLM_TARGET_DEVICE=rocm
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

ensure_amdsmi_python() {
    if python3 - <<'PY'
import amdsmi

print(f"amdsmi already importable from {amdsmi.__file__}")
PY
    then
        return
    fi

    # ROCm ships the Python binding under /opt/rocm/share/amd_smi. Prefer
    # that over PyPI so the Python wrapper matches the image's ROCm runtime.
    if [ -d /opt/rocm/share/amd_smi ]; then
        if ! python3 -m pip install --no-deps /opt/rocm/share/amd_smi; then
            python3 -m pip install --no-deps amdsmi
        fi
    else
        python3 -m pip install --no-deps amdsmi
    fi

    python3 - <<'PY'
import amdsmi

print(f"amdsmi installed from {amdsmi.__file__}")
PY
}

install_tilelang_runtime_deps() {
    # DeepSeek-V4 mHC kernels import tilelang lazily during the vLLM profile
    # run. vLLM's ROCm requirements do not include it yet, while the unpinned
    # package can fall back to a source build or try to resolve CUDA torch
    # dependencies. Use binary wheels only, skip dependency resolution, and
    # install the small direct runtime deps we need. TileLang 0.1.9 is required
    # for T.pdl_sync used by mhc.py. Do not install torch-c-dlpack-ext on ROCm;
    # its wheel expects CUDA libraries.
    python3 -m pip install \
        -c /tmp/rocm-pins.txt \
        --no-deps \
        --only-binary=:all: \
        apache-tvm-ffi==0.1.9 \
        z3-solver==4.15.4.0 \
        tilelang==0.1.9

    python3 - <<'PY'
import tilelang
import tilelang.language as T

print(f"tilelang {tilelang.__version__} imported from {tilelang.__file__}")
if not hasattr(T, "pdl_sync"):
    raise SystemExit("tilelang.language.pdl_sync is required by vLLM mhc.py")
PY
}

patch_vllm_rocm_platform_detection() {
    # vLLM detects ROCm with amdsmi. On this MI355X/ATOM stack, amdsmi can be
    # unavailable or return no handles even when PyTorch sees HIP devices. Fall
    # back to torch ROCm visibility so current_platform is RocmPlatform. Also
    # avoid rocm.py's warning_once path during module import; it imports
    # distributed modules while current_platform is still being initialized.
    python3 - <<'PY'
from pathlib import Path

path = Path("vllm/platforms/__init__.py")
text = path.read_text()
start = text.index("def rocm_platform_plugin() -> str | None:")
end = text.index("\n\ndef xpu_platform_plugin() -> str | None:", start)
new = '''def rocm_platform_plugin() -> str | None:
    is_rocm = False
    logger.debug("Checking if ROCm platform is available.")
    try:
        import amdsmi

        amdsmi.amdsmi_init()
        try:
            if len(amdsmi.amdsmi_get_processor_handles()) > 0:
                is_rocm = True
                logger.debug("Confirmed ROCm platform is available via amdsmi.")
            else:
                logger.debug("ROCm platform is not available because no GPU is found by amdsmi.")
        finally:
            amdsmi.amdsmi_shut_down()
    except Exception as e:
        logger.debug("ROCm platform is not available via amdsmi because: %s", str(e))

    if not is_rocm:
        try:
            import torch

            is_rocm = (
                torch.version.hip is not None
                and torch.cuda.is_available()
                and torch.cuda.device_count() > 0
            )
            if is_rocm:
                logger.debug("Confirmed ROCm platform is available via torch HIP.")
            else:
                logger.debug("ROCm platform is not available via torch HIP.")
        except Exception as e:
            logger.debug("ROCm torch HIP fallback failed because: %s", str(e))

    return "vllm.platforms.rocm.RocmPlatform" if is_rocm else None
'''
path.write_text(text[:start] + new + text[end:])
print(f"Patched ROCm platform detection fallback in {path}")

path = Path("vllm/platforms/rocm.py")
text = path.read_text()
start = text.index("def _get_gcn_arch() -> str:")
end = text.index("\n\n# Resolve once at module load.", start)
new = '''def _get_gcn_arch() -> str:
    """
    Get GCN arch via amdsmi when available, otherwise use torch.cuda.
    Avoid warning_once during module import because it can re-enter
    vllm.platforms.current_platform initialization.
    """
    try:
        return _query_gcn_arch_from_amdsmi()
    except Exception as e:
        logger.debug("Failed to get GCN arch via amdsmi: %s", e)

    try:
        props = torch.cuda.get_device_properties(0)
        gcn_arch = getattr(props, "gcnArchName", "")
        if gcn_arch:
            logger.debug("Got GCN arch via torch.cuda: %s", gcn_arch)
            return gcn_arch
    except Exception as e:
        logger.debug("Failed to get GCN arch via torch.cuda: %s", e)

    # This benchmark is MI355X-only. Keep a deterministic fallback instead of
    # failing ROCm platform import when amdsmi is absent.
    logger.warning("Falling back to gfx950 for MI355X ROCm platform detection.")
    return "gfx950"
'''
path.write_text(text[:start] + new + text[end:])
print(f"Patched ROCm GCN arch fallback in {path}")

text = path.read_text()

def replace_block(text: str, start_marker: str, end_marker: str, replacement: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[:start] + replacement + text[end:]

text = replace_block(
    text,
    "    @classmethod\n    @with_amdsmi_context\n    def is_fully_connected",
    "    @classmethod\n    @with_amdsmi_context\n    @lru_cache(maxsize=8)\n    def get_device_name",
    '''    @classmethod
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool:
        """
        Query if the set of GPUs are fully connected by XGMI (1 hop).
        Fall back to disabling custom allreduce when amdsmi is unavailable.
        """
        if "amdsmi_init" not in globals():
            logger.warning(
                "amdsmi is unavailable; treating ROCm GPU topology as not "
                "fully connected for custom allreduce."
            )
            return False

        try:
            amdsmi_init()
            try:
                handles = [
                    amdsmi_get_processor_handles()[i] for i in physical_device_ids
                ]
                for i, handle in enumerate(handles):
                    for j, peer_handle in enumerate(handles):
                        if i < j:
                            link_type = amdsmi_topo_get_link_type(
                                handle, peer_handle
                            )
                            # type is 2 for XGMI
                            if link_type["hops"] != 1 or link_type["type"] != 2:
                                return False
                return True
            finally:
                amdsmi_shut_down()
        except Exception as error:
            logger.warning(
                "AMD 1 hop XGMI detection failed; treating ROCm GPU topology "
                "as not fully connected for custom allreduce.",
                exc_info=error,
            )
            return False

''',
)

text = replace_block(
    text,
    "    @classmethod\n    @with_amdsmi_context\n    @lru_cache(maxsize=8)\n    def get_device_name",
    "    @classmethod\n    @with_amdsmi_context\n    def get_device_uuid",
    '''    @classmethod
    @lru_cache(maxsize=8)
    def get_device_name(cls, device_id: int = 0) -> str:
        if "amdsmi_init" in globals():
            try:
                amdsmi_init()
                try:
                    physical_device_id = cls.device_id_to_physical_device_id(device_id)
                    handle = amdsmi_get_processor_handles()[physical_device_id]
                    asic_info = amdsmi_get_gpu_asic_info(handle)
                    asic_info_device_id: str = asic_info["device_id"]
                    if asic_info_device_id in _ROCM_DEVICE_ID_NAME_MAP:
                        return _ROCM_DEVICE_ID_NAME_MAP[asic_info_device_id]
                    return asic_info["market_name"]
                finally:
                    amdsmi_shut_down()
            except Exception as error:
                logger.debug(
                    "amdsmi device name query failed; falling back to torch.cuda.",
                    exc_info=error,
                )

        return torch.cuda.get_device_name(device_id)

''',
)

text = replace_block(
    text,
    "    @classmethod\n    @with_amdsmi_context\n    def get_device_uuid",
    "    @classmethod\n    def get_device_total_memory",
    '''    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        if "amdsmi_init" in globals():
            try:
                amdsmi_init()
                try:
                    device = amdsmi_get_processor_handles()[device_id]
                    return amdsmi_get_gpu_device_uuid(device)
                finally:
                    amdsmi_shut_down()
            except Exception as error:
                logger.debug(
                    "amdsmi device uuid query failed; falling back to torch.cuda.",
                    exc_info=error,
                )

        try:
            props = torch.cuda.get_device_properties(device_id)
            device_uuid = getattr(props, "uuid", None)
            if device_uuid:
                return str(device_uuid)
        except Exception as error:
            logger.debug("torch.cuda device uuid fallback failed.", exc_info=error)
        return f"cuda:{device_id}"

''',
)

path.write_text(text)
print(f"Patched ROCm amdsmi runtime fallbacks in {path}")
PY
}

check_vllm_rocm_platform_detection() {
    VLLM_LOGGING_LEVEL=DEBUG python3 - <<'PY'
import torch
from vllm.platforms import current_platform

print(f"torch.version.hip={torch.version.hip}")
print(f"torch.cuda.is_available={torch.cuda.is_available()}")
print(f"torch.cuda.device_count={torch.cuda.device_count()}")
print(
    "vllm.current_platform="
    f"{current_platform.__class__.__module__}.{current_platform.__class__.__name__} "
    f"device_type={current_platform.device_type}"
)
if not current_platform.is_rocm():
    raise SystemExit("vLLM did not detect ROCm platform")
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

    patch_vllm_rocm_platform_detection
    sanitize_stale_triton_test_metadata
    ensure_amdsmi_python

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
    install_tilelang_runtime_deps
)

python3 -c "import vllm; print(f'vLLM {vllm.__version__} from {vllm.__path__[0]}')"
check_vllm_rocm_platform_detection

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
    --gpu-memory-utilization 0.90 \
    --max-model-len $MAX_MODEL_LEN \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --enforce-eager \
    --moe-backend "triton_unfused" \
    --no-enable-prefix-caching \
    --max-num-seqs 32 \
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
