"""Auto-install Kimi-K3 aiter patches in vLLM worker processes only.

Installs the N=6288 GEMM chunk split and the custom-all-reduce IPC-meta sequence
fix. Must NOT run in ROCm helper CLIs (rocm_agent_enumerator, rocminfo, hipcc, ...):
those are python processes that inherit PYTHONPATH; eager install previously
fork-bombed by re-entering this file on every helper spawn.
"""
from __future__ import annotations

import os
import sys


_PATCHES = (
    ("AITER_N6288_CHUNK_PATCH", "patch_gemm_n6288_chunk"),
    ("AITER_CA_FLUSH_SYNC_PATCH", "patch_ca_graph_flush_sync"),
    ("KV_OFFLOAD_BATCH_CHUNK_PATCH", "patch_kv_offload_batch_chunk"),
    ("KV_OFFLOAD_BLOCK_CAP_PATCH", "patch_kv_offload_block_cap"),
)


def _enabled_patches() -> list[str]:
    return [mod for env, mod in _PATCHES if os.environ.get(env, "1") != "0"]


def _should_skip() -> bool:
    if not _enabled_patches():
        return True
    if os.environ.get("_AITER_N6288_INSTALLING") == "1":
        return True
    if os.environ.get("_AITER_CA_FLUSH_SYNC_INSTALLING") == "1":
        return True
    argv0 = (sys.argv[0] if sys.argv else "") or ""
    base = os.path.basename(argv0)
    joined = " ".join(sys.argv)
    # Denylist: ROCm / build / package helpers that must never import aiter here.
    deny = (
        "rocm_agent_enumerator",
        "rocminfo",
        "rocm-smi",
        "amd-smi",
        "hipcc",
        "amdclang",
        "clang",
        "pip",
        "pip3",
        "setuptools",
        "egg_info",
        "wheel",
        "ptxas",
        "ld.lld",
    )
    if any(tok in base for tok in deny):
        return True
    # python -c: only allow vLLM / multiprocessing worker snippets; skip one-shot
    # probes used by apply_k3_container_patches.
    if sys.argv and (argv0 == "-c" or base == "-c"):
        code = sys.argv[1] if len(sys.argv) > 1 else ""
        allow_c = (
            "multiprocessing",
            "spawn_main",
            "EngineCore",
            "VllmWorker",
            "WorkerProc",
            "runpy",
            "vllm",
        )
        return not any(tok in code for tok in allow_c)
    # Allowlist for normal entrypoints.
    if not base:
        return True
    allow = (
        "vllm",
        "EngineCore",
        "VllmWorker",
        "kimik3_fp4_mi355x",
        "multiprocessing",
        "torch",
    )
    if any(tok in base or tok in joined for tok in allow):
        return False
    # Default deny for unknown python entrypoints.
    return True


try:
    if not _should_skip():
        _repo = os.environ.get("GITHUB_WORKSPACE", "/workspace")
        _patch_dir = os.path.join(_repo, "experimental/kimik3-v4/aiter")
        if _patch_dir not in sys.path:
            sys.path.insert(0, _patch_dir)
        for _mod_name in _enabled_patches():
            try:
                __import__(_mod_name).install()
            except Exception:
                pass
except Exception:
    # Never break host python tools if patch fails.
    pass


# libtvm_ffi.so installs its own SIGSEGV handler from a library constructor, which
# clobbers Python's faulthandler and leaves only unsymbolized C frames. Re-enabling
# on a timer wins that race no matter when tvm_ffi gets imported. The dump target is
# a real file because vLLM replaces sys.stderr with a wrapper that has no fileno().
def _keep_python_faulthandler() -> None:
    import faulthandler
    import time

    try:
        fh = open("/tmp/fh_%d.log" % os.getpid(), "w", buffering=1)
    except Exception:
        return
    globals()["_FH_FILE"] = fh  # handler needs this fd to stay open
    fh.write("armed pid=%d argv=%r\n" % (os.getpid(), sys.argv[:2]))
    armed = False
    while True:
        try:
            faulthandler.disable()
            faulthandler.enable(file=fh, all_threads=True)
            if not armed:
                fh.write("faulthandler.enable ok\n")
                armed = True
        except Exception as exc:
            fh.write("enable failed: %r\n" % (exc,))
        time.sleep(0.25)


try:
    if os.environ.get("FORCE_PY_FAULTHANDLER", "0") == "1" and not _should_skip():
        import threading

        threading.Thread(
            target=_keep_python_faulthandler,
            name="keep-faulthandler",
            daemon=True,
        ).start()
except Exception:
    pass
