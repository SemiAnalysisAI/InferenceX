"""Op-attribution profiling for inference engines, loaded through PYTHONPATH (sitecustomize)."""

import atexit
import importlib
import os

from . import graphs, imports, launchers, modules, runtime, steps, windows

ENGINES = {"vllm": "vllm", "sglang": "sglang", "trtllm": "trtllm", "trt": "trtllm"}
STACK_DUMP_S = int(os.environ.get("INFX_PROF_STACK_DUMP_S") or 300)


def dump_stacks_periodically():
    """A daemon thread writes every thread's stack to stacks/<pid>.log every STACK_DUMP_S seconds."""
    import faulthandler
    import threading
    import time

    if STACK_DUMP_S <= 0:
        return

    def run():
        path = os.path.join(runtime.DIR, "stacks", f"{os.getpid()}.log")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", buffering=1) as stream:
            while True:
                time.sleep(STACK_DUMP_S)
                stream.write(f"=== {time.strftime('%Y-%m-%dT%H:%M:%S')} pid {os.getpid()}\n")
                faulthandler.dump_traceback(file=stream, all_threads=True)

    def start():
        threading.Thread(target=run, name="infx_prof_stacks", daemon=True).start()

    start()
    os.register_at_fork(after_in_child=start)


ROCTRACER_MAX_EVENTS = 10_000_000


def raise_rocm_trace_limit():
    """Kineto's ROCm tracer keeps at most ROCTRACER_MAX_EVENTS events per session (1M by default in older releases); a long graph capture overran it and later graphs lost their launches. Set through Kineto's base config, before torch loads Kineto, unless one is already configured."""
    if "KINETO_CONFIG" in os.environ or not os.path.isdir("/opt/rocm"):
        return
    path = os.path.join(runtime.DIR, "kineto.conf")
    os.makedirs(runtime.DIR, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"ROCTRACER_MAX_EVENTS={ROCTRACER_MAX_EVENTS}\n")
    os.environ["KINETO_CONFIG"] = path


def boot():
    if not runtime.DIR:
        return
    try:
        raise_rocm_trace_limit()
    except Exception:
        runtime.write_error("kineto config")
    try:
        name = ENGINES.get(os.environ.get("INFX_PROF_ENGINE", "vllm"), "vllm")
        engine = importlib.import_module(f".engines.{name}", __name__).ENGINE
        registry = imports.HookRegistry()
        graphs.install(registry)
        modules.install(registry)
        launchers.install(registry, engine.caller_roots, engine.launcher_modules, engine.launcher_packages)
        if engine.window_control == "file":
            steps.controller = windows.FileWindows()
        engine.install(registry)
        registry.install()
        atexit.register(runtime.flush_all)
    except Exception:
        runtime.write_error("boot")
    try:
        dump_stacks_periodically()
    except Exception:
        runtime.write_error("stack dumps")
