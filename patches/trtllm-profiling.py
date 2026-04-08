"""
Patch for dynamo.trtllm to add torch profiler support.

Wraps inference with torch.profiler.profile() and writes chrome trace
JSON to SGLANG_TORCH_PROFILER_DIR.

Environment variables:
  PROFILING_MODE          - "prefill" or "decode" (from srtctl)
  PROFILE_PREFILL_START_STEP / PROFILE_DECODE_START_STEP - start step
  PROFILE_PREFILL_STOP_STEP  / PROFILE_DECODE_STOP_STEP  - stop step
  SGLANG_TORCH_PROFILER_DIR - output dir for traces

Applied automatically via sitecustomize.py when PROFILING_MODE is set.
Uses a post-import hook so the patch applies after dynamo.trtllm loads.
"""

import importlib
import logging
import os
import sys

logger = logging.getLogger("trtllm-profiling-patch")


def _apply_patch():
    """Actually monkey-patch HandlerBase.generate with profiler wrapping."""
    mode = os.environ.get("PROFILING_MODE", "")
    start_step = int(os.environ.get(f"PROFILE_{mode.upper()}_START_STEP",
                     os.environ.get("PROFILE_START_STEP", "5")))
    stop_step = int(os.environ.get(f"PROFILE_{mode.upper()}_STOP_STEP",
                    os.environ.get("PROFILE_STOP_STEP", "50")))
    output_dir = os.environ.get("SGLANG_TORCH_PROFILER_DIR", "/tmp/trtllm_profiles")

    import torch.profiler
    from dynamo.trtllm.request_handlers.handler_base import HandlerBase

    os.makedirs(output_dir, exist_ok=True)

    _orig_generate = HandlerBase.generate
    _state = {"step": 0, "started": False, "stopped": False}

    _state["profiler"] = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
    )

    async def _patched_generate(self, request, *args, **kwargs):
        _state["step"] += 1
        step = _state["step"]

        if step == start_step and not _state["started"]:
            _state["profiler"].__enter__()
            _state["started"] = True
            print(f"[trtllm-patch] Step {step}: profiler started", file=sys.stderr, flush=True)

        result = _orig_generate(self, request, *args, **kwargs)
        if hasattr(result, '__aiter__'):
            async for chunk in result:
                yield chunk
        else:
            yield await result

        if step == stop_step and not _state["stopped"]:
            import torch.cuda
            torch.cuda.synchronize()
            _state["profiler"].__exit__(None, None, None)
            _state["stopped"] = True
            print(f"[trtllm-patch] Step {step}: profiler stopped, traces in {output_dir}", file=sys.stderr, flush=True)

    HandlerBase.generate = _patched_generate
    print(f"[trtllm-patch] Patched HandlerBase.generate (steps {start_step}-{stop_step}, output={output_dir})", file=sys.stderr, flush=True)


def patch():
    """Install a post-import hook that patches HandlerBase after dynamo.trtllm loads."""
    mode = os.environ.get("PROFILING_MODE", "")
    if not mode:
        return

    # If the module is already imported, patch immediately
    if "dynamo.trtllm.request_handlers.handler_base" in sys.modules:
        try:
            _apply_patch()
        except Exception as e:
            print(f"[trtllm-patch] Failed: {e}", file=sys.stderr, flush=True)
        return

    # Otherwise, install a meta path finder that triggers after the module loads
    class _PatchFinder:
        _patched = False

        def find_module(self, fullname, path=None):
            if fullname == "dynamo.trtllm.request_handlers.handler_base" and not self._patched:
                return self
            return None

        def load_module(self, fullname):
            # Remove ourselves to avoid recursion
            self._patched = True
            # Let the real import happen
            if self in sys.meta_path:
                sys.meta_path.remove(self)
            mod = importlib.import_module(fullname)
            # Now apply the patch
            try:
                _apply_patch()
            except Exception as e:
                print(f"[trtllm-patch] Failed: {e}", file=sys.stderr, flush=True)
            return mod

    sys.meta_path.insert(0, _PatchFinder())
    print(f"[trtllm-patch] Installed post-import hook for HandlerBase", file=sys.stderr, flush=True)
