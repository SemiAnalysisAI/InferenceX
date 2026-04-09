"""
Patch for dynamo.trtllm v0.8.1 to add torch profiler support.

Wraps HandlerBase.generate_locally() with torch.profiler.profile() and
writes chrome trace JSON to SGLANG_TORCH_PROFILER_DIR.

Target: dynamo.trtllm.request_handlers.handler_base.HandlerBase.generate_locally
  - Async generator: async def generate_locally(self, request, context, embeddings=None)
  - Called by PrefillHandler.generate(), DecodeHandler.generate(), AggregatedHandler.generate()
  - Each call = one request; "step" here counts requests processed

Environment variables:
  PROFILING_MODE          - "prefill" or "decode" (from srtctl)
  PROFILE_PREFILL_START_STEP / PROFILE_DECODE_START_STEP - start step
  PROFILE_PREFILL_STOP_STEP  / PROFILE_DECODE_STOP_STEP  - stop step
  SGLANG_TORCH_PROFILER_DIR - output dir for traces

Applied by appending import+call to handler_base.py via setup script.
"""

import os
import sys


def _apply_patch():
    """Monkey-patch HandlerBase.generate_locally with profiler wrapping."""
    mode = os.environ.get("PROFILING_MODE", "")
    start_step = int(os.environ.get(f"PROFILE_{mode.upper()}_START_STEP",
                     os.environ.get("PROFILE_START_STEP", "5")))
    stop_step = int(os.environ.get(f"PROFILE_{mode.upper()}_STOP_STEP",
                    os.environ.get("PROFILE_STOP_STEP", "50")))
    output_dir = os.environ.get("SGLANG_TORCH_PROFILER_DIR", "/tmp/trtllm_profiles")

    import torch.profiler
    from dynamo.trtllm.request_handlers.handler_base import HandlerBase

    if not hasattr(HandlerBase, "generate_locally"):
        methods = [m for m in dir(HandlerBase)
                   if not m.startswith('_') and callable(getattr(HandlerBase, m, None))]
        print(f"[trtllm-patch] ERROR: HandlerBase has no generate_locally. "
              f"Available: {methods}", file=sys.stderr, flush=True)
        return

    os.makedirs(output_dir, exist_ok=True)

    _orig_generate_locally = HandlerBase.generate_locally
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

    async def _patched_generate_locally(self, request, context, embeddings=None):
        _state["step"] += 1
        step = _state["step"]

        if step == start_step and not _state["started"]:
            _state["profiler"].__enter__()
            _state["started"] = True
            print(f"[trtllm-patch] Step {step}: profiler started",
                  file=sys.stderr, flush=True)

        async for chunk in _orig_generate_locally(self, request, context, embeddings):
            yield chunk

        if step == stop_step and not _state["stopped"]:
            import torch.cuda
            torch.cuda.synchronize()
            _state["profiler"].__exit__(None, None, None)
            _state["stopped"] = True
            print(f"[trtllm-patch] Step {step}: profiler stopped, "
                  f"traces in {output_dir}", file=sys.stderr, flush=True)

    HandlerBase.generate_locally = _patched_generate_locally
    print(f"[trtllm-patch] Patched HandlerBase.generate_locally "
          f"(steps {start_step}-{stop_step}, output={output_dir})",
          file=sys.stderr, flush=True)


def patch():
    """Apply the profiling patch to HandlerBase.

    When appended to handler_base.py, HandlerBase is already defined,
    so the import succeeds and we patch immediately.
    """
    mode = os.environ.get("PROFILING_MODE", "")
    if not mode:
        return

    try:
        _apply_patch()
    except Exception as e:
        print(f"[trtllm-patch] Failed to apply patch: {e}",
              file=sys.stderr, flush=True)
