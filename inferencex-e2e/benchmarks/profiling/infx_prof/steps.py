"""Per-step record and marker."""

import time

from . import routing, runtime

controller = None


class Step:
    def __init__(self, rank, k, record=None, kind="step"):
        self.rank, self.k, self.kind = rank, k, kind
        self.record = record if record is not None else {}
        self.rf = None

    def __enter__(self):
        if controller is not None:
            controller.on_step_begin(self.rank, self.k)
        self.t0 = time.time_ns()
        if runtime.profiling():
            try:
                import torch

                self.rf = torch.autograd.profiler.record_function(f"infx_{self.kind}#{self.k}")
                self.rf.__enter__()
                if self.kind == "step":
                    routing.reset()
            except Exception:
                self.rf = None
        return self.record

    def __exit__(self, *exc):
        if self.rf is not None:
            try:
                self.rf.__exit__(None, None, None)
                if self.kind == "step":
                    rows = self.record.get("route_rows") or self.record.get("total_tokens")
                    routing.snapshot(self.k, rows, self.t0)
            except Exception:
                runtime.write_error("step close")
        try:
            self.record.update({"dummy" if self.kind == "dummy" else "step": self.k,
                                "t0_ns": self.t0, "t1_ns": time.time_ns()})
            runtime.sink("steps", self.rank).write(self.record)
        except Exception:
            runtime.write_error("step write")
        return False
