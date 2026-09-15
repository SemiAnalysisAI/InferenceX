#!/usr/bin/env python3
"""Apply a configurable MRV2 PP scheduler request cap with shape telemetry."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import py_compile


cap = int(os.environ.get("K3_PP_BATCH_CAP", "40"))
if cap < 1:
    raise ValueError(f"K3_PP_BATCH_CAP must be positive, got {cap}")

spec = importlib.util.find_spec("vllm")
if spec is None or spec.origin is None:
    raise SystemExit("vllm not found")
path = Path(spec.origin).parent / "v1/core/sched/scheduler.py"
source = path.read_text()
marker = "def _get_k3_max_num_scheduled_reqs(self) -> int:"
if marker in source:
    print(f"{path}: K3 PP batch cap already applied")
    raise SystemExit(0)


def replace_once(old: str, new: str) -> None:
    global source
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected one anchor, found {count}: {old[:100]!r}")
    source = source.replace(old, new, 1)


replace_once(
    """        self.current_step += 1
        # NOTE(woosuk) on the scheduling algorithm:
""",
    """        self.current_step += 1
        max_num_scheduled_reqs = self._get_k3_max_num_scheduled_reqs()
        # NOTE(woosuk) on the scheduling algorithm:
""",
)
replace_once(
    """        while req_index < len(self.running) and token_budget > 0:
""",
    """        while (
            req_index < len(self.running)
            and token_budget > 0
            and len(num_scheduled_tokens) < max_num_scheduled_reqs
        ):
""",
)
replace_once(
    """            while (self.waiting or self.skipped_waiting) and token_budget > 0:
""",
    """            while (
                (self.waiting or self.skipped_waiting)
                and token_budget > 0
                and len(num_scheduled_tokens) < max_num_scheduled_reqs
            ):
""",
)
replace_once(
    """        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
""",
    """        assert len(self.running) <= self.max_num_running_reqs
        assert len(num_scheduled_tokens) <= max_num_scheduled_reqs
        if self.use_pp and self.use_v2_model_runner:
            batch_size = len(num_scheduled_tokens)
            histogram = getattr(self, "_k3_pp_batch_histogram", None)
            if histogram is None:
                histogram = {}
                self._k3_pp_batch_histogram = histogram
            histogram[batch_size] = histogram.get(batch_size, 0) + 1
            if self.current_step % 128 == 0:
                logger.info(
                    "K3 PP batch cap telemetry: cap=%d step=%d "
                    "batch=%d histogram=%s unfinished=%d",
                    max_num_scheduled_reqs,
                    self.current_step,
                    batch_size,
                    sorted(histogram.items()),
                    self.get_num_unfinished_requests(),
                )
        # Since some requests in the RUNNING queue may not be scheduled in
""",
)
replace_once(
    """    def _build_kv_connector_meta(
""",
    f"""    def _get_k3_max_num_scheduled_reqs(self) -> int:
        \"\"\"Cap MRV2 PP requests per scheduler wave for controlled cohorts.\"\"\"
        if not self.use_pp or not self.use_v2_model_runner:
            return self.max_num_running_reqs
        return min(
            self.max_num_running_reqs,
            self.get_num_unfinished_requests(),
            {cap},
        )

    def _build_kv_connector_meta(
""",
)
path.write_text(source)
py_compile.compile(str(path), doraise=True)
print(f"{path}: applied K3 PP batch cap={cap}; py_compile OK")
