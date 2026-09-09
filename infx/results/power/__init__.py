"""Power-metric transformations independent of telemetry and artifact formats."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


# The unprefixed joules_per_* fields silently switched from role-local to
# whole-deployment energy when multinode aggregation landed, and the values
# alone cannot distinguish the two. Stamp the semantics so consumers fail
# closed on unversioned rows instead of guessing.
POWER_METRIC_SCHEMA_VERSION = 2

WHOLE_METRIC_KEYS = (
    "avg_power_w",
    "avg_total_gpu_power_w",
    "total_gpu_energy_j",
    "joules_per_successful_query",
    "joules_per_input_token",
    "joules_per_output_token",
    "joules_per_total_token",
)
ROLE_METRIC_KEYS = (
    "prefill_gpu_energy_j",
    "decode_gpu_energy_j",
    "prefill_avg_power_w",
    "decode_avg_power_w",
    "prefill_joules_per_input_token",
    "decode_joules_per_output_token",
)
ALL_POWER_METRIC_KEYS = WHOLE_METRIC_KEYS + ROLE_METRIC_KEYS


def with_power_metrics(
    result: Mapping[str, Any],
    *,
    metric_keys: Iterable[str],
    schema_version: int,
    power_valid: bool,
    metrics: Mapping[str, float],
) -> dict[str, Any]:
    """Return a result with stale metrics replaced by one validated metric set.

    The caller supplies the metric family and schema version. Detailed invalid
    reasons belong in its validation sidecar, not the numeric metric payload.
    Neither the input result nor the supplied metrics is mutated.
    """
    # Do not coerce malformed JSON (such as a list of pairs) into an object.
    # Leave non-mappings to fail on the same operations as the existing adapters.
    data = dict(result) if isinstance(result, Mapping) else result
    for key in metric_keys:
        data.pop(key, None)
    data["power_metric_schema_version"] = schema_version
    data["power_valid"] = int(power_valid)
    data.pop("power_invalid_reasons", None)
    if power_valid:
        for key, value in metrics.items():
            if value is None or not math.isfinite(value):
                raise ValueError(f"non-finite power metric: {key}")
            precision = 3 if key.endswith(("_w", "_j")) else 6
            data[key] = round(value, precision)
    return data
