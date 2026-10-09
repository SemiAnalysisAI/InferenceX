# SPDX-License-Identifier: GPL-3.0-only
"""The power_model command-line interface for all supported system families."""

import argparse
import json
from collections.abc import Sequence
from math import isfinite

from power_model import WorkloadState, create_power_model
from power_model.base import validate_watts
from power_model.models.advanced.systems.catalog import SYSTEMS, system_name
from power_model.models.catalog import MODELS, model_name
from power_model.reporting import format_power_breakdown_per_chassis


def _help_catalog() -> str:
    models = "\n".join(
        f"  {name:<14} {model_class.__name__}" for name, model_class in MODELS.items()
    )
    systems = "\n".join(
        f"  {name:<14} {system.__name__}" + (" (H100 / H200)" if name == "hopper" else "")
        for name, system in SYSTEMS.items()
    )
    return f"Models:\n{models}\n\nSystems:\n{systems}"


def _measured_watts(value: str) -> float:
    try:
        return validate_watts(float(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected finite watts >= 0, got {value!r}") from error


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m power_model",
        description="Estimate all-in facility watts per GPU for the selected system and model.",
        epilog=_help_catalog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--gpu_level_power_per_gpu",
        "--gpu-level-power-per-gpu",
        type=float,
        required=True,
        help="GPU electrical power in watts per GPU",
    )
    parser.add_argument(
        "--cpu-socket-measured-power",
        type=_measured_watts,
        help="ACPI Grace Power Socket average in watts per socket (GB200/GB300 NVL72), "
        "never DCGM field 1130 CPU power; replaces modeled Grace CPU and LPDDR5X",
    )
    parser.add_argument(
        "--system",
        type=system_name,
        choices=tuple(SYSTEMS),
        metavar="SYSTEM",
        required=True,
        help="System from the list below; also selects cooling",
    )
    parser.add_argument(
        "--model",
        type=model_name,
        choices=tuple(MODELS),
        default="oss",
        help="Power model (default: oss)",
    )
    parser.add_argument(
        "--workload",
        choices=tuple(state.value for state in WorkloadState if state != WorkloadState.IDLE),
        default="fixed-seq-len",
        help="Select CPU and memory workload profile (default: fixed-seq-len)",
    )
    parser.add_argument(
        "--scale-out-enabled",
        "--using-scale-out",
        dest="using_scale_out",
        action="store_true",
        help="Use active NIC and switch power (default: off)",
    )
    common = parser.add_argument_group("oss systems")
    common.add_argument("--systems", type=int, default=1, help="Quantity of the selected system")
    common.add_argument(
        "--power-breakdown-per-chassis",
        action="store_true",
        help="Print a nested power BoM with unit watts, quantity, and totals per chassis or rack",
    )
    args = parser.parse_args(argv)
    try:
        model = create_power_model(
            model=args.model,
            system=args.system,
            workload_state=args.workload,
            using_scale_out=args.using_scale_out,
            systems=args.systems,
        )
        result = model.estimate_breakdown(
            args.gpu_level_power_per_gpu, cpu_socket_measured_power=args.cpu_socket_measured_power
        )
        ratio = (
            result.AllInPower_per_gpu / result.gpu_level_power_per_gpu
            if result.gpu_level_power_per_gpu
            else None
        )
        if ratio is not None and not isfinite(ratio):
            ratio = None
        if args.power_breakdown_per_chassis:
            ratio_text = f"{ratio:.4f}x" if ratio is not None else "n/a"
            output = (
                f"{format_power_breakdown_per_chassis(result)}\n"
                f"Facility / GPU power ratio: {ratio_text}"
            )
        else:
            output = json.dumps(
                result.model_dump(mode="json", exclude_none=True)
                | {"facility_to_gpu_power_ratio": ratio},
                indent=2,
                allow_nan=False,
            )
    except ValueError as error:
        parser.error(str(error))
    print(output)
