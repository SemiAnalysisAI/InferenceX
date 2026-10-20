# SPDX-License-Identifier: GPL-3.0-only
"""GB200 NVL72 and its endpoint-based external fabric allocation."""

from typing import ClassVar

from power_model.models.advanced.networking import NetworkGroup, QM9700NDRInfiniBandSwitch

from .compute_tray import GB200ComputeTray
from .rack_scale import RackScaleSystem


class GB200NVL72RackScaleSystem(RackScaleSystem):
    system_type: ClassVar[str] = "GB200NVL72RackScaleSystem"
    compute_tray: GB200ComputeTray = GB200ComputeTray()
    default_networking: ClassVar[tuple[NetworkGroup, ...]] = (
        NetworkGroup(
            gear=QM9700NDRInfiniBandSwitch(),
            quantity=3.375,
            details=(
                ("topology", "two-tier"),
                ("switches_per_rack", 3.375),
                ("allocation_basis", "User-specified: 3.375 QM9700 switches per rack"),
            ),
        ),
    )
