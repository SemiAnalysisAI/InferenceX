# SPDX-License-Identifier: GPL-3.0-only
"""GB300 NVL72 and its endpoint-based external fabric allocation."""

from typing import ClassVar

from power_model.models.advanced.networking import Generic512TEthernetSwitch, NetworkGroup

from .compute_tray import GB300ComputeTray
from .rack_scale import RackScaleSystem


class GB300NVL72RackScaleSystem(RackScaleSystem):
    system_type: ClassVar[str] = "GB300NVL72RackScaleSystem"
    compute_tray: GB300ComputeTray = GB300ComputeTray()
    default_networking: ClassVar[tuple[NetworkGroup, ...]] = (
        NetworkGroup(
            gear=Generic512TEthernetSwitch(),
            quantity=3.375,
            details=(
                ("topology", "two-tier"),
                ("switches_per_rack", 3.375),
                ("allocation_basis", "User-specified: 3.375 51.2T Ethernet switches per rack"),
            ),
        ),
    )
