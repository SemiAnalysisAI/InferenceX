# SPDX-License-Identifier: GPL-3.0-only
"""GPU system families and their common grouping interface."""

from power_model.models.advanced.systems.base import GPUSystem, SystemGroup
from power_model.models.advanced.systems.hgx import (
    B200HGXSystemChassis,
    B300HGXSystemChassis,
    HGXSystemChassis,
    HopperHGXSystemChassis,
    MI300HGXSystemChassis,
    MI325HGXSystemChassis,
    MI355HGXSystemChassis,
)
from power_model.models.advanced.systems.rack_scale import (
    GB200NVL72RackScaleSystem,
    GB300NVL72RackScaleSystem,
    RackScaleSystem,
)

__all__ = [
    "B200HGXSystemChassis",
    "B300HGXSystemChassis",
    "GB200NVL72RackScaleSystem",
    "GB300NVL72RackScaleSystem",
    "GPUSystem",
    "HGXSystemChassis",
    "HopperHGXSystemChassis",
    "MI300HGXSystemChassis",
    "MI325HGXSystemChassis",
    "MI355HGXSystemChassis",
    "RackScaleSystem",
    "SystemGroup",
]
