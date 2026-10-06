# SPDX-License-Identifier: GPL-3.0-only
"""System selection shared by the model API and presentation layers."""

from power_model.models.advanced.systems.base import GPUSystem
from power_model.models.advanced.systems.hgx import (
    B200HGXSystemChassis,
    B300HGXSystemChassis,
    HopperHGXSystemChassis,
    MI300HGXSystemChassis,
    MI325HGXSystemChassis,
    MI355HGXSystemChassis,
)
from power_model.models.advanced.systems.rack_scale import (
    GB200NVL72RackScaleSystem,
    GB300NVL72RackScaleSystem,
)

SYSTEMS: dict[str, type[GPUSystem]] = {
    "mi300": MI300HGXSystemChassis,
    "mi325": MI325HGXSystemChassis,
    "mi355": MI355HGXSystemChassis,
    "hopper": HopperHGXSystemChassis,
    "b200": B200HGXSystemChassis,
    "b300": B300HGXSystemChassis,
    "gb200-nvl72": GB200NVL72RackScaleSystem,
    "gb300-nvl72": GB300NVL72RackScaleSystem,
}


def system_name(value: str) -> str:
    aliases = {system.__name__.lower(): name for name, system in SYSTEMS.items()}
    aliases.update(
        {"h100": "hopper", "h200": "hopper", "gb200": "gb200-nvl72", "gb300": "gb300-nvl72"}
    )
    return aliases.get(value.lower(), value.lower())


def get_system_class(system: str | type[GPUSystem]) -> type[GPUSystem]:
    if isinstance(system, type) and issubclass(system, GPUSystem):
        return system
    if isinstance(system, str) and system_name(system) in SYSTEMS:
        return SYSTEMS[system_name(system)]
    raise ValueError(f"Unknown system: {system}")
