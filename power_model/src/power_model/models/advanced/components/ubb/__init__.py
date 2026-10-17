# SPDX-License-Identifier: GPL-3.0-only
"""UBB/HGX board families, with each implementation in its own module."""

from power_model.models.advanced.components.ubb.amd_8way_ubb import AMD8WayUBB
from power_model.models.advanced.components.ubb.b200_hgx import B200HGXBoard
from power_model.models.advanced.components.ubb.b300_hgx import B300HGXBoard
from power_model.models.advanced.components.ubb.blackwell_hgx_nvswitch import BlackwellHGXNVSwitch
from power_model.models.advanced.components.ubb.hopper_hgx import HopperHGXBoard
from power_model.models.advanced.components.ubb.hopper_hgx_nvswitch import HopperHGXNVSwitch
from power_model.models.advanced.components.ubb.mi300_ubb import MI300UBB
from power_model.models.advanced.components.ubb.mi325_ubb import MI325UBB
from power_model.models.advanced.components.ubb.mi355_ubb import MI355UBB
from power_model.models.advanced.components.ubb.nvidia_8way_hgx import NVIDIA8WayHGXBoard
from power_model.models.advanced.components.ubb.ubb import UniversalBaseBoard

__all__ = [
    "AMD8WayUBB",
    "B200HGXBoard",
    "B300HGXBoard",
    "BlackwellHGXNVSwitch",
    "HopperHGXBoard",
    "HopperHGXNVSwitch",
    "MI300UBB",
    "MI325UBB",
    "MI355UBB",
    "NVIDIA8WayHGXBoard",
    "UniversalBaseBoard",
]
