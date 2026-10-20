# SPDX-License-Identifier: GPL-3.0-only
"""External network devices and allocations, separate from chassis power."""

from power_model.models.advanced.networking.base import NetworkGroup, ScaleOutNetworkingGear
from power_model.models.advanced.networking.ethernet_512t import (
    AMD_TWO_TIER_NETWORK,
    B300_TWO_TIER_NETWORK,
    Generic512TEthernetSwitch,
)
from power_model.models.advanced.networking.qm9700 import QM9700NDRInfiniBandSwitch
from power_model.models.advanced.networking.qm9790 import (
    QM9790_TWO_TIER_NETWORK,
    QM9790NDRInfiniBandSwitch,
)

__all__ = [
    "AMD_TWO_TIER_NETWORK",
    "B300_TWO_TIER_NETWORK",
    "Generic512TEthernetSwitch",
    "NetworkGroup",
    "QM9700NDRInfiniBandSwitch",
    "ScaleOutNetworkingGear",
    "QM9790NDRInfiniBandSwitch",
    "QM9790_TWO_TIER_NETWORK",
]
