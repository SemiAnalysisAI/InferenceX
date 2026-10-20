# SPDX-License-Identifier: GPL-3.0-only
"""Advanced cluster and equipment power modeling."""

from power_model.models.advanced.cluster import Cluster
from power_model.models.advanced.model import OSSAllinPowerModel
from power_model.models.advanced.networking import (
    Generic512TEthernetSwitch,
    NetworkGroup,
    QM9700NDRInfiniBandSwitch,
    QM9790NDRInfiniBandSwitch,
    ScaleOutNetworkingGear,
)
from power_model.models.advanced.systems.base import SystemGroup

__all__ = [
    "Cluster",
    "Generic512TEthernetSwitch",
    "NetworkGroup",
    "OSSAllinPowerModel",
    "QM9700NDRInfiniBandSwitch",
    "QM9790NDRInfiniBandSwitch",
    "ScaleOutNetworkingGear",
    "SystemGroup",
]
