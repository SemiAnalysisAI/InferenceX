# SPDX-License-Identifier: GPL-3.0-only
"""Scale-out NIC types and their shared two-state interface."""

from power_model.models.advanced.components.scaleoutnic.connectx7 import ConnectX7NICCard
from power_model.models.advanced.components.scaleoutnic.connectx8 import ConnectX8NICCard
from power_model.models.advanced.components.scaleoutnic.nic import NICState, ScaleOutNICCard
from power_model.models.advanced.components.scaleoutnic.pollara import PollaraNICCard
from power_model.models.advanced.components.scaleoutnic.thor2 import Thor2NICCard

__all__ = [
    "ConnectX7NICCard",
    "ConnectX8NICCard",
    "NICState",
    "PollaraNICCard",
    "ScaleOutNICCard",
    "Thor2NICCard",
]
