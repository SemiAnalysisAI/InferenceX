# SPDX-License-Identifier: GPL-3.0-only
"""ConnectX-8 modeled at 800 Gbit/s."""

from typing import ClassVar

from power_model.base import Provenance
from power_model.models.advanced.components.scaleoutnic.nic import ScaleOutNICCard
from power_model.models.advanced.profiles import nic_power_baseline


class ConnectX8NICCard(ScaleOutNICCard):
    bandwidth_gbps: ClassVar[int] = 800
    idle_power_w: ClassVar[float] = 30.0
    active_power_w: ClassVar[float] = 50.0
    provenance: Provenance = nic_power_baseline("connectx8", idle_power_w, active_power_w)
