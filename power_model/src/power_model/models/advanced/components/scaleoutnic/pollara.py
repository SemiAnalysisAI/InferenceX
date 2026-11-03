# SPDX-License-Identifier: GPL-3.0-only
"""Pollara modeled at 400 Gbit/s."""

from typing import ClassVar

from power_model.base import Provenance
from power_model.models.advanced.components.scaleoutnic.nic import ScaleOutNICCard
from power_model.models.advanced.profiles import POLLARA_NIC_BASELINE


class PollaraNICCard(ScaleOutNICCard):
    bandwidth_gbps: ClassVar[int] = 400
    idle_power_w: ClassVar[float] = 34.5
    active_power_w: ClassVar[float] = 34.5
    provenance: Provenance = POLLARA_NIC_BASELINE
