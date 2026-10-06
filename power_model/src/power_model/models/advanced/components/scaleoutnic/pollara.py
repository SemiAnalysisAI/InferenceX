# SPDX-License-Identifier: GPL-3.0-only
"""Pollara modeled at 400 Gbit/s."""

from typing import ClassVar

from power_model.base import Provenance
from power_model.models.advanced.components.scaleoutnic.nic import ScaleOutNICCard
from power_model.models.advanced.profiles import nic_power_baseline


class PollaraNICCard(ScaleOutNICCard):
    bandwidth_gbps: ClassVar[int] = 400
    idle_power_w: ClassVar[float] = 20.0
    active_power_w: ClassVar[float] = 30.0
    provenance: Provenance = nic_power_baseline("pollara", idle_power_w, active_power_w)
