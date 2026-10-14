# SPDX-License-Identifier: GPL-3.0-only
"""NVSwitch tray management CPU; independent of inference workload."""

from typing import Literal

from power_model.base import Provenance
from power_model.models.advanced.components.base import FixedPowerComponent


class EPYCEmbedded3151CPU(FixedPowerComponent):
    power_w: Literal[45.0] = 45.0
    provenance: Provenance = Provenance(
        profile_id="epyc-embedded-3151",
        version="1",
        kind="assumed",
        source="User-specified 45 W operating budget",
        input_boundary="One management CPU electrical budget after tray DC/DC; excludes memory",
        assumptions=(
            "The 45 W TDP is used as a fixed intermediate-rail operating budget in this model.",
            "Local CPU regulation is treated as included in this provisional budget.",
        ),
    )
