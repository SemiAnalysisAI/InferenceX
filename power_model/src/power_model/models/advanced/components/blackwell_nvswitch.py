# SPDX-License-Identifier: GPL-3.0-only
"""Rack NVLink switch ASIC, separate from external scale-out gear."""

from power_model.base import Details, Provenance, Watts
from power_model.models.advanced.components.base import FixedPowerComponent


class BlackwellNVSwitch(FixedPowerComponent):
    power_w: Watts = 200.0
    provenance: Provenance = Provenance(
        profile_id="rack-blackwell-nvswitch",
        version="1",
        kind="estimated",
        source="Project Blackwell NVSwitch budget",
        input_boundary="One rack NVSwitch electrical load downstream of tray conversion",
        assumptions=(
            "28.8 Tbit/s; 200 W borrowed provisionally from the HGX switch assumption.",
            "Always powered; external scale-out state does not disable NVLink.",
            "Local switch regulation is included in this provisional electrical budget.",
        ),
    )

    def _details(self) -> Details:
        return (("bandwidth_tbps", 28.8), ("power_policy", "always_on"))
