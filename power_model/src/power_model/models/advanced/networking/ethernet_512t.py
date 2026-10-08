# SPDX-License-Identifier: GPL-3.0-only
"""Generic 51.2 Tbit/s Ethernet switch and two-tier allocations for AMD and B300."""

from typing import ClassVar, Literal

from power_model.base import Provenance
from power_model.models.advanced.networking.base import NetworkGroup
from power_model.models.advanced.networking.optical_switch import OpticalScaleOutSwitch

ETHERNET_512T_BASELINE = Provenance(
    profile_id="generic-51.2t-ethernet-switch",
    version="1",
    source="User-specified model baseline",
    kind="assumed",
    input_boundary="External switch ASIC plus switch-side optics power budget; excludes PUE",
    assumptions=(
        "Generic Ethernet switch bandwidth is 51.2 Tbit/s.",
        "The ASIC uses 539 W idle or 900 W active, excluding separately modeled optics.",
        "64 always-powered 800G SR8 optics use the shared 15 W per-optic assumption.",
        "Use the specified ASIC-plus-optics budget as external IT power; "
        "no HGX fan/PSU model applies.",
    ),
)


class Generic512TEthernetSwitch(OpticalScaleOutSwitch):
    name: Literal["Generic512TEthernetSwitch"] = "Generic512TEthernetSwitch"
    bandwidth_tbps: ClassVar[float] = 51.2
    idle_power_w: ClassVar[float] = 539.0
    active_power_w: ClassVar[float] = 900.0
    optics_count: ClassVar[int] = 64
    box_name: ClassVar[str] = "Generic 51.2T Ethernet switch ASIC"
    protocol: ClassVar[str] = "Ethernet"
    switch_provenance: ClassVar[Provenance] = ETHERNET_512T_BASELINE
    provenance: Provenance = ETHERNET_512T_BASELINE


AMD_TWO_TIER_NETWORK = NetworkGroup(
    gear=Generic512TEthernetSwitch(),
    quantity=0.1875,
    details=(("topology", "two-tier"), ("switches_per_chassis", 0.1875)),
)

B300_TWO_TIER_NETWORK = NetworkGroup(
    gear=Generic512TEthernetSwitch(),
    quantity=0.375,
    details=(("topology", "two-tier"), ("switches_per_chassis", 0.375)),
)
