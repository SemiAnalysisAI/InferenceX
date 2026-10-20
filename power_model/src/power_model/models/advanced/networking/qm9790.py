# SPDX-License-Identifier: GPL-3.0-only
"""QM9790 NDR switch box, its optics, and the agreed two-tier allocation."""

from typing import ClassVar, Literal

from power_model.base import Provenance
from power_model.models.advanced.networking.base import NetworkGroup
from power_model.models.advanced.networking.optical_switch import OpticalScaleOutSwitch

QM9790_BASELINE = Provenance(
    profile_id="qm9790-ndr-infiniband-switch",
    version="1",
    source="User-specified model baseline",
    kind="assumed",
    input_boundary="External switch AC power budget including switch-side optics; excludes PUE",
    assumptions=(
        "QM9790 NDR InfiniBand switch bandwidth is 25.6 Tbit/s.",
        "The switch box uses 600 W idle or 783 W active, excluding separately modeled optics.",
        "32 always-powered 800G SR8 optics use the shared 15 W per-optic assumption.",
        "The box budget includes its cooling and conversion losses; do not add HGX fan/PSU models.",
    ),
)


class QM9790NDRInfiniBandSwitch(OpticalScaleOutSwitch):
    name: Literal["QM9790NDRInfiniBandSwitch"] = "QM9790NDRInfiniBandSwitch"
    bandwidth_tbps: ClassVar[float] = 25.6
    idle_power_w: ClassVar[float] = 600.0
    active_power_w: ClassVar[float] = 783.0
    optics_count: ClassVar[int] = 32
    box_name: ClassVar[str] = "QM9790 switch box"
    protocol: ClassVar[str] = "NDR InfiniBand"
    switch_provenance: ClassVar[Provenance] = QM9790_BASELINE
    provenance: Provenance = QM9790_BASELINE


QM9790_TWO_TIER_NETWORK = NetworkGroup(
    gear=QM9790NDRInfiniBandSwitch(),
    quantity=0.375,
    details=(("topology", "two-tier"), ("switches_per_chassis", 0.375)),
)
