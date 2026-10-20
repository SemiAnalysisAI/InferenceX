# SPDX-License-Identifier: GPL-3.0-only
"""QM9700 NDR switch with the project's provisional Quantum box-power budget."""

from typing import ClassVar, Literal

from power_model.base import Provenance
from power_model.models.advanced.networking.optical_switch import OpticalScaleOutSwitch

QM9700_BASELINE = Provenance(
    profile_id="qm9700-ndr-infiniband-switch",
    version="1",
    source="User-specified QM9700 fabric; existing project Quantum power assumptions",
    kind="estimated",
    input_boundary="External switch AC power budget including switch-side optics; excludes PUE",
    assumptions=(
        "QM9700 NDR InfiniBand switch bandwidth is 25.6 Tbit/s, with 32 OSFP cages.",
        "Reuses the project Quantum box budget of 600 W idle / 783 W active, excluding optics.",
        "These state watts are a proxy, not QM9700 measurements or NVIDIA typical-load ratings.",
        "32 always-powered 800G SR8 optics use the shared 15 W per-optic assumption.",
        "Box power includes cooling and conversion; do not apply rack or HGX PSU models.",
    ),
)


class QM9700NDRInfiniBandSwitch(OpticalScaleOutSwitch):
    name: Literal["QM9700NDRInfiniBandSwitch"] = "QM9700NDRInfiniBandSwitch"
    bandwidth_tbps: ClassVar[float] = 25.6
    idle_power_w: ClassVar[float] = 600.0
    active_power_w: ClassVar[float] = 783.0
    optics_count: ClassVar[int] = 32
    box_name: ClassVar[str] = "QM9700 switch box"
    protocol: ClassVar[str] = "NDR InfiniBand"
    switch_provenance: ClassVar[Provenance] = QM9700_BASELINE
    provenance: Provenance = QM9700_BASELINE
