# SPDX-License-Identifier: GPL-3.0-only
"""Versioned provenance for explicit inputs and the agreed model baselines."""

from power_model.base import Provenance


def nic_power_baseline(nic_name: str, idle_power_w: float, active_power_w: float) -> Provenance:
    return Provenance(
        profile_id=f"{nic_name}-state-power",
        version="1",
        source="User-specified model baseline",
        kind="assumed",
        input_boundary="One scale-out NIC; excludes separately modeled transceivers and switches",
        assumptions=(
            f"Idle power is {idle_power_w:g} W per card.",
            f"Power when using scale-out networking is {active_power_w:g} W per card.",
        ),
    )


def system_side_transceiver_baseline(bandwidth_gbps: int, power_w: float) -> Provenance:
    return Provenance(
        profile_id=f"system-side-{bandwidth_gbps}g-transceiver",
        version="1",
        source="User-specified model baseline",
        kind="assumed",
        input_boundary="One system-side optical transceiver; excludes NIC and switch-side optics",
        assumptions=(
            f"Power is {power_w:g} W per {bandwidth_gbps} Gbit/s transceiver.",
            "One transceiver per scale-out NIC, including board-mounted NICs.",
            "Always powered, including when scale-out networking is off.",
        ),
    )


COMPONENT_INPUT = Provenance(
    profile_id="explicit-component-input",
    version="1",
    source="Caller-supplied component power",
    kind="input",
    input_boundary="Component electrical input; excludes separately modeled components",
)
GPU_INPUT = Provenance(
    profile_id="gpu-input",
    version="1",
    source="gpu_level_power_per_gpu argument",
    kind="input",
    input_boundary="GPU electrical power, excluding board and host overhead",
)
NVME_BASELINE = Provenance(
    profile_id="hgx-idle-nvme",
    version="1",
    source="User-specified model baseline",
    kind="assumed",
    input_boundary="One NVMe drive",
    assumptions=("Idle is the only supported state.", "Idle power is 5 W per drive."),
)
DDR5_BASELINE = Provenance(
    profile_id="ddr5-dimm-offload",
    version="1",
    source="User-specified model baseline",
    kind="assumed",
    input_boundary="One DDR5 system-memory DIMM; excludes CPU and fan power",
    assumptions=(
        "Idle power is 3.8 W per DIMM.",
        "Active power during CPU offloading is 7 W per DIMM.",
        "The model workload_state determines CPU offloading and the state of all DDR5 DIMMs.",
    ),
)
X86_CPU_BASELINE = Provenance(
    profile_id="x86-cpu-state",
    version="1",
    source="User-specified model baseline",
    kind="assumed",
    input_boundary="One x86 CPU; excludes separately modeled DIMMs and fans",
    assumptions=(
        "Idle power is 80 W per CPU.",
        "FixedSeqLen power is 120 W per CPU.",
        "Agentic power is 150 W per CPU, or 200 W with CPU offloading enabled.",
    ),
)
FAN_INPUT = Provenance(
    profile_id="explicit-fan-curve",
    version="1",
    source="Caller-supplied fan relationship",
    kind="estimated",
    input_boundary="System fans, excluding facility cooling",
)
HOPPER_BOARD_BASELINE = Provenance(
    profile_id="hopper-hgx-board",
    version="1",
    source="User-specified model baseline",
    kind="assumed",
    input_boundary="HGX board; excludes host components, fans, and conversion losses",
    assumptions=(
        "Eight GPUs at the supplied GPU-level power.",
        "Four Hopper HGX NVSwitches and eight PCIe 5.0 x16 retimers.",
    ),
)
B200_BOARD_BASELINE = Provenance(
    profile_id="b200-hgx-board",
    version="1",
    source="User-specified model baseline",
    kind="assumed",
    input_boundary="HGX board; excludes chassis PCIe switches, host components, fans, and losses",
    assumptions=(
        "Eight B200 GPUs at the supplied GPU-level power.",
        "Two Blackwell HGX NVSwitches at 200 W each, rated at 28.8 Tbit/s per switch.",
        "Eight PCIe 5.0 x16 retimers at 12 W each; non-GPU board power totals 496 W.",
    ),
)
B300_BOARD_BASELINE = Provenance(
    profile_id="b300-hgx-board",
    version="1",
    source="User-specified model baseline",
    kind="assumed",
    input_boundary="HGX board including NICs; excludes other host components, fans, and losses",
    assumptions=(
        "Eight B300 GPUs at the supplied GPU-level power.",
        "Two Blackwell HGX NVSwitches at 200 W and 28.8 Tbit/s each.",
        "Eight board-mounted ConnectX-8 NICs: 30 W idle or 50 W active per card.",
        "No PCIe retimers, chassis-level NICs, or chassis PCIe switches in this B300 inventory.",
    ),
)
BLACKWELL_NVSWITCH_BASELINE = Provenance(
    profile_id="blackwell-hgx-nvswitch",
    version="1",
    source="User-specified model baseline",
    kind="assumed",
    input_boundary="One Blackwell HGX NVSwitch",
    assumptions=("Power is 200 W per switch.", "Bandwidth is 28.8 Tbit/s per switch."),
)
AMD_UBB_BASELINE = Provenance(
    profile_id="amd-8way-ubb",
    version="1",
    source="User-specified model baseline",
    kind="assumed",
    input_boundary="AMD UBB; excludes chassis PCIe switches, host components, fans, and losses",
    assumptions=(
        "MI300, MI325, and MI355 boards each contain eight GPUs at the supplied GPU-level power.",
        "Eight PCIe 5.0 x16 retimers at 12 W each; non-GPU board power totals 96 W.",
    ),
)
HOPPER_NVSWITCH_BASELINE = Provenance(
    profile_id="hopper-hgx-nvswitch",
    version="1",
    source="User-specified model baseline",
    kind="assumed",
    input_boundary="One Hopper HGX NVSwitch",
    assumptions=("Power is 100 W per switch.", "Bandwidth is 12.8 Tbit/s per switch."),
)
PCIE5_RETIMER_BASELINE = Provenance(
    profile_id="pcie5-x16-retimer",
    version="1",
    source="User-specified model baseline",
    kind="assumed",
    input_boundary="One PCIe 5.0 x16 retimer",
    assumptions=("Power is 12 W per retimer.",),
)
PEX89144_BASELINE = Provenance(
    profile_id="broadcom-pex89144",
    version="1",
    source="User-specified model baseline",
    kind="assumed",
    input_boundary="One chassis-owned Broadcom PEX89144 PCIe switch; excludes UBB and fans",
    assumptions=(
        "PCIe 5.0 with 144 lanes.",
        "Always active at 45 W per switch.",
        "Four switches per Hopper, MI300, MI325, MI355, and B200 HGX chassis.",
    ),
)
