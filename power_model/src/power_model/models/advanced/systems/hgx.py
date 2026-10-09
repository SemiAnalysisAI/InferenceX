# SPDX-License-Identifier: GPL-3.0-only
"""HGX chassis with a GPU-bearing UBB and a shared CPU, NIC, DIMM, and drive inventory."""

from typing import ClassVar, Self

from pydantic import Field, JsonValue, SerializeAsAny, model_validator

from power_model.base import (
    DEFAULT_OPERATING_STATE,
    OperatingState,
    PositiveCount,
    PowerComponentBreakdown,
    WorkloadState,
    sum_watts,
)
from power_model.cooling import CoolingProfile
from power_model.models.advanced.components.base import ComponentGroup
from power_model.models.advanced.components.fan import FanPower
from power_model.models.advanced.components.generic import (
    CPU,
    DDR5DIMM,
    X86CPU,
    NVMeDrive,
    PCIeSwitch,
    PEX89144PCIeSwitch,
    SystemSide400GTransceiver,
    SystemSide800GTransceiver,
    SystemSideTransceiver,
)
from power_model.models.advanced.components.hgx_fans import (
    HGXFanPolicy,
    NormalizedHGXFanPower,
)
from power_model.models.advanced.components.psu import (
    B200PSUEfficiency,
    B300PSUEfficiency,
    HopperPSUEfficiency,
    MI300PSUEfficiency,
    MI325PSUEfficiency,
    MI355PSUEfficiency,
    PSUEfficiencyModel,
)
from power_model.models.advanced.components.scaleoutnic import (
    ConnectX7NICCard,
    PollaraNICCard,
    ScaleOutNICCard,
    Thor2NICCard,
)
from power_model.models.advanced.components.ubb import (
    MI300UBB,
    MI325UBB,
    MI355UBB,
    B200HGXBoard,
    B300HGXBoard,
    HopperHGXBoard,
    UniversalBaseBoard,
)
from power_model.models.advanced.networking import (
    AMD_TWO_TIER_NETWORK,
    B300_TWO_TIER_NETWORK,
    QM9790_TWO_TIER_NETWORK,
    NetworkGroup,
)
from power_model.models.advanced.systems.base import GPUSystem, _profile_parameters

DESIGN_OPERATING_STATE = OperatingState(
    workload_state=WorkloadState.AGENTIC_CPU_OFFLOADING, using_scale_out=True
)


class HGXSystemChassis(GPUSystem):
    default_cooling: ClassVar[CoolingProfile] = CoolingProfile(mode="air")
    cpu_count: ClassVar[int] = 2
    dimm_count: ClassVar[int] = 32
    dimm_capacity_gb: ClassVar[int] = 64
    components: tuple[ComponentGroup, ...] = ()
    default_nic_type: ClassVar[type[ScaleOutNICCard] | None]
    system_nic_count: ClassVar[int] = 8
    default_transceiver_type: ClassVar[type[SystemSideTransceiver]] = SystemSide400GTransceiver
    transceiver_count: ClassVar[int] = 8
    pex89144_count: ClassVar[int] = 0
    gpu_count: PositiveCount = 8
    nvme_count: PositiveCount = 10
    ubb: SerializeAsAny[UniversalBaseBoard]
    psu: SerializeAsAny[PSUEfficiencyModel]
    fan_policy: HGXFanPolicy = HGXFanPolicy()

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        if self.gpu_count != 8 or self.ubb.gpu_count != self.gpu_count:
            raise ValueError("An HGX chassis and its UBB must both contain exactly 8 GPUs")
        if self.nvme_count != 10:
            raise ValueError("An HGX chassis must contain exactly 10 NVMe drives")
        for group in self.components:
            if isinstance(
                group.component,
                (
                    CPU,
                    DDR5DIMM,
                    ScaleOutNICCard,
                    SystemSideTransceiver,
                    NVMeDrive,
                    PEX89144PCIeSwitch,
                ),
            ):
                raise ValueError(
                    f"{type(group.component).__name__} inventory is owned by {self.system_type}; "
                    "pass workload_state and using_scale_out to the power model instead"
                )
        return self

    def _component_groups(self, state: OperatingState) -> tuple[ComponentGroup, ...]:
        groups = (
            ComponentGroup(
                component=X86CPU.for_workload(state.workload_state), quantity=self.cpu_count
            ),
        )
        if self.system_nic_count:
            if self.default_nic_type is None:
                raise ValueError("Chassis-level NIC inventory requires a NIC type")
            groups += (
                ComponentGroup(
                    component=self.default_nic_type(state=state.nic_state),
                    quantity=self.system_nic_count,
                ),
            )
        groups += (
            ComponentGroup(
                component=DDR5DIMM(capacity_gb=self.dimm_capacity_gb).for_cpu_offload(
                    state.cpu_offload
                ),
                quantity=self.dimm_count,
            ),
            ComponentGroup(component=NVMeDrive(), quantity=self.nvme_count),
        )
        if self.pex89144_count:
            groups += (
                ComponentGroup(component=PEX89144PCIeSwitch(), quantity=self.pex89144_count),
            )
        groups += (
            ComponentGroup(
                component=self.default_transceiver_type(), quantity=self.transceiver_count
            ),
        )
        return (*groups, *self.components)

    def _finish_system(
        self,
        system_components: tuple[PowerComponentBreakdown, ...],
        *,
        operating_state: OperatingState,
    ) -> PowerComponentBreakdown:
        state = OperatingState.model_validate(operating_state)
        groups = self._component_groups(state)
        components = self._power_components(
            (
                *system_components,
                PowerComponentBreakdown.group(
                    "Generic components",
                    tuple(group.estimate_breakdown() for group in groups),
                ),
            )
        )
        configuration = self._configuration(state)
        configuration["components"] = [_profile_parameters(group) for group in groups]
        return PowerComponentBreakdown.group(
            self.system_type,
            components,
            provenance=self.provenance,
            details=(("gpu_count_per_system", self.gpu_count),),
            configuration=configuration,
        )

    def _configuration(self, state: OperatingState) -> dict[str, JsonValue]:
        return super()._configuration(state) | {
            "ubb": _profile_parameters(self.ubb.for_operating_state(state)),
        }

    @property
    def resolved_fan_power(self) -> FanPower:
        if self.ubb.gpu_tdp_w is None:
            raise ValueError("Automatic HGX fan sizing requires a GPU TDP on the UBB")
        board = self.ubb.for_operating_state(DESIGN_OPERATING_STATE)
        host_watts = sum_watts(
            (
                board.estimate_breakdown(0).power_w,
                *(
                    group.estimate_breakdown().power_w
                    for group in self._component_groups(DESIGN_OPERATING_STATE)
                ),
            )
        )
        return NormalizedHGXFanPower(
            gpu_tdp_w=board.gpu_tdp_w,
            gpu_count=self.gpu_count,
            design_non_gpu_power_w=host_watts,
            policy=self.fan_policy,
        )

    def _power_components(
        self, components: tuple[PowerComponentBreakdown, ...]
    ) -> tuple[PowerComponentBreakdown, ...]:
        component_dc_w = sum_watts(tuple(node.power_w for node in components))
        fans = self.resolved_fan_power.estimate_breakdown(component_dc_w)
        dc_load_w = sum_watts((component_dc_w, fans.power_w))
        losses = self.psu.estimate_loss_breakdown(dc_load_w)
        return (*components, losses, fans)

    def estimate_it_power(
        self,
        gpu_level_power_per_gpu: float,
        *,
        cpu_and_dram_measured_power_per_socket: float | None = None,
        operating_state: OperatingState = DEFAULT_OPERATING_STATE,
    ) -> PowerComponentBreakdown:
        state = OperatingState.model_validate(operating_state)
        board = self.ubb.for_operating_state(state)
        return self._finish_system(
            (board.estimate_breakdown(gpu_level_power_per_gpu),), operating_state=state
        )


class MI300HGXSystemChassis(HGXSystemChassis):
    default_networking: ClassVar[tuple[NetworkGroup, ...]] = (AMD_TWO_TIER_NETWORK,)
    psu: SerializeAsAny[PSUEfficiencyModel] = Field(default_factory=MI300PSUEfficiency)
    pex89144_count: ClassVar[int] = 4
    system_type: ClassVar[str] = "MI300HGXSystemChassis"
    default_nic_type: ClassVar[type[ScaleOutNICCard]] = Thor2NICCard
    ubb: SerializeAsAny[MI300UBB] = Field(default_factory=MI300UBB)


class MI325HGXSystemChassis(HGXSystemChassis):
    default_networking: ClassVar[tuple[NetworkGroup, ...]] = (AMD_TWO_TIER_NETWORK,)
    psu: SerializeAsAny[PSUEfficiencyModel] = Field(default_factory=MI325PSUEfficiency)
    pex89144_count: ClassVar[int] = 4
    system_type: ClassVar[str] = "MI325HGXSystemChassis"
    default_nic_type: ClassVar[type[ScaleOutNICCard]] = Thor2NICCard
    ubb: SerializeAsAny[MI325UBB] = Field(default_factory=MI325UBB)


class MI355HGXSystemChassis(HGXSystemChassis):
    default_networking: ClassVar[tuple[NetworkGroup, ...]] = (AMD_TWO_TIER_NETWORK,)
    psu: SerializeAsAny[PSUEfficiencyModel] = Field(default_factory=MI355PSUEfficiency)
    pex89144_count: ClassVar[int] = 4
    system_type: ClassVar[str] = "MI355HGXSystemChassis"
    default_nic_type: ClassVar[type[ScaleOutNICCard]] = PollaraNICCard
    ubb: SerializeAsAny[MI355UBB] = Field(default_factory=MI355UBB)


class HopperHGXSystemChassis(HGXSystemChassis):
    """Shared HGX chassis model for H100 and H200 GPUs on a Hopper board."""

    psu: SerializeAsAny[PSUEfficiencyModel] = Field(default_factory=HopperPSUEfficiency)
    default_networking: ClassVar[tuple[NetworkGroup, ...]] = (QM9790_TWO_TIER_NETWORK,)
    pex89144_count: ClassVar[int] = 4
    system_type: ClassVar[str] = "HopperHGXSystemChassis"
    default_nic_type: ClassVar[type[ScaleOutNICCard]] = ConnectX7NICCard
    ubb: SerializeAsAny[HopperHGXBoard] = Field(default_factory=HopperHGXBoard)


class B200HGXSystemChassis(HGXSystemChassis):
    psu: SerializeAsAny[PSUEfficiencyModel] = Field(default_factory=B200PSUEfficiency)
    default_networking: ClassVar[tuple[NetworkGroup, ...]] = (QM9790_TWO_TIER_NETWORK,)
    pex89144_count: ClassVar[int] = 4
    system_type: ClassVar[str] = "B200HGXSystemChassis"
    default_nic_type: ClassVar[type[ScaleOutNICCard]] = ConnectX7NICCard
    ubb: SerializeAsAny[B200HGXBoard] = Field(default_factory=B200HGXBoard)


class B300HGXSystemChassis(HGXSystemChassis):
    default_networking: ClassVar[tuple[NetworkGroup, ...]] = (B300_TWO_TIER_NETWORK,)
    psu: SerializeAsAny[PSUEfficiencyModel] = Field(default_factory=B300PSUEfficiency)
    system_type: ClassVar[str] = "B300HGXSystemChassis"
    default_nic_type: ClassVar[type[ScaleOutNICCard] | None] = None
    system_nic_count: ClassVar[int] = 0
    default_transceiver_type: ClassVar[type[SystemSideTransceiver]] = SystemSide800GTransceiver
    ubb: SerializeAsAny[B300HGXBoard] = Field(default_factory=B300HGXBoard)

    @model_validator(mode="after")
    def no_chassis_pcie_switches(self) -> Self:
        if any(isinstance(group.component, PCIeSwitch) for group in self.components):
            raise ValueError("B300 has no chassis-level PCIe switches")
        return self
