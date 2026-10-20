# SPDX-License-Identifier: GPL-3.0-only
"""Shared state selection and accounting for switches with always-powered optics."""

from typing import ClassVar, Literal, Self

from pydantic import model_validator

from power_model.base import OperatingState, PowerComponentBreakdown, Provenance, sum_watts
from power_model.models.advanced.components.base import ComponentGroup
from power_model.models.advanced.components.generic import SwitchSide800GSR8Transceiver
from power_model.models.advanced.networking.base import ScaleOutNetworkingGear


class OpticalScaleOutSwitch(ScaleOutNetworkingGear):
    state: Literal["idle", "active"] = "idle"
    bandwidth_tbps: ClassVar[float]
    idle_power_w: ClassVar[float]
    active_power_w: ClassVar[float]
    optics_count: ClassVar[int]
    box_name: ClassVar[str]
    protocol: ClassVar[str]
    switch_provenance: ClassVar[Provenance]

    @classmethod
    def _components(cls, state: str) -> tuple[PowerComponentBreakdown, ...]:
        return (
            PowerComponentBreakdown(
                name=cls.box_name,
                power_w=cls.active_power_w if state == "active" else cls.idle_power_w,
                provenance=cls.switch_provenance,
                details=(("state", state), ("bandwidth_tbps", cls.bandwidth_tbps)),
            ),
            ComponentGroup(
                component=SwitchSide800GSR8Transceiver(), quantity=cls.optics_count
            ).estimate_breakdown(),
        )

    @model_validator(mode="before")
    @classmethod
    def derive_switch_power(cls, values: object) -> object:
        if isinstance(values, dict) and "power_w" not in values:
            state = values.get("state", "idle")
            if isinstance(state, str) and state in ("idle", "active"):
                return values | {
                    "power_w": sum_watts(tuple(node.power_w for node in cls._components(state)))
                }
        return values

    @model_validator(mode="after")
    def enforce_switch_power(self) -> Self:
        expected = sum_watts(tuple(node.power_w for node in self._components(self.state)))
        if self.power_w != expected:
            raise ValueError(
                f"{self.name} power is determined by switch state "
                f"and its {self.optics_count} optics"
            )
        return self

    def for_operating_state(self, state: OperatingState) -> Self:
        return type(self).model_validate(
            self.model_dump(exclude={"power_w"}) | {"state": state.nic_state}
        )

    def estimate_breakdown(self) -> PowerComponentBreakdown:
        return PowerComponentBreakdown.group(
            self.name,
            self._components(self.state),
            provenance=self.provenance,
            details=(
                ("state", self.state),
                ("bandwidth_tbps", self.bandwidth_tbps),
                ("protocol", self.protocol),
                ("optics_per_switch", self.optics_count),
            ),
        )
