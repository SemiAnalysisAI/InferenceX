# SPDX-License-Identifier: GPL-3.0-only
"""Shared inputs, auditable results, and the per-GPU estimation contract."""

from abc import ABC, abstractmethod
from enum import StrEnum
from math import fsum, isclose
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    field_validator,
    model_validator,
)

from power_model.cooling import CoolingMode, CoolingProfile

Watts = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
PositiveCount = Annotated[int, Field(strict=True, gt=0)]
PositiveQuantity = PositiveCount | Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
NonemptyString = Annotated[str, Field(min_length=1)]
Details = tuple[tuple[str, str | int | float], ...]
Scope = Literal["per_gpu_reference", "cluster"]
_WATTS = TypeAdapter(Watts)
_CPU_OFFLOAD = TypeAdapter(bool)
MODEL_VERSION = "0.1.0"


def validate_watts(value: float) -> float:
    """Reject invalid power, including non-finite arithmetic results."""
    return _WATTS.validate_python(value)


def validate_cpu_offload(value: bool) -> bool:
    """Require a boolean even when evaluating a system directly."""
    return _CPU_OFFLOAD.validate_python(value, strict=True)


def sum_watts(values: tuple[float, ...]) -> float:
    try:
        return validate_watts(fsum(values))
    except OverflowError as error:
        raise ValueError("Power total exceeds the supported finite range") from error


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", validate_default=True)


class WorkloadState(StrEnum):
    IDLE = "idle"
    FIXED_SEQ_LEN = "fixed-seq-len"
    AGENTIC = "agentic"
    AGENTIC_CPU_OFFLOADING = "agentic-cpu-offloading"


class OperatingState(FrozenModel):
    """One workload and scale-out policy shared by the whole modeled cluster."""

    workload_state: WorkloadState = WorkloadState.FIXED_SEQ_LEN
    using_scale_out: bool = False

    @field_validator("workload_state", mode="before")
    @classmethod
    def parse_workload(cls, value: object) -> WorkloadState:
        if not isinstance(value, str):
            raise ValueError("workload_state must be a supported workload name")
        return WorkloadState(value)

    @property
    def cpu_offload(self) -> bool:
        return self.workload_state == WorkloadState.AGENTIC_CPU_OFFLOADING

    @property
    def nic_state(self) -> Literal["active", "idle"]:
        return "active" if self.using_scale_out else "idle"


DEFAULT_OPERATING_STATE = OperatingState()


class Provenance(FrozenModel):
    """Source and measurement boundary attached to an input or modeling assumption."""

    profile_id: NonemptyString
    version: NonemptyString
    source: NonemptyString
    kind: Literal["input", "assumed", "measured", "estimated"]
    input_boundary: NonemptyString
    assumptions: tuple[str, ...] = ()


class PowerComponentBreakdown(FrozenModel):
    """Totals across this scope; quantities may be fractional shares of external equipment."""

    name: NonemptyString
    power_w: Watts
    quantity: PositiveQuantity = 1
    children: tuple["PowerComponentBreakdown", ...] = ()
    provenance: Provenance | None = None
    details: Details = ()
    configuration: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def reconcile_children(self) -> Self:
        if self.children and not isclose(
            self.power_w,
            sum_watts(tuple(child.power_w for child in self.children)),
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise ValueError("Parent power must equal the sum of child power")
        return self

    @classmethod
    def group(
        cls,
        name: str,
        children: tuple["PowerComponentBreakdown", ...],
        *,
        provenance: Provenance | None = None,
        details: Details = (),
        configuration: dict[str, JsonValue] | None = None,
    ) -> "PowerComponentBreakdown":
        return cls(
            name=name,
            power_w=sum_watts(tuple(child.power_w for child in children)),
            children=children,
            provenance=provenance,
            details=details,
            configuration=configuration,
        )

    def scaled(self, quantity: int | float) -> "PowerComponentBreakdown":
        """Scale a whole tree so every parent still sums its children exactly once."""
        count = TypeAdapter(PositiveQuantity).validate_python(quantity)
        return PowerComponentBreakdown(
            name=self.name,
            power_w=self.power_w * count,
            quantity=self.quantity * count,
            children=tuple(child.scaled(count) for child in self.children),
            provenance=self.provenance,
            details=self.details,
            configuration=self.configuration,
        )


class ITPowerBreakdown(FrozenModel):
    scope: Scope
    gpu_count: PositiveCount
    components: tuple[PowerComponentBreakdown, ...]

    @property
    def it_power_w(self) -> float:
        return sum_watts(tuple(component.power_w for component in self.components))


class PowerEstimate(FrozenModel):
    model_name: str
    model_version: str
    cpu_offload: bool
    workload_state: WorkloadState
    using_scale_out: bool
    gpu_level_power_per_gpu: Watts
    AllInPower_per_gpu: Watts
    scope: Scope
    gpu_count: PositiveCount
    cooling_mode: CoolingMode
    pue: Annotated[float, Field(ge=1, allow_inf_nan=False)]
    it_power_w: Watts
    facility_overhead_w: Watts
    facility_power_w: Watts
    components: tuple[PowerComponentBreakdown, ...]

    @model_validator(mode="after")
    def reconcile_totals(self) -> Self:
        if self.cpu_offload != (self.workload_state == WorkloadState.AGENTIC_CPU_OFFLOADING):
            raise ValueError("CPU offloading must agree with workload_state")
        checks = (
            (self.it_power_w, sum_watts(tuple(c.power_w for c in self.components))),
            (self.facility_power_w, self.it_power_w * self.pue),
            (self.facility_power_w, self.it_power_w + self.facility_overhead_w),
            (self.facility_power_w, self.AllInPower_per_gpu * self.gpu_count),
        )
        if any(not isclose(a, b, rel_tol=1e-12, abs_tol=1e-9) for a, b in checks):
            raise ValueError("Power estimate totals do not reconcile")
        return self


class PowerModel(OperatingState, ABC):
    cooling: CoolingProfile

    @property
    def operating_state(self) -> OperatingState:
        return OperatingState(
            workload_state=self.workload_state, using_scale_out=self.using_scale_out
        )

    def estimate(self, gpu_level_power_per_gpu: float) -> float:
        """Return AllInPower_per_gpu in watts per GPU."""
        return self.estimate_breakdown(gpu_level_power_per_gpu).AllInPower_per_gpu

    def estimate_breakdown(self, gpu_level_power_per_gpu: float) -> PowerEstimate:
        gpu_power = validate_watts(gpu_level_power_per_gpu)
        breakdown = self._estimate_it_power(gpu_power)
        it_power = breakdown.it_power_w
        facility_power = validate_watts(it_power * self.cooling.pue)
        return PowerEstimate(
            model_name=type(self).__name__,
            model_version=MODEL_VERSION,
            cpu_offload=self.cpu_offload,
            workload_state=self.workload_state,
            using_scale_out=self.using_scale_out,
            gpu_level_power_per_gpu=gpu_power,
            AllInPower_per_gpu=facility_power / breakdown.gpu_count,
            scope=breakdown.scope,
            gpu_count=breakdown.gpu_count,
            cooling_mode=self.cooling.mode,
            pue=self.cooling.pue,
            it_power_w=it_power,
            facility_overhead_w=facility_power - it_power,
            facility_power_w=facility_power,
            components=breakdown.components,
        )

    @abstractmethod
    def _estimate_it_power(self, gpu_level_power_per_gpu: float) -> ITPowerBreakdown:
        """Return scoped IT watts and the matching GPU count before PUE."""
