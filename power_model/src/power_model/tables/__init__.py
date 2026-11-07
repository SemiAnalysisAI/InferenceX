# SPDX-License-Identifier: GPL-3.0-only
"""The power model's constants and provenance as validated data tables.

Each YAML file in this directory holds one kind of table; systems/<id>.yaml holds one
system per catalog key. load_tables() validates every table's shape and every
cross-reference (provenance ids, parts, curves, assemblies, detail templates) before
anything is evaluated, so a bad table fails at load rather than inside an estimate.
"""

import json
import re
from functools import cache
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import Field, JsonValue, model_validator

from power_model.base import (
    FrozenModel,
    NonemptyString,
    PositiveCount,
    Provenance,
    Watts,
    WorkloadState,
)

TABLES_DIR = Path(__file__).parent
TABLE_FILES = ("parts", "network", "curves", "assemblies", "facility")

Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Fraction = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
PositiveFraction = Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
Category = Literal["cpu", "memory", "nic", "nvswitch", "storage", "other", "network"]
Details = dict[str, JsonValue]
Curve = tuple[tuple[PositiveFraction, PositiveFraction], ...]
TEMPLATE = re.compile(r"^\$(\w+)$")


def _increasing(points: Curve, what: str) -> None:
    if len(points) < 2 or any(a[0] >= b[0] for a, b in zip(points, points[1:], strict=False)):
        raise ValueError(f"{what} needs at least two points with increasing load fractions")


class Part(FrozenModel):
    """One unit of a leaf component, at a fixed power or a power per state."""

    name: NonemptyString
    category: Category
    provenance: NonemptyString
    power_w: Watts | None = None
    state_by: Literal["scale_out", "cpu_offload", "workload"] | None = None
    states_w: dict[str, Watts] = {}
    workload_states: dict[WorkloadState, str] = {}
    details: Details = {}

    @model_validator(mode="after")
    def one_power_source(self) -> Self:
        if (self.power_w is None) == (self.state_by is None):
            raise ValueError(f"{self.name} needs exactly one of power_w or state_by")
        if self.state_by == "workload":
            labels = set(self.workload_states.values())
            if set(self.workload_states) != set(WorkloadState) or labels != set(self.states_w):
                raise ValueError(f"{self.name} needs a state and watts for every workload")
        elif self.state_by is not None:
            if set(self.states_w) != {"active", "idle"} or self.workload_states:
                raise ValueError(f"{self.name} needs exactly active and idle watts")
        elif self.states_w or self.workload_states:
            raise ValueError(f"{self.name} has a fixed power_w and per-state watts")
        return self

    def variables(self) -> set[str]:
        """Names a details template may use besides the referencing entry's vars."""
        if self.state_by is None:
            return {"power_w"}
        return {"state", "power_w", *(f"{label}_w" for label in self.states_w)}


class PartUse(FrozenModel):
    part: NonemptyString
    count: PositiveCount
    vars: Details = {}


class TrayPartUse(PartUse):
    heat: Literal["air", "liquid"]


class Gear(FrozenModel):
    """One external switch: a box and its optics, in the switch's scale-out state."""

    name: NonemptyString
    provenance: NonemptyString
    details: Details = {}
    parts: tuple[PartUse, ...]


class EfficiencyCurve(FrozenModel):
    provenance: NonemptyString
    points: Curve

    @model_validator(mode="after")
    def ordered(self) -> Self:
        _increasing(self.points, "An efficiency curve")
        return self


class DCConverter(FrozenModel):
    provenance: NonemptyString
    module_capacity_w: Positive
    input_voltage_v: Positive
    no_load_loss_w: Watts
    efficiency: Curve

    @model_validator(mode="after")
    def full_range(self) -> Self:
        _increasing(self.efficiency, "A converter curve")
        if self.efficiency[-1][0] != 1:
            raise ValueError("A converter curve must reach full output")
        losses = [loss for _, loss in self.loss_points()]
        if any(a > b for a, b in zip(losses, losses[1:], strict=False)):
            raise ValueError("Converter loss must not decrease with output")
        return self

    def loss_points(self) -> tuple[tuple[float, float], ...]:
        """[output fraction, loss watts] from no load through full output."""
        capacity = self.module_capacity_w
        return ((0.0, self.no_load_loss_w),) + tuple(
            (fraction, fraction * capacity / efficiency - fraction * capacity)
            for fraction, efficiency in self.efficiency
        )


class HGXFans(FrozenModel):
    name: NonemptyString
    provenance: NonemptyString
    fan_complement: NonemptyString
    min_pwm: Fraction
    normal_max_pwm: PositiveFraction
    exponent: Positive
    ratio_at_design: PositiveFraction

    @model_validator(mode="after")
    def ordered_pwm(self) -> Self:
        if self.min_pwm > self.normal_max_pwm:
            raise ValueError("Minimum fan PWM must not exceed normal maximum PWM")
        return self


class RackFan(FrozenModel):
    provenance: NonemptyString
    rated_w: Watts
    min_speed: Fraction
    max_speed: Fraction

    @model_validator(mode="after")
    def ordered_speed(self) -> Self:
        if self.min_speed > self.max_speed:
            raise ValueError("Minimum fan speed must not exceed maximum fan speed")
        return self


class Quadratic(FrozenModel):
    a: float
    b1: float
    c: float


class LPDDR5X(FrozenModel):
    name: NonemptyString
    provenance: NonemptyString
    capacity_gb: PositiveInt
    frequency_mhz: PositiveInt
    max_bandwidth_gbps: Watts
    milliwatts: Quadratic


class GracePoint(FrozenModel):
    cpu_w: Watts
    memory_bandwidth_gbps: Watts


class GraceWorkload(FrozenModel):
    """Per-socket CPU watts and DRAM bandwidth; a missing workload is uncalibrated."""

    name: NonemptyString
    provenance: NonemptyString
    states: dict[WorkloadState, GracePoint]


class Curves(FrozenModel):
    psu_efficiency: dict[str, EfficiencyCurve]
    dc_converters: dict[str, DCConverter]
    hgx_fans: dict[str, HGXFans]
    rack_fans: dict[str, RackFan]
    lpddr5x: dict[str, LPDDR5X]
    grace_workload: dict[str, GraceWorkload]


class FanUse(FrozenModel):
    curve: NonemptyString
    count: PositiveCount
    design_air_heat_w: Positive


class Tray(FrozenModel):
    name: NonemptyString
    provenance: NonemptyString
    parts: tuple[TrayPartUse, ...]
    fans: FanUse
    converter: NonemptyString


class PowerShelves(FrozenModel):
    provenance: NonemptyString
    installed_shelves: PositiveCount
    active_shelves: PositiveCount
    surviving_shelves: PositiveCount
    psus_per_shelf: PositiveCount
    psu_capacity_w: Positive
    efficiency: NonemptyString
    psu_fans: FanUse
    controller_w: Watts

    @model_validator(mode="after")
    def ordered_banks(self) -> Self:
        if not self.surviving_shelves <= self.active_shelves <= self.installed_shelves:
            raise ValueError("Shelf counts require surviving <= active <= installed")
        return self


class Assemblies(FrozenModel):
    trays: dict[str, Tray]
    power_shelves: dict[str, PowerShelves]


class Labeled(FrozenModel):
    name: NonemptyString
    provenance: NonemptyString | Provenance


class PUE(FrozenModel):
    provenance: NonemptyString
    air: Annotated[float, Field(ge=1, allow_inf_nan=False)]
    liquid: Annotated[float, Field(ge=1, allow_inf_nan=False)]


class ExampleModel(FrozenModel):
    non_gpu_ratio: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    gpu: Labeled
    non_gpu: Labeled


class Facility(FrozenModel):
    pue: PUE
    example: ExampleModel


class NetworkAllocation(FrozenModel):
    gear: NonemptyString
    per_system: Positive
    details: Details = {}


class SystemTable(FrozenModel):
    id: NonemptyString
    name: NonemptyString
    aliases: tuple[str, ...] = ()
    provenance: NonemptyString
    cooling: Literal["air", "liquid"]
    gpu_count: PositiveCount
    network: NetworkAllocation


class HGXBoard(FrozenModel):
    name: NonemptyString
    provenance: NonemptyString
    gpu_tdp_w: PositiveInt
    parts: tuple[PartUse, ...]


class PSUBank(FrozenModel):
    name: NonemptyString
    provenance: NonemptyString
    efficiency: NonemptyString
    installed: PositiveCount
    load_sharing: PositiveCount
    redundant_capacity: PositiveCount
    capacity_w: Positive
    system_max_w: Positive | None = None
    redundancy: NonemptyString

    @model_validator(mode="after")
    def ordered_bank(self) -> Self:
        if not self.redundant_capacity <= self.load_sharing <= self.installed:
            raise ValueError("PSU counts must satisfy redundant <= load-sharing <= installed")
        return self


class HGXSystem(SystemTable):
    """An eight-GPU chassis: board, host parts, a PSU bank and normalized fans."""

    family: Literal["hgx-chassis"]
    board: HGXBoard
    host: tuple[PartUse, ...]
    psu: PSUBank
    fans: NonemptyString


class BiancaBoard(FrozenModel):
    name: NonemptyString
    provenance: NonemptyString
    count: PositiveCount
    gpu_count: PositiveCount
    gpu_tdp_w: PositiveInt
    gpu_family: NonemptyString
    cpu: NonemptyString
    memory: NonemptyString


class ComputeTray(Tray):
    count: PositiveCount
    board: BiancaBoard


class TrayCount(FrozenModel):
    tray: NonemptyString
    count: PositiveCount


class NVL72System(SystemTable):
    """A rack of compute and switch trays behind shared power shelves."""

    family: Literal["nvl72-rack"]
    compute_tray: ComputeTray
    switch_trays: TrayCount
    power_shelves: NonemptyString

    @model_validator(mode="after")
    def gpu_inventory(self) -> Self:
        tray = self.compute_tray
        if self.gpu_count != tray.count * tray.board.count * tray.board.gpu_count:
            raise ValueError(f"{self.id} GPU count must equal its compute trays' GPUs")
        return self


System = Annotated[HGXSystem | NVL72System, Field(discriminator="family")]


def _references(value: object, key: str) -> list[str]:
    """Every string stored under key anywhere in a dumped table."""
    if isinstance(value, dict):
        own = [value[key]] if isinstance(value.get(key), str) else []
        return own + [ref for item in value.values() for ref in _references(item, key)]
    if isinstance(value, list):
        return [ref for item in value for ref in _references(item, key)]
    return []


def _part_uses(value: object) -> list[dict]:
    if isinstance(value, dict):
        own = [value] if isinstance(value.get("part"), str) else []
        return own + [use for item in value.values() for use in _part_uses(item)]
    if isinstance(value, list):
        return [use for item in value for use in _part_uses(item)]
    return []


def _templates(details: Details) -> set[str]:
    return {
        match.group(1)
        for value in details.values()
        if isinstance(value, str) and (match := TEMPLATE.match(value))
    }


class TableSet(FrozenModel):
    provenance: dict[str, Provenance]
    parts: dict[str, Part]
    network: dict[str, Gear]
    curves: Curves
    assemblies: Assemblies
    facility: Facility
    systems: dict[str, System]

    @model_validator(mode="after")
    def resolve_references(self) -> Self:
        tables = self.model_dump(mode="json", exclude={"provenance"})
        curves, assemblies = self.curves, self.assemblies
        missing = [
            f"provenance {ref}"
            for ref in _references(tables, "provenance")
            if ref not in self.provenance
        ]
        missing += [f"part {ref}" for ref in _references(tables, "part") if ref not in self.parts]
        for use in _part_uses(tables):
            if part := self.parts.get(use["part"]):
                unknown = _templates(part.details) - part.variables() - set(use["vars"])
                missing += [f"{use['part']} detail variable {name}" for name in sorted(unknown)]
        missing += [
            f"{gear.name} detail variable {name}"
            for gear in self.network.values()
            for name in sorted(_templates(gear.details) - {"state"})
        ]
        refs = [(tray.converter, curves.dc_converters) for tray in assemblies.trays.values()]
        refs += [(tray.fans.curve, curves.rack_fans) for tray in assemblies.trays.values()]
        for shelves in assemblies.power_shelves.values():
            refs += [
                (shelves.efficiency, curves.psu_efficiency),
                (shelves.psu_fans.curve, curves.rack_fans),
            ]
        for system in self.systems.values():
            refs.append((system.network.gear, self.network))
            if isinstance(system, HGXSystem):
                refs += [(system.psu.efficiency, curves.psu_efficiency)]
                refs += [(system.fans, curves.hgx_fans)]
            else:
                tray = system.compute_tray
                refs += [
                    (tray.converter, curves.dc_converters),
                    (tray.fans.curve, curves.rack_fans),
                    (tray.board.cpu, curves.grace_workload),
                    (tray.board.memory, curves.lpddr5x),
                    (system.switch_trays.tray, assemblies.trays),
                    (system.power_shelves, assemblies.power_shelves),
                ]
        missing += [f"table entry {ref}" for ref, table in refs if ref not in table]
        if missing:
            raise ValueError(f"Unresolved table references: {', '.join(missing)}")
        self._check_domains()
        return self

    def _check_domains(self) -> None:
        for key, system in self.systems.items():
            if system.id != key:
                raise ValueError(f"systems/{key}.yaml declares id {system.id}")
        names = [self.alias_key(name) for name in self.aliases()]
        if len(names) != len(set(names)):
            raise ValueError("System ids, names and aliases must be unique")
        for shelves in self.assemblies.power_shelves.values():
            if self.curves.psu_efficiency[shelves.efficiency].points[-1][0] != 1:
                raise ValueError("A power-shelf efficiency curve must reach full load")
        for system in self.systems.values():
            if isinstance(system, NVL72System):
                board = system.compute_tray.board
                limit = self.curves.lpddr5x[board.memory].max_bandwidth_gbps
                workload = self.curves.grace_workload[board.cpu]
                if any(p.memory_bandwidth_gbps > limit for p in workload.states.values()):
                    raise ValueError(f"{workload.name} bandwidth exceeds {board.memory}")

    @staticmethod
    def alias_key(name: str) -> str:
        return name.lower()

    def aliases(self) -> list[str]:
        return [
            name
            for key, system in self.systems.items()
            for name in (key, system.name, *system.aliases)
        ]

    def system(self, name: str) -> HGXSystem | NVL72System:
        """The system for a catalog key, alias or system name, case-insensitively."""
        wanted = self.alias_key(name)
        for key, system in self.systems.items():
            if wanted in {self.alias_key(n) for n in (key, system.name, *system.aliases)}:
                return system
        raise ValueError(f"Unknown system: {name}")

    def provenance_of(self, ref: str | Provenance) -> Provenance:
        return ref if isinstance(ref, Provenance) else self.provenance[ref]


def _read(path: Path) -> object:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_tables(root: Path = TABLES_DIR) -> TableSet:
    """Read and validate every table under root."""
    raw: dict[str, object] = {name: _read(root / f"{name}.yaml") for name in TABLE_FILES}
    raw["provenance"] = {
        key: {"profile_id": key, **value} for key, value in _read(root / "provenance.yaml").items()
    }
    raw["systems"] = {path.stem: _read(path) for path in sorted(root.glob("systems/*.yaml"))}
    return TableSet.model_validate_json(json.dumps(raw))


@cache
def default_tables() -> TableSet:
    return load_tables()
