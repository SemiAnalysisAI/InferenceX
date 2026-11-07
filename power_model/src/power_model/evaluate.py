# SPDX-License-Identifier: GPL-3.0-only
"""One evaluator for every system in the power tables.

estimate_breakdown() returns the same PowerEstimate as create_power_model(...)
.estimate_breakdown(...): the same component tree, totals and errors. Constants and
provenance come from power_model.tables; this module holds only the arithmetic each
system family applies to them, in the class graph's order of operations.
"""

from dataclasses import dataclass
from itertools import pairwise

from pydantic import JsonValue, TypeAdapter

from power_model.base import (
    MODEL_VERSION,
    Details,
    ITPowerBreakdown,
    OperatingState,
    PositiveCount,
    PowerEstimate,
    WorkloadState,
    sum_watts,
    validate_watts,
)
from power_model.base import PowerComponentBreakdown as Node
from power_model.tables import (
    TEMPLATE,
    HGXSystem,
    NetworkAllocation,
    PartUse,
    PSUBank,
    TableSet,
    default_tables,
)

MODELS = {"oss": "OSSAllinPowerModel", "example": "ExamplePowerModel"}
# Fan sizing assumes the most demanding host state, whatever the scenario.
DESIGN_STATE = OperatingState(
    workload_state=WorkloadState.AGENTIC_CPU_OFFLOADING, using_scale_out=True
)
GPU_INPUT = "gpu-input"
# A measured x86 package replaces the modeled CPU; CPU-and-DRAM also drops the DIMMs.
X86_CPU_INPUT = ("CPU", "x86-cpu-socket-input")
X86_CPU_AND_DRAM_INPUT = ("CPUAndDRAM", "x86-cpu-and-dram-socket-input")
_COUNT = TypeAdapter(PositiveCount)


@dataclass(frozen=True)
class Inputs:
    state: OperatingState
    gpu_w: float
    cpu_and_dram_w: float | None
    cpu_w: float | None


def estimate_breakdown(
    system: str,
    gpu_level_power_per_gpu: float,
    *,
    workload_state: WorkloadState | str = WorkloadState.FIXED_SEQ_LEN,
    using_scale_out: bool = False,
    systems: int = 1,
    model: str = "oss",
    cpu_and_dram_measured_power_per_socket: float | None = None,
    cpu_measured_power_per_socket: float | None = None,
    tables: TableSet | None = None,
) -> PowerEstimate:
    """Facility power for systems units of one catalog system at the given inputs."""
    tables = tables or default_tables()
    selected = {name.lower(): key for key, name in MODELS.items()}.get(model.lower(), model.lower())
    if selected not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    table = tables.system(system)
    state = OperatingState(workload_state=workload_state, using_scale_out=using_scale_out)
    if selected == "example" and state.cpu_offload:
        raise ValueError("CPU offloading requires OSSAllinPowerModel with DDR5 inventory")
    count = _COUNT.validate_python(systems)
    inputs = Inputs(
        state=state,
        gpu_w=validate_watts(gpu_level_power_per_gpu),
        cpu_and_dram_w=_optional_watts(cpu_and_dram_measured_power_per_socket),
        cpu_w=_optional_watts(cpu_measured_power_per_socket),
    )
    it = (
        _example(tables, inputs)
        if selected == "example"
        else _cluster(tables, table, count, inputs)
    )
    pue = getattr(tables.facility.pue, table.cooling)
    it_power = it.it_power_w
    facility = validate_watts(it_power * pue)
    return PowerEstimate(
        model_name=MODELS[selected],
        model_version=MODEL_VERSION,
        cpu_offload=state.cpu_offload,
        workload_state=state.workload_state,
        using_scale_out=state.using_scale_out,
        gpu_level_power_per_gpu=inputs.gpu_w,
        cpu_and_dram_measured_power_per_socket=inputs.cpu_and_dram_w,
        cpu_measured_power_per_socket=inputs.cpu_w,
        AllInPower_per_gpu=facility / it.gpu_count,
        scope=it.scope,
        gpu_count=it.gpu_count,
        cooling_mode=table.cooling,
        pue=pue,
        it_power_w=it_power,
        facility_overhead_w=facility - it_power,
        facility_power_w=facility,
        components=it.components,
    )


def _optional_watts(value: float | None) -> float | None:
    return None if value is None else validate_watts(value)


def _example(t: TableSet, inputs: Inputs) -> ITPowerBreakdown:
    if inputs.cpu_and_dram_w is not None:
        raise ValueError(
            "cpu_and_dram_measured_power_per_socket requires the oss model's Grace inventory"
        )
    if inputs.cpu_w is not None:
        raise ValueError("cpu_measured_power_per_socket requires the oss model's x86 CPU inventory")
    example = t.facility.example
    return ITPowerBreakdown(
        scope="per_gpu_reference",
        gpu_count=1,
        components=(
            Node(
                name=example.gpu.name,
                power_w=inputs.gpu_w,
                provenance=t.provenance_of(example.gpu.provenance),
            ),
            Node(
                name=example.non_gpu.name,
                power_w=inputs.gpu_w * example.non_gpu_ratio,
                provenance=t.provenance_of(example.non_gpu.provenance),
            ),
        ),
    )


def _cluster(t: TableSet, table: HGXSystem, count: int, inputs: Inputs) -> ITPowerBreakdown:
    unit = _hgx(t, table, inputs)
    return ITPowerBreakdown(
        scope="cluster",
        gpu_count=count * table.gpu_count,
        components=(
            Node.group("GPU systems", (unit.scaled(count),)),
            Node.group("Scale-out networking", (_network(t, table.network, count, inputs.state),)),
        ),
    )


def _configuration(state: OperatingState) -> dict[str, JsonValue]:
    return state.model_dump(mode="json") | {"cpu_offload": state.cpu_offload}


def _details(template: dict[str, JsonValue], values: dict[str, JsonValue]) -> Details:
    return tuple(
        (key, values[match.group(1)])
        if isinstance(value, str) and (match := TEMPLATE.match(value))
        else (key, value)
        for key, value in template.items()
    )


def _part(t: TableSet, use: PartUse, state: OperatingState) -> Node:
    """count units of one part in the state the scenario selects."""
    part = t.parts[use.part]
    values: dict[str, JsonValue] = {}
    if part.state_by is None:
        watts = part.power_w
    else:
        if part.state_by == "workload":
            label = part.workload_states[state.workload_state]
        elif part.state_by == "scale_out":
            label = state.nic_state
        else:
            label = "active" if state.cpu_offload else "idle"
        watts = part.states_w[label]
        values = {"state": label} | {f"{name}_w": w for name, w in part.states_w.items()}
    return Node(
        name=part.name,
        power_w=watts,
        provenance=t.provenance_of(part.provenance),
        details=_details(part.details, values | {"power_w": watts} | use.vars),
    ).scaled(use.count)


def _measured(t: TableSet, label: tuple[str, str], watts: float, count: int) -> Node:
    name, provenance = label
    return Node(name=name, power_w=watts, provenance=t.provenance_of(provenance)).scaled(count)


def _interpolate(points: tuple[tuple[float, float], ...], x: float) -> float:
    for (x0, y0), (x1, y1) in pairwise(points):
        if x <= x1:
            weight = (x - x0) / (x1 - x0)
            return y0 + weight * (y1 - y0)
    raise ValueError("Curve does not cover the supplied load")


def _network(t: TableSet, allocation: NetworkAllocation, count: int, state: OperatingState):
    """The scenario's share of external switches, allocated per system."""
    gear = t.network[allocation.gear]
    node = Node.group(
        gear.name,
        tuple(_part(t, use, state) for use in gear.parts),
        provenance=t.provenance_of(gear.provenance),
        details=_details(gear.details, {"state": state.nic_state}),
    ).scaled(allocation.per_system * count)
    return Node(
        name=node.name,
        power_w=node.power_w,
        quantity=node.quantity,
        children=node.children,
        provenance=node.provenance,
        details=(*node.details, *allocation.details.items()),
    )


# HGX chassis: GPU board, host parts, normalized fans and a PSU bank.


def _hgx(t: TableSet, s: HGXSystem, inputs: Inputs) -> Node:
    board = _hgx_board(t, s, inputs.state, inputs.gpu_w)
    host = _hgx_host(t, s, inputs.state, inputs.cpu_and_dram_w, inputs.cpu_w)
    components = (board, Node.group("Generic components", host))
    component_dc_w = sum_watts(tuple(node.power_w for node in components))
    fans = _hgx_fans(t, s, component_dc_w)
    losses = _psu_loss(t, s.psu, sum_watts((component_dc_w, fans.power_w)))
    return Node.group(
        s.name,
        (*components, losses, fans),
        provenance=t.provenance_of(s.provenance),
        details=(("gpu_count_per_system", s.gpu_count),),
        configuration=_configuration(inputs.state),
    )


def _hgx_board(t: TableSet, s: HGXSystem, state: OperatingState, gpu_w: float) -> Node:
    per_gpu = validate_watts(gpu_w)
    gpus = Node(
        name="GPUs",
        power_w=s.gpu_count * per_gpu,
        quantity=s.gpu_count,
        provenance=t.provenance_of(GPU_INPUT),
        details=(("gpu_level_power_per_gpu", per_gpu), ("gpu_tdp_w", s.board.gpu_tdp_w)),
    )
    return Node.group(
        s.board.name,
        (gpus, *(_part(t, use, state) for use in s.board.parts)),
        provenance=t.provenance_of(s.board.provenance),
    )


def _hgx_host(
    t: TableSet,
    s: HGXSystem,
    state: OperatingState,
    cpu_and_dram_w: float | None,
    cpu_w: float | None,
) -> tuple[Node, ...]:
    nodes: list[Node] = []
    for use in s.host:
        category = t.parts[use.part].category
        if category == "cpu" and cpu_w is not None:
            nodes.append(_measured(t, X86_CPU_INPUT, cpu_w, use.count))
        elif category == "cpu" and cpu_and_dram_w is not None:
            nodes.append(_measured(t, X86_CPU_AND_DRAM_INPUT, cpu_and_dram_w, use.count))
        elif not (category == "memory" and cpu_w is None and cpu_and_dram_w is not None):
            nodes.append(_part(t, use, state))
    return tuple(nodes)


def _hgx_fans(t: TableSet, s: HGXSystem, component_dc_w: float) -> Node:
    """Size the fans for TDP plus design-state host watts, then run them at this load."""
    fans = t.curves.hgx_fans[s.fans]
    design_w = sum_watts(
        (
            _hgx_board(t, s, DESIGN_STATE, 0.0).power_w,
            *(node.power_w for node in _hgx_host(t, s, DESIGN_STATE, None, None)),
        )
    )
    tdp = float(s.board.gpu_tdp_w)
    full_load_w = sum_watts((s.gpu_count * tdp, design_w))
    nameplate_w = validate_watts(full_load_w * fans.ratio_at_design / fans.normal_max_pwm**3)
    load = validate_watts(component_dc_w)
    fraction = min(1.0, load / full_load_w)
    pwm = fans.min_pwm + (fans.normal_max_pwm - fans.min_pwm) * (fraction**fans.exponent)
    pwm = min(fans.normal_max_pwm, max(fans.min_pwm, pwm))
    return Node(
        name="FanPower",
        power_w=validate_watts(nameplate_w * pwm**3),
        provenance=t.provenance_of(fans.provenance),
        details=(
            ("model", fans.name),
            ("non_fan_power_w_per_system", load),
            ("mode", "thermal_load_curve"),
            ("fan_pwm_fraction", pwm),
            ("electrical_nameplate_w", nameplate_w),
            ("fan_complement", fans.fan_complement),
            ("cooling_load_fraction", fraction),
            ("gpu_tdp_w", tdp),
            ("gpu_count_per_system", s.gpu_count),
            ("design_non_gpu_power_w", design_w),
            ("full_cooling_load_w", full_load_w),
            ("fan_power_ratio_at_design", fans.ratio_at_design),
            ("min_pwm_frac", fans.min_pwm),
            ("normal_max_pwm_frac", fans.normal_max_pwm),
            ("fan_curve_exponent", fans.exponent),
        ),
    )


def _psu_loss(t: TableSet, psu: PSUBank, dc_load_w: float) -> Node:
    """AC-to-DC loss of a load-sharing PSU bank, limited to its redundant capacity."""
    load = validate_watts(dc_load_w)
    installed_w = psu.installed * psu.capacity_w
    sharing_w = psu.load_sharing * psu.capacity_w
    redundant_w = psu.redundant_capacity * psu.capacity_w
    modeled_w = redundant_w if psu.system_max_w is None else min(psu.system_max_w, redundant_w)
    if load > modeled_w:
        raise ValueError(
            f"DC load {load:g} W exceeds modeled PSU capacity {modeled_w:g} W ({psu.redundancy})"
        )
    points = t.curves.psu_efficiency[psu.efficiency].points
    fraction = load / sharing_w
    if fraction <= points[0][0]:
        efficiency = points[0][1]
    elif fraction >= points[-1][0]:
        efficiency = points[-1][1]
    else:
        efficiency = _interpolate(points, fraction)
    ac_w = validate_watts(load / efficiency)
    return Node(
        name="Power conversion losses",
        power_w=ac_w - load,
        provenance=t.provenance_of(psu.provenance),
        details=(
            ("model", psu.name),
            ("dc_load_w_per_system", load),
            ("ac_wall_w_per_system", ac_w),
            ("efficiency", efficiency),
            ("load_sharing_load_fraction", load / sharing_w),
            ("load_sharing_capacity_w", sharing_w),
            ("redundant_capacity_w", redundant_w),
            ("modeled_capacity_w", modeled_w),
            ("installed_capacity_w", installed_w),
            ("n_installed_psu", psu.installed),
            ("n_load_sharing_psu", psu.load_sharing),
            ("redundancy", psu.redundancy),
        ),
    )
