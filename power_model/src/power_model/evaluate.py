# SPDX-License-Identifier: GPL-3.0-only
"""One evaluator for every system in the power tables.

estimate_breakdown() returns the same PowerEstimate as create_power_model(...)
.estimate_breakdown(...): the same component tree, totals and errors, except that a
system's configuration records only the scenario. Constants and
provenance come from power_model.tables; this module holds only the arithmetic each
system family applies to them, in the class graph's order of operations.
"""

from collections.abc import Callable
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
    BiancaBoard,
    DCConverter,
    FanUse,
    HGXSystem,
    Labeled,
    NetworkAllocation,
    NVL72System,
    PartUse,
    PowerShelves,
    PSUBank,
    TableSet,
    Tray,
    default_tables,
)

MODELS = {"oss": "OSSAllinPowerModel", "example": "ExamplePowerModel"}
# Fan sizing assumes the most demanding host state, whatever the scenario.
DESIGN_STATE = OperatingState(
    workload_state=WorkloadState.AGENTIC_CPU_OFFLOADING, using_scale_out=True
)
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
    # The example model is one GPU and ignores systems, as the class graph does.
    count = _COUNT.validate_python(systems) if selected == "oss" else 1
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


def _cluster(
    t: TableSet, table: HGXSystem | NVL72System, count: int, inputs: Inputs
) -> ITPowerBreakdown:
    unit = _hgx(t, table, inputs) if isinstance(table, HGXSystem) else _nvl72(t, table, inputs)
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
        elif part.state_by == "scale-out":
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


def _measured(t: TableSet, label: Labeled, watts: float, count: int) -> Node:
    return Node(
        name=label.name, power_w=watts, provenance=t.provenance_of(label.provenance)
    ).scaled(count)


def _interpolate(points: tuple[tuple[float, float], ...], x: float) -> float:
    for (x0, y0), (x1, y1) in pairwise(points):
        if x <= x1:
            weight = (x - x0) / (x1 - x0)
            return y0 + weight * (y1 - y0)
    raise ValueError("Curve does not cover the supplied load")


def _network(t: TableSet, allocation: NetworkAllocation, count: int, state: OperatingState) -> Node:
    """The scenario's share of external switches, allocated per system."""
    gear = t.network[allocation.gear]
    return Node.group(
        gear.name,
        tuple(_part(t, use, state) for use in gear.parts),
        provenance=t.provenance_of(gear.provenance),
        details=(
            *_details(gear.details, {"state": state.nic_state}),
            *allocation.details.items(),
        ),
    ).scaled(allocation.per_system * count)


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
        name=t.inputs.gpus.name,
        power_w=s.gpu_count * per_gpu,
        quantity=s.gpu_count,
        provenance=t.provenance_of(t.inputs.gpus.provenance),
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
            nodes.append(_measured(t, t.inputs.x86_cpu, cpu_w, use.count))
        elif category == "cpu" and cpu_and_dram_w is not None:
            nodes.append(_measured(t, t.inputs.x86_cpu_and_dram, cpu_and_dram_w, use.count))
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


# NVL72 racks: compute and NVSwitch trays on a rack bus fed by shared power shelves.


def _nvl72(t: TableSet, s: NVL72System, inputs: Inputs) -> Node:
    per_gpu = validate_watts(inputs.gpu_w)
    tray = s.compute_tray
    board = _bianca(t, tray.board, inputs, per_gpu)
    compute = _tray(
        t,
        tray,
        inputs.state,
        board=board.scaled(tray.board.count),
        board_air_w=tray.board.count * dict(board.details)["air_heat_w"],
        gpus=tray.board.count * tray.board.gpu_count,
    ).scaled(tray.count)
    switch = t.assemblies.trays[s.switch_trays.tray]
    switches = _tray(t, switch, inputs.state).scaled(s.switch_trays.count)
    shelves = _power_shelves(
        t,
        t.assemblies.power_shelves[s.power_shelves],
        sum_watts((compute.power_w, switches.power_w)),
    )
    return Node.group(
        s.name,
        (compute, switches, shelves),
        provenance=t.provenance_of(s.provenance),
        details=(("gpu_count_per_system", s.gpu_count), ("system_unit", "rack")),
        configuration=_configuration(inputs.state),
    )


def _bianca(t: TableSet, board: BiancaBoard, inputs: Inputs, per_gpu: float) -> Node:
    """GPUs and one Grace socket; only the modeled LPDDR5X share heats tray air."""
    workload = t.curves.grace_workload[board.cpu]
    state = inputs.state.workload_state
    point = workload.states.get(state)
    if point is None:
        raise ValueError(f"Grace {state} power requires an explicit calibrated workload profile")
    memory = _lpddr5x(t, board.memory, point.memory_bandwidth_gbps)
    if inputs.cpu_and_dram_w is None:
        source = workload if inputs.cpu_w is None else t.inputs.grace_cpu
        cpu_w = point.cpu_w if inputs.cpu_w is None else inputs.cpu_w
        grace = (
            Node(
                name=workload.name,
                power_w=cpu_w,
                provenance=t.provenance_of(source.provenance),
                details=(
                    ("architecture", "Arm"),
                    ("workload_state", state.value),
                    ("per_cpu_power_w", cpu_w),
                ),
            ),
            memory,
        )
    else:
        socket_w = validate_watts(inputs.cpu_and_dram_w)
        grace = (
            Node(
                name=t.inputs.grace_socket.name,
                power_w=socket_w,
                provenance=t.provenance_of(t.inputs.grace_socket.provenance),
                details=(("cpu_and_dram_measured_power_per_socket", socket_w),),
            ),
        )
    gpus = Node(
        name=t.inputs.gpus.name,
        quantity=board.gpu_count,
        power_w=board.gpu_count * per_gpu,
        provenance=t.provenance_of(t.inputs.gpus.provenance),
        details=(
            ("gpu_family", board.gpu_family),
            ("gpu_tdp_w", board.gpu_tdp_w),
            ("gpu_level_power_per_gpu", per_gpu),
            ("input_boundary", "GPU module, after tray DC/DC"),
        ),
    )
    return Node.group(
        board.name,
        (gpus, *grace),
        provenance=t.provenance_of(board.provenance),
        details=(("air_heat_w", memory.power_w),),
    )


def _lpddr5x(t: TableSet, memory_id: str, bandwidth: float) -> Node:
    memory = t.curves.lpddr5x[memory_id]
    q = memory.milliwatts
    return Node(
        name=memory.name,
        power_w=validate_watts((q.a * bandwidth * bandwidth + q.b1 * bandwidth + q.c) / 1000),
        provenance=t.provenance_of(memory.provenance),
        details=(
            ("capacity_gb_per_pool", memory.capacity_gb),
            ("frequency_mhz", memory.frequency_mhz),
            ("bandwidth_gbps", bandwidth),
            ("scope", "one Grace memory pool"),
        ),
    )


def _tray(
    t: TableSet,
    tray: Tray,
    state: OperatingState,
    *,
    board: Node | None = None,
    board_air_w: float = 0.0,
    gpus: int = 0,
) -> Node:
    """Tray loads, the fans their air heat needs, and the tray converter's loss."""
    parts = tuple(_part(t, use, state) for use in tray.parts)
    components = (*((board,) if board else ()), *parts)
    load = sum_watts(tuple(node.power_w for node in components))
    air = sum_watts(
        (
            board_air_w,
            *(
                node.power_w
                for node, use in zip(parts, tray.parts, strict=True)
                if use.heat == "air"
            ),
        )
    )
    converter = t.curves.dc_converters[tray.converter]
    fans = _solve_cooling(t, tray.fans, load, air, lambda watts: _dc_loss_w(converter, watts))
    loss_w = _dc_loss_w(converter, load + fans.power_w)
    loss = Node(
        name="DC/DC converter loss",
        power_w=loss_w,
        provenance=t.provenance_of(converter.provenance),
        details=(
            ("output_power_w", load + fans.power_w),
            ("input_power_w", load + fans.power_w + loss_w),
            ("input_voltage_v", converter.input_voltage_v),
            ("module_capacity_w", converter.module_capacity_w),
        ),
    )
    return Node.group(
        tray.name,
        (*components, fans, loss),
        provenance=t.provenance_of(tray.provenance),
        details=(("gpu_count_per_tray", gpus), ("dc_component_power_w", load)) if board else (),
    )


def _fan_module(t: TableSet, fans: FanUse, air_heat_w: float) -> Node:
    """Fan speed linear in air heat up to the design heat; watts cubic in speed."""
    fan = t.curves.rack_fans[fans.curve]
    heat = validate_watts(air_heat_w)
    speed = fan.min_speed + (fan.max_speed - fan.min_speed) * min(heat / fans.design_air_heat_w, 1)
    return Node(
        name="Fan module",
        quantity=fans.count,
        power_w=validate_watts(fans.count * fan.rated_w * speed**3),
        provenance=t.provenance_of(fan.provenance),
        details=(
            ("speed_fraction", speed),
            ("air_heat_w", heat),
            ("rated_power_w_per_fan", fan.rated_w),
            ("design_air_heat_w", fans.design_air_heat_w),
        ),
    )


def _solve_cooling(
    t: TableSet,
    fans: FanUse,
    load_w: float,
    air_w: float,
    conversion_loss_w: Callable[[float], float],
) -> Node:
    """Fans for air heat plus the conversion loss of carrying the fans themselves."""
    load = validate_watts(load_w)
    air = validate_watts(air_w)
    fan = _fan_module(t, fans, air)
    for _ in range(128):
        loss = validate_watts(conversion_loss_w(validate_watts(load + fan.power_w)))
        updated = _fan_module(t, fans, validate_watts(air + loss))
        if abs(updated.power_w - fan.power_w) < 1e-8:
            return updated
        fan = updated
    raise ValueError("Fan and conversion-loss iteration did not converge")


def _dc_loss_w(converter: DCConverter, output_w: float) -> float:
    output = validate_watts(output_w)
    fraction = output / converter.module_capacity_w
    if fraction > 1:
        raise ValueError("DC converter output exceeds installed continuous capacity")
    return validate_watts(_interpolate(converter.loss_points(), fraction))


def _power_shelves(t: TableSet, shelves: PowerShelves, rack_bus_w: float) -> Node:
    """Normal operation on the active shelves, checked against the surviving bank."""
    load = validate_watts(rack_bus_w)
    redundant_w = shelves.surviving_shelves * shelves.psus_per_shelf * shelves.psu_capacity_w
    if load > redundant_w:
        raise ValueError("Rack bus load exceeds redundant power-shelf capacity")
    # The surviving bank carries its own fan load, not the normal operating fan watts.
    reserve = _shelf(t, shelves, load / shelves.surviving_shelves)
    current = _shelf(t, shelves, load / shelves.active_shelves).scaled(shelves.active_shelves)
    return Node(
        name="Power shelves",
        power_w=current.power_w,
        quantity=current.quantity,
        children=current.children,
        provenance=t.provenance_of(shelves.provenance),
        details=(
            ("rack_bus_w", load),
            ("rack_ac_w", load + current.power_w),
            ("installed_shelves", shelves.installed_shelves),
            ("active_shelves", shelves.active_shelves),
            ("redundant_capacity_w", redundant_w),
            ("feed_loss_ac_w", load + reserve.power_w * shelves.surviving_shelves),
        ),
    )


def _shelf(t: TableSet, shelves: PowerShelves, delivered_w: float) -> Node:
    provenance = t.provenance_of(shelves.provenance)
    load = validate_watts(validate_watts(delivered_w) + shelves.controller_w)
    psus = _rack_psu(t, shelves, load / shelves.psus_per_shelf).scaled(shelves.psus_per_shelf)
    controller = Node(name="Shelf controller", power_w=shelves.controller_w, provenance=provenance)
    return Node.group("Power shelf overhead", (psus, controller), provenance=provenance)


def _rack_psu(t: TableSet, shelves: PowerShelves, delivered_w: float) -> Node:
    provenance = t.provenance_of(shelves.provenance)
    fan = _solve_cooling(
        t, shelves.psu_fans, delivered_w, 0.0, lambda watts: _rack_psu_loss_w(t, shelves, watts)
    )
    output = validate_watts(delivered_w + fan.power_w)
    efficiency = _rack_psu_efficiency(t, shelves, output)
    loss = Node(
        name="AC/DC conversion loss",
        power_w=_rack_psu_loss_w(t, shelves, output),
        provenance=provenance,
    )
    return Node.group(
        "Rack PSU overhead",
        (loss, fan),
        provenance=provenance,
        details=(
            ("dc_output_w_per_psu", output),
            ("efficiency", efficiency),
            ("load_fraction", output / shelves.psu_capacity_w),
            ("capacity_w", shelves.psu_capacity_w),
        ),
    )


def _rack_psu_efficiency(t: TableSet, shelves: PowerShelves, output_w: float) -> float:
    points = t.curves.psu_efficiency[shelves.efficiency].points
    load = validate_watts(output_w) / shelves.psu_capacity_w
    if load < points[0][0]:
        raise ValueError(
            f"Rack PSU load below {points[0][0]:.0%} requires a calibrated low-load profile"
        )
    if load > 1:
        raise ValueError("Rack PSU output exceeds continuous capacity")
    return _interpolate(points, load)


def _rack_psu_loss_w(t: TableSet, shelves: PowerShelves, output_w: float) -> float:
    output = validate_watts(output_w)
    return validate_watts(output * (1 / _rack_psu_efficiency(t, shelves, output) - 1))
