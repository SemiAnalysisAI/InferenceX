# SPDX-License-Identifier: GPL-3.0-only
"""Behavioral checks for composed rack inventory and electrical boundaries."""

import json

import pytest

from power_model import OperatingState, create_power_model
from power_model.cli import main
from power_model.models.advanced.components.bianca import GB200BiancaBoard, GB300BiancaBoard
from power_model.models.advanced.components.dc_converter import DCConverterAssembly, DCLossPoint
from power_model.models.advanced.components.grace_cpu import (
    GraceWorkloadPoint,
    GraceWorkloadProfile,
)
from power_model.models.advanced.components.lpddr5x import LPDDR5XMemory
from power_model.models.advanced.components.psu import PSUEfficiencyPoint
from power_model.models.advanced.components.rack_fans import RackFanAssembly, solve_cooling
from power_model.models.advanced.components.rack_power_supply import (
    PowerShelf,
    RackPowerSupply,
    RackPSU,
)
from power_model.models.advanced.systems import GB200NVL72RackScaleSystem
from power_model.models.advanced.systems.rack_scale.compute_tray import GB200ComputeTray
from power_model.models.advanced.systems.rack_scale.nvswitch_tray import NVSwitchTray
from power_model.reporting import format_power_breakdown_per_chassis


def nodes(tree):
    yield tree
    for child in tree.children:
        yield from nodes(child)


def named(tree, name):
    return [node for node in nodes(tree) if node.name == name]


def ideal_converter(capacity=8000):
    return DCConverterAssembly(
        module_capacity_w=capacity,
        loss_curve=(
            DCLossPoint(output_fraction=0, loss_w=0),
            DCLossPoint(output_fraction=1, loss_w=0),
        ),
    )


def fixed_fans(count, power=0):
    return RackFanAssembly(
        fan_count=count,
        rated_power_w_per_fan=power,
        minimum_speed_fraction=1,
        maximum_speed_fraction=1,
    )


@pytest.mark.parametrize(
    "board_class,system,tdp", [(GB200BiancaBoard, "gb200", 1200), (GB300BiancaBoard, "gb300", 1400)]
)
def test_rack_gpu_tdp_is_fixed_metadata_and_actual_gpu_power_remains_the_input(
    board_class, system, tdp
):
    board = board_class()
    assert board.gpu_tdp_w == tdp
    assert board.model_dump()["gpu_tdp_w"] == tdp
    for actual in (400, tdp, tdp + 1):
        gpu = named(board.estimate_breakdown(actual), "GPUs")[0]
        assert gpu.power_w == 2 * actual
        assert dict(gpu.details)["gpu_tdp_w"] == tdp
    with pytest.raises(ValueError):
        board_class(gpu_tdp_w=tdp - 100)
    # The shared electrical profile supports the specified design loads in its busiest state.
    result = create_power_model(
        system=system, workload_state="agentic-cpu-offloading", using_scale_out=True
    ).estimate_breakdown(tdp)
    assert result.gpu_count == 72
    assert result.it_power_w > 72 * tdp


@pytest.mark.parametrize(
    "workload,cpu,memory",
    [
        ("fixed-seq-len", 100, 4.8009623125),
        ("agentic", 140, 6.20584925),
        ("agentic-cpu-offloading", 180, 17.44223125),
    ],
)
def test_bianca_counts_gpu_cpu_and_the_whole_memory_pool_without_a_converter(workload, cpu, memory):
    tree = GB200BiancaBoard().estimate_breakdown(
        400, operating_state=OperatingState(workload_state=workload)
    )
    assert named(tree, "GPUs")[0].power_w == 800
    assert named(tree, "GPUs")[0].quantity == 2
    assert named(tree, "GraceCPU")[0].power_w == cpu
    assert named(tree, "LPDDR5XMemory")[0].power_w == pytest.approx(memory)
    assert named(tree, "LPDDR5XMemory")[0].quantity == 1
    assert tree.power_w == pytest.approx(800 + cpu + memory)
    assert len(tree.children) == 3
    assert dict(named(tree, "LPDDR5XMemory")[0].details)["capacity_gb_per_pool"] == 512


@pytest.mark.parametrize("bandwidth,power", [(0, 3.396), (100, 9.015397), (384, 24.9679084032)])
def test_memory_equation_and_domain_edges(bandwidth, power):
    assert LPDDR5XMemory(bandwidth_gbps=bandwidth).estimate_w() == pytest.approx(power)


@pytest.mark.parametrize("bandwidth", [-1, 385, float("nan"), float("inf"), True, "25"])
def test_memory_rejects_unsupported_bandwidth(bandwidth):
    with pytest.raises(ValueError):
        LPDDR5XMemory(bandwidth_gbps=bandwidth)


def test_explicit_idle_calibration_and_missing_profile_error():
    state = OperatingState(workload_state="idle")
    with pytest.raises(ValueError, match="calibrated workload"):
        GB200BiancaBoard().estimate_breakdown(0, operating_state=state)
    board = GB200BiancaBoard(
        workload_profile=GraceWorkloadProfile(
            idle=GraceWorkloadPoint(cpu_power_w=50, memory_bandwidth_gbps=0)
        ),
    )
    assert board.estimate_breakdown(0, operating_state=state).power_w == pytest.approx(53.396)


def test_converter_idle_interpolation_and_overload():
    converter = DCConverterAssembly(
        module_capacity_w=1000,
        loss_curve=(
            DCLossPoint(output_fraction=0, loss_w=5),
            DCLossPoint(output_fraction=0.5, loss_w=20),
            DCLossPoint(output_fraction=1, loss_w=40),
        ),
    )
    assert converter.loss_w(0) == 5
    assert converter.loss_w(250) == 12.5
    result = converter.estimate_loss_breakdown(750)
    assert result.power_w == 30
    assert dict(result.details)["input_power_w"] == 780
    assert result.quantity == 1
    with pytest.raises(ValueError, match="capacity"):
        converter.loss_w(1001)


@pytest.mark.parametrize(
    "points",
    [
        ((0.1, 5), (1.0, 40)),
        ((0.0, 5), (0.5, 20)),
        ((0.0, 5), (0.5, 20), (0.5, 30), (1.0, 40)),
        ((0.0, 20), (1.0, 10)),
    ],
)
def test_converter_rejects_incomplete_or_nonphysical_loss_curves(points):
    with pytest.raises(ValueError):
        DCConverterAssembly(
            loss_curve=tuple(DCLossPoint(output_fraction=x, loss_w=y) for x, y in points)
        )


def test_fans_use_speed_cubed_and_coupled_loss_heat():
    fans = RackFanAssembly(fan_count=8, minimum_speed_fraction=0.6, maximum_speed_fraction=0.6)
    assert fans.estimate_breakdown(500).power_w == pytest.approx(52.25472)
    fan = RackFanAssembly(
        fan_count=1,
        rated_power_w_per_fan=100,
        minimum_speed_fraction=0,
        maximum_speed_fraction=1,
        design_air_heat_w=1000,
    )
    # 400 W direct air heat + 100 W conversion heat => half speed => 12.5 W.
    result = solve_cooling(1000, 400, fan, lambda output: 100)
    assert result.power_w == pytest.approx(12.5)
    assert dict(result.details)["air_heat_w"] == 500
    # A deliberately discontinuous thermal profile cannot silently return an arbitrary iterate.
    with pytest.raises(ValueError, match="did not converge"):
        solve_cooling(1000, 0, fan, lambda output: 1000 if output < 1050 else 0)


def test_gpu_liquid_heat_does_not_drive_tray_fans_as_air_heat():
    tray = GB200ComputeTray(converter=ideal_converter())
    low = tray.estimate_breakdown(100)
    high = tray.estimate_breakdown(800)
    assert named(high, "Fan module")[0].power_w == named(low, "Fan module")[0].power_w
    assert high.power_w - low.power_w == pytest.approx(2800)


@pytest.mark.parametrize(
    "workload,scale_out,expected",
    [
        ("fixed-seq-len", False, 2020),
        ("agentic-cpu-offloading", True, 2060),
    ],
)
def test_controlled_compute_tray_is_additive_and_resolves_state(workload, scale_out, expected):
    point = GraceWorkloadPoint(cpu_power_w=100, memory_bandwidth_gbps=0)
    board = GB200BiancaBoard(
        workload_profile=GraceWorkloadProfile(fixed_seq_len=point, agentic_cpu_offloading=point),
    )
    converter = DCConverterAssembly(
        module_capacity_w=8000,
        loss_curve=(
            DCLossPoint(output_fraction=0, loss_w=60),
            DCLossPoint(output_fraction=1, loss_w=60),
        ),
    )
    tray = GB200ComputeTray(board=board, converter=converter, fans=fixed_fans(8, 3.5))
    result = tray.estimate_breakdown(
        400, operating_state=OperatingState(workload_state=workload, using_scale_out=scale_out)
    )
    # 1600 GPU + 200 CPU + 6.792 memory + 60/100 NIC + 32 optics + 40 NVMe + 28 fans
    # plus exactly one converter's 60 W loss, covering both boards and all tray auxiliaries.
    assert result.power_w == pytest.approx(expected + 6.792)
    loss = named(result, "DC/DC converter loss")
    assert len(loss) == 1
    assert (loss[0].quantity, loss[0].power_w) == (1, 60)
    assert dict(loss[0].details)["output_power_w"] == pytest.approx(expected - 60 + 6.792)


def test_controlled_switch_tray_counts_switches_cpu_fans_and_loss():
    tray = NVSwitchTray(converter=ideal_converter(), fans=fixed_fans(4, 10))
    assert tray.estimate_breakdown().power_w == 485


@pytest.mark.parametrize("active,expected", [(8, 104575.99188743821), (4, 104202.19014057822)])
def test_titanium_bank_uses_participating_capacity_separately_from_redundancy(active, expected):
    bank = RackPowerSupply(
        active_shelves=active,
        shelf=PowerShelf(controller_power_w=0, psu=RackPSU(fans=fixed_fans(1))),
    )
    result = bank.estimate_overhead(100000)
    assert result.power_w + 100000 == pytest.approx(expected)
    assert result.quantity == active
    assert named(result, "Rack PSU overhead")[0].quantity == 6 * active
    with pytest.raises(ValueError, match="capacity"):
        bank.estimate_overhead(132001)
    with pytest.raises(ValueError, match="low-load"):
        bank.estimate_overhead(0)


def test_shelf_auxiliaries_are_counted_once_and_receive_conversion():
    psu = RackPSU(
        capacity_w=1000,
        fans=fixed_fans(1, 10),
        efficiency_curve=(
            PSUEfficiencyPoint(load_fraction=0.05, efficiency=0.8),
            PSUEfficiencyPoint(load_fraction=1, efficiency=0.8),
        ),
    )
    shelf = PowerShelf(psu_count=2, psu=psu, controller_power_w=20)
    # 1000 payload + 20 control + 20 fan = 1040 DC, 1300 AC. Only overhead is additive.
    result = shelf.estimate_overhead(1000)
    assert result.power_w == pytest.approx(300)
    assert named(result, "AC/DC conversion loss")[0].power_w == 260
    assert named(result, "Fan module")[0].power_w == 20


@pytest.mark.parametrize(
    "system,nic,optics,switch,bandwidth,network,active_network",
    [
        (
            "gb200",
            "ConnectX7NICCard",
            "SystemSide400GTransceiver",
            "QM9700NDRInfiniBandSwitch",
            25.6,
            3645,
            4262.625,
        ),
        (
            "gb300",
            "ConnectX8NICCard",
            "SystemSide800GTransceiver",
            "Generic512TEthernetSwitch",
            51.2,
            5059.125,
            6277.5,
        ),
    ],
)
def test_rack_tree_scaling_network_and_state(
    system, nic, optics, switch, bandwidth, network, active_network
):
    model = create_power_model(system=system)
    result = model.estimate_breakdown(400)
    rack, fabric = result.components
    tree = rack.children[0]
    expected_quantities = {
        "GPUs": 72,
        "GraceCPU": 36,
        "LPDDR5XMemory": 36,
        nic: 72,
        optics: 72,
        "NVMeDrive": 144,
        "BlackwellNVSwitch": 18,
        "EPYCEmbedded3151CPU": 9,
    }
    for name, quantity in expected_quantities.items():
        assert sum(node.quantity for node in named(tree, name)) == quantity
    assert sum(node.power_w for node in named(tree, "GPUs")) == 28800
    assert named(tree, "EPYCEmbedded3151CPU")[0].power_w == 405
    # One converter per physical tray, never per Bianca board or Grace CPU.
    compute_trays, switch_trays, _ = tree.children
    assert [node.quantity for node in named(compute_trays, "DC/DC converter loss")] == [18]
    assert [node.quantity for node in named(switch_trays, "DC/DC converter loss")] == [9]
    assert fabric.power_w == network
    allocation = named(fabric, switch)[0]
    assert allocation.quantity == 3.375
    assert dict(allocation.details)["bandwidth_tbps"] == bandwidth
    active = create_power_model(
        system=system, workload_state="agentic-cpu-offloading", using_scale_out=True
    ).estimate_breakdown(400)
    assert active.components[1].power_w == active_network
    active_rack = active.components[0].children[0]
    assert named(active_rack, "GraceCPU")[0].power_w == 6480
    assert named(active_rack, "LPDDR5XMemory")[0].power_w == pytest.approx(627.920325)
    assert named(active_rack, optics)[0].power_w == named(tree, optics)[0].power_w
    assert named(active_rack, "BlackwellNVSwitch")[0].power_w == 3600
    assert (result.gpu_count, result.cooling_mode, result.pue) == (72, "liquid", 1.1)
    assert result.facility_power_w == pytest.approx(1.1 * (tree.power_w + network))
    doubled = create_power_model(system=system, systems=2).estimate_breakdown(400)
    assert doubled.gpu_count == 144
    assert doubled.it_power_w == pytest.approx(result.it_power_w * 2)
    assert doubled.AllInPower_per_gpu == pytest.approx(result.AllInPower_per_gpu)
    assert named(doubled.components[1], switch)[0].quantity == 6.75
    assert sum(node.quantity for node in named(doubled.components[0], "DC/DC converter loss")) == 54
    for component in nodes(doubled.components[0]):
        if component.children:
            assert component.power_w == pytest.approx(sum(c.power_w for c in component.children))
    # Serialized results preserve nested quantities and the estimation assumptions.
    encoded = json.loads(doubled.model_dump_json())
    exported_rack = encoded["components"][0]["children"][0]
    assert exported_rack["provenance"]["kind"] == "estimated"
    config = exported_rack["configuration"]
    assert config["compute_tray"]["converter"]["loss_curve"][0] == {
        "output_fraction": 0.0,
        "loss_w": 25.0,
    }
    assert config["switch_tray"]["converter"]["module_capacity_w"] == 1600
    assert config["power_supply"]["shelf"]["psu"]["efficiency_curve"][0] == {
        "load_fraction": 0.05,
        "efficiency": 0.8864,
    }


@pytest.mark.parametrize("system", ["gb200", "gb300"])
def test_measured_grace_socket_replaces_modeled_cpu_and_memory_once(system):
    model = create_power_model(system=system)
    modeled = model.estimate_breakdown(594.191)
    measured = model.estimate_breakdown(594.191, cpu_and_dram_measured_power_per_socket=98.066)
    modeled_rack = modeled.components[0].children[0]
    measured_rack = measured.components[0].children[0]
    cpu = named(modeled_rack, "GraceCPU")[0]
    memory = named(modeled_rack, "LPDDR5XMemory")[0]
    sockets = cpu.quantity
    assert sockets == memory.quantity == 36
    grace_delta = sockets * 98.066 - (cpu.power_w + memory.power_w)

    socket = named(measured_rack, "Grace socket (measured)")
    assert [(node.quantity, node.provenance.kind) for node in socket] == [(36, "measured")]
    assert socket[0].power_w == pytest.approx(36 * 98.066)
    assert dict(socket[0].details)["cpu_and_dram_measured_power_per_socket"] == 98.066
    assert measured.cpu_and_dram_measured_power_per_socket == 98.066
    assert not named(measured_rack, "GraceCPU") + named(measured_rack, "LPDDR5XMemory")

    def tray_load(result):
        tray = result.components[0].children[0].children[0]
        return tray.quantity * dict(tray.details)["dc_component_power_w"]

    assert tray_load(measured) - tray_load(modeled) == pytest.approx(grace_delta)
    # Tray conversion, fans and shelves stay modeled, so IT power moves further than the load.
    assert measured.it_power_w - modeled.it_power_w < grace_delta < 0


@pytest.mark.parametrize("socket_w", [None, 98.066])
def test_tray_air_heat_takes_the_workload_memory_share_but_no_grace_chip_heat(socket_w):
    tray = GB200ComputeTray(converter=ideal_converter())
    result = tray.estimate_breakdown(
        400,
        cpu_and_dram_measured_power_per_socket=socket_w,
        operating_state=OperatingState(workload_state="agentic-cpu-offloading"),
    )
    fans = named(result, "Fan module")[0]
    # 2 x 17.44223125 W LPDDR5X at 250 GB/s + 4 x 15 W idle NICs + 4 x 8 W optics + 8 x 5 W NVMe
    assert dict(fans.details)["air_heat_w"] == pytest.approx(166.8844625)


def test_rack_cli_bom_marks_the_measured_grace_socket(capsys):
    main(
        [
            "--system=gb200",
            "--gpu-level-power-per-gpu=594.191",
            "--cpu-and-dram-measured-power-per-socket=98.066",
            "--power-breakdown-per-chassis",
        ]
    )
    output = " ".join(capsys.readouterr().out.split())
    assert "Grace socket input (measured): 98.07 W/socket" in output
    assert "Grace socket (measured) 98.07 36 3,530.38" in output
    assert "GraceCPU" not in output
    assert "LPDDR5XMemory" not in output


def test_rack_cli_bom_is_normalized_to_one_rack(capsys):
    main(
        [
            "--system=gb300",
            "--gpu-level-power-per-gpu=400",
            "--systems=2",
            "--power-breakdown-per-chassis",
            "--scale-out-enabled",
            "--workload=agentic-cpu-offloading",
        ]
    )
    output = capsys.readouterr().out
    assert "Power BoM per rack" in output
    assert "one rack (configured quantity: 2)" in output
    assert "GPUs 400.00 72 28,800.00" in " ".join(output.split())
    assert "GraceCPU 180.00 36 6,480.00" in " ".join(output.split())
    assert "Cluster totals (2 racks, 144 GPUs)" in output
    assert "Allocated external network per rack" in output


@pytest.mark.parametrize(
    "overrides",
    [
        {"gpu_count": 8},
        {"compute_tray_count": 17},
        {"switch_tray_count": 10},
        {"compute_tray_count": True},
        {"power_conversion_loss_w": 0},
    ],
)
def test_rack_rejects_inconsistent_inventory_and_flat_hgx_inputs(overrides):
    with pytest.raises(ValueError):
        GB200NVL72RackScaleSystem(**overrides)


def test_rack_capacity_and_input_errors_are_reported_by_cli(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--system=gb200", "--gpu-level-power-per-gpu=2000"])
    assert error.value.code == 2
    assert "capacity" in capsys.readouterr().err
    with pytest.raises(ValueError, match="eight fan modules"):
        GB200ComputeTray(fans=fixed_fans(7))


def test_rack_report_preserves_identical_unit_bom_when_cluster_is_repeated():
    one = create_power_model(system="gb200").estimate_breakdown(400)
    two = create_power_model(system="gb200", systems=2).estimate_breakdown(400)

    def rows(result):
        return [
            line
            for line in format_power_breakdown_per_chassis(result).splitlines()
            if "├─" in line or "└─" in line
        ]

    assert rows(one) == rows(two)
