# SPDX-License-Identifier: GPL-3.0-only
import pytest
from power_fixtures import constant_efficiency_psu, controlled_fan_policy
from pydantic import ValidationError

from power_model import OperatingState
from power_model.models.advanced.components import (
    ComponentGroup,
    PCIe4x16Retimer,
    PCIe5x16Retimer,
)
from power_model.models.advanced.components.scaleoutnic import ConnectX7NICCard, ConnectX8NICCard
from power_model.models.advanced.components.ubb import (
    MI300UBB,
    MI325UBB,
    MI355UBB,
    B200HGXBoard,
    B300HGXBoard,
    BlackwellHGXNVSwitch,
    HopperHGXBoard,
    HopperHGXNVSwitch,
)
from power_model.models.advanced.systems import (
    B200HGXSystemChassis,
    B300HGXSystemChassis,
    HopperHGXSystemChassis,
    MI300HGXSystemChassis,
    MI325HGXSystemChassis,
    MI355HGXSystemChassis,
    SystemGroup,
)


@pytest.mark.parametrize(
    "board_type, switch_name, switch_count, bandwidth, per_switch_w",
    [
        (HopperHGXBoard, "HopperHGXNVSwitch", 4, 12.8, 100),
        (B200HGXBoard, "BlackwellHGXNVSwitch", 2, 28.8, 200),
    ],
)
@pytest.mark.parametrize("gpu_power, expected", [(0, 496), (100, 1296)])
def test_nvidia_board_sums_gpus_switches_and_retimers_once(
    board_type, switch_name, switch_count, bandwidth, per_switch_w, gpu_power, expected
):
    result = board_type().estimate_breakdown(gpu_power)
    assert result.power_w == expected
    gpus, switches, retimers = result.children
    assert (gpus.name, gpus.quantity) == ("GPUs", 8)
    assert (switches.name, switches.quantity, switches.power_w) == (switch_name, switch_count, 400)
    assert dict(switches.details)["bandwidth_tbps"] == bandwidth
    assert dict(switches.details)["per_switch_power_w"] == per_switch_w
    assert (retimers.quantity, retimers.power_w) == (8, 96)
    assert dict(retimers.details)["pcie_generation"] == 5
    assert dict(retimers.details)["lane_count"] == 16


@pytest.mark.parametrize("board_type", [MI300UBB, MI325UBB, MI355UBB])
@pytest.mark.parametrize("gpu_power, expected", [(0, 96), (100, 896)])
def test_amd_board_sums_only_eight_gpus_and_eight_retimers(board_type, gpu_power, expected):
    result = board_type().estimate_breakdown(gpu_power)
    assert result.power_w == expected
    gpus, retimers = result.children
    assert (gpus.name, gpus.quantity) == ("GPUs", 8)
    assert gpus.power_w == expected - 96
    assert (retimers.name, retimers.quantity, retimers.power_w) == ("PCIe5x16Retimer", 8, 96)
    assert dict(retimers.details) == {
        "pcie_generation": 5,
        "lane_count": 16,
        "per_retimer_power_w": 12,
    }
    assert result.provenance.kind == "assumed"


@pytest.mark.parametrize(
    "chassis_type, board_name, expected, fan_w",
    [
        (MI300HGXSystemChassis, "MI300UBB", 4919.563227194, 24.763227194),
        (MI325HGXSystemChassis, "MI325UBB", 4910.016100019, 15.216100019),
        (MI355HGXSystemChassis, "MI355UBB", 4903.201263802, 8.401263802),
    ],
)
def test_amd_chassis_defaults_to_its_board_and_scales_retimers_once(
    chassis_type, board_name, expected, fan_w
):
    chassis = chassis_type(
        fan_policy=controlled_fan_policy(),
        psu=constant_efficiency_psu(),
    )
    result = SystemGroup(system=chassis, quantity=3).estimate_it_power(
        100, operating_state=OperatingState(workload_state="idle")
    )
    assert result.power_w == pytest.approx(expected)
    board = result.children[0]
    assert (board.name, board.quantity, board.power_w) == (board_name, 3, 2688)
    gpus, retimers = board.children
    assert (gpus.quantity, gpus.power_w) == (24, 2400)
    assert (retimers.quantity, retimers.power_w) == (24, 288)
    assert dict(retimers.details)["per_retimer_power_w"] == 12
    assert result.children[1].children[-2].power_w == 540
    assert result.children[-1].power_w == pytest.approx(fan_w)


@pytest.mark.parametrize(
    "chassis_type, board_name, switch_name, switch_count, bandwidth, expected",
    [
        (HopperHGXSystemChassis, "HopperHGXBoard", "HopperHGXNVSwitch", 12, 12.8, 6020.338144575),
        (B200HGXSystemChassis, "B200HGXBoard", "BlackwellHGXNVSwitch", 6, 28.8, 6000.440079767),
    ],
)
def test_chassis_quantity_scales_nvidia_components_and_preserves_per_device_metadata(
    chassis_type, board_name, switch_name, switch_count, bandwidth, expected
):
    chassis = chassis_type(
        fan_policy=controlled_fan_policy(),
        psu=constant_efficiency_psu(),
    )
    result = SystemGroup(system=chassis, quantity=3).estimate_it_power(
        100, operating_state=OperatingState(workload_state="idle")
    )
    assert result.power_w == pytest.approx(expected)
    board = result.children[0]
    assert (board.name, board.quantity, board.power_w) == (board_name, 3, 3888)
    assert (board.children[0].quantity, board.children[0].power_w) == (24, 2400)
    switches = board.children[1]
    retimers = board.children[2]
    assert (switches.name, switches.quantity, switches.power_w) == (switch_name, switch_count, 1200)
    assert dict(switches.details)["bandwidth_tbps"] == bandwidth
    assert (retimers.quantity, retimers.power_w) == (24, 288)
    assert dict(retimers.details)["per_retimer_power_w"] == 12
    pcie_switches = result.children[1].children[-2]
    assert pcie_switches.name == "PEX89144PCIeSwitch"
    assert (pcie_switches.quantity, pcie_switches.power_w) == (12, 540)
    assert dict(pcie_switches.details)["per_switch_power_w"] == 45


@pytest.mark.parametrize(
    "component_type, inputs",
    [
        (HopperHGXBoard, {"non_gpu_power_w": 0}),
        (HopperHGXBoard, {"gpu_count": 4}),
        (HopperHGXBoard, {"gpu_tdp_w": 750}),
        (B200HGXBoard, {"non_gpu_power_w": 80}),
        (B200HGXBoard, {"gpu_count": 4}),
        (B200HGXBoard, {"gpu_tdp_w": 1200}),
        (B300HGXBoard, {"non_gpu_power_w": 80}),
        (B300HGXBoard, {"gpu_count": 4}),
        (B300HGXBoard, {"gpu_tdp_w": 1000}),
        (B300HGXBoard, {"nic": ConnectX7NICCard(state="idle")}),
        (
            B300HGXBoard,
            {"nic": ConnectX8NICCard(state="active"), "non_gpu_power_w": 640},
        ),
        (MI300UBB, {"non_gpu_power_w": 0}),
        (MI325UBB, {"non_gpu_power_w": 80}),
        (MI355UBB, {"non_gpu_power_w": 97}),
        (MI300UBB, {"gpu_count": 4}),
        (MI300UBB, {"gpu_tdp_w": 700}),
        (MI325UBB, {"gpu_tdp_w": 750}),
        (MI355UBB, {"gpu_tdp_w": 1000}),
        (HopperHGXNVSwitch, {"power_w": 90}),
        (BlackwellHGXNVSwitch, {"power_w": 100}),
        (PCIe5x16Retimer, {"power_w": 10}),
        (PCIe4x16Retimer, {}),
    ],
)
def test_incomplete_or_inconsistent_board_component_power_is_rejected(component_type, inputs):
    with pytest.raises(ValidationError):
        component_type(**inputs)


@pytest.mark.parametrize(
    "component, quantity, expected, generation",
    [(PCIe4x16Retimer(power_w=7), 3, 21, 4), (PCIe5x16Retimer(), 2, 24, 5)],
)
def test_retimers_can_be_used_as_generic_component_groups(
    component, quantity, expected, generation
):
    result = ComponentGroup(component=component, quantity=quantity).estimate_breakdown()
    assert result.power_w == expected
    assert dict(result.details)["pcie_generation"] == generation
    assert dict(result.details)["lane_count"] == 16


@pytest.mark.parametrize(
    "state, gpu_power, expected, nic_power",
    [
        ("idle", 0, 640, 240),
        ("idle", 100, 1440, 240),
        ("active", 0, 800, 400),
        ("active", 100, 1600, 400),
    ],
)
def test_b300_board_owns_eight_connectx8_nics_and_two_blackwell_switches(
    state, gpu_power, expected, nic_power
):
    result = B300HGXBoard(nic=ConnectX8NICCard(state=state)).estimate_breakdown(gpu_power)
    assert result.power_w == expected
    gpus, switches, nics = result.children
    assert (gpus.name, gpus.quantity) == ("GPUs", 8)
    assert (switches.name, switches.quantity, switches.power_w) == ("BlackwellHGXNVSwitch", 2, 400)
    assert dict(switches.details)["bandwidth_tbps"] == 28.8
    assert (nics.name, nics.quantity, nics.power_w) == ("ConnectX8NICCard", 8, nic_power)
    assert dict(nics.details)["state"] == state
    assert dict(nics.details)["bandwidth_gbps"] == 800


@pytest.mark.parametrize(
    "state, expected, board_w, nic_w, fan_w",
    [
        ("idle", 5691.004609342, 4320, 720, 16.204609342),
        ("active", 6175.474200356, 4800, 1200, 20.674200356),
    ],
)
def test_b300_chassis_scales_board_nics_and_fans_without_chassis_nics_or_switches(
    state, expected, board_w, nic_w, fan_w
):
    chassis = B300HGXSystemChassis(
        fan_policy=controlled_fan_policy(),
        psu=constant_efficiency_psu(),
    )
    result = SystemGroup(system=chassis, quantity=3).estimate_it_power(
        100,
        operating_state=OperatingState(workload_state="idle", using_scale_out=state == "active"),
    )
    assert result.power_w == pytest.approx(expected)
    board = result.children[0]
    assert (board.name, board.quantity, board.power_w) == ("B300HGXBoard", 3, board_w)
    assert (board.children[2].quantity, board.children[2].power_w) == (24, nic_w)
    assert dict(board.children[2].details)["state"] == state
    assert [node.name for node in result.children[1].children] == [
        "X86CPU",
        "DDR5DIMM",
        "NVMeDrive",
        "SystemSide800GTransceiver",
    ]
    assert result.children[-1].power_w == pytest.approx(fan_w)


@pytest.mark.parametrize(
    "chassis_type, board",
    [
        (HopperHGXSystemChassis, B200HGXBoard()),
        (B200HGXSystemChassis, B300HGXBoard()),
        (B300HGXSystemChassis, HopperHGXBoard()),
        (MI300HGXSystemChassis, MI325UBB()),
        (MI325HGXSystemChassis, MI355UBB()),
        (MI355HGXSystemChassis, MI300UBB()),
    ],
)
def test_chassis_rejects_an_incompatible_board(chassis_type, board):
    with pytest.raises(ValidationError):
        chassis_type(
            ubb=board,
            fan_policy=controlled_fan_policy(),
            psu=constant_efficiency_psu(),
        )
