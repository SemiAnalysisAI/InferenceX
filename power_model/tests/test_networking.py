# SPDX-License-Identifier: GPL-3.0-only
import json

import pytest
from pydantic import ValidationError

from power_model import OperatingState, create_power_model
from power_model.cli import main
from power_model.models.advanced import AdvancedAllInPowerModel, Cluster
from power_model.models.advanced.components import SwitchSide800GSR8Transceiver
from power_model.models.advanced.networking import (
    AMD_TWO_TIER_NETWORK,
    B300_TWO_TIER_NETWORK,
    QM9790_TWO_TIER_NETWORK,
    Generic512TEthernetSwitch,
    NetworkGroup,
    QM9700NDRInfiniBandSwitch,
    QM9790NDRInfiniBandSwitch,
)


@pytest.mark.parametrize(
    "switch_type, state, box_w, optics_count, optics_w, total_w, bandwidth, protocol",
    [
        (QM9700NDRInfiniBandSwitch, "idle", 600, 32, 480, 1080, 25.6, "NDR InfiniBand"),
        (QM9700NDRInfiniBandSwitch, "active", 783, 32, 480, 1263, 25.6, "NDR InfiniBand"),
        (QM9790NDRInfiniBandSwitch, "idle", 600, 32, 480, 1080, 25.6, "NDR InfiniBand"),
        (QM9790NDRInfiniBandSwitch, "active", 783, 32, 480, 1263, 25.6, "NDR InfiniBand"),
        (Generic512TEthernetSwitch, "idle", 539, 64, 960, 1499, 51.2, "Ethernet"),
        (Generic512TEthernetSwitch, "active", 900, 64, 960, 1860, 51.2, "Ethernet"),
    ],
)
def test_switch_sums_state_dependent_box_and_always_powered_optics(
    switch_type, state, box_w, optics_count, optics_w, total_w, bandwidth, protocol
):
    switch = switch_type(state=state)
    result = switch.estimate_breakdown()
    assert switch.power_w == total_w
    assert switch_type.model_validate(switch.model_dump()).estimate_breakdown() == result
    box, optics = result.children
    assert result.power_w == total_w
    assert box.power_w == box_w
    assert (optics.quantity, optics.power_w) == (optics_count, optics_w)
    assert dict(optics.details)["per_transceiver_power_w"] == 15
    assert dict(optics.details)["optical_standard"] == "SR8"
    assert dict(result.details)["bandwidth_tbps"] == bandwidth
    assert dict(result.details)["protocol"] == protocol
    assert dict(result.details)["state"] == state
    assert box.provenance == result.provenance


@pytest.mark.parametrize(
    "profile, enabled, chassis_count, switch_count, total_w, box_w, optics_count, optics_w",
    [
        (QM9790_TWO_TIER_NETWORK, False, 1, 0.375, 405, 225, 12, 180),
        (QM9790_TWO_TIER_NETWORK, True, 1, 0.375, 473.625, 293.625, 12, 180),
        (QM9790_TWO_TIER_NETWORK, True, 8, 3, 3789, 2349, 96, 1440),
        (AMD_TWO_TIER_NETWORK, False, 1, 0.1875, 281.0625, 101.0625, 12, 180),
        (AMD_TWO_TIER_NETWORK, True, 1, 0.1875, 348.75, 168.75, 12, 180),
        (AMD_TWO_TIER_NETWORK, True, 16, 3, 5580, 2700, 192, 2880),
        (B300_TWO_TIER_NETWORK, False, 1, 0.375, 562.125, 202.125, 24, 360),
        (B300_TWO_TIER_NETWORK, True, 8, 3, 5580, 2700, 192, 2880),
    ],
)
def test_two_tier_allocation_scales_whole_switch_tree_without_rounding_device_shares(
    profile, enabled, chassis_count, switch_count, total_w, box_w, optics_count, optics_w
):
    result = profile.scaled(chassis_count).estimate_breakdown(
        operating_state=OperatingState(using_scale_out=enabled)
    )
    assert (result.quantity, result.power_w) == (switch_count, total_w)
    box, optics = result.children
    assert (box.quantity, box.power_w) == (switch_count, box_w)
    assert (optics.quantity, optics.power_w) == (optics_count, optics_w)
    assert dict(result.details)["topology"] == "two-tier"
    exported = json.loads(result.model_dump_json())
    assert exported["quantity"] == switch_count
    assert exported["children"][1]["power_w"] == optics_w


@pytest.mark.parametrize(
    "system, idle_w, active_w, idle_facility_delta, active_facility_delta",
    [
        ("h100", 810, 947.25, 1053, 1231.425),
        ("h200", 810, 947.25, 1053, 1231.425),
        ("b200", 810, 947.25, 1053, 1231.425),
        ("mi300", 562.125, 697.5, 730.7625, 906.75),
        ("mi325", 562.125, 697.5, 730.7625, 906.75),
        ("mi355", 562.125, 697.5, 730.7625, 906.75),
        ("b300", 1124.25, 1395, 1461.525, 1813.5),
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_hgx_default_network_adds_power_before_pue_without_heating_chassis(
    system, enabled, idle_w, active_w, idle_facility_delta, active_facility_delta
):
    network_w = active_w if enabled else idle_w
    facility_delta = active_facility_delta if enabled else idle_facility_delta
    automatic = create_power_model(system=system, systems=2, using_scale_out=enabled)
    without_network = AdvancedAllInPowerModel(
        cooling=automatic.cooling,
        using_scale_out=enabled,
        cluster=Cluster(systems=automatic.cluster.systems, networking=()),
    )
    result = automatic.estimate_breakdown(400)
    baseline = without_network.estimate_breakdown(400)
    assert result.components[0] == baseline.components[0]
    assert result.components[1].power_w == network_w
    assert result.it_power_w - baseline.it_power_w == pytest.approx(network_w)
    assert result.facility_power_w - baseline.facility_power_w == pytest.approx(facility_delta)
    assert result.gpu_count == 16


@pytest.mark.parametrize("flag", ["--scale-out-enabled", "--using-scale-out"])
@pytest.mark.parametrize(
    "system, nic_w, network_w",
    [("h100", 200, 473.625), ("mi300", 240, 348.75), ("b300", 400, 697.5)],
)
def test_scale_out_cli_flags_select_active_switch_and_nic_power(
    capsys, flag, system, nic_w, network_w
):
    main(["--gpu-level-power-per-gpu=400", f"--system={system}", flag])
    result = json.loads(capsys.readouterr().out)
    assert result["using_scale_out"] is True
    chassis = result["components"][0]["children"][0]
    nic = (
        chassis["children"][0]["children"][2]
        if system == "b300"
        else chassis["children"][1]["children"][1]
    )
    assert nic["power_w"] == nic_w
    assert dict(nic["details"])["state"] == "active"
    networking = result["components"][1]
    assert networking["power_w"] == network_w
    assert dict(networking["children"][0]["details"])["state"] == "active"


@pytest.mark.parametrize(
    "system, switch_row, box_row, optics_row, total_w, system_optics_row",
    [
        (
            "b200",
            "QM9790NDRInfiniBandSwitch 1,080.00 0.375 405.00",
            "├─ QM9790 switch box 600.00 0.375 225.00",
            "└─ SwitchSide800GSR8Transceiver 15.00 12 180.00",
            "3,240.00",
            "│ └─ SystemSide400GTransceiver 8.00 8 64.00",
        ),
        (
            "mi300",
            "Generic512TEthernetSwitch 1,499.00 0.1875 281.06",
            "├─ Generic 51.2T Ethernet switch ASIC 539.00 0.1875 101.06",
            "└─ SwitchSide800GSR8Transceiver 15.00 12 180.00",
            "2,248.50",
            "│ └─ SystemSide400GTransceiver 8.00 8 64.00",
        ),
        (
            "b300",
            "Generic512TEthernetSwitch 1,499.00 0.375 562.12",
            "├─ Generic 51.2T Ethernet switch ASIC 539.00 0.375 202.12",
            "└─ SwitchSide800GSR8Transceiver 15.00 24 360.00",
            "4,497.00",
            "│ └─ SystemSide800GTransceiver 15.00 8 120.00",
        ),
    ],
)
def test_network_bom_displays_fractional_switch_and_optics_shares_per_chassis(
    capsys, system, switch_row, box_row, optics_row, total_w, system_optics_row
):
    main(
        [
            "--gpu-level-power-per-gpu=400",
            f"--system={system}",
            "--systems=8",
            "--power-breakdown-per-chassis",
        ]
    )
    output = capsys.readouterr().out
    rows = [" ".join(line.split()) for line in output.splitlines()]
    assert "Allocated external network per chassis (cluster average)" in output
    assert switch_row in rows
    assert box_row in rows
    assert optics_row in rows
    assert f"Shared external network AC power: {total_w} W" in output
    assert system_optics_row in rows


@pytest.mark.parametrize("quantity", [0, -1, float("inf"), float("nan"), True, "0.375"])
def test_network_allocation_rejects_invalid_quantities(quantity):
    with pytest.raises(ValidationError):
        NetworkGroup(gear=QM9790NDRInfiniBandSwitch(), quantity=quantity)


@pytest.mark.parametrize(
    "switch_type, inputs",
    [
        (QM9790NDRInfiniBandSwitch, {"state": "sleep"}),
        (QM9790NDRInfiniBandSwitch, {"power_w": 600}),
        (QM9790NDRInfiniBandSwitch, {"state": "active", "power_w": 1080}),
        (Generic512TEthernetSwitch, {"state": "sleep"}),
        (Generic512TEthernetSwitch, {"power_w": 539}),
        (Generic512TEthernetSwitch, {"state": "active", "power_w": 1499}),
    ],
)
def test_switch_rejects_invalid_state_and_inconsistent_box_plus_optics_power(switch_type, inputs):
    with pytest.raises(ValidationError):
        switch_type(**inputs)


def test_switch_side_optics_reject_power_override():
    with pytest.raises(ValidationError, match="must be 15 W"):
        SwitchSide800GSR8Transceiver(power_w=8)
