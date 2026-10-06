# SPDX-License-Identifier: GPL-3.0-only
import pytest
from pydantic import ValidationError

from power_model.models.advanced import NetworkGroup, ScaleOutNetworkingGear
from power_model.models.advanced.components import (
    CPU,
    DDR5DIMM,
    X86CPU,
    ComponentGroup,
    NVMeDrive,
    PEX89144PCIeSwitch,
    SystemSide400GTransceiver,
    SystemSide800GTransceiver,
    UniversalBaseBoard,
)
from power_model.models.advanced.components.scaleoutnic import (
    ConnectX7NICCard,
    ConnectX8NICCard,
    NICState,
    PollaraNICCard,
    Thor2NICCard,
)
from power_model.models.advanced.networking import QM9790NDRInfiniBandSwitch


@pytest.mark.parametrize(
    "state, expected",
    [("Idle", 160), ("FixedSeqLen", 240), ("Agentic", 300), ("AgenticOffloadingOn", 400)],
)
def test_x86_state_selects_fixed_power_for_two_cpus(state, expected):
    result = ComponentGroup(component=X86CPU(state=state), quantity=2).estimate_breakdown()
    assert result.power_w == expected
    assert dict(result.details)["state"] == state
    assert dict(result.details)["architecture"] == "x86"
    assert result.provenance.kind == "assumed"


@pytest.mark.parametrize(
    "inputs",
    [{"state": "sleep"}, {"state": True}, {"state": "Agentic", "power_w": 80}],
)
def test_invalid_x86_state_or_power_override_is_rejected(inputs):
    with pytest.raises(ValidationError):
        X86CPU(**inputs)


@pytest.mark.parametrize(
    "nic_class, active_w, idle_w, total_w, bandwidth",
    [
        (ConnectX7NICCard, 50, 15, 65, 400),
        (ConnectX8NICCard, 100, 30, 130, 800),
        (Thor2NICCard, 60, 20, 80, 400),
        (PollaraNICCard, 60, 20, 80, 400),
    ],
)
def test_nic_state_selects_fixed_power_without_changing_capacity_metadata(
    nic_class, active_w, idle_w, total_w, bandwidth
):
    active = nic_class(state=NICState.ACTIVE)
    idle = nic_class(state="idle")
    active_group = ComponentGroup(component=active, quantity=2).estimate_breakdown()
    idle_group = ComponentGroup(component=idle).estimate_breakdown()
    assert active_group.power_w == active_w
    assert idle_group.power_w == idle_w
    assert active_group.power_w + idle_group.power_w == total_w
    assert (
        dict(active_group.details)["bandwidth_gbps"]
        == dict(idle_group.details)["bandwidth_gbps"]
        == bandwidth
    )
    assert dict(idle_group.details)["state"] == "idle"
    assert dict(active_group.details)["state"] == "active"
    assert dict(idle_group.details)["idle_power_w"] == idle_w
    assert dict(active_group.details)["active_power_w"] == active_w / 2
    assert active_group.provenance.kind == "assumed"


@pytest.mark.parametrize("state, expected", [("idle", 121.6), ("active", 224)])
def test_ddr5_state_selects_power_for_32_dimms(state, expected):
    result = ComponentGroup(component=DDR5DIMM(state=state), quantity=32).estimate_breakdown()
    assert result.power_w == pytest.approx(expected)
    assert dict(result.details)["state"] == state
    assert dict(result.details)["capacity_gb_per_dimm"] == 64


@pytest.mark.parametrize(
    "inputs", [{"state": "sleep"}, {"state": True}, {"power_w": 4}, {"active_power_w": 8}]
)
def test_invalid_ddr5_state_or_power_override_is_rejected(inputs):
    with pytest.raises(ValidationError):
        DDR5DIMM(**inputs)


@pytest.mark.parametrize(
    "inputs",
    [
        {},
        {"state": "sleep"},
        {"state": True},
        {"state": "active", "active_power_w": 25},
        {"state": "idle", "idle_power_w": 5},
    ],
)
def test_missing_invalid_nic_state_or_power_override_is_rejected(inputs):
    with pytest.raises(ValidationError):
        ConnectX8NICCard(**inputs)


@pytest.mark.parametrize("inputs", [{"state": "active"}, {"power_w": 4}])
def test_unsupported_nvme_profile_is_rejected(inputs):
    with pytest.raises(ValidationError):
        NVMeDrive(**inputs)


def test_pex89144_is_always_active_and_scales_by_switch_quantity():
    switch = PEX89144PCIeSwitch()
    assert switch.estimate_w() == 45
    result = ComponentGroup(component=switch, quantity=4).estimate_breakdown()
    assert (result.quantity, result.power_w) == (4, 180)
    assert dict(result.details) == {
        "vendor": "Broadcom",
        "part_number": "PEX89144",
        "pcie_generation": 5,
        "lane_count": 144,
        "state": "active",
        "per_switch_power_w": 45,
    }
    assert result.provenance.kind == "assumed"


@pytest.mark.parametrize(
    "inputs", [{"state": "idle"}, {"state": True}, {"power_w": 0}, {"power_w": 50}]
)
def test_pex89144_rejects_inactive_states_and_power_overrides(inputs):
    with pytest.raises(ValidationError):
        PEX89144PCIeSwitch(**inputs)


@pytest.mark.parametrize(
    "component_type, bandwidth, per_device_w, total_w",
    [(SystemSide400GTransceiver, 400, 8, 64), (SystemSide800GTransceiver, 800, 15, 120)],
)
def test_system_side_optics_scale_power_and_preserve_per_device_metadata(
    component_type, bandwidth, per_device_w, total_w
):
    component = component_type()
    assert component.estimate_w() == per_device_w
    result = ComponentGroup(component=component, quantity=8).estimate_breakdown()
    assert (result.quantity, result.power_w) == (8, total_w)
    assert dict(result.details) == {
        "bandwidth_gbps": bandwidth,
        "per_transceiver_power_w": per_device_w,
        "power_policy": "always_on",
    }
    assert result.provenance.kind == "assumed"


@pytest.mark.parametrize("component_type", [SystemSide400GTransceiver, SystemSide800GTransceiver])
@pytest.mark.parametrize("inputs", [{"power_w": 0}, {"power_w": 9}, {"state": "off"}])
def test_fixed_transceiver_power_cannot_be_overridden_or_disabled(component_type, inputs):
    with pytest.raises(ValidationError):
        component_type(**inputs)


@pytest.mark.parametrize("power", [float("inf"), "5", True])
def test_component_inputs_are_strict_finite_numbers(power):
    with pytest.raises(ValidationError):
        CPU(power_w=power)


@pytest.mark.parametrize(
    "base, inputs",
    [
        (UniversalBaseBoard, {"non_gpu_power_w": 80}),
        (ScaleOutNetworkingGear, {"name": "Unmodeled switch", "power_w": 40}),
    ],
)
def test_equipment_interfaces_require_a_concrete_inventory(base, inputs):
    with pytest.raises(TypeError, match="abstract"):
        base(**inputs)


def test_network_quantity_scales_allocated_equipment_power():
    result = NetworkGroup(gear=QM9790NDRInfiniBandSwitch(), quantity=3).estimate_breakdown()
    assert result.power_w == 3240
    assert result.quantity == 3
