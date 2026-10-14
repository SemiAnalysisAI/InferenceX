# SPDX-License-Identifier: GPL-3.0-only
import pytest
from power_fixtures import constant_efficiency_psu, controlled_fan_policy
from pydantic import ValidationError

from power_model import OperatingState
from power_model.models.advanced.components import (
    CPU,
    DDR5DIMM,
    X86CPU,
    ComponentGroup,
    FrontendDPU,
    NVMeDrive,
    PCIe4x16Retimer,
    PCIeSwitch,
    PEX89144PCIeSwitch,
    SystemSide400GTransceiver,
    SystemSide800GTransceiver,
    SystemSideTransceiver,
)
from power_model.models.advanced.components.scaleoutnic import ConnectX7NICCard, ConnectX8NICCard
from power_model.models.advanced.systems import (
    B200HGXSystemChassis,
    B300HGXSystemChassis,
    HopperHGXSystemChassis,
    MI300HGXSystemChassis,
    MI325HGXSystemChassis,
    MI355HGXSystemChassis,
    SystemGroup,
)


def chassis(chassis_type=HopperHGXSystemChassis, **kwargs):
    return chassis_type(
        fan_policy=controlled_fan_policy(),
        psu=constant_efficiency_psu(),
        **kwargs,
    )


@pytest.mark.parametrize(
    "chassis_type, expected, expected_components",
    [
        (
            HopperHGXSystemChassis,
            2088.683050750,
            [
                ("X86CPU", 2, 240),
                ("ConnectX7NICCard", 8, 120),
                ("DDR5DIMM", 32, 121.6),
                ("NVMeDrive", 10, 50),
                ("PEX89144PCIeSwitch", 4, 180),
                ("SystemSide400GTransceiver", 8, 64),
            ],
        ),
        (
            B300HGXSystemChassis,
            1977.716258373,
            [
                ("X86CPU", 2, 240),
                ("DDR5DIMM", 32, 121.6),
                ("NVMeDrive", 10, 50),
                ("SystemSide800GTransceiver", 8, 120),
            ],
        ),
    ],
)
def test_chassis_owns_its_inventory_without_caller_supplied_host_components(
    chassis_type, expected, expected_components
):
    result = chassis(chassis_type).estimate_it_power(100)
    assert result.power_w == pytest.approx(expected)
    actual = result.children[1].children
    assert [(node.name, node.quantity) for node in actual] == [
        (name, count) for name, count, _ in expected_components
    ]
    assert [node.power_w for node in actual] == pytest.approx(
        [watts for _, _, watts in expected_components]
    )
    memory = next(node for node in actual if node.name == "DDR5DIMM")
    assert dict(memory.details)["capacity_gb_per_dimm"] == 64


@pytest.mark.parametrize(
    "component",
    [
        CPU(power_w=100),
        X86CPU(state="Idle"),
        ConnectX7NICCard(state="idle"),
        DDR5DIMM(capacity_gb=32),
        NVMeDrive(),
        PEX89144PCIeSwitch(),
        SystemSideTransceiver(power_w=5),
        SystemSide400GTransceiver(),
        SystemSide800GTransceiver(),
    ],
)
def test_automatic_inventory_cannot_be_added_or_overridden_as_extra_components(component):
    with pytest.raises(ValidationError, match="inventory is owned by HopperHGXSystemChassis"):
        chassis(components=(ComponentGroup(component=component),))


@pytest.mark.parametrize(
    "field",
    [
        "cpu_count",
        "dimm_count",
        "dimm_capacity_gb",
        "system_nic_count",
        "transceiver_count",
        "pex89144_count",
    ],
)
def test_fixed_inventory_counts_cannot_be_overridden(field):
    with pytest.raises(ValidationError, match=field):
        chassis(**{field: 1})


def test_extra_components_feed_fans_without_replacing_automatic_inventory():
    result = chassis(
        components=(
            ComponentGroup(component=FrontendDPU(power_w=100)),
            ComponentGroup(component=PCIe4x16Retimer(power_w=5), quantity=2),
        )
    ).estimate_it_power(100)
    assert result.power_w == pytest.approx(2200.956583420)
    assert result.children[1].power_w == pytest.approx(885.6)
    assert result.children[-1].power_w == pytest.approx(19.356583420)
    assert [(node.name, node.power_w) for node in result.children[1].children[-2:]] == [
        ("FrontendDPU", 100),
        ("PCIe4x16Retimer", 10),
    ]


@pytest.mark.parametrize(
    "component, message",
    [
        (ConnectX8NICCard(state="active"), "inventory is owned by B300HGXSystemChassis"),
        (PEX89144PCIeSwitch(), "inventory is owned by B300HGXSystemChassis"),
        (PCIeSwitch(power_w=45), "B300 has no chassis-level PCIe switches"),
    ],
)
def test_b300_rejects_chassis_nics_and_pcie_switches(component, message):
    with pytest.raises(ValidationError, match=message):
        chassis(B300HGXSystemChassis, components=(ComponentGroup(component=component),))


def test_b300_scenario_updates_board_nics_without_mutating_the_board():
    system = chassis(B300HGXSystemChassis)
    active = system.estimate_it_power(100, operating_state=OperatingState(using_scale_out=True))
    idle = system.estimate_it_power(100)
    assert active.children[0].children[2].power_w == 400
    assert idle.children[0].children[2].power_w == 240
    assert active.power_w == pytest.approx(2139.329413290)
    assert idle.power_w == pytest.approx(1977.716258373)
    assert system.ubb.estimate_breakdown(100).power_w == 1440


@pytest.mark.parametrize(
    "chassis_type, optics_name, expected_optics_w",
    [
        (MI300HGXSystemChassis, "SystemSide400GTransceiver", 192),
        (MI325HGXSystemChassis, "SystemSide400GTransceiver", 192),
        (MI355HGXSystemChassis, "SystemSide400GTransceiver", 192),
        (HopperHGXSystemChassis, "SystemSide400GTransceiver", 192),
        (B200HGXSystemChassis, "SystemSide400GTransceiver", 192),
        (B300HGXSystemChassis, "SystemSide800GTransceiver", 360),
    ],
)
def test_one_optic_per_nic_stays_powered_and_scales_with_chassis_count(
    chassis_type, optics_name, expected_optics_w
):
    group = SystemGroup(system=chassis(chassis_type), quantity=3)
    for using_scale_out in (False, True):
        result = group.estimate_it_power(
            100, operating_state=OperatingState(using_scale_out=using_scale_out)
        )
        optics = [node for node in result.children[1].children if node.name == optics_name]
        assert [(node.quantity, node.power_w) for node in optics] == [(24, expected_optics_w)]
        nic_nodes = [
            node
            for section in result.children
            for node in section.children
            if node.name.endswith("NICCard")
        ]
        assert sum(node.quantity for node in nic_nodes) == optics[0].quantity
        assert dict(optics[0].details)["power_policy"] == "always_on"
