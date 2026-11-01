# SPDX-License-Identifier: GPL-3.0-only
import pytest
from power_fixtures import constant_efficiency_psu, controlled_fan_policy

from power_model import CoolingProfile
from power_model.cli import main
from power_model.models.advanced import Cluster, OSSAllinPowerModel, SystemGroup
from power_model.models.advanced.systems import HopperHGXSystemChassis, MI300HGXSystemChassis
from power_model.reporting import format_power_breakdown_per_chassis


@pytest.mark.parametrize(
    "system, chassis, board_row, nic_row, memory_row, fan_row, psu_row, network_w, "
    "it_total, facility_total, per_gpu, ratio",
    [
        (
            "h100",
            "HopperHGXSystemChassis",
            "├─ HopperHGXBoard 3,696.00 1 3,696.00",
            "│ ├─ ConnectX7NICCard 15.00 8 120.00",
            "│ ├─ DDR5DIMM 2.75 32 88.00",
            "└─ FanPower 169.43 1 169.43",
            "├─ Power conversion losses 261.94 1 261.94",
            "810.00",
            "10,728.72",
            "13,947.34",
            "871.71",
            "2.1793x",
        ),
        (
            "mi300",
            "MI300HGXSystemChassis",
            "├─ MI300UBB 3,296.00 1 3,296.00",
            "│ ├─ Thor2NICCard 20.00 8 160.00",
            "│ ├─ DDR5DIMM 2.75 16 44.00",
            "└─ FanPower 142.57 1 142.57",
            "├─ Power conversion losses 237.98 1 237.98",
            "562.12",
            "9,571.22",
            "12,442.59",
            "777.66",
            "1.9442x",
        ),
    ],
)
def test_cli_bom_reports_one_chassis_with_nested_inventory_and_separate_cluster_totals(
    capsys,
    system,
    chassis,
    board_row,
    nic_row,
    memory_row,
    fan_row,
    psu_row,
    network_w,
    it_total,
    facility_total,
    per_gpu,
    ratio,
):
    main(
        [
            "--gpu-level-power-per-gpu=400",
            f"--system={system}",
            "--systems=2",
            "--power-breakdown-per-chassis",
        ]
    )
    output = capsys.readouterr().out
    rows = [" ".join(line.split()) for line in output.splitlines()]
    assert f"{chassis} — one chassis (configured quantity: 2)" in output
    assert "Component Power / unit (W) Quantity Extended power (W)" in rows
    assert board_row in rows
    assert "│ ├─ GPUs 400.00 8 3,200.00" in rows
    assert "│ └─ PCIe5x16Retimer 12.00 8 96.00" in rows
    assert nic_row in rows
    assert "│ ├─ X86CPU 165.00 2 330.00" in rows
    assert memory_row in rows
    assert "│ ├─ NVMeDrive 5.00 10 50.00" in rows
    assert "│ ├─ PEX89144PCIeSwitch 45.00 4 180.00" in rows
    assert "│ └─ SystemSide400GTransceiver 8.00 8 64.00" in rows
    assert fan_row in rows
    assert psu_row in rows
    assert "Cluster totals (2 chassis, 16 GPUs)" in output
    assert f"Shared external network AC power: {network_w} W" in output
    assert f"IT AC power: {it_total} W" in output
    assert f"All in Utility Power: {facility_total} W" in output
    assert f"AllInPower_per_gpu: {per_gpu} W/GPU" in output
    assert f"Facility / GPU power ratio: {ratio}" in output


def test_cli_b300_bom_nests_active_nics_on_board_and_uses_offloading_component_power(capsys):
    main(
        [
            "--gpu-level-power-per-gpu=400",
            "--system=b300",
            "--workload=agentic-cpu-offloading",
            "--using-scale-out",
            "--power-breakdown-per-chassis",
        ]
    )
    output = capsys.readouterr().out
    rows = [" ".join(line.split()) for line in output.splitlines()]
    assert "Workload: agentic-cpu-offloading | Scale-out: on" in output
    assert "├─ B300HGXBoard 4,000.00 1 4,000.00" in rows
    assert "│ ├─ GPUs 400.00 8 3,200.00" in rows
    assert "│ ├─ BlackwellHGXNVSwitch 200.00 2 400.00" in rows
    assert "│ └─ ConnectX8NICCard 50.00 8 400.00" in rows
    assert "├─ Generic components 648.00 1 648.00" in rows
    assert "│ ├─ X86CPU 195.00 2 390.00" in rows
    assert "│ ├─ DDR5DIMM 2.75 32 88.00" in rows
    assert "│ └─ SystemSide800GTransceiver 15.00 8 120.00" in rows
    assert "PCIe5x16Retimer" not in output
    assert "PEX89144PCIeSwitch" not in output


def test_mixed_system_report_normalizes_each_group_using_its_own_chassis_count():
    model = OSSAllinPowerModel(
        cooling=CoolingProfile(mode="air"),
        cluster=Cluster(
            systems=(
                SystemGroup(
                    system=HopperHGXSystemChassis(
                        fan_policy=controlled_fan_policy(), psu=constant_efficiency_psu()
                    ),
                    quantity=2,
                ),
                SystemGroup(
                    system=MI300HGXSystemChassis(
                        fan_policy=controlled_fan_policy(), psu=constant_efficiency_psu()
                    ),
                    quantity=3,
                ),
            ),
            networking=(),
        ),
    )
    output = format_power_breakdown_per_chassis(model.estimate_breakdown(125))
    rows = [" ".join(line.split()) for line in output.splitlines()]
    assert "HopperHGXSystemChassis — one chassis (configured quantity: 2)" in output
    assert "MI300HGXSystemChassis — one chassis (configured quantity: 3)" in output
    assert "HopperHGXSystemChassis 2,353.26 1 2,353.26" in rows
    assert "MI300HGXSystemChassis 1,938.27 1 1,938.27" in rows
    assert rows.count("│ ├─ GPUs 125.00 8 1,000.00") == 2
    assert "Cluster totals (5 chassis, 40 GPUs)" in output
    assert "IT AC power: 10,521.33 W" in output
    assert "All in Utility Power: 13,677.73 W" in output
