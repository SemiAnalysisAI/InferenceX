# SPDX-License-Identifier: GPL-3.0-only
import json

import pytest
from power_fixtures import controlled_fan_policy
from pydantic import ValidationError

from power_model import CoolingProfile, OperatingState
from power_model.cli import main
from power_model.models.advanced import Cluster, OSSAllinPowerModel, SystemGroup
from power_model.models.advanced.components import (
    AffinityFanPower,
    B200PSUEfficiency,
    B300PSUEfficiency,
    ComponentGroup,
    FrontendDPU,
    HGXFanPolicy,
    HopperPSUEfficiency,
    MI300PSUEfficiency,
    MI325PSUEfficiency,
    MI355PSUEfficiency,
    NormalizedHGXFanPower,
    PSUEfficiencyModel,
    PSUEfficiencyPoint,
)
from power_model.models.advanced.systems import HopperHGXSystemChassis


@pytest.mark.parametrize(
    "tdp, load, expected_w",
    [
        (100, 0, 1),
        (100, 500, 15.625),
        (100, 1000, 64),
        (100, 2000, 64),
        (225, 0, 2),
        (225, 1000, 31.25),
        (225, 2000, 128),
        (225, 4000, 128),
    ],
)
def test_common_fan_policy_scales_watts_with_design_heat_and_caps_at_normal_pwm(
    tdp, load, expected_w
):
    # Controlled 1000/2000 W designs have 125/250 W nameplates. At half design
    # heat both run at 50% PWM: equal relative overhead, different absolute watts.
    fan = NormalizedHGXFanPower(
        gpu_tdp_w=tdp,
        gpu_count=8,
        design_non_gpu_power_w=200,
        policy=HGXFanPolicy(min_pwm_frac=0.2, fan_power_ratio_at_design=0.064),
    )
    assert fan.estimate_w(load) == pytest.approx(expected_w)


def test_fan_affinity_interpolation_and_floor():
    fan = AffinityFanPower(
        electrical_nameplate_w=1000,
        min_pwm_frac=0.2,
        normal_max_pwm_frac=0.8,
        full_cooling_load_w=2000,
        fan_curve_exponent=1,
    )
    assert fan.estimate_w(0) == pytest.approx(8)
    result = fan.estimate_breakdown(1000)
    assert result.power_w == pytest.approx(125)
    assert dict(result.details)["fan_pwm_fraction"] == pytest.approx(0.5)
    assert dict(result.details)["cooling_load_fraction"] == 0.5


@pytest.mark.parametrize(
    "changes",
    [
        {"min_pwm_frac": 0.9},
        {"normal_max_pwm_frac": 0},
        {"fan_curve_exponent": -1},
        {"fan_power_ratio_at_design": 0},
        {"fan_power_ratio_at_design": float("nan")},
    ],
)
def test_invalid_fan_policy_is_rejected(changes):
    with pytest.raises(ValidationError):
        HGXFanPolicy(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"gpu_tdp_w": 0},
        {"gpu_tdp_w": float("inf")},
        {"gpu_count": 0},
        {"design_non_gpu_power_w": -1},
    ],
)
def test_invalid_fan_config_is_rejected(changes):
    with pytest.raises(ValidationError):
        NormalizedHGXFanPower(
            **{"gpu_tdp_w": 100, "gpu_count": 8, "design_non_gpu_power_w": 200} | changes
        )


@pytest.mark.parametrize("load", [-1, float("inf"), float("nan"), True, "100"])
def test_fan_and_psu_reject_invalid_dc_loads(load):
    with pytest.raises(ValidationError):
        NormalizedHGXFanPower(gpu_tdp_w=100, gpu_count=8, design_non_gpu_power_w=200).estimate_w(
            load
        )
    with pytest.raises(ValidationError):
        HopperPSUEfficiency().estimate_loss_breakdown(load)


@pytest.mark.parametrize(
    "psu_type, sharing_w, limit_w",
    [
        (HopperPSUEfficiency, 19800, 13200),
        (B200PSUEfficiency, 15750, 15750),
        (B300PSUEfficiency, 39600, 15000),
        (MI300PSUEfficiency, 18000, 9000),
        (MI325PSUEfficiency, 31500, 15750),
        (MI355PSUEfficiency, 39600, 26400),
    ],
)
def test_family_psus_share_efficiency_quality_but_use_their_own_bank_and_capacity(
    psu_type, sharing_w, limit_w
):
    psu = psu_type()
    low = psu.estimate_loss_breakdown(sharing_w * 0.01)
    assert dict(low.details)["efficiency"] == 0.8864
    midpoint = psu.estimate_loss_breakdown(sharing_w * 0.15)
    assert dict(midpoint.details)["efficiency"] == pytest.approx(0.9343)
    assert dict(midpoint.details)["load_sharing_load_fraction"] == pytest.approx(0.15)
    assert dict(midpoint.details)["load_sharing_capacity_w"] == sharing_w
    assert dict(midpoint.details)["modeled_capacity_w"] == limit_w
    assert psu.estimate_loss_breakdown(0).power_w == 0
    assert psu.estimate_loss_breakdown(limit_w).power_w > 0
    with pytest.raises(ValueError, match="exceeds modeled PSU capacity"):
        psu.estimate_loss_breakdown(limit_w + 1)


@pytest.mark.parametrize(
    "dc_load, efficiency, expected_ac",
    [(0, 0.8, 0), (100, 0.8, 125), (400, 0.9, 444.4444444444444), (800, 1, 800), (1000, 1, 1000)],
)
def test_psu_efficiency_interpolates_and_clamps_without_rounding_accounting(
    dc_load, efficiency, expected_ac
):
    psu = PSUEfficiencyModel(
        n_installed_psu=1,
        n_load_sharing_psu=1,
        n_redundant_capacity_psu=1,
        psu_capacity_w=1000,
        redundancy="none",
        efficiency_curve=(
            PSUEfficiencyPoint(load_fraction=0.2, efficiency=0.8),
            PSUEfficiencyPoint(load_fraction=0.6, efficiency=1),
        ),
    )
    assert psu.efficiency(dc_load) == pytest.approx(efficiency)
    assert dict(psu.estimate_loss_breakdown(dc_load).details)[
        "ac_wall_w_per_system"
    ] == pytest.approx(expected_ac)
    assert psu.estimate_loss_breakdown(dc_load).power_w == pytest.approx(expected_ac - dc_load)


@pytest.mark.parametrize(
    "changes",
    [
        {"n_load_sharing_psu": 7},
        {"n_redundant_capacity_psu": 7},
        {"psu_capacity_w": 0},
        {"system_max_w": -1},
        {"efficiency_curve": (PSUEfficiencyPoint(load_fraction=0.5, efficiency=0.9),)},
        {
            "efficiency_curve": (
                PSUEfficiencyPoint(load_fraction=0.5, efficiency=0.9),
                PSUEfficiencyPoint(load_fraction=0.2, efficiency=0.8),
            )
        },
    ],
)
def test_invalid_psu_config_is_rejected(changes):
    with pytest.raises(ValidationError):
        HopperPSUEfficiency(**changes)


@pytest.mark.parametrize("efficiency", [0, 1.1, float("nan"), True])
def test_psu_rejects_invalid_efficiency(efficiency):
    with pytest.raises(ValidationError):
        PSUEfficiencyPoint(load_fraction=0.5, efficiency=efficiency)


@pytest.mark.parametrize(
    "system, tdp, design_heat, design_host, expected_fan, expected_ac, network_w, per_gpu",
    [
        ("hopper", 700, 7068, 1468, 419.0208, 7829.969787, 473.625, 1349.334153),
        ("h100", 700, 7068, 1468, 419.0208, 7829.969787, 473.625, 1349.334153),
        ("h200", 700, 7068, 1468, 419.0208, 7829.969787, 473.625, 1349.334153),
        ("b200", 1000, 9468, 1468, 561.302905, 10428.700474, 473.625, 1771.62789),
        ("b300", 1200, 11048, 1448, 654.971958, 12307.078518, 697.5, 2113.244009),
        ("mi300", 750, 7064, 1064, 418.783663, 7805.905589, 348.75, 1325.131533),
        ("mi325", 1000, 9086, 1086, 538.656337, 10114.66261, 348.75, 1700.304549),
        ("mi355", 1400, 12286, 1086, 728.365811, 13655.730139, 348.75, 2275.728023),
    ],
)
def test_cli_sizes_fans_from_system_tdp_and_host_inventory_before_psu_and_pue(
    capsys, system, tdp, design_heat, design_host, expected_fan, expected_ac, network_w, per_gpu
):
    # Max-state host subtotals: Hopper/B200 1614 W, AMD 1254 W, B300 1594 W.
    # Full-load fans use the common 563.2 / 9500 ratio. AC fixtures use the
    # 20-50% efficiency segment, except B200 on the 50-100% segment.
    main(
        [
            f"--system={system}",
            f"--gpu_level_power_per_gpu={tdp}",
            "--workload=agentic-cpu-offloading",
            "--using-scale-out",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    chassis = result["components"][0]["children"][0]
    losses, fan = chassis["children"][-2:]
    assert fan["power_w"] == pytest.approx(expected_fan)
    details = dict(fan["details"])
    assert details["non_fan_power_w_per_system"] == design_heat
    assert details["full_cooling_load_w"] == design_heat
    assert details["design_non_gpu_power_w"] == design_host
    assert details["fan_pwm_fraction"] == 0.8
    configuration = chassis["configuration"]
    assert configuration["ubb"]["gpu_tdp_w"] == tdp
    assert configuration["fan_policy"]["normal_max_pwm_frac"] == 0.8
    assert fan["provenance"]["kind"] == "assumed"
    assert chassis["power_w"] == pytest.approx(expected_ac)
    assert result["components"][1]["power_w"] == network_w
    assert result["it_power_w"] == pytest.approx(expected_ac + network_w)
    assert result["AllInPower_per_gpu"] == pytest.approx(per_gpu)
    assert losses["provenance"]["kind"] == "estimated"


def test_workload_and_actual_gpu_watts_change_draw_without_resizing_fan_capacity():
    hardware = HopperHGXSystemChassis()
    states = [
        OperatingState(workload_state="idle"),
        OperatingState(workload_state="fixed-seq-len"),
        OperatingState(workload_state="agentic-cpu-offloading", using_scale_out=True),
    ]
    draws = []
    for state, heat in zip(states, (2158, 2328, 2468), strict=True):
        result = hardware.estimate_it_power(125, operating_state=state)
        fan = result.children[-1]
        details = dict(fan.details)
        assert details["non_fan_power_w_per_system"] == pytest.approx(heat)
        assert details["full_cooling_load_w"] == 7068
        assert details["electrical_nameplate_w"] == pytest.approx(818.4)
        draws.append(fan.power_w)
    assert draws[0] < draws[1] < draws[2]

    # TDP sizes cooling; it must not replace or clamp the caller's GPU watts.
    above_tdp = hardware.estimate_it_power(800)
    assert above_tdp.children[0].children[0].power_w == 6400
    fan = above_tdp.children[-1]
    assert dict(fan.details)["non_fan_power_w_per_system"] == pytest.approx(7728)
    assert dict(fan.details)["full_cooling_load_w"] == 7068
    assert fan.power_w == pytest.approx(419.0208)


def test_optional_components_resize_design_heat_but_system_quantity_does_not():
    hardware = HopperHGXSystemChassis(
        components=(ComponentGroup(component=FrontendDPU(power_w=100)),),
    )
    result = SystemGroup(system=hardware, quantity=2).estimate_it_power(125)
    fan = result.children[-1]
    details = dict(fan.details)
    assert details["full_cooling_load_w"] == 7168
    assert details["design_non_gpu_power_w"] == 1568
    assert details["non_fan_power_w_per_system"] == pytest.approx(2428)
    assert fan.power_w == pytest.approx(119.901158981)


def test_psu_converts_fans_and_components_before_pue_and_quantity_scaling():
    psu = HopperPSUEfficiency(
        efficiency_curve=(
            PSUEfficiencyPoint(load_fraction=0, efficiency=0.8),
            PSUEfficiencyPoint(load_fraction=1, efficiency=0.8),
        )
    )
    hardware = HopperHGXSystemChassis(
        fan_policy=controlled_fan_policy(),
        psu=psu,
    )
    model = OSSAllinPowerModel(
        cooling=CoolingProfile(mode="air"),
        using_scale_out=True,
        cluster=Cluster(systems=(SystemGroup(system=hardware, quantity=2),), networking=()),
    )
    result = model.estimate_breakdown(125)
    chassis = result.components[0].children[0]
    losses, fan = chassis.children[-2:]
    # One chassis: 2408 W components and a 7068 W cooling design.
    # The controlled cubic profile adds 27.949649504 W of fans; 80% PSU
    # efficiency adds 608.987412376 W before quantity scaling and PUE.
    assert fan.power_w == pytest.approx(55.89929900788)
    assert losses.power_w == pytest.approx(1217.97482475197)
    assert result.it_power_w == pytest.approx(6089.874123759843)
    assert result.facility_power_w == pytest.approx(7916.836360887796)
    assert result.AllInPower_per_gpu == pytest.approx(494.802272555487)
    assert dict(fan.details)["non_fan_power_w_per_system"] == pytest.approx(2408)
    assert dict(losses.details)["dc_load_w_per_system"] == pytest.approx(2435.94964950394)
    assert dict(losses.details)["efficiency"] == 0.8


def test_cli_reports_psu_capacity_exceeded(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--system=mi300", "--gpu_level_power_per_gpu=2000"])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "exceeds modeled PSU capacity 9000 W" in captured.err
