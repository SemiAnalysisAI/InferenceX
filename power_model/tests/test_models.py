# SPDX-License-Identifier: GPL-3.0-only
import json

import pytest
from power_fixtures import constant_efficiency_psu, controlled_fan_policy
from pydantic import ValidationError

from power_model import (
    CoolingProfile,
    ExamplePowerModel,
    OperatingState,
    PowerEstimate,
    create_power_model,
)
from power_model.models.advanced import (
    Cluster,
    NetworkGroup,
    OSSAllinPowerModel,
    SystemGroup,
)
from power_model.models.advanced.components import (
    ComponentGroup,
    NVMeDrive,
)
from power_model.models.advanced.networking import QM9790NDRInfiniBandSwitch
from power_model.models.advanced.systems import (
    B300HGXSystemChassis,
    GB200NVL72RackScaleSystem,
    GB300NVL72RackScaleSystem,
    HopperHGXSystemChassis,
)


def hgx(**changes):
    inputs = {
        "fan_policy": controlled_fan_policy(),
        "psu": constant_efficiency_psu(0.8),
    }
    return HopperHGXSystemChassis(**(inputs | changes))


def advanced(
    system=None,
    *,
    quantity=1,
    network_switches=None,
    cooling="liquid",
    workload_state="fixed-seq-len",
    using_scale_out=True,
):
    networking = (
        ()
        if network_switches is None
        else (NetworkGroup(gear=QM9790NDRInfiniBandSwitch(), quantity=network_switches),)
    )
    return OSSAllinPowerModel(
        cooling=CoolingProfile(mode=cooling),
        workload_state=workload_state,
        using_scale_out=using_scale_out,
        cluster=Cluster(
            systems=(SystemGroup(system=hgx() if system is None else system, quantity=quantity),),
            networking=networking,
        ),
    )


def nodes(components):
    """Inspect the public component tree as a result consumer would."""
    for component in components:
        yield component
        yield from nodes(component.children)


def named(result, name):
    return [node for node in nodes(result.components) if node.name == name]


@pytest.mark.parametrize(
    "system_type, network_name, network_w",
    [
        (HopperHGXSystemChassis, "QM9790NDRInfiniBandSwitch", 947.25),
        (B300HGXSystemChassis, "Generic512TEthernetSwitch", 1395),
    ],
)
def test_model_constructs_equipment_and_resolves_scenario_without_the_cli(
    system_type, network_name, network_w
):
    model = OSSAllinPowerModel.for_system(
        system_type,
        workload_state="agentic-cpu-offloading",
        using_scale_out=True,
        systems=2,
    )
    result = model.estimate_breakdown(125)
    assert result.gpu_count == 16
    assert (result.cooling_mode, result.pue) == ("air", 1.3)
    assert named(result, "X86CPU")[0].power_w == 800
    assert named(result, "DDR5DIMM")[0].power_w == 448
    assert named(result, network_name)[0].power_w == network_w
    losses = named(result, "Power conversion losses")[0]
    assert losses.power_w > 0
    assert dict(losses.details)["dc_load_w_per_system"] > 2594


def test_public_factory_uses_model_owned_defaults_and_family_nics():
    result = create_power_model(system="MI300HGXSystemChassis").estimate_breakdown(100)
    assert result.workload_state == "fixed-seq-len"
    assert result.using_scale_out is False
    assert result.cpu_offload is False
    assert named(result, "Thor2NICCard")[0].power_w == 160
    assert named(result, "X86CPU")[0].power_w == 240
    assert named(result, "Generic512TEthernetSwitch")[0].power_w == 281.0625
    fan = named(result, "FanPower")[0]
    assert dict(fan.details)["non_fan_power_w_per_system"] == pytest.approx(1711.6)


@pytest.mark.parametrize("factory", [create_power_model, OSSAllinPowerModel.for_system])
@pytest.mark.parametrize(
    "argument",
    ["fan_base_power_w", "fan_watts_per_watt", "power_conversion_loss_w", "network_power_w"],
)
def test_scenario_factories_reject_retired_hardware_power_overrides(factory, argument):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        factory(system="h100", **{argument: 0})


@pytest.mark.parametrize("cooling, expected", [("air", 1950.0), ("liquid", 1650.0)])
def test_example_formula_and_breakdown(cooling, expected):
    model = ExamplePowerModel(cooling=CoolingProfile(mode=cooling))
    result = model.estimate_breakdown(1000)
    assert model.estimate(1000) == pytest.approx(expected)
    assert result.AllInPower_per_gpu == pytest.approx(expected)
    assert result.it_power_w == 1500
    assert result.scope == "per_gpu_reference"
    assert result.gpu_count == 1
    assert result.components[1].power_w == 500


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), "100", True, None])
def test_invalid_per_gpu_input_is_rejected(value):
    model = ExamplePowerModel(cooling=CoolingProfile(mode="air"))
    with pytest.raises(ValueError):
        model.estimate(value)


def test_finite_input_cannot_produce_infinite_facility_power():
    model = ExamplePowerModel(cooling=CoolingProfile(mode="air"))
    with pytest.raises(ValueError):
        model.estimate(1e308)


def test_unsupported_cooling_is_rejected():
    with pytest.raises(ValidationError):
        CoolingProfile(mode="hybrid")


def test_hgx_complete_power_and_serializable_provenance():
    model = advanced()
    result = model.estimate_breakdown(125)
    assert result.gpu_count == 8
    assert result.it_power_w == pytest.approx(2970.735473339775)
    assert result.facility_power_w == pytest.approx(3267.8090206737525)
    assert model.estimate(125) == pytest.approx(408.476127584219)
    assert named(result, "GPUs")[0].power_w == 1000
    assert named(result, "HopperHGXBoard")[0].power_w == 1496
    switch = named(result, "HopperHGXNVSwitch")[0]
    assert (switch.quantity, switch.power_w) == (4, 400)
    assert dict(switch.details)["bandwidth_tbps"] == 12.8
    retimer = named(result, "PCIe5x16Retimer")[0]
    assert (retimer.quantity, retimer.power_w) == (8, 96)
    pcie_switch = named(result, "PEX89144PCIeSwitch")[0]
    assert (pcie_switch.quantity, pcie_switch.power_w) == (4, 180)
    assert dict(pcie_switch.details)["state"] == "active"
    assert named(result, "NVMeDrive")[0].power_w == 50
    assert named(result, "NVMeDrive")[0].quantity == 10
    optics = named(result, "SystemSide400GTransceiver")[0]
    assert (optics.quantity, optics.power_w) == (8, 64)
    cpu = named(result, "X86CPU")[0]
    assert (cpu.quantity, cpu.power_w) == (2, 240)
    assert dict(cpu.details)["architecture"] == "x86"
    memory = named(result, "DDR5DIMM")[0]
    assert memory.quantity == 32
    assert memory.power_w == pytest.approx(121.6)
    assert dict(memory.details)["state"] == "idle"
    assert dict(memory.details)["capacity_gb_per_dimm"] == 64
    assert dict(memory.details)["memory_type"] == "DDR5"
    assert sum(nic.quantity for nic in named(result, "ConnectX7NICCard")) == 8
    fan = named(result, "FanPower")[0]
    assert fan.power_w == pytest.approx(24.988378671820)
    assert dict(fan.details)["non_fan_power_w_per_system"] == 2351.6
    exported = json.loads(result.model_dump_json())
    assert exported["AllInPower_per_gpu"] == pytest.approx(408.476127584219)
    config = result.components[0].children[0].configuration
    assert config["components"][1]["component"]["state"] == "active"
    assert config["fan_policy"]["fan_power_ratio_at_design"] == 0.1
    assert config["psu"]["efficiency_curve"] == [
        {"load_fraction": 0.0, "efficiency": 0.8},
        {"load_fraction": 1.0, "efficiency": 0.8},
    ]
    assert exported["components"][0]["children"][0]["configuration"] == config
    assert PowerEstimate.model_validate_json(result.model_dump_json()).it_power_w == pytest.approx(
        2970.735473339775
    )
    assert config["ubb"]["non_gpu_power_w"] == 496
    assert config["components"][-1]["component"]["power_w"] == 8
    assert named(result, "NVMeDrive")[0].provenance.kind == "assumed"


def test_idle_nics_reduce_both_direct_power_and_fans():
    result = advanced(using_scale_out=False).estimate_breakdown(125)
    assert result.it_power_w == pytest.approx(2867.654856429474)
    assert named(result, "FanPower")[0].power_w == pytest.approx(22.523885143580)
    assert named(result, "ConnectX7NICCard")[0].power_w == 120
    assert result.AllInPower_per_gpu == pytest.approx(394.302542759053)


@pytest.mark.parametrize(
    "workload, offload, state, cpu_w, memory_w, fan_w, it_w, per_gpu",
    [
        (
            "fixed-seq-len",
            False,
            "FixedSeqLen",
            240,
            121.6,
            24.988378671820,
            2970.735473339775,
            408.476127584219,
        ),
        (
            "agentic",
            False,
            "Agentic",
            300,
            121.6,
            26.950296628687,
            3048.187870785858,
            419.125832233056,
        ),
        (
            "agentic-cpu-offloading",
            True,
            "AgenticOffloadingOn",
            400,
            224,
            34.321358824742,
            3310.401698530927,
            455.180233548002,
        ),
    ],
)
def test_workload_resolves_cpus_dimms_fans_and_exported_state_without_mutating_hardware(
    workload, offload, state, cpu_w, memory_w, fan_w, it_w, per_gpu
):
    system = hgx()
    result = advanced(system, workload_state=workload).estimate_breakdown(125)
    assert result.cpu_offload is offload
    assert result.workload_state == workload
    assert result.using_scale_out is True
    cpu = named(result, "X86CPU")[0]
    assert (cpu.quantity, cpu.power_w) == (2, cpu_w)
    assert dict(cpu.details)["state"] == state
    assert named(result, "DDR5DIMM")[0].power_w == pytest.approx(memory_w)
    assert named(result, "FanPower")[0].power_w == pytest.approx(fan_w)
    assert result.it_power_w == pytest.approx(it_w)
    assert result.AllInPower_per_gpu == pytest.approx(per_gpu)
    config = result.components[0].children[0].configuration
    assert config["workload_state"] == workload
    assert config["using_scale_out"] is True
    assert config["cpu_offload"] is offload
    assert config["components"][0]["component"]["state"] == state
    assert config["components"][0]["component"]["power_w"] == cpu_w / 2
    assert advanced(system).estimate(125) == pytest.approx(408.476127584219)


@pytest.mark.parametrize("value", ["false", 1, None])
def test_using_scale_out_requires_a_boolean(value):
    with pytest.raises(ValidationError):
        advanced(using_scale_out=value)
    with pytest.raises(ValidationError):
        OperatingState(using_scale_out=value)


@pytest.mark.parametrize("value", ["unknown", 1, None, True])
def test_workload_requires_a_supported_name(value):
    with pytest.raises(ValidationError):
        advanced(workload_state=value)


def test_example_model_rejects_offloading_workload_without_a_memory_inventory():
    with pytest.raises(ValidationError, match="CPU offloading requires"):
        ExamplePowerModel(
            cooling=CoolingProfile(mode="air"), workload_state="agentic-cpu-offloading"
        )


@pytest.mark.parametrize(
    "quantity, count, nvme_w, fan_w, expected",
    [
        (1, 8, 50, 24.988378671820, 582.138627584219),
        (2, 16, 100, 49.976757343640, 495.307377584219),
        (3, 24, 150, 74.965136015460, 466.363627584219),
    ],
)
def test_system_quantities_scale_nonlinear_fans_but_not_shared_networking(
    quantity, count, nvme_w, fan_w, expected
):
    result = advanced(quantity=quantity, network_switches=1).estimate_breakdown(125)
    assert result.gpu_count == count
    assert result.AllInPower_per_gpu == pytest.approx(expected)
    assert named(result, "NVMeDrive")[0].power_w == nvme_w
    assert named(result, "FanPower")[0].power_w == pytest.approx(fan_w)
    assert named(result, "QM9790NDRInfiniBandSwitch")[0].power_w == 1263
    for node in nodes(result.components):
        if node.children:
            assert node.power_w == pytest.approx(sum(child.power_w for child in node.children))


def test_mixed_chassis_and_rack_use_their_actual_gpu_counts():
    model = OSSAllinPowerModel(
        cooling=CoolingProfile(mode="liquid"),
        cluster=Cluster(
            systems=(SystemGroup(system=hgx()), SystemGroup(system=GB200NVL72RackScaleSystem())),
            networking=(),
        ),
    )
    result = model.estimate_breakdown(125)
    assert result.gpu_count == 80
    assert [node.power_w for node in named(result, "GPUs")] == [1000, 9000]
    assert [node.power_w for node in named(result, "X86CPU")] == [240]
    assert [node.power_w for node in named(result, "GraceCPU")] == [3600]
    assert [node.power_w for node in named(result, "NVMeDrive")] == [50, 720]
    assert result.facility_power_w == pytest.approx(result.AllInPower_per_gpu * 80)


def test_external_network_power_does_not_heat_the_chassis_fan_model():
    low = advanced(network_switches=1).estimate_breakdown(125)
    high = advanced(network_switches=2).estimate_breakdown(125)
    assert named(low, "FanPower")[0].power_w == pytest.approx(24.988378671820)
    assert named(high, "FanPower")[0].power_w == pytest.approx(24.988378671820)
    assert high.AllInPower_per_gpu == pytest.approx(755.801127584219)


def test_cooling_only_changes_facility_accounting():
    air = advanced(cooling="air").estimate_breakdown(125)
    liquid = advanced(cooling="liquid").estimate_breakdown(125)
    assert air.it_power_w == liquid.it_power_w == pytest.approx(2970.735473339775)
    assert air.AllInPower_per_gpu == pytest.approx(482.744514417713)
    assert liquid.AllInPower_per_gpu == pytest.approx(408.476127584219)


def test_zero_gpu_input_preserves_advanced_overhead_and_gpu_count():
    assert ExamplePowerModel(cooling=CoolingProfile(mode="air")).estimate(0) == 0
    result = advanced().estimate_breakdown(0)
    assert result.gpu_count == 8
    assert result.it_power_w == pytest.approx(1695.430649885949)
    assert result.AllInPower_per_gpu == pytest.approx(233.121714359318)


@pytest.mark.parametrize(
    "changes",
    [
        {"gpu_count": 7},
        {"gpu_count": 8.0},
        {"gpu_count": True},
        {"nvme_count": 9},
        {"power_conversion_loss_w": 0},
        {"components": (ComponentGroup(component=NVMeDrive()),)},
    ],
)
def test_invalid_hgx_inventory_is_rejected(changes):
    with pytest.raises(ValidationError):
        hgx(**changes)


@pytest.mark.parametrize("rack_class", [GB200NVL72RackScaleSystem, GB300NVL72RackScaleSystem])
@pytest.mark.parametrize("gpu_count", [71, 73, 72.0, True])
def test_nvl72_systems_reject_gpu_counts_other_than_exactly_72(rack_class, gpu_count):
    with pytest.raises(ValidationError, match="gpu_count"):
        rack_class(gpu_count=gpu_count)


@pytest.mark.parametrize("quantity", [0, -1, 1.5, True])
def test_invalid_system_quantity_is_rejected(quantity):
    with pytest.raises(ValidationError):
        SystemGroup(system=hgx(), quantity=quantity)


def test_empty_cluster_is_rejected():
    with pytest.raises(ValidationError):
        Cluster(systems=(), networking=())


def test_frozen_inputs_cannot_change_after_validation():
    system = hgx()
    with pytest.raises(ValidationError):
        system.ubb.non_gpu_power_w = -1
    assert advanced(system).estimate(125) == pytest.approx(408.476127584219)


def test_inconsistent_exported_estimate_is_rejected():
    values = advanced().estimate_breakdown(125).model_dump()
    values["gpu_count"] = 16
    with pytest.raises(ValidationError, match="do not reconcile"):
        PowerEstimate.model_validate(values)
