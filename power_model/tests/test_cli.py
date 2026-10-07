# SPDX-License-Identifier: GPL-3.0-only
import json

import pytest

from power_model import create_power_model
from power_model.cli import main


def hgx_args(system="hopper"):
    # Numerical fixtures use the shared fan policy and each family's PSU bank,
    # evaluated per chassis before doubling and adding the family's two-tier network.
    return [
        "--gpu_level_power_per_gpu=125",
        f"--system={system}",
        "--systems=2",
    ]


def rack_args(system="gb200-nvl72"):
    return ["--gpu_level_power_per_gpu=125", f"--system={system}"]


@pytest.mark.parametrize(
    "flags, offload, cpu_state, cpu_w, state, memory_w, fan_w, it_w, per_gpu",
    [
        ([], False, "FixedSeqLen", 480, "idle", 243.2, 109.045912, 5826.055387, 473.367000),
        (
            ["--workload=fixed-seq-len"],
            False,
            "FixedSeqLen",
            480,
            "idle",
            243.2,
            109.045912,
            5826.055387,
            473.367000,
        ),
        (
            ["--workload=agentic"],
            False,
            "Agentic",
            600,
            "idle",
            243.2,
            113.012483,
            5956.068674,
            483.930580,
        ),
        (
            ["--workload=agentic-cpu-offloading"],
            True,
            "AgenticOffloadingOn",
            800,
            "active",
            448,
            127.100680,
            6394.029590,
            519.514904,
        ),
    ],
)
def test_cli_hopper_workloads_select_cpu_and_memory_power_with_air_cooling(
    capsys, flags, offload, cpu_state, cpu_w, state, memory_w, fan_w, it_w, per_gpu
):
    main(hgx_args() + flags)
    result = json.loads(capsys.readouterr().out)
    assert result["model_name"] == "OSSAllinPowerModel"
    assert result["cpu_offload"] is offload
    assert result["cooling_mode"] == "air"
    assert result["pue"] == 1.3
    assert result["gpu_count"] == 16
    assert result["it_power_w"] == pytest.approx(it_w)
    assert result["AllInPower_per_gpu"] == pytest.approx(per_gpu)
    assert result["facility_to_gpu_power_ratio"] == pytest.approx(per_gpu / 125)
    system = result["components"][0]["children"][0]
    assert system["name"] == "HopperHGXSystemChassis"
    nic = system["children"][1]["children"][1]
    assert nic["name"] == "ConnectX7NICCard"
    assert nic["quantity"] == 16
    assert nic["power_w"] == 240
    assert dict(nic["details"])["state"] == "idle"
    cpu = system["children"][1]["children"][0]
    assert cpu["quantity"] == 4
    assert cpu["power_w"] == cpu_w
    assert dict(cpu["details"])["state"] == cpu_state
    memory = system["children"][1]["children"][2]
    assert memory["quantity"] == 64
    assert memory["power_w"] == pytest.approx(memory_w)
    assert dict(memory["details"])["state"] == state
    assert system["children"][-1]["power_w"] == pytest.approx(fan_w)
    pcie_switches = system["children"][1]["children"][-2]
    assert pcie_switches["name"] == "PEX89144PCIeSwitch"
    assert (pcie_switches["quantity"], pcie_switches["power_w"]) == (8, 360)
    assert dict(pcie_switches["details"])["state"] == "active"


@pytest.mark.parametrize(
    "system_name, chassis_name, board_name, board_w, nic_name, idle_w, active_w, "
    "idle_per_gpu, active_per_gpu",
    [
        (
            "mi300",
            "MI300HGXSystemChassis",
            "MI300UBB",
            2192,
            "Thor2NICCard",
            320,
            480,
            388.915798,
            413.989391,
        ),
        (
            "mi325",
            "MI325HGXSystemChassis",
            "MI325UBB",
            2192,
            "Thor2NICCard",
            320,
            480,
            400.229637,
            425.289895,
        ),
        (
            "mi355",
            "MI355HGXSystemChassis",
            "MI355UBB",
            2192,
            "PollaraNICCard",
            320,
            480,
            403.919921,
            429.428243,
        ),
        (
            "b200",
            "B200HGXSystemChassis",
            "B200HGXBoard",
            2992,
            "ConnectX7NICCard",
            240,
            400,
            469.949645,
            494.895946,
        ),
    ],
)
@pytest.mark.parametrize("nic_state", ["active", "idle"])
def test_cli_selects_each_hgx_family_board_and_nic(
    capsys,
    system_name,
    chassis_name,
    board_name,
    board_w,
    nic_name,
    idle_w,
    active_w,
    idle_per_gpu,
    active_per_gpu,
    nic_state,
):
    scale_out_args = ["--using-scale-out"] if nic_state == "active" else []
    main(hgx_args(system_name) + scale_out_args)
    nic_w = active_w if nic_state == "active" else idle_w
    per_gpu = active_per_gpu if nic_state == "active" else idle_per_gpu
    result = json.loads(capsys.readouterr().out)
    assert result["AllInPower_per_gpu"] == pytest.approx(per_gpu)
    assert result["gpu_count"] == 16
    assert result["cooling_mode"] == "air"
    system = result["components"][0]["children"][0]
    assert system["name"] == chassis_name
    assert system["children"][0]["name"] == board_name
    assert system["children"][0]["power_w"] == board_w
    if system_name in ("mi300", "mi325", "mi355"):
        gpus, retimers = system["children"][0]["children"]
        assert (gpus["quantity"], gpus["power_w"]) == (16, 2000)
        assert (retimers["name"], retimers["quantity"], retimers["power_w"]) == (
            "PCIe5x16Retimer",
            16,
            192,
        )
    elif system_name == "b200":
        gpus, switches, retimers = system["children"][0]["children"]
        assert (gpus["quantity"], gpus["power_w"]) == (16, 2000)
        assert (switches["name"], switches["quantity"], switches["power_w"]) == (
            "BlackwellHGXNVSwitch",
            4,
            800,
        )
        assert dict(switches["details"])["bandwidth_tbps"] == 28.8
        assert dict(switches["details"])["per_switch_power_w"] == 200
        assert (retimers["quantity"], retimers["power_w"]) == (16, 192)
    nic = system["children"][1]["children"][1]
    assert nic["name"] == nic_name
    assert nic["quantity"] == 16
    assert nic["power_w"] == nic_w
    assert dict(nic["details"])["state"] == nic_state
    assert dict(nic["details"])["bandwidth_gbps"] == 400
    memory = system["children"][1]["children"][2]
    assert memory["quantity"] == 64
    assert memory["power_w"] == pytest.approx(243.2)
    assert dict(memory["details"])["capacity_gb_per_dimm"] == 64
    pcie_switches = [
        component
        for component in system["children"][1]["children"]
        if component["name"] == "PEX89144PCIeSwitch"
    ]
    assert len(pcie_switches) == 1
    assert (pcie_switches[0]["quantity"], pcie_switches[0]["power_w"]) == (8, 360)
    assert dict(pcie_switches[0]["details"])["state"] == "active"


@pytest.mark.parametrize(
    "state, workload, expected, it_w, fan_w, board_w, nic_w, cpu_w, memory_w",
    [
        ("idle", "fixed-seq-len", 496.128120, 6106.192251, 95.308226, 3280, 480, 480, 243.2),
        ("active", "fixed-seq-len", 546.473509, 6725.827799, 102.615940, 3600, 800, 480, 243.2),
        (
            "idle",
            "agentic-cpu-offloading",
            542.530850,
            6677.302765,
            107.483510,
            3280,
            480,
            800,
            448,
        ),
        (
            "active",
            "agentic-cpu-offloading",
            592.608181,
            7293.639155,
            115.393343,
            3600,
            800,
            800,
            448,
        ),
    ],
)
def test_cli_b300_routes_scale_out_to_board_nics_and_preserves_workload_accounting(
    capsys, state, workload, expected, it_w, fan_w, board_w, nic_w, cpu_w, memory_w
):
    flags = ["--using-scale-out"] if state == "active" else []
    main(hgx_args("b300") + [f"--workload={workload}", *flags])
    result = json.loads(capsys.readouterr().out)
    assert result["AllInPower_per_gpu"] == pytest.approx(expected)
    assert result["it_power_w"] == pytest.approx(it_w)
    assert result["gpu_count"] == 16
    system = result["components"][0]["children"][0]
    assert system["name"] == "B300HGXSystemChassis"
    board = system["children"][0]
    assert (board["name"], board["power_w"]) == ("B300HGXBoard", board_w)
    gpus, switches, nics = board["children"]
    assert (gpus["quantity"], gpus["power_w"]) == (16, 2000)
    assert (switches["name"], switches["quantity"], switches["power_w"]) == (
        "BlackwellHGXNVSwitch",
        4,
        800,
    )
    assert dict(switches["details"])["bandwidth_tbps"] == 28.8
    assert (nics["name"], nics["quantity"], nics["power_w"]) == ("ConnectX8NICCard", 16, nic_w)
    assert dict(nics["details"])["state"] == state
    assert dict(nics["details"])["bandwidth_gbps"] == 800
    cpu, memory, nvme, optics = system["children"][1]["children"]
    assert [cpu["name"], memory["name"], nvme["name"]] == ["X86CPU", "DDR5DIMM", "NVMeDrive"]
    assert cpu["power_w"] == cpu_w
    assert memory["power_w"] == pytest.approx(memory_w)
    assert (nvme["quantity"], nvme["power_w"]) == (20, 100)
    assert (optics["name"], optics["quantity"], optics["power_w"]) == (
        "SystemSide800GTransceiver",
        16,
        240,
    )
    assert system["children"][-1]["power_w"] == pytest.approx(fan_w)
    config = system["configuration"]
    assert config["ubb"]["nic"]["state"] == state
    assert config["components"][-1]["component"]["power_w"] == 15


@pytest.mark.parametrize(
    "system_name, system_class",
    [("gb200-nvl72", "GB200NVL72RackScaleSystem"), ("gb300-nvl72", "GB300NVL72RackScaleSystem")],
)
@pytest.mark.parametrize("workload", ["fixed-seq-len", "agentic-cpu-offloading"])
def test_cli_nvl72_reports_rack_power(capsys, system_name, system_class, workload):
    main(rack_args(system_name) + [f"--workload={workload}"])
    result = json.loads(capsys.readouterr().out)
    assert result["gpu_count"] == 72
    assert result["components"][0]["children"][0]["name"] == system_class
    assert result["pue"] == 1.1
    assert result["AllInPower_per_gpu"] > 125
    assert result["facility_power_w"] == pytest.approx(result["it_power_w"] * 1.1)


@pytest.mark.parametrize(
    "system, expected, ratio", [("hopper", 1950, 1.95), ("gb300-nvl72", 1650, 1.65)]
)
def test_cli_example_formula_uses_the_selected_system_cooling(capsys, system, expected, ratio):
    main(["--model=example", "--gpu_level_power_per_gpu=1000", f"--system={system}"])
    result = json.loads(capsys.readouterr().out)
    assert result["model_name"] == "ExamplePowerModel"
    assert result["AllInPower_per_gpu"] == pytest.approx(expected)
    assert result["facility_to_gpu_power_ratio"] == pytest.approx(ratio)
    assert result["cpu_offload"] is False
    assert result["scope"] == "per_gpu_reference"
    model_result = create_power_model(model="example", system=system).estimate_breakdown(1000)
    assert "facility_to_gpu_power_ratio" not in model_result.model_dump()


@pytest.mark.parametrize("power", ["0", "5e-324"])
@pytest.mark.parametrize("bom", [False, True])
def test_cli_ratio_is_unavailable_for_zero_input_or_unrepresentable_ratio(capsys, power, bom):
    args = ["--system=h100", f"--gpu-level-power-per-gpu={power}"]
    if bom:
        args.append("--power-breakdown-per-chassis")
    main(args)
    output = capsys.readouterr().out
    if bom:
        assert "Facility / GPU power ratio: n/a" in output
    else:
        assert json.loads(output)["facility_to_gpu_power_ratio"] is None


def test_cli_accepts_full_class_names_and_the_hyphenated_gpu_flag(capsys):
    main(
        [
            "--gpu-level-power-per-gpu=125",
            *hgx_args("HopperHGXSystemChassis")[1:],
            "--model=OSSAllinPowerModel",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert result["AllInPower_per_gpu"] == pytest.approx(473.367000)


def test_cli_help_lists_models_and_systems(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "agentic-cpu-offloading" in help_text
    assert "GB200NVL72RackScaleSystem" in help_text
    assert "GB300NVL72RackScaleSystem" in help_text
    assert "Models:" in help_text
    assert "OSSAllinPowerModel" in help_text
    assert "ExamplePowerModel" in help_text
    assert "Systems:" in help_text
    assert "HopperHGXSystemChassis (H100 / H200)" in help_text
    assert "--power-breakdown-per-chassis" in help_text
    assert "--fan-base-power-w" not in help_text
    assert "--fan-watts-per-watt" not in help_text
    assert "--power-conversion-loss-w" not in help_text
    assert "--network-power-w" not in help_text


@pytest.mark.parametrize(
    "args, message",
    [
        (["--gpu_level_power_per_gpu=125"], "--system"),
        (["--gpu_level_power_per_gpu=125", "--system=unknown"], "invalid choice"),
        (["--gpu_level_power_per_gpu=125", "--system=hopper", "--model=unknown"], "invalid choice"),
        (
            ["--gpu_level_power_per_gpu=125", "--system=hopper", "--workload=idle"],
            "invalid choice",
        ),
        (hgx_args() + ["--network-power-w=0"], "unrecognized arguments: --network-power-w=0"),
        (
            [
                "--model=example",
                "--gpu_level_power_per_gpu=400",
                "--system=h100",
                "--power-breakdown-per-chassis",
            ],
            "--power-breakdown-per-chassis requires the oss model",
        ),
        (
            [
                "--model=example",
                "--gpu_level_power_per_gpu=125",
                "--system=hopper",
                "--workload=agentic-cpu-offloading",
            ],
            "CPU offloading requires OSSAllinPowerModel",
        ),
        (
            [
                "--model=example",
                "--gpu_level_power_per_gpu=125",
                "--system=hopper",
                "--cooling=air",
            ],
            "unrecognized arguments: --cooling=air",
        ),
        (hgx_args() + ["--cpu-power-w=120"], "unrecognized arguments: --cpu-power-w=120"),
        (hgx_args() + ["--fan-base-power-w=0"], "unrecognized arguments: --fan-base-power-w=0"),
        (
            hgx_args() + ["--fan-watts-per-watt=0.1"],
            "unrecognized arguments: --fan-watts-per-watt=0.1",
        ),
        (
            hgx_args() + ["--power-conversion-loss-w=50"],
            "unrecognized arguments: --power-conversion-loss-w=50",
        ),
        (hgx_args() + ["--cpu-offload=on"], "unrecognized arguments: --cpu-offload=on"),
        (hgx_args() + ["--cpu-offloading=on"], "unrecognized arguments: --cpu-offloading=on"),
        (
            hgx_args("b300") + ["--ubb-non-gpu-power-w=80"],
            "unrecognized arguments: --ubb-non-gpu-power-w=80",
        ),
        (hgx_args() + ["--nic=connectx7"], "unrecognized arguments: --nic=connectx7"),
        (hgx_args() + ["--nic-state=active"], "unrecognized arguments: --nic-state=active"),
        (
            hgx_args() + ["--nic-active-power-w=25"],
            "unrecognized arguments: --nic-active-power-w=25",
        ),
        (hgx_args() + ["--nic-idle-power-w=15"], "unrecognized arguments: --nic-idle-power-w=15"),
        (rack_args() + ["--gpu-count=72"], "unrecognized arguments: --gpu-count=72"),
        (
            rack_args() + ["--rack-components-power-w=100"],
            "unrecognized arguments: --rack-components-power-w=100",
        ),
        (
            rack_args() + ["--scale-up-fabric-power-w=100"],
            "unrecognized arguments: --scale-up-fabric-power-w=100",
        ),
        (hgx_args() + ["--dimm-count=32"], "unrecognized arguments: --dimm-count=32"),
        (hgx_args() + ["--dimm-capacity-gb=64"], "unrecognized arguments: --dimm-capacity-gb=64"),
    ],
)
def test_cli_rejects_missing_or_invalid_scenarios(capsys, args, message):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert message in captured.err
