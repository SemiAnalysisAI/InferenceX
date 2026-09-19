"""Behavior at the matrix, recipe, preparation, and scheduler-client boundaries."""

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from infx.srt_slurm.contracts import digest, load_mapping, resolve_reference
from infx.srt_slurm.job import intent_id, parse_job
from infx.srt_slurm.launch import execute, publish_outputs, verify_bundle
from infx.srt_slurm.render import ClientPolicy, PilotSite, render_recipe

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs(tmp_path):
    tmp_path = tmp_path.resolve()
    root = tmp_path / "checkout"
    paths = (
        "benchmarks/srt-slurm/phase1/h100-dsv41flash.yaml",
        "runners/srt-slurm/h100-phase1.yaml",
        "benchmarks/srt-slurm/phase1/client-policy.json",
        "golden_al_distribution/dsv41flash_dspark.yaml",
    )
    fixture_names = ("recipe.yaml", "profile.yaml", "client-policy.json", "golden.yaml")
    for name, fixture_name in zip(paths, fixture_names, strict=True):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "utils/fixtures/native_pilot" / fixture_name, target)
    (root / "runtime.json").write_text("{}")
    reference = {
        "runtime": "srt-slurm",
        "contract-version": 1,
        "recipe": paths[0],
        "profile": paths[1],
        "client-policy": paths[2],
        "runtime-lock": "runtime.json",
    }
    row = {
        "image": "example.invalid/pilot:v1",
        "model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "model-prefix": "dsv41flash",
        "precision": "fp4",
        "framework": "vllm",
        "runner": "cluster:h100-dgxc",
        "tp": 8,
        "pp": 1,
        "dcp-size": 1,
        "pcp-size": 1,
        "ep": 1,
        "dp-attn": False,
        "spec-decoding": "mtp",
        "conc": 1,
        "kv-offloading": "none",
        "total-cpu-dram-gb": 0,
        "duration": 3600,
        "exp-name": "pilot",
        "scenario-type": "agentic-coding",
        "execution": reference,
    }
    scheduling = {"priority": "-2.1", "queue-token": "explicit-queue", "node-count": 1}
    site = PilotSite(
        schema_version=1,
        cluster="h100-dgxc",
        native_python=str(tmp_path / "runtime/python"),
        native_source=str(tmp_path / "runtime/src"),
        wrapper_python=str(tmp_path / "wrapper/python"),
        shared_root=str(tmp_path),
        model_snapshot=str(tmp_path / ("d" * 40)),
        model_revision="d" * 40,
        image={"path": str(tmp_path / "model.sqsh"), "sha256": "e" * 64},
        image_reference=row["image"],
        client_sites={"agentx": "/agentx.json", "eval": "/eval.json"},
        mounts={str(tmp_path): str(tmp_path)},
        reader_revision="a" * 40,
        collector_revision="b" * 40,
    )
    return root, row, scheduling, site


def test_mount_aliases_and_workspace_outputs_are_rejected(inputs, tmp_path):
    _, _, _, site = inputs
    alias = tmp_path / "alias"
    target = tmp_path / "actual"
    target.mkdir()
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="canonical paths"):
        PilotSite.model_validate(
            {**site.model_dump(), "mounts": {str(alias): str(alias)}}
        )
    with pytest.raises(ValueError, match="under /workspace"):
        PilotSite.model_validate(
            {**site.model_dump(), "shared_root": "/workspace/pilot"}
        )


def test_interpreter_requires_mounted_base_and_expected_python(inputs, tmp_path):
    _, _, _, site = inputs
    shared = Path(site.shared_root)
    identity = {
        "python_version": "3.12.9",
        "python_paths": {
            "executable": str(shared / "venv/bin/python"),
            "executable_resolved": "/unmounted/python/bin/python3.12",
            "prefix": str(shared / "venv"),
            "base_prefix": "/unmounted/python",
        },
    }
    with pytest.raises(ValueError, match="not mounted"):
        site.require_interpreter(identity, python_minor="3.12")
    identity["python_paths"].update(
        executable_resolved=str(shared / "python/bin/python3.12"),
        base_prefix=str(shared / "python"),
    )
    with pytest.raises(ValueError, match="must use Python 3.11"):
        site.require_interpreter(identity, python_minor="3.11")
    link = shared / "external-cache"
    link.symlink_to("/unmounted/cache", target_is_directory=True)
    with pytest.raises(ValueError, match="not mounted"):
        site.require_visible(str(link / "data"))


def test_policy_symlink_cannot_read_outside_checkout(inputs, tmp_path):
    root, row, _, _ = inputs
    outside = tmp_path / "not-a-policy"
    outside.write_text("[malformed yaml")
    policy = root / row["execution"]["client-policy"]
    policy.unlink()
    policy.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes checkout"):
        resolve_reference(row["execution"], root)


def test_golden_resource_and_scheduling_have_distinct_identities(inputs):
    root, row, scheduling, _ = inputs
    first = parse_job(row, root, scheduling)
    second = parse_job(
        row, root, {**scheduling, "priority": "999", "queue-token": "other"}
    )
    assert first.point_id == second.point_id
    assert intent_id("owner/repo", "10", "1", first.point_id) != intent_id(
        "owner/repo", "10", "2", first.point_id
    )
    curve = root / "golden_al_distribution/dsv41flash_dspark.yaml"
    curve.write_text(curve.read_text().replace("5: 3.51", "5: 3.52"))
    assert parse_job(row, root, scheduling).point_id != first.point_id
    with pytest.raises(ValueError, match="changed after matrix"):
        resolve_reference(first.row.execution.model_dump(by_alias=True), root)


@pytest.mark.parametrize(
    "concurrency,sequences,capture",
    [
        (1, 2, 16),
        (2, 4, 32),
        (4, 8, 64),
        (8, 16, 128),
        (16, 32, 256),
        (20, 40, 256),
        (24, 48, 512),
        (28, 56, 512),
    ],
)
def test_h100_recipe_preserves_real_serving_parameters(
    inputs, concurrency, sequences, capture
):
    root, row, scheduling, site = inputs
    job = parse_job({**row, "conc": concurrency}, root, scheduling)
    policy = ClientPolicy.model_validate(
        load_mapping(root / row["execution"]["client-policy"])
    )
    recipe, profile = render_recipe(
        job, root, site, policy, root / "spec.json", root / "outputs"
    )
    args = recipe["roles"]["agg"]["args"]
    assert args["max-num-seqs"] == sequences
    assert args["max-cudagraph-capture-size"] == capture
    assert args["max-model-len"] == 1048576
    assert args["max-num-batched-tokens"] == 4096
    assert args["speculative-config"]["synthetic_acceptance_length"] == 3.51
    assert args["speculative-config"]["enable_adaptive_verification"] is False
    assert profile["default_time_limit"] == "08:00:00"
    assert profile["use_exclusive_sbatch_directive"] is True
    assert recipe["benchmark"]["argv"][-4:] == [
        "--spec",
        str(root / "spec.json"),
        "--artifact-root",
        str(root / "outputs"),
    ]


def test_real_eval_has_no_synthetic_acceptance(inputs):
    root, row, scheduling, site = inputs
    job = parse_job(
        {
            **row,
            "conc": 28,
            "run-eval": True,
            "eval-only": True,
            "eval-framework": "lm-eval",
        },
        root,
        scheduling,
    )
    policy = ClientPolicy.model_validate(
        load_mapping(root / row["execution"]["client-policy"])
    )
    recipe, _ = render_recipe(
        job, root, site, policy, root / "spec.json", root / "outputs"
    )
    spec = recipe["roles"]["agg"]["args"]["speculative-config"]
    assert spec == {
        "method": "dspark",
        "num_speculative_tokens": 5,
        "draft_sample_method": "probabilistic",
        "rejection_sample_method": "block",
        "enable_adaptive_verification": True,
    }


@pytest.mark.parametrize(
    "changes",
    [{"tp": 4}, {"conc": 32}, {"run-eval": True}, {"rogue": 1}, {"duration": 1200}],
)
def test_unqualified_inputs_are_rejected_before_runtime(inputs, changes):
    root, row, scheduling, _ = inputs
    with pytest.raises(ValueError):
        parse_job({**row, **changes}, root, scheduling)


def test_missing_queue_demand_and_duplicate_yaml_fail(inputs):
    root, row, scheduling, _ = inputs
    del scheduling["node-count"]
    with pytest.raises(ValueError):
        parse_job(row, root, scheduling)
    path = root / "duplicate.yaml"
    path.write_text("nodes: 1\nnodes: 2\n")
    with pytest.raises(ValueError, match="Duplicate YAML key"):
        load_mapping(path)


def test_bundle_mutation_fails_before_scheduler(inputs, tmp_path, monkeypatch):
    from infx.srt_slurm import launch

    file = tmp_path / "prepared.json"
    file.write_text('{"approved":true}')
    from infx.srt_slurm.job import file_digest

    bundle = {"files": {str(file): file_digest(file)}}
    bundle["bundle_digest"] = digest(bundle)
    verify_bundle(bundle)
    file.write_text('{"approved":false}')
    calls = []
    monkeypatch.setattr(launch, "native", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="prepared input changed"):
        execute(bundle, tmp_path)
    assert calls == []


def test_client_descendants_cannot_be_published_as_success(inputs, tmp_path):
    root, row, scheduling, _ = inputs
    directory = tmp_path / "run"
    audit = directory / "client-output/diagnostics/client-audit.json"
    audit.parent.mkdir(parents=True)
    audit.write_text(
        json.dumps(
            {
                "errors": [],
                "status": {
                    "returncode": 0,
                    "cancelled_by_signal": None,
                    "timed_out": False,
                    "orphaned_descendants": True,
                },
            }
        )
    )
    job = parse_job(row, root, scheduling)
    bundle = {
        "directory": str(directory),
        "point_id": job.point_id,
        "job": job.model_dump(by_alias=True),
    }
    with pytest.raises(ValueError, match="closed result"):
        publish_outputs(bundle, {}, tmp_path / "publish")
    assert not (tmp_path / "publish").exists()


def test_native_unknown_acceptance_is_not_retried(inputs, tmp_path, monkeypatch):
    from infx.srt_slurm import launch

    _, _, _, site = inputs
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("{}")
    bundle = {
        "files": {},
        "site": site.model_dump(),
        "execution_id": "unique",
        "prepared": {"prepared_dir": str(tmp_path)},
    }
    bundle["bundle_digest"] = digest(bundle)
    calls = []

    def external(_site, command, *args, **kwargs):
        calls.append(command)
        if command == "intent-path":
            return {"state": "intent", "receipt_path": str(receipt_path)}
        if command == "submit-prepared":
            raise launch.NativeCommandError(
                {"state": "unknown", "receipt_path": str(receipt_path)}, "unknown"
            )
        return {"state": "unknown"}

    monkeypatch.setattr(launch, "native", external)
    monkeypatch.setattr(launch, "verify_execution_clients", lambda *_args: None)
    monkeypatch.setattr(launch, "copy_diagnostics", lambda *_args, **_kwargs: None)
    with pytest.raises(RuntimeError, match="intent stays fenced"):
        execute(bundle, tmp_path, reconcile_timeout=0)
    assert calls == ["intent-path", "submit-prepared", "reconcile"]


def test_installed_native_runtime_prepares_and_renders_the_entire_pilot(
    inputs, tmp_path
):
    """Exercise the real cross-repository boundary; never allocate or run clients.

    Opt in with INFX_NATIVE_PHASE1_PYTHON and INFX_NATIVE_PHASE1_SOURCE pointing
    to an installed, source-matching native runtime and its clean source tree.
    """
    import yaml

    from infx.srt_slurm.launch import native

    python = os.environ.get("INFX_NATIVE_PHASE1_PYTHON")
    source = os.environ.get("INFX_NATIVE_PHASE1_SOURCE")
    if not python or not source:
        pytest.skip("requires explicitly provisioned native runtime Python and source")
    root, row, scheduling, original_site = inputs
    root = root.resolve()
    shared = tmp_path.resolve()
    model = shared / ("d" * 40)
    model.mkdir()
    image = shared / "model.sqsh"
    image.write_text("Inert test placeholder: no container is started.\n")
    site = PilotSite.model_validate(
        {
            **original_site.model_dump(),
            "native_python": python,
            "native_source": source,
            "wrapper_python": str(shared / "wrapper/python"),
            "shared_root": str(shared),
            "model_snapshot": str(model),
            "image": {"path": str(image), "sha256": "e" * 64},
            "mounts": {str(shared): str(shared)},
        }
    )
    policy = ClientPolicy.model_validate(
        load_mapping(root / row["execution"]["client-policy"])
    )
    probe = ROOT / "utils/fixtures/native_pilot_probe.py"
    inspections = []
    cases = [
        (False, 1, 2, 16),
        (False, 2, 4, 32),
        (False, 4, 8, 64),
        (False, 8, 16, 128),
        (False, 16, 32, 256),
        (False, 20, 40, 256),
        (False, 24, 48, 512),
        (False, 28, 56, 512),
        (True, 28, 56, 512),
    ]
    for evaluation, concurrency, sequences, capture in cases:
        kind = "eval" if evaluation else "agentx"
        current_row = {**row, "conc": concurrency}
        if evaluation:
            current_row.update(
                {"run-eval": True, "eval-only": True, "eval-framework": "lm-eval"}
            )
        job = parse_job(current_row, root, scheduling)
        point = shared / f"{kind}-c{concurrency}"
        point.mkdir()
        spec_path = point / "client.json"
        output = point / "client-output"
        recipe, profile = render_recipe(job, root, site, policy, spec_path, output)
        # This is the guarded argv installed by launch.prepare; no guard or
        # benchmark executes in this test.
        recipe["benchmark"]["argv"][3:4] = [
            "infx.srt_slurm.client_guard",
            "--bundle",
            str(point / "bundle.json"),
            "--client",
            kind,
        ]
        for name, data in (("recipe.yaml", recipe), ("profile.yaml", profile)):
            (point / name).write_text(yaml.safe_dump(data, sort_keys=False))
        prepared = native(
            site,
            "prepare",
            "--recipe",
            str(point / "recipe.yaml"),
            "--profile",
            str(point / "profile.yaml"),
            "--output",
            str(point / "native"),
            "--expected-nodes",
            "1",
            "--runtime-python",
            python,
        )
        assert prepared["state"] == "prepared"
        assert prepared["resources"] == {
            "nodes": 1,
            "gpus_per_node": 8,
            "serving_gpus": 8,
            "workers": 1,
            "cardinality": 1,
        }
        result = subprocess.run(
            [python, "-I", str(probe), prepared["prepared_dir"]],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        inspected = json.loads(result.stdout)
        inspections.append({"kind": kind, "concurrency": concurrency, **inspected})
        argv = inspected["server_argv"]

        def value(flag, command=argv):
            assert command.count(flag) == 1
            return command[command.index(flag) + 1]

        assert argv[:3] == ["vllm", "serve", str(model)]
        assert value("--tensor-parallel-size") == "8"
        assert value("--device-ids") == "0,1,2,3,4,5,6,7"
        assert value("--max-num-seqs") == str(sequences)
        assert value("--max-cudagraph-capture-size") == str(capture)
        assert value("--max-model-len") == "1048576"
        assert value("--max-num-batched-tokens") == "4096"
        assert value("--gpu-memory-utilization") == "0.92"
        assert value("--served-model-name") == row["model"]
        assert value("--tokenizer-mode") == "deepseek_v41"
        assert value("--tool-call-parser") == "deepseek_v41"
        assert value("--reasoning-parser") == "deepseek_v41"
        assert "--language-model-only" in argv
        assert "--enable-auto-tool-choice" in argv
        assert "--disable-uvicorn-access-log" in argv
        assert json.loads(value("--engram-config")) == {"cpu_offload": True}
        speculative = json.loads(value("--speculative-config"))
        assert speculative == {
            "method": "dspark",
            "num_speculative_tokens": 5,
            "draft_sample_method": "probabilistic",
            "rejection_sample_method": "block" if evaluation else "synthetic",
            "enable_adaptive_verification": evaluation,
            **({} if evaluation else {"synthetic_acceptance_length": 3.51}),
        }
        assert inspected["processes"] == [
            {"node": "simulated-h100", "mode": "agg", "gpus": list(range(8))}
        ]
        assert inspected["frontend_type"] == "vllm"
        assert inspected["discovery_services"] == []
        assert inspected["profiling"] is False
        assert inspected["worker_env"] == recipe["roles"]["agg"]["env"]
        assert inspected["client_argv"] == recipe["benchmark"]["argv"]
        assert inspected["client_env_unset"] == [
            "PYTHONPATH",
            "PYTHONHOME",
            "BASH_ENV",
            "ENV",
        ]
        assert inspected["client_cwd"] == str(point)
        environment = inspected["client_env"]
        assert environment["SRT_ENDPOINT"] == f"http://127.0.0.1:{value('--port')}"
        assert (
            environment["AIPERF_SERVER_METRICS_URLS"]
            == environment["SRT_ENDPOINT"] + "/metrics"
        )
        assert environment["SRT_MODEL_NAME"] == row["model"]
        assert environment["SRT_LOG_DIR"] == "/logs"
        assert (
            environment["HF_HUB_OFFLINE"] == environment["HF_DATASETS_OFFLINE"] == "1"
        )
        assert inspected["mounts"][str(shared)] == str(shared)
        assert inspected["exit_code"] == 0
        assert len(inspected["srun_argv"]) == 1
        srun = inspected["srun_argv"][0]
        assert srun[0] == "srun"
        assert srun[srun.index("--container-image") + 1] == str(image)
        assert f"{shared}:{shared}" in srun[srun.index("--container-mounts") + 1].split(
            ","
        )
        assert srun[-3:-1] == ["bash", "-c"]
        assert (
            shlex.split(srun[-1].rsplit(" && exec ", 1)[1]) == inspected["client_argv"]
        )
        assert inspected["time_limit"] == "08:00:00"
        directives = {}
        for line in (point / "native/job.slurm").read_text().splitlines():
            if line.startswith("#SBATCH --"):
                key, _, val = line.removeprefix("#SBATCH --").partition("=")
                assert key not in directives
                directives[key] = val
        assert {
            name: directives[name]
            for name in ("nodes", "ntasks", "gpus-per-node", "time", "exclusive")
        } == {
            "nodes": "1",
            "ntasks": "1",
            "gpus-per-node": "8",
            "time": "08:00:00",
            "exclusive": "",
        }
    (shared / "cross-repo-inspection.json").write_text(
        json.dumps(inspections, indent=2)
    )
    frozen = point / "native/config.yaml"
    frozen.chmod(0o644)
    frozen.write_text(frozen.read_text() + "# changed after preparation\n")
    tampered = subprocess.run(
        [python, "-I", str(probe), str(point / "native")],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert tampered.returncode != 0
    assert "Prepared input changed: config.yaml" in tampered.stderr
