"""Real diagnostic writes/signals with an external native scheduler collaborator."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from infx.srt_slurm.qualify_cancellation import (
    DraftSite,
    qualify,
    render_probe,
    verify_closed_writer,
)

# This executable replaces only the external native/Slurm boundary. Real
# preparation, path validation, trigger selection, cleanup orchestration and
# diagnostic client code run from the production module.
NATIVE_COLLABORATOR = r"""
import json, os, signal, subprocess, sys, time
from pathlib import Path
import yaml
control_path = Path(__file__).with_name("control.json")
control = json.loads(control_path.read_text())
state_path = Path(__file__).with_name("state.json")
state = json.loads(state_path.read_text()) if state_path.exists() else {}
args = sys.argv[1:]
def value(name): return args[args.index(name) + 1]
def persist(): state_path.write_text(json.dumps(state))
def output(payload, code=0):
    persist()
    print(json.dumps(payload))
    raise SystemExit(code)
def receipt_path(): return Path(state["receipt"])
def load_receipt(): return json.loads(receipt_path().read_text())
if args[:2] == ["-I", "-c"]:
    if "--output" in args:
        os.execv(sys.executable, [sys.executable, *args])
    state.setdefault("calls", []).append("worker-observation")
    receipt = load_receipt()
    logs = Path(receipt["output_dir"]) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "direct-vllm-worker.json").write_text(json.dumps({
        "pid": 100, "start_ticks": 200, "pid_namespace": [1, 2], "net_namespace": [1, 3]}))
    steps = {"agg_0_node": "71.0"}
    if control["scenario"] == "foreign-step": steps = {"agg_0_node": "999.0"}
    if control["scenario"] in ("client", "unclosed-writer"):
        recipe = state["recipe"]
        if not state.get("writer_pid"):
            environment = {**os.environ, "SRT_JOB_ID": "71", "SRT_ENDPOINT": "http://node:8000"}
            child = subprocess.Popen(recipe["benchmark"]["argv"], cwd=recipe["benchmark"]["cwd"],
                       env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       start_new_session=True)
            state["writer_pid"] = child.pid
            persist()
        started = Path(recipe["benchmark"]["cwd"]) / "writer/started.json"
        deadline = time.monotonic() + 5
        while not started.exists() and time.monotonic() < deadline: time.sleep(0.01)
        steps["benchmark-client"] = "71.1"
    output({"observation": {"state": "active", "terminal": False,
                             "job_id": "71", "slurm_state": "RUNNING"}, "steps": steps})
command = args[3]
state.setdefault("calls", []).append(command)
if command == "capabilities":
    output({"state": "supported", "capabilities": control["capabilities"]})
if command == "prepare":
    state["recipe"] = yaml.safe_load(Path(value("--recipe")).read_text())
    profile = yaml.safe_load(Path(value("--profile")).read_text())
    state["profile"] = profile
    state["prepared_dir"] = value("--output")
    state["output_root"] = profile["output_dir"]
    Path(state["prepared_dir"]).mkdir()
    resources = {"nodes": 1, "gpus_per_node": 8, "serving_gpus": 8, "workers": 1, "cardinality": 1}
    if control["scenario"] == "wrong-resources": resources["nodes"] = 2
    output({"state": "prepared", "prepared_dir": state["prepared_dir"], "resources": resources,
            "output_root": state["output_root"], "manifest_sha256": "a" * 64})
if command == "intent-path":
    path = Path(value("--journal-dir")) / "owned-intent/receipt.json"
    if control["scenario"] == "escaping-intent": path = control_path.parent / "foreign-receipt.json"
    state["receipt"] = str(path)
    output({"state": "intent", "receipt_path": str(path)})
if command == "submit-prepared":
    path = receipt_path()
    path.parent.mkdir(parents=True)
    receipt = {"state": "accepted", "job_id": "71", "accepted_ids": ["71"],
               "receipt_path": str(path), "output_dir": str(Path(state["output_root"]) / "71")}
    if control["scenario"] in ("ambiguous", "unresolved"):
        receipt.update(state="unknown", accepted_ids=[])
    path.write_text(json.dumps(receipt))
    if control["scenario"] == "interrupt-submit":
        persist()
        control_path.with_name("submit-pending").touch()
        time.sleep(30)
    output(receipt, 0 if receipt["state"] == "accepted" else 2)
if command == "wait":
    output({"state": "unknown", "terminal": False, "slurm_state": "RUNNING"}, 2)
if command == "reconcile":
    receipt = load_receipt()
    if control["scenario"] == "ambiguous":
        receipt["accepted_ids"] = ["71", "72"]
        receipt_path().write_text(json.dumps(receipt))
    output(receipt, 0 if receipt["state"] == "accepted" else 2)
if command == "cancel-known":
    receipt = load_receipt()
    state["cancelled_ids"] = receipt["accepted_ids"]
    if state.get("writer_pid"):
        os.kill(state["writer_pid"], signal.SIGTERM)
        closed = Path(state["recipe"]["benchmark"]["cwd"]) / "writer/closed.json"
        deadline = time.monotonic() + 5
        while not closed.exists() and time.monotonic() < deadline: time.sleep(0.01)
        if control["scenario"] == "unclosed-writer" and closed.exists(): closed.unlink()
    output({"state": "cancellation_requested", "jobs": [
            {"state": "cancellation_requested", "job_id": job,
             "already_terminal": control["scenario"] == "already-terminal"} for job in receipt["accepted_ids"]]})
if command == "wait-known":
    closed = control["scenario"] != "cleanup-unresolved"
    output({"state": "closed" if closed else "unknown", "terminal": closed,
            "jobs": [{"job_id": job, "terminal": closed, "slurm_state": "CANCELLED" if closed else "COMPLETING"}
                     for job in load_receipt()["accepted_ids"]]}, 0 if closed else 2)
raise SystemExit("unexpected external native command: " + command)
"""


def store(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def pilot(tmp_path):
    # Resolve macOS /var aliases because production requires canonical mounts.
    root = tmp_path.resolve()
    checkout = root / "checkout"
    checkout.mkdir()
    source = root / "native-source"
    source.mkdir()
    (source / "uv.lock").write_text("controlled-runtime-lock\n")
    for arguments in (
        ["init", "--quiet"],
        ["add", "uv.lock"],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
    ):
        subprocess.run(
            ["git", "-C", str(source), *arguments], check=True, capture_output=True
        )
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    native_python = root / "native-python"
    native_python.write_text(f"#!{sys.executable}\n" + NATIVE_COLLABORATOR)
    native_python.chmod(0o755)
    image = root / "image.sqsh"
    image.write_bytes(b"controlled squash contents")
    shared = root / "shared"
    shared.mkdir()
    snapshot = root / "hub/models--controlled--model/snapshots" / ("b" * 40)
    snapshot.mkdir(parents=True)
    draft = {
        "schema_version": 1,
        "cluster": "h100-dgxc",
        "native_python": str(native_python),
        "native_source": str(source),
        "wrapper_python": str(native_python),
        "shared_root": str(shared),
        "model_snapshot": str(snapshot),
        "model_revision": "b" * 40,
        "image": {
            "path": str(image),
            "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        },
        "image_reference": "controlled/image:retained",
        "client_sites": {
            "agentx": str(root / "agentx.json"),
            "eval": str(root / "eval.json"),
        },
        "mounts": {str(root): str(root)},
    }
    capabilities = ["prepared-v1", "prepared-direct-listener-ownership-v1"]
    store(
        checkout
        / "benchmarks/multi_node/srt-slurm-recipes/configs/prepared-runtime-lock.json",
        {
            "schema_version": 1,
            "repository": "https://example.test/native.git",
            "revision": revision,
            "uv_lock_sha256": hashlib.sha256(
                (source / "uv.lock").read_bytes()
            ).hexdigest(),
            "capabilities": capabilities,
        },
    )
    recipe = {
        "schema": 2,
        "name": "controlled",
        "engine": "vllm",
        "slurm": {"time_limit": "08:00:00"},
        "model": {
            "path": "controlled/model",
            "container": "controlled/image:retained",
            "precision": "fp4",
        },
        "frontend": {"type": "vllm"},
        "roles": {
            "agg": {
                "nodes": 1,
                "workers": 1,
                "gpus": 8,
                "env": {"UNCHANGED": "retained"},
                "args": {
                    "tensor-parallel-size": 8,
                    "max-model-len": 8192,
                    "max-num-batched-tokens": 256,
                },
            }
        },
    }
    recipe_path = (
        checkout
        / "benchmarks/multi_node/srt-slurm-recipes/dsv41flash/vllm/h100-fp4/agentx/agg-tp8-dspark5.yaml"
    )
    recipe_path.parent.mkdir(parents=True)
    recipe_path.write_text(yaml.safe_dump(recipe))
    profile_path = checkout / "runners/srt-slurm/h100-phase1.yaml"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(
        yaml.safe_dump(
            {
                "use_exclusive_sbatch_directive": True,
                "default_account": "controlled-account",
                "default_partition": "controlled-partition",
                "gpus_per_node": 8,
            }
        )
    )
    store(root / "site-draft.json", draft)
    store(root / "control.json", {"scenario": "startup", "capabilities": capabilities})
    yield root, checkout, draft
    state = root / "state.json"
    if state.exists() and (pid := json.loads(state.read_text()).get("writer_pid")):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_probe(pilot, *, scenario="startup", mode="startup", namespace="probe"):
    root, checkout, _ = pilot
    control = json.loads((root / "control.json").read_text())
    store(root / "control.json", {**control, "scenario": scenario})
    return qualify(
        checkout,
        root / "site-draft.json",
        root / "artifacts",
        namespace,
        mode=mode,
        walltime_seconds=300 if mode == "startup" else 1800,
        observation_timeout_seconds=1 if scenario == "foreign-step" else 10,
        cleanup_timeout_seconds=1 if scenario == "unresolved" else 10,
    )


def test_startup_cancels_the_entered_worker_and_preserves_diagnostic_evidence(pilot):
    root, _, _ = pilot
    report = run_probe(pilot)
    state = json.loads((root / "state.json").read_text())
    assert report["state"] == "passed"
    assert report["lifecycle_qualified"] is True
    assert report["qualification_complete"] is False
    assert report["accepted_benchmark"] is False
    assert report["trigger"]["aggregate_steps"] == {"agg_0_node": "71.0"}
    assert state["cancelled_ids"] == ["71"]
    assert state["calls"].count("submit-prepared") == 1
    assert state["recipe"]["roles"]["agg"]["args"] == {
        "tensor-parallel-size": 8,
        "max-model-len": 8192,
        "max-num-batched-tokens": 256,
    }
    assert state["profile"]["default_account"] == "controlled-account"
    assert state["profile"]["default_partition"] == "controlled-partition"
    assert state["profile"]["default_time_limit"] == "00:05:00"
    assert state["recipe"]["slurm"]["time_limit"] == "00:05:00"
    retained = root / "artifacts/evidence/native-output/71/logs/direct-vllm-worker.json"
    assert json.loads(retained.read_text())["start_ticks"] == 200
    with pytest.raises(FileExistsError):
        qualify(
            pilot[1],
            root / "site-draft.json",
            root / "another-output",
            "probe",
            mode="startup",
            walltime_seconds=300,
            observation_timeout_seconds=10,
            cleanup_timeout_seconds=10,
        )


def test_client_cancellation_requires_a_real_signal_closed_writer(pilot):
    root, _, _ = pilot
    report = run_probe(pilot, scenario="client", mode="client")
    assert report["state"] == "passed"
    assert report["writer_closure"]["signal"] == signal.SIGTERM
    assert report["writer_closure"]["writer_closed"] is True
    assert report["writer_closure"]["records"] >= 1
    assert report["trigger"]["writer"]["pid"] == report["writer_closure"]["pid"]
    assert report["writer_closure"]["endpoint"] == "http://node:8000"
    assert report["cleanup"]["closure"]["terminal"] is True
    heartbeat = root / "artifacts/evidence/writer/heartbeat.jsonl"
    assert (
        len(heartbeat.read_text().splitlines()) == report["writer_closure"]["records"]
    )


@pytest.mark.parametrize(
    "scenario, expected_cancelled, message",
    [
        ("ambiguous", ["71", "72"], "uniquely accepted"),
        ("unresolved", None, "uniquely accepted"),
        ("cleanup-unresolved", ["71"], "physical terminal closure"),
        ("already-terminal", ["71"], "new cancellation request"),
        ("foreign-step", ["71"], "cancellation trigger"),
        ("unclosed-writer", ["71"], "closed.json"),
    ],
)
def test_failures_remain_unqualified_and_cancel_only_known_native_ownership(
    pilot, scenario, expected_cancelled, message
):
    root, _, _ = pilot
    with pytest.raises(RuntimeError, match="inspect"):
        run_probe(
            pilot,
            scenario=scenario,
            mode="client" if scenario == "unclosed-writer" else "startup",
        )
    report = json.loads((root / "artifacts/qualification.json").read_text())
    state = json.loads((root / "state.json").read_text())
    assert report["state"] == "failed"
    assert report["lifecycle_qualified"] is False
    assert report["accepted_benchmark"] is False
    assert state.get("cancelled_ids") == expected_cancelled
    assert state["calls"].count("submit-prepared") == 1
    assert message in report.get("error", "") + report.get("cleanup_error", "")
    if scenario == "ambiguous":
        assert report["cleanup"]["closure"]["terminal"] is True
        assert "worker-observation" not in state["calls"]
    if scenario == "unresolved":
        assert "fenced" in report["cleanup_error"]


@pytest.mark.parametrize("scenario", ["wrong-resources", "escaping-intent"])
def test_native_preflight_rejects_wrong_demand_or_unowned_journal_before_submit(
    pilot, scenario
):
    root, _, _ = pilot
    with pytest.raises(RuntimeError, match="inspect"):
        run_probe(pilot, scenario=scenario)
    state = json.loads((root / "state.json").read_text())
    assert "submit-prepared" not in state["calls"]
    assert "cancel-known" not in state["calls"]
    assert (
        json.loads((root / "artifacts/qualification.json").read_text())["state"]
        == "failed"
    )


def test_unqualified_pin_and_deployment_pins_fail_before_native_submission(pilot):
    root, _, draft = pilot
    with pytest.raises(ValueError, match="Extra inputs"):
        DraftSite.model_validate({**draft, "reader_revision": "a" * 40})
    source = Path(draft["native_source"])
    (source / "uv.lock").write_text("changed dependency resolution\n")
    with pytest.raises(ValueError, match="differs from selected pin"):
        run_probe(pilot)
    assert not (root / "state.json").exists()


def test_owned_output_cannot_alias_the_shared_generation(pilot):
    root, checkout, draft = pilot
    owned = Path(draft["shared_root"]) / "cancellation-qualification/probe"
    with pytest.raises(ValueError, match="disjoint"):
        qualify(
            checkout,
            root / "site-draft.json",
            owned / "artifacts",
            "probe",
            mode="startup",
            walltime_seconds=300,
            observation_timeout_seconds=10,
            cleanup_timeout_seconds=10,
        )
    assert not owned.exists()
    assert not (root / "state.json").exists()


def test_literal_writer_accepts_endpoint_and_rejects_incomplete_closed_evidence(pilot):
    _, checkout, draft = pilot
    directory = Path(draft["shared_root"]) / "standalone-writer"
    directory.mkdir()
    recipe, _ = render_probe(
        checkout, DraftSite.model_validate(draft), directory, "writer-test", 1800
    )
    argv = [*recipe["benchmark"]["argv"], "--endpoint", "http://explicit:8000"]
    with subprocess.Popen(
        argv, cwd=directory, env={**os.environ, "SRT_JOB_ID": "81"}
    ) as child:
        try:
            deadline = time.monotonic() + 5
            while not (directory / "writer/started.json").exists():
                if child.poll() is not None or time.monotonic() >= deadline:
                    pytest.fail("diagnostic writer did not produce its start record")
                time.sleep(0.01)
            child.send_signal(signal.SIGTERM)
            assert child.wait(timeout=5) == 128 + signal.SIGTERM
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
    receipt = {"job_id": "81"}
    closed = verify_closed_writer(directory, receipt, "writer-test")
    assert closed["endpoint"] == "http://explicit:8000"
    assert closed["pid"] == child.pid
    with (directory / "writer/heartbeat.jsonl").open("a") as stream:
        stream.write('{"sequence":99,"token":"writer-test"}\n')
    with pytest.raises(ValueError, match="incomplete or foreign"):
        verify_closed_writer(directory, receipt, "writer-test")


def test_signal_before_submission_stdout_recovers_receipt_and_closes_allocation(pilot):
    root, checkout, _ = pilot
    control = json.loads((root / "control.json").read_text())
    store(root / "control.json", {**control, "scenario": "interrupt-submit"})
    argv = [
        sys.executable,
        "-m",
        "infx.srt_slurm.qualify_cancellation",
        "--root",
        str(checkout),
        "--site-draft",
        str(root / "site-draft.json"),
        "--output",
        str(root / "artifacts"),
        "--namespace",
        "probe",
        "--mode",
        "startup",
        "--walltime-seconds",
        "300",
        "--observation-timeout-seconds",
        "10",
        "--cleanup-timeout-seconds",
        "10",
    ]
    with subprocess.Popen(
        argv,
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as process:
        try:
            deadline = time.monotonic() + 10
            while not (root / "submit-pending").exists():
                if process.poll() is not None or time.monotonic() >= deadline:
                    pytest.fail(
                        "external native submission did not reach the acceptance window"
                    )
                time.sleep(0.01)
            process.send_signal(signal.SIGTERM)
            _, stderr = process.communicate(timeout=15)
            assert process.returncode != 0
            assert "cancellation qualification failed" in stderr
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)
    report = json.loads((root / "artifacts/qualification.json").read_text())
    state = json.loads((root / "state.json").read_text())
    assert report["error_type"] == "QualificationInterruptedError"
    assert report["cleanup"]["closure"]["terminal"] is True
    assert report["lifecycle_qualified"] is False
    assert state["cancelled_ids"] == ["71"]
    assert state["calls"].count("submit-prepared") == 1
