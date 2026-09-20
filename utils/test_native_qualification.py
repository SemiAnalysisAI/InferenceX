"""Exercise diagnostic-only PR execution and revalidation with controlled workload evidence."""

import copy
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_native_pilot import inputs as _pilot_inputs
from test_python_benchmark_clients import (
    bound_file,
    raw_agentx,
)
from test_python_benchmark_clients import (
    eval_case as _eval_case,
)
from test_python_benchmark_clients import (
    spec_inputs as _spec_inputs,
)

from infx.benchmarks import agentx
from infx.benchmarks import eval as real_eval
from infx.benchmarks.common import read_json, write_json
from infx.benchmarks.spec import RuntimeSpec
from infx.srt_slurm import qualification as q
from infx.srt_slurm import workflow
from infx.srt_slurm.contracts import digest, load_mapping
from infx.srt_slurm.job import file_digest, intent_id
from infx.srt_slurm.launch import effective_identity
from infx.srt_slurm.render import ClientPolicy, PreparedSite, client_spec
from infx.workflows.sweep_runs import has_reusable_result_artifacts


@pytest.fixture
def pilot_inputs(tmp_path):
    return _pilot_inputs.__wrapped__(tmp_path)


@pytest.fixture
def spec_inputs(tmp_path):
    return _spec_inputs.__wrapped__(tmp_path)


@pytest.fixture
def eval_case(tmp_path, spec_inputs):
    return _eval_case.__wrapped__(tmp_path, spec_inputs)


@pytest.fixture
def pr_environment(tmp_path):
    event = tmp_path / "event.json"
    event.write_text(
        json.dumps(
            {
                "number": 3299,
                "pull_request": {
                    "head": {"repo": {"full_name": "SemiAnalysisAI/InferenceX"}},
                    "base": {"repo": {"full_name": "SemiAnalysisAI/InferenceX"}},
                },
            }
        )
    )
    return {
        "GITHUB_EVENT_NAME": "pull_request",
        "GITHUB_EVENT_PATH": str(event),
        "GITHUB_REPOSITORY": "SemiAnalysisAI/InferenceX",
        "GITHUB_REF": "refs/pull/3299/merge",
        "GITHUB_SHA": "a" * 40,
        "GITHUB_RUN_ID": "1234",
        "GITHUB_RUN_ATTEMPT": "2",
    }


@pytest.fixture
def plan(pilot_inputs):
    root, row, scheduling, _ = pilot_inputs
    write_json(
        root / "runtime.json",
        {
            "schema_version": 1,
            "repository": "https://example.invalid/native.git",
            "revision": "c" * 40,
            "uv_lock_sha256": "d" * 64,
            "capabilities": ["prepared-v1"],
        },
    )
    row = row | scheduling
    return {
        "single_node": {"agentic": [row | {"conc": c} for c in q.CONCURRENCIES]},
        "agentic_evals": [row | {"conc": 28}],
    }


def test_deployment_free_site_requires_actual_pr(pilot_inputs, pr_environment):
    *_, site = pilot_inputs
    environment = pr_environment | {
        "NATIVE_PURPOSE": q.PURPOSE,
        "NATIVE_PREPARED_SITE_JSON": json.dumps(
            site.model_dump(exclude={"reader_revision", "collector_revision"})
        ),
    }
    actual = workflow.load_site(environment)
    assert type(actual) is PreparedSite
    assert actual.native_python == site.native_python
    with pytest.raises(ValueError, match="repository variables"):
        workflow.load_site(environment | {"NATIVE_PURPOSE": "publication"})
    with pytest.raises(ValueError, match="deployment-free"):
        workflow.load_site(
            environment | {"NATIVE_PREPARED_SITE_JSON": site.model_dump_json()}
        )
    for name in ("push", "workflow_dispatch", "pull_request_target"):
        with pytest.raises(ValueError, match="pull_request event"):
            workflow.load_site(environment | {"GITHUB_EVENT_NAME": name})
    event = json.loads(Path(environment["GITHUB_EVENT_PATH"]).read_text())
    event["pull_request"]["head"]["repo"]["full_name"] = "fork/InferenceX"
    Path(environment["GITHUB_EVENT_PATH"]).write_text(json.dumps(event))
    with pytest.raises(ValueError, match="same-repository"):
        workflow.load_site(environment)


def test_matrix_requires_exact_grid_and_isolates_publication(
    pilot_inputs, plan, pr_environment
):
    root, *_ = pilot_inputs
    assert q.select_qualification(plan, root, pr_environment)
    assert not q.select_qualification(
        plan, root, pr_environment | {"GITHUB_EVENT_NAME": "push"}
    )
    for mutate in (
        lambda value: value["single_node"]["agentic"].pop(),
        lambda value: value["agentic_evals"].clear(),
        lambda value: value["single_node"]["agentic"].append(
            value["single_node"]["agentic"][0]
        ),
        lambda value: value["single_node"]["agentic"][0].pop("execution"),
    ):
        bad = copy.deepcopy(plan)
        mutate(bad)
        with pytest.raises(ValueError):
            q.select_qualification(bad, root, pr_environment)
    normal = {"results_bmk", "eval_results_all", "bmk_agentic_point"}
    assert has_reusable_result_artifacts(normal)
    assert not has_reusable_result_artifacts(normal | {"native-qualification-run"})


def test_workflow_exports_only_diagnostics(
    pilot_inputs, pr_environment, monkeypatch, tmp_path
):
    root, row, scheduling, site = pilot_inputs
    env_path = tmp_path / "github-env"
    environment = pr_environment | {
        "GITHUB_WORKSPACE": str(tmp_path),
        "NATIVE_CHECKOUT_ROOT": str(root),
        "GITHUB_ENV": str(env_path),
        "NATIVE_PURPOSE": q.PURPOSE,
        "NATIVE_PREPARED_SITE_JSON": json.dumps(
            site.model_dump(exclude={"reader_revision", "collector_revision"})
        ),
        "NATIVE_CONFIG_JSON": json.dumps(row),
        "NATIVE_AGENTX_FAST": "false",
        "NATIVE_EVAL_LIMIT": "",
        "NATIVE_REQUIRE_POWER": "false",
        "NATIVE_RUN_EVAL": "false",
        "NATIVE_EVAL_ONLY": "false",
        "NATIVE_PRIORITY": scheduling["priority"],
        "NATIVE_QUEUE_TOKEN": scheduling["queue-token"],
    }
    calls = []

    def prepare(job, prepared_site, checkout, source):
        assert type(prepared_site) is PreparedSite
        assert source["purpose"] == q.PURPOSE and checkout == root
        return {
            "point_id": "e" * 64,
            "execution_id": "f" * 64,
            "bundle_digest": "d" * 64,
            "prepared": {"manifest_sha256": "c" * 64},
        }

    monkeypatch.setattr(workflow.os, "environ", environment)
    monkeypatch.setattr(
        workflow.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="a" * 40)
    )
    monkeypatch.setattr(workflow, "prepare", prepare)
    monkeypatch.setattr(
        workflow, "execute", lambda bundle, output: calls.append(output)
    )
    assert workflow.main() == 0
    assert calls == [root / "native-qualification"]
    assert dict(line.split("=", 1) for line in env_path.read_text().splitlines()) == {
        "NATIVE_POINT_ID": "e" * 64,
        "GPU_COUNT": "8",
    }
    assert (
        read_json(calls[0] / "native-execution/prepared.json")["source"]["purpose"]
        == q.PURPOSE
    )

    def failed_prepare(*args):
        raise ValueError("prepared image digest differs")

    monkeypatch.setattr(workflow, "prepare", failed_prepare)
    with pytest.raises(ValueError, match="image digest"):
        workflow.main()
    failure = read_json(root / "native-qualification/preparation-failure.json")
    assert failure["error_type"] == "ValueError"
    assert failure["source"]["purpose"] == q.PURPOSE
    assert (
        failure["requested_point_id"]
        == env_path.read_text().splitlines()[-1].split("=", 1)[1]
    )
    assert "RESULT_FILENAME=" not in env_path.read_text()


@pytest.fixture
def evidence(pilot_inputs, plan, pr_environment, spec_inputs, eval_case, tmp_path):
    root, _, _, original_site = pilot_inputs
    site = PreparedSite.model_validate(
        original_site.model_dump(exclude={"reader_revision", "collector_revision"})
    )
    jobs = q.qualification_jobs(plan, root)
    artifacts = tmp_path / "downloaded"
    artifacts.mkdir()
    write_json(
        artifacts / "native-qualification-run/qualification-intent.json",
        {
            "purpose": q.PURPOSE,
            "publication_eligible": False,
            "run_id": "1234",
            "attempt": "2",
        },
    )
    source = {
        "repository": pr_environment["GITHUB_REPOSITORY"],
        "run_id": 1234,
        "attempt": 2,
        "head_sha": "a" * 40,
        "purpose": q.PURPOSE,
        "event": "pull_request",
        "pull_request": 3299,
    }
    eval_spec, _, sample_file, eval_result, _ = eval_case
    policy = ClientPolicy.model_validate(
        load_mapping(root / next(iter(jobs.values())).row.execution.client_policy)
    )

    def build(job, variant=""):
        original = tmp_path / "shared" / (job.mode + str(job.row.conc) + variant)
        original.mkdir(parents=True)
        runtime_data = copy.deepcopy(spec_inputs["runtime"])
        runtime_data["env"]["HF_DATASETS_CACHE"] += variant
        identity = original / "client/identity.json"
        identity.parent.mkdir()
        shutil.copyfile(spec_inputs["runtime"]["identity"]["path"], identity)
        runtime_data["identity"] = bound_file(identity)
        if job.mode == "eval":
            runtime_data["distributions"] = ["lm-eval"]
            task = original / "client/gsm8k.yaml"
            documents = original / "client/documents.json"
            shutil.copyfile(eval_spec.task.path, task)
            shutil.copyfile(eval_spec.document_identities.path, documents)
            resources = {
                "task": bound_file(task),
                "document_identities": bound_file(documents),
            }
        else:
            resources = {"dataset_revision": "f" * 40}
        runtime = RuntimeSpec.model_validate(runtime_data)
        write_json(original / "client/prepared-resources.json", resources)
        write_json(original / "native/manifest.json", {"nodes": 1})
        installed = {
            "runtime_lock": read_json(root / "runtime.json"),
            "wrapper_identity": {"distributions": {"infx": {"files": {}}}},
        }
        point, curve = effective_identity(job, site, installed, runtime, resources)
        spec = client_spec(job, policy, runtime, resources, point)
        write_json(original / "client.json", spec.model_dump(mode="json"))
        bundle = {
            "schema_version": 1,
            "directory": str(original),
            "source": source,
            "requested_point_id": job.point_id,
            "point_id": point,
            "effective_curve_id": curve,
            "execution_id": intent_id(source["repository"], "1234", "2", job.point_id),
            "job": job.model_dump(mode="json", by_alias=True),
            "site": site.model_dump(),
            "identity": installed,
            "prepared": {
                "manifest_sha256": file_digest(original / "native/manifest.json"),
                "capabilities": ["prepared-v1"],
                "resources": {
                    "nodes": 1,
                    "gpus_per_node": 8,
                    "serving_gpus": 8,
                    "workers": 1,
                    "cardinality": 1,
                },
            },
            "files": {
                str(path): file_digest(path)
                for path in original.rglob("*")
                if path.is_file()
            },
        }
        bundle["bundle_digest"] = digest(bundle)
        output = artifacts / (q.PREFIX + point)
        shutil.copytree(original, output / "native-execution/prepared")
        write_json(output / "native-execution/bundle.json", bundle)
        write_json(
            output / "native-execution/execution.json",
            {
                key: bundle[key]
                for key in ("point_id", "execution_id", "bundle_digest", "source")
            }
            | {
                "mode": job.mode,
                "client_exit_code": 0,
                "native_receipt": {
                    "state": "COMPLETED",
                    "job_id": "123",
                    "manifest_sha256": bundle["prepared"]["manifest_sha256"],
                },
            },
        )
        write_json(output / "native-execution/output-state.json", {"complete": True})
        endpoint = (
            "http://worker:9000" if job.mode == "eval" else "http://worker.example:9123"
        )
        write_json(
            output / "results/diagnostics/client-audit.json",
            {
                "errors": [],
                "endpoint": endpoint,
                "status": {
                    "returncode": 0,
                    "cancelled_by_signal": None,
                    "timed_out": False,
                    "orphaned_descendants": False,
                },
            },
        )
        if job.mode == "eval":
            result = copy.deepcopy(eval_result)
            result["config"]["model_args"]["model"] = job.row.model
            write_json(output / "results_case_conc28.json", result)
            shutil.copyfile(sample_file, output / "samples_case_conc28.jsonl")
            write_json(
                output / "meta_env.json", real_eval.eval_metadata(spec, complete=True)
            )
        else:
            raw = raw_agentx(output / "results")
            aggregate = read_json(raw / "profile_export_aiperf.json")
            aggregate["input_config"]["models"]["items"][0]["name"] = job.row.model
            aggregate["input_config"]["tokenizer"]["name"] = job.row.model
            aggregate["input_config"]["phases"][0]["concurrency"] = job.row.conc
            write_json(raw / "profile_export_aiperf.json", aggregate)
            normalized = agentx.normalize(spec, output / "results")
            shutil.copyfile(normalized, output / normalized.name)
        return output

    return (
        root,
        artifacts,
        jobs,
        source,
        build,
        Path(eval_spec.document_identities.path),
    )


def test_downloaded_point_revalidates_runtime_raw_settings_and_digests(evidence):
    root, _, jobs, source, build, _ = evidence
    job = next(job for job in jobs.values() if job.mode == "throughput")
    output = build(job)
    assert q.validate_point(output, jobs, source, root)["concurrency"] == 1
    raw = output / "results/aiperf_artifacts/profile_export_aiperf.json"
    data = read_json(raw)
    data["input_config"]["phases"][0]["duration"] = 1200
    write_json(raw, data)
    with pytest.raises(ValueError, match="profiling.duration"):
        q.validate_point(output, jobs, source, root)
    (output / "native-execution/prepared/client.json").write_text("{}")
    with pytest.raises(ValueError, match="file digest"):
        q.validate_point(output, jobs, source, root)


def test_full_controlled_grid_and_eval_score_are_rechecked(
    evidence, plan, pr_environment, monkeypatch, tmp_path
):
    root, artifacts, jobs, _, build, identities = evidence
    # Replace only the independent corpus oracle with a controlled 1,319-document split.
    oracle = tmp_path / "oracle/resources"
    oracle.mkdir(parents=True)
    shutil.copyfile(identities, oracle / "gsm8k-test-doc-hashes.json")
    monkeypatch.setattr(
        "infx.results.publication_receipt.files", lambda _: oracle.parent
    )
    outputs = [build(job) for job in jobs.values()]
    summary = q.summarize(artifacts, plan, root, pr_environment)
    assert summary["complete"] is True and summary["publication_eligible"] is False
    assert len(summary["points"]) == 9
    eval_output = next(path for path in outputs if list(path.glob("samples*.jsonl")))
    result = next(eval_output.glob("results*.json"))
    data = read_json(result)
    data["results"]["gsm8k"]["exact_match,strict-match"] = 0.1
    write_json(result, data)
    with pytest.raises(ValueError, match="threshold or disagrees"):
        q.summarize(artifacts, plan, root, pr_environment)
    shutil.rmtree(eval_output)
    with pytest.raises(ValueError, match="exactly nine"):
        q.summarize(artifacts, plan, root, pr_environment)


def test_isolated_checkout_ignores_stale_parent_git_state(tmp_path):
    parent = tmp_path / "runner"
    parent.mkdir()
    subprocess.run(["git", "init", str(parent)], check=True, capture_output=True)
    (parent / ".git/index.lock").write_text("stale")
    corrupt = parent / ".git/modules/utils/aiperf"
    corrupt.mkdir(parents=True)
    (corrupt / "HEAD").write_text("invalid")
    fresh = parent / "native-candidate-1234-2-token"
    subprocess.run(["git", "init", str(fresh)], check=True, capture_output=True)
    (fresh / "candidate.txt").write_text("candidate")
    subprocess.run(["git", "-C", str(fresh), "add", "candidate.txt"], check=True)
    assert (
        subprocess.check_output(
            ["git", "-C", str(fresh), "diff", "--cached", "--name-only"], text=True
        ).strip()
        == "candidate.txt"
    )
    assert (parent / ".git/index.lock").read_text() == "stale"


def test_foreign_source_and_duplicate_matrix_points_are_rejected(
    evidence, plan, pr_environment
):
    root, artifacts, jobs, source, build, _ = evidence
    throughput = [job for job in jobs.values() if job.mode == "throughput"]
    first = build(throughput[0])
    with pytest.raises(ValueError, match="source identity"):
        q.validate_point(first, jobs, source | {"attempt": 3}, root)
    for job in throughput[1:]:
        build(job)
    build(throughput[0], variant="duplicate")
    with pytest.raises(ValueError, match="duplicate or missing matrix points"):
        q.summarize(artifacts, plan, root, pr_environment)


def test_canonical_eval_corpus_cannot_be_replaced_by_self_consistent_documents(
    evidence,
):
    root, _, jobs, source, build, _ = evidence
    evaluation = next(job for job in jobs.values() if job.mode == "eval")
    output = build(evaluation)
    with pytest.raises(ValueError, match="document/target differs"):
        q.validate_point(output, jobs, source, root)


def test_repeated_sigterm_cannot_interrupt_owned_cleanup(pilot_inputs, tmp_path):
    import os
    import signal
    import sys
    import time

    _, _, _, site = pilot_inputs
    native = tmp_path / "native-control.py"
    native.write_text("""import json, pathlib, sys, time
root = pathlib.Path(sys.argv[1])
command = sys.argv[2]
receipt = root / 'receipt.json'
if command == 'intent-path':
    result = {'receipt_path': str(receipt)}
elif command == 'submit-prepared':
    result = {'receipt_path': str(receipt), 'state': 'accepted', 'accepted_ids': ['123'], 'job_id': '123'}
    receipt.write_text(json.dumps(result))
elif command == 'wait':
    (root / 'waiting').touch()
    while True: time.sleep(0.01)
elif command == 'reconcile':
    result = {'accepted_ids': ['123']}
elif command == 'cancel-known':
    (root / 'cancelling').touch()
    while not (root / 'release').exists(): time.sleep(0.01)
    result = {'accepted_ids': ['123']}
elif command == 'wait-known':
    (root / 'terminal').touch()
    result = {'terminal': True}
else:
    raise ValueError(command)
print(json.dumps(result), flush=True)
""")
    code = tmp_path / "execute.py"
    code.write_text(f"""import pathlib, sys
sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
from infx.srt_slurm import launch
root = pathlib.Path({str(tmp_path)!r})
site = {site.model_dump()!r}
bundle = {{'files': {{}}, 'site': site, 'source': {{}}, 'execution_id': 'owned', 'prepared': {{'prepared_dir': str(root)}}}}
bundle['bundle_digest'] = launch.digest(bundle)
launch.verify_execution_clients = lambda *a: None
launch.native = lambda site, command, *a, **kw: launch.checked_json([sys.executable, {str(native)!r}, str(root), command], timeout=10)
launch.copy_diagnostics = lambda *a, **kw: (root / 'diagnostics-kept').write_text(str(kw['complete']))
try:
    launch.execute(bundle, root)
except launch.WorkflowCancelledError:
    sys.exit(0)
raise RuntimeError('cancellation did not interrupt native wait')
""")
    process = subprocess.Popen(
        [sys.executable, str(code)], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    def wait_for(name):
        deadline = time.monotonic() + 8
        while not (tmp_path / name).exists():
            if process.poll() is not None or time.monotonic() >= deadline:
                raise AssertionError(f"execution did not reach {name}")
            time.sleep(0.01)

    try:
        wait_for("waiting")
        os.kill(process.pid, signal.SIGTERM)
        wait_for("cancelling")
        os.kill(process.pid, signal.SIGTERM)
        (tmp_path / "release").touch()
        _, stderr = process.communicate(timeout=8)
        assert process.returncode == 0, stderr.decode()
        assert (tmp_path / "terminal").exists()
        assert (tmp_path / "diagnostics-kept").read_text() == "False"
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=8)
