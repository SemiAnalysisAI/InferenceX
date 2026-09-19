"""Nominal tokenizer lookup must resolve to the prepared serving snapshot."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from test_benchmark_preparation import client_site, installed_child  # noqa: F401
from test_native_pilot import inputs as pilot_inputs  # noqa: F401

from infx.benchmarks.common import verify_model_snapshot_assets
from infx.benchmarks.prepare import bind_file
from infx.benchmarks.spec import RuntimeSpec
from infx.srt_slurm import launch
from infx.srt_slurm.job import parse_job


@pytest.fixture
def model_cache(tmp_path):
    cache = tmp_path / "hub/models--example--model"
    reference = cache / "refs/main"
    reference.parent.mkdir(parents=True)
    reference.write_text("a" * 40)
    snapshot = cache / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    blob = cache / "blobs/tokenizer"
    blob.parent.mkdir()
    blob.write_bytes(b'[{"id": 7, "piece": "prepared"}]')
    tokenizer = snapshot / "tokenizer.json"
    tokenizer.symlink_to("../../blobs/tokenizer")
    config = snapshot / "config.json"
    config.write_text('{"model_type":"fixture"}')
    identity = tmp_path / "identity.json"
    identity.write_text("{}")
    runtime = RuntimeSpec(
        python=str(tmp_path / "python"),
        identity=bind_file(identity),
        distributions=["fixture-client"],
        env={
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_CACHE": str(tmp_path / "hub"),
            "HF_DATASETS_CACHE": str(tmp_path / "datasets"),
        },
        env_unset=[],
        assets=[bind_file(path) for path in (reference, config, tokenizer)],
        timeout_seconds=10,
        terminate_grace_seconds=1,
    )
    return runtime, reference, snapshot, tokenizer


def test_bound_model_cache_resolves_to_serving_snapshot(model_cache):
    runtime, _, snapshot, _ = model_cache
    resolved = verify_model_snapshot_assets(
        runtime,
        "example/model",
        expected_revision="a" * 40,
        expected_snapshot=snapshot,
    )
    assert resolved == snapshot.resolve()
    assert (
        resolved / "tokenizer.json"
    ).read_bytes() == b'[{"id": 7, "piece": "prepared"}]'


def test_model_cache_requires_bound_tokenizer(model_cache):
    runtime, _, snapshot, tokenizer = model_cache
    runtime = runtime.model_copy(
        update={
            "assets": [
                asset for asset in runtime.assets if asset.path != str(tokenizer)
            ]
        }
    )
    with pytest.raises(ValueError, match="model snapshot/ref contains content absent"):
        verify_model_snapshot_assets(
            runtime,
            "example/model",
            expected_revision="a" * 40,
            expected_snapshot=snapshot,
        )


def test_model_cache_rejects_distinct_serving_path_with_same_revision(
    model_cache, tmp_path
):
    runtime, _, snapshot, _ = model_cache
    other_snapshot = tmp_path / "other" / snapshot.name
    other_snapshot.mkdir(parents=True)
    with pytest.raises(
        ValueError, match="cache snapshot differs from the serving model"
    ):
        verify_model_snapshot_assets(
            runtime,
            "example/model",
            expected_revision="a" * 40,
            expected_snapshot=other_snapshot,
        )


@pytest.mark.parametrize(
    "reference_revision,bind_reference,error",
    [
        ("b" * 40, True, "offline model main ref does not match"),
        ("a" * 40, False, "model snapshot/ref contains content absent"),
    ],
)
def test_bad_model_cache_fails_preparation_before_native_boundary(
    pilot_inputs, client_site, monkeypatch, reference_revision, bind_reference, error
):
    root, row, scheduling, site = pilot_inputs
    cache = (
        Path(client_site.env["HF_HUB_CACHE"])
        / "models--deepseek-ai--DeepSeek-V4.1-Flash"
    )
    reference = cache / "refs/main"
    reference.parent.mkdir(parents=True)
    reference.write_text(reference_revision)
    snapshot = cache / "snapshots" / ("a" * 40)
    snapshot.parent.mkdir()
    shutil.move(client_site.model_path, snapshot)
    client_site = client_site.model_copy(
        update={
            "model_path": str(snapshot),
            "asset_roots": [str(snapshot), client_site.asset_roots[1]],
            "asset_files": [str(reference)] if bind_reference else [],
        }
    )
    client_file = Path(site.shared_root) / "client-site.json"
    client_file.write_text(client_site.model_dump_json())
    native_source = Path(site.native_source)
    native_source.mkdir(parents=True)
    (native_source / "uv.lock").write_text("controlled dependency lock")
    Path(site.image.path).write_bytes(b"controlled image")
    site = site.model_copy(
        update={
            "model_snapshot": str(snapshot),
            "model_revision": "a" * 40,
            "image": bind_file(Path(site.image.path)),
            "client_sites": {"agentx": str(client_file), "eval": str(client_file)},
        }
    )
    (root / "runtime.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repository": "https://example.invalid/native.git",
                "revision": "c" * 40,
                "uv_lock_sha256": bind_file(native_source / "uv.lock").sha256,
                "capabilities": [],
            }
        )
    )
    actual_run = subprocess.run
    native_calls = []

    def external_process(argv, **kwargs):
        if argv[0] == "git":
            return subprocess.CompletedProcess(
                argv, 0, "c" * 40 if "rev-parse" in argv else "", ""
            )
        if argv[0] in {site.wrapper_python, site.native_python}:
            if "-m" in argv:
                native_calls.append(argv)
                raise AssertionError("invalid cache reached native execution")
            distribution = argv[argv.index("--distribution") + 1]
            identity = {
                "python_version": "3.12.0",
                "python_paths": {
                    key: site.shared_root
                    for key in (
                        "executable",
                        "executable_resolved",
                        "prefix",
                        "base_prefix",
                    )
                },
                "distributions": {distribution: {"files": {}}},
            }
            return subprocess.CompletedProcess(argv, 0, json.dumps(identity), "")
        return actual_run(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", external_process)
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
    with pytest.raises(ValueError, match=error):
        launch.prepare(
            job,
            site,
            root,
            {
                "repository": "example/pilot",
                "run_id": 10,
                "attempt": 1,
                "head_sha": "d" * 40,
            },
        )
    assert native_calls == []
