"""Preparation and derived-cache behavior under file mutation and process contention."""

from __future__ import annotations

import fcntl
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from infx.benchmarks.cache import MmapCache
from infx.benchmarks.common import read_json
from infx.benchmarks.identity import (
    capture_identity,
    require_source_revision,
    verify_runtime,
)
from infx.benchmarks.prepare import ClientSite, collect_assets, prepare
from infx.benchmarks.spec import RuntimeSpec


@pytest.fixture
def installed_child(tmp_path):
    environment = tmp_path / "child-env"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(environment)], check=True
    )
    python = environment / "bin/python"
    result = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "import sysconfig; print(sysconfig.get_path('purelib'))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    site = Path(result.stdout.strip())
    package = site / "lm_eval"
    package.mkdir()
    implementation = package / "__init__.py"
    implementation.write_text("meaning = 42\n")
    metadata = site / "lm_eval-0.1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: lm-eval\nVersion: 0.1\n"
    )
    (metadata / "direct_url.json").write_text(
        json.dumps(
            {
                "url": "https://github.com/EleutherAI/lm-evaluation-harness.git",
                "vcs_info": {
                    "vcs": "git",
                    "commit_id": "b315ef3b05176acc9732bb7fdec116abe1ecc476",
                },
            }
        )
    )
    (metadata / "RECORD").write_text(
        "\n".join(
            f"{path.relative_to(site)},,"
            for path in [implementation, *metadata.iterdir(), metadata / "RECORD"]
        )
        + "\n"
    )
    return python, implementation


@pytest.fixture
def client_site(tmp_path, installed_child):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"fixture"}')
    (model / "tokenizer.json").write_text('{"type":"fixture"}')
    (model / "model.safetensors.index.json").write_text(
        '{"weight_map":{"weight":"model-00001.safetensors"}}'
    )
    (model / "model-00001.safetensors").write_bytes(b"controlled model bytes")
    dataset = tmp_path / "hub/datasets--openai--gsm8k"
    reference = dataset / "refs/main"
    reference.parent.mkdir(parents=True)
    reference.write_text("b" * 40)
    payload = dataset / "snapshots" / ("b" * 40) / "train.parquet"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"controlled cached dataset")
    return ClientSite(
        python=str(installed_child[0]),
        distributions=["lm-eval"],
        env={
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_CACHE": str(tmp_path / "hub"),
            "HF_DATASETS_CACHE": str(tmp_path / "datasets"),
        },
        env_unset=[],
        asset_roots=[str(model), str(dataset)],
        asset_files=[],
        model_path=str(model),
        timeout_seconds=10,
        terminate_grace_seconds=1,
    )


def test_preparation_binds_installed_bytes_and_packaged_resources_from_other_cwd(
    tmp_path, installed_child, client_site, monkeypatch
):
    outside = tmp_path / "unrelated"
    outside.mkdir()
    monkeypatch.chdir(outside)
    prepared = tmp_path / "prepared"
    resources = prepare(client_site, "eval", prepared)
    runtime = RuntimeSpec.model_validate(read_json(prepared / "runtime.json"))
    original = verify_runtime(runtime, dataset_loader=None)
    assert original["distributions"]["lm-eval"]["version"] == "0.1"
    # Both files are real installed resources copied outside the checkout. A changed
    # task is detected through its bound content, before the harness can execute it.
    task = Path(resources["task"]["path"])
    assert task.parent == prepared
    task.chmod(0o644)
    task.write_text("task: unrelated\n")
    from infx.benchmarks.common import verify_file
    from infx.benchmarks.spec import PreparedFile

    with pytest.raises(ValueError, match="prepared file missing or changed"):
        verify_file(PreparedFile.model_validate(resources["task"]))
    # Same package name and version, different actual implementation bytes.
    installed_child[1].write_text("meaning = -1\n")
    with pytest.raises(ValueError, match="identity differs"):
        verify_runtime(runtime, dataset_loader=None)


def test_model_config_alone_cannot_pass_asset_preparation(client_site):
    (Path(client_site.model_path) / "model-00001.safetensors").unlink()
    with pytest.raises(ValueError, match="shard is missing or unbound"):
        collect_assets(client_site)


def test_installed_interpreter_identity_exposes_external_base_paths(installed_child):
    python, _implementation = installed_child
    identity = capture_identity(str(python), ["lm-eval"], dataset_loader=None)
    paths = identity["python_paths"]
    assert paths["executable"] == str(python)
    assert paths["executable_resolved"] == str(python.resolve())
    assert paths["prefix"] == str(python.parent.parent)
    assert paths["base_prefix"] != paths["prefix"]
    assert Path(paths["base_prefix"]).is_dir()
    # A same-path mount of the venv alone cannot make its external interpreter
    # and standard library available inside a client container.
    assert not Path(paths["executable_resolved"]).is_relative_to(paths["prefix"])


def test_client_source_and_secret_inputs_are_rejected(client_site):
    with pytest.raises(ValueError, match="immutable reviewed revision"):
        require_source_revision(
            {"distributions": {"lm-eval": {"version": "0.1"}}}, "lm-eval", "a" * 40
        )
    with pytest.raises(ValueError, match="secret-bearing"):
        ClientSite.model_validate(
            {
                **client_site.model_dump(),
                "env": {**client_site.env, "HF_TOKEN": "not-persisted"},
            }
        )


def test_installed_wheel_prepares_and_verifies_resources_without_checkout(
    tmp_path, client_site
):
    uv = shutil.which("uv") or str(Path(sys.executable).with_name("uv"))
    repository = Path(__file__).resolve().parents[1]
    wheels = tmp_path / "wheels"
    subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(wheels)],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    environment = tmp_path / "installed-wrapper"
    subprocess.run(
        [uv, "venv", "--python", sys.executable, str(environment)],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    python = environment / "bin/python"
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            str(next(wheels.glob("*.whl"))),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    site_path = tmp_path / "site.json"
    site_path.write_text(client_site.model_dump_json())
    outside = tmp_path / "outside-checkout"
    outside.mkdir()
    code = """
import json
from pathlib import Path
from infx.benchmarks.prepare import ClientSite, prepare
from infx.benchmarks.common import read_json, verify_file
from infx.benchmarks.spec import PreparedFile
import sys
resources = prepare(ClientSite.model_validate(read_json(Path(sys.argv[1]))), 'eval', Path('prepared'))
task = PreparedFile.model_validate(resources['task'])
Path(task.path).chmod(0o644)
Path(task.path).write_text('task: changed-after-preparation\\n')
try:
    verify_file(task)
except ValueError as error:
    print(json.dumps({'error': str(error), 'resource_directory': str(Path(task.path).parent)}))
else:
    raise SystemExit('mutated resource was accepted')
"""
    result = subprocess.run(
        [str(python), "-I", "-c", code, str(site_path)],
        cwd=outside,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    observed = json.loads(result.stdout)
    assert "prepared file missing or changed" in observed["error"]
    assert observed["resource_directory"] == str(outside / "prepared")


def populate(root, payload=b"client-generated bytes"):
    key = "a" * 32
    entry = root / key
    entry.mkdir()
    (entry / "dataset.dat").write_bytes(payload)
    (entry / "index.dat").write_bytes(b"index bytes")
    (entry / "manifest.json").write_text(
        json.dumps({"cache_key": key, "compressed": False})
    )
    return entry


def test_cache_cold_warm_copy_and_corrupt_snapshot_rebuild(tmp_path):
    first = MmapCache(
        tmp_path / "cache", {"dataset": "prepared"}, lock_timeout_seconds=0.1
    )
    source = populate(first.prepare())
    first.publish()
    first.close()
    canonical_payload = first.canonical / source.name / "dataset.dat"
    warm = MmapCache(
        tmp_path / "cache", {"dataset": "prepared"}, lock_timeout_seconds=0.1
    )
    restored = warm.prepare() / source.name / "dataset.dat"
    assert restored.read_bytes() == b"client-generated bytes"
    assert restored.stat().st_ino != canonical_payload.stat().st_ino
    restored.write_bytes(b"private mutation")
    assert canonical_payload.read_bytes() == b"client-generated bytes"
    warm.close()
    canonical_payload.write_bytes(b"truncated")
    repair = MmapCache(
        tmp_path / "cache", {"dataset": "prepared"}, lock_timeout_seconds=0.1
    )
    assert list(repair.prepare().iterdir()) == []
    assert len(list(repair.root.glob("quarantine-*"))) == 1
    populate(repair.run_dir, b"rebuilt bytes")
    repair.publish()
    repair.close()
    assert canonical_payload.read_bytes() == b"rebuilt bytes"


def test_incomplete_cache_receipt_and_lock_contention_never_authorize_unlocked_publication(
    tmp_path,
):
    cache = MmapCache(
        tmp_path / "cache", {"dataset": "prepared"}, lock_timeout_seconds=0.05
    )
    cache.canonical.mkdir()
    populate(cache.canonical)
    # A producer crashed before committing its integrity record.
    assert list(cache.prepare().iterdir()) == []
    cache.close()
    lock = cache.root / "publication.lock"
    with lock.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        other = MmapCache(
            tmp_path / "cache", {"dataset": "prepared"}, lock_timeout_seconds=0.05
        )
        populate(other.prepare())
        other.publish()
        assert not other.canonical.exists()
        assert any("independent cold cache" in event for event in other.events)
        assert any("publication skipped" in event for event in other.events)
        other.close()
