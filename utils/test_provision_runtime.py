"""Shared provisioning boundaries with tiny assets and public-download collaborators."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from infx.benchmarks.common import (
    read_json,
    verify_model_snapshot_assets,
    verify_snapshot_assets,
)
from infx.benchmarks.prepare import bind_file, collect_assets
from infx.benchmarks.spec import RuntimeSpec
from infx.srt_slurm.provision import ProvisionConfig, snapshot
from infx.srt_slurm.provision_runtime import (
    Commands,
    ProvisionStepError,
    _GSM_SCRIPT,
    _offline_env,
    _sites,
    installer_environment,
    owned_generation,
    publish_evidence,
    require_clean_checkout,
    verify_clients,
)


@pytest.fixture
def assets(tmp_path):
    root = tmp_path.resolve()
    config = ProvisionConfig(
        schema_version=1,
        shared_root=str(root / "prepared"),
        hub_cache=str(root / "legacy-hub"),
        image_path=str(root / "image.sqsh"),
        image_reference="fixture/image@sha256:abc",
        model_repository="fixture/model",
        model_revision="a" * 40,
        dataset_repository="semianalysisai/cc-traces-weka-062126",
        dataset_revision="b" * 40,
    )
    Path(config.image_path).write_bytes(b"hsqs inert test payload")
    model = snapshot(config, dataset=False)
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"model_type":"fixture"}')
    (model / "tokenizer.json").write_text('{"fixture":true}')
    (model / "model.safetensors.index.json").write_text(
        '{"weight_map":{"weight":"one.safetensors"}}'
    )
    (model / "one.safetensors").write_bytes(b"weight")
    trace = snapshot(config, dataset=True)
    trace.mkdir(parents=True)
    (trace / "train.parquet").write_bytes(b"trace")
    for source in (model, trace):
        (source.parent.parent / "refs").mkdir()
        (source.parent.parent / "refs/main").write_text("c" * 40)
        (source.parent / ("d" * 40)).mkdir()
    return config


def test_private_views_bind_original_snapshot_without_mutating_legacy_refs(assets):
    with owned_generation(Path(assets.shared_root), "run-1") as generation:
        env = _offline_env(generation)
        sites = _sites(
            assets, {"agentx": Path(sys.executable), "eval": Path(sys.executable)}, env
        )
        gsm = Path(env["HF_HUB_CACHE"]) / "datasets--openai--gsm8k"
        (gsm / "snapshots" / ("e" * 40)).mkdir(parents=True)
        (gsm / "snapshots" / ("e" * 40) / "test.parquet").write_bytes(b"gsm")
        (gsm / "refs").mkdir()
        (gsm / "refs/main").write_text("e" * 40)
        (Path(env["HF_DATASETS_CACHE"]) / "test.arrow").write_bytes(b"arrow")
        identity = generation / "identity.json"
        identity.write_text("{}")
        runtime = RuntimeSpec(
            python=sites["agentx"].python,
            identity=bind_file(identity),
            distributions=["aiperf"],
            env=sites["agentx"].env,
            env_unset=sites["agentx"].env_unset,
            assets=collect_assets(sites["agentx"]),
            timeout_seconds=10,
            terminate_grace_seconds=1,
        )
        original_model = snapshot(assets, dataset=False)
        assert (
            verify_model_snapshot_assets(
                runtime,
                "fixture/model",
                expected_revision="a" * 40,
                expected_snapshot=original_model,
            )
            == original_model
        )
        assert (
            verify_snapshot_assets(
                runtime,
                assets.dataset_repository,
                expected_revision="b" * 40,
                only_snapshot=True,
            )
            == "b" * 40
        )
        assert (
            verify_snapshot_assets(
                runtime, "openai/gsm8k", expected_revision="e" * 40, only_snapshot=True
            )
            == "e" * 40
        )
        assert (original_model.parent.parent / "refs/main").read_text() == "c" * 40
        assert (
            snapshot(assets, dataset=True).parent.parent / "refs/main"
        ).read_text() == "c" * 40
        assert (original_model.parent / ("d" * 40)).is_dir()
        assert sites["eval"].env["HF_HUB_OFFLINE"] == "1"
        assert "HF_HUB_DISABLE_IMPLICIT_TOKEN" not in sites["eval"].env


def test_owned_generation_rejects_overlap_and_reuse_and_retains_failure(tmp_path):
    root = tmp_path.resolve() / "prepared"
    with pytest.raises(RuntimeError, match="controlled failure"):
        with owned_generation(root, "attempt-1") as generation:
            (generation / "evidence/progress.txt").write_text("completed stage")
            with pytest.raises(ValueError, match="another provisioning"):
                with owned_generation(root, "attempt-2"):
                    pytest.fail("lock was bypassed")
            raise RuntimeError("controlled failure")
    assert read_json(generation / "state.json") == {
        "state": "failed",
        "error_type": "RuntimeError",
        "qualification_complete": False,
    }
    assert (generation / "evidence/progress.txt").read_text() == "completed stage"
    with pytest.raises(FileExistsError):
        with owned_generation(root, "attempt-1"):
            pytest.fail("failed generation was silently reused")
    assert not (root / "generations/attempt-2").exists()


@pytest.mark.parametrize("namespace", ["../escape", "", "/absolute"])
def test_namespace_cannot_escape_owned_root(tmp_path, namespace):
    with pytest.raises(ValueError, match="namespace"):
        with owned_generation(tmp_path.resolve() / "prepared", namespace):
            pytest.fail("unsafe namespace accepted")


def test_child_gets_no_ambient_credentials_and_failure_logs_are_publishable(tmp_path):
    with owned_generation(tmp_path.resolve() / "prepared", "run") as generation:
        ambient = {
            "PATH": os.environ["PATH"],
            "HF_TOKEN": "sensitive-hf",
            "GITHUB_TOKEN": "sensitive-git",
            "PYTHONPATH": "injected",
            "GIT_CONFIG_COUNT": "99",
            "AWS_ACCESS_KEY_ID": "sensitive-aws",
        }
        commands = Commands(generation, installer_environment(generation, ambient))
        result = commands.run(
            "environment",
            [
                sys.executable,
                "-I",
                "-c",
                "import json,os; print(json.dumps({k:os.environ.get(k) for k in ['HF_TOKEN','GITHUB_TOKEN','PYTHONPATH','AWS_ACCESS_KEY_ID','GIT_CONFIG_COUNT','HF_HUB_DISABLE_IMPLICIT_TOKEN']}))",
            ],
            cwd=generation,
            timeout=10,
        )
        assert json.loads(result) == {
            "HF_TOKEN": None,
            "GITHUB_TOKEN": None,
            "PYTHONPATH": None,
            "AWS_ACCESS_KEY_ID": None,
            "GIT_CONFIG_COUNT": None,
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        }
        with pytest.raises(ProvisionStepError, match="controlled-exit"):
            commands.run(
                "controlled-exit",
                [
                    sys.executable,
                    "-I",
                    "-c",
                    "print('download collaborator failed'); raise SystemExit(7)",
                ],
                cwd=generation,
                timeout=10,
            )
        output = tmp_path / "artifact"
        publish_evidence(generation, output)
        status = read_json(output / "logs/02-controlled-exit.json")
        assert status["returncode"] == 7
        assert not status["timed_out"]
        assert (
            "download collaborator failed"
            in (output / "logs/02-controlled-exit.log").read_text()
        )
        assert "sensitive" not in (output / "logs/01-environment.log").read_text()


def git(checkout, *arguments):
    return subprocess.run(
        ["git", *arguments], cwd=checkout, capture_output=True, text=True, check=True
    ).stdout.strip()


def test_checkout_exclusion_allows_only_explicit_untracked_artifact_directory(tmp_path):
    checkout = tmp_path.resolve() / "checkout"
    checkout.mkdir()
    git(checkout, "init")
    git(checkout, "config", "user.email", "test@example.invalid")
    git(checkout, "config", "user.name", "Fixture")
    (checkout / "tracked.txt").write_text("source")
    git(checkout, "add", "tracked.txt")
    git(checkout, "commit", "-m", "fixture")
    expected_revision = git(checkout, "rev-parse", "HEAD")
    output = checkout / "report"
    output.mkdir()
    (output / "inventory.json").write_text("{}")
    with owned_generation(tmp_path.resolve() / "shared", "run") as generation:
        commands = Commands(generation, installer_environment(generation, os.environ))
        assert require_clean_checkout(commands, checkout, output) == expected_revision
        (checkout / "unrelated.txt").write_text("not an artifact")
        with pytest.raises(ValueError, match="must be clean"):
            require_clean_checkout(commands, checkout, output)
        (checkout / "unrelated.txt").unlink()
        (checkout / "tracked.txt").write_text("changed source")
        with pytest.raises(ValueError, match="must be clean"):
            require_clean_checkout(commands, checkout, output)
        with pytest.raises(ValueError, match="outside source"):
            require_clean_checkout(commands, checkout, checkout / "infx/report")
        with pytest.raises(ValueError, match="tracked repository"):
            require_clean_checkout(commands, checkout, checkout / "tracked.txt")


@pytest.fixture
def isolated_python(tmp_path):
    environment = tmp_path / "python"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(environment)], check=True
    )
    python = environment / "bin/python"
    purelib = Path(
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    return python, purelib


def test_installed_client_probes_retain_behavior_and_reject_invalid_tokenization(
    assets, isolated_python
):
    python, purelib = isolated_python
    for package in ("aiperf", "aiperf/common", "lm_eval", "lm_eval/models"):
        directory = purelib / package
        directory.mkdir(exist_ok=True)
        (directory / "__init__.py").touch()
    metadata = purelib / "transformers-5.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: transformers\nVersion: 5.0\n")
    (purelib / "huggingface_hub.py").write_text(
        "import os,pathlib\n"
        "def snapshot_download(repository, *, revision, local_files_only):\n"
        "    assert revision == 'main' and local_files_only\n"
        "    root=pathlib.Path(os.environ['HF_HUB_CACHE'])/('models--'+repository.replace('/','--'))\n"
        "    return str(root/'snapshots'/(root/'refs/main').read_text().strip())\n"
    )
    (purelib / "aiperf/common/tokenizer.py").write_text(
        "import os\n"
        "class Tokenizer:\n"
        "    @classmethod\n"
        "    def from_pretrained(cls, repository, *, trust_remote_code):\n"
        "        assert repository == 'fixture/model' and trust_remote_code\n"
        "        assert os.environ['HF_HUB_OFFLINE'] == '1'\n"
        "        return cls()\n"
        "    def encode(self, text): return [17,19]\n"
        "    def decode(self, tokens): return 'decoded text'\n"
        "    def encode_lengths_batch(self, texts):\n"
        "        return [2,2] if 'FIXTURE_INVALID_LENGTHS' not in os.environ else [1,2]\n"
    )
    (purelib / "lm_eval/models/openai_completions.py").write_text(
        "class LocalChatCompletion:\n"
        "    def __init__(self, **kwargs):\n"
        "        assert kwargs['tokenized_requests'] is False\n"
        "        self.tokenizer_backend=None\n"
        "    def apply_chat_template(self, messages): return messages\n"
        "    def create_message(self, batch): return batch[0]\n"
    )
    with owned_generation(Path(assets.shared_root), "client-probes") as generation:
        env = _offline_env(generation)
        clients = {"agentx": python, "eval": python}
        _sites(assets, clients, env)
        commands = Commands(generation, installer_environment(generation, os.environ))
        verify_clients(commands, assets, clients, env)
        agent = read_json(generation / "evidence/agentx-behavior.json")
        assert agent["tokens"] == [[17, 19], [17, 19]]
        assert agent["batch_lengths"] == [2, 2]
        assert agent["snapshot"] == str(snapshot(assets, dataset=False))
        assert agent["configuration"] == {"config.json": {"model_type": "fixture"}}
        evaluation = read_json(generation / "evidence/eval-behavior.json")
        assert evaluation["messages"] == [
            {"role": "user", "content": "InferenceX client preparation."}
        ]
        assert evaluation["tokenizer_backend"] is None
        with pytest.raises(ProvisionStepError, match="offline-agentx-behavior"):
            verify_clients(
                commands, assets, clients, {**env, "FIXTURE_INVALID_LENGTHS": "1"}
            )
        assert (
            "batch lengths differ"
            in (generation / "logs/03-offline-agentx-behavior.log").read_text()
        )


def test_gsm_materialization_pins_online_revision_then_checks_nominal_offline_lookup(
    tmp_path, isolated_python
):
    python, purelib = isolated_python
    payload = tmp_path / "dataset.json"
    payload.write_text(
        json.dumps(
            {
                "train": [{"question": "one?", "answer": "#### 1"}] * 5,
                "test": [{"question": "one?", "answer": "#### 1"}] * 1319,
            }
        )
    )
    (purelib / "datasets.py").write_text(
        "import json,os,pathlib\n"
        "def load_dataset(repository, name, **kwargs):\n"
        "    assert repository == 'openai/gsm8k' and name == 'main'\n"
        "    if os.environ['HF_HUB_OFFLINE'] == '0':\n"
        "        assert kwargs['revision'] == 'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee'\n"
        "    else:\n"
        "        assert 'revision' not in kwargs\n"
        f"    return json.loads(pathlib.Path({str(payload)!r}).read_text())\n"
    )
    (purelib / "huggingface_hub.py").write_text(
        "import pathlib\n"
        "def snapshot_download(repository, *, repo_type, revision, cache_dir):\n"
        "    assert (repository,repo_type,revision) == ('openai/gsm8k','dataset','main')\n"
        "    path=pathlib.Path(cache_dir)/'datasets--openai--gsm8k'/'snapshots'/('e'*40)\n"
        "    path.mkdir(parents=True)\n"
        "    return str(path)\n"
    )
    expected = tmp_path / "hashes.json"
    expected.write_text(
        json.dumps(
            {
                str(
                    i
                ): "198133a7f6c2d658eaef3a9bbd3695f496257a5407021ef31471834fbb2c8fe4"
                for i in range(1319)
            }
        )
    )
    script = tmp_path / "materialize.py"
    script.write_text(_GSM_SCRIPT)
    output = tmp_path / "receipt.json"
    for mode in ("online", "offline"):
        subprocess.run(
            [str(python), "-I", str(script), mode, str(expected), str(output)],
            env={
                **os.environ,
                "HF_HUB_CACHE": str(tmp_path / "hub"),
                "HF_DATASETS_CACHE": str(tmp_path / "datasets"),
                "HF_HUB_OFFLINE": "0" if mode == "online" else "1",
            },
            capture_output=True,
            text=True,
            check=True,
        )
    assert read_json(output) == {
        "revision": "e" * 40,
        "test_documents": 1319,
        "train_documents": 5,
        "offline": True,
    }
    data = read_json(payload)
    data["test"][0]["answer"] = "changed document"
    payload.write_text(json.dumps(data))
    failed = subprocess.run(
        [str(python), "-I", str(script), "offline", str(expected), str(output)],
        env={
            **os.environ,
            "HF_HUB_CACHE": str(tmp_path / "hub"),
            "HF_DATASETS_CACHE": str(tmp_path / "datasets"),
            "HF_HUB_OFFLINE": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode != 0
    assert "GSM8K document differs" in failed.stderr
