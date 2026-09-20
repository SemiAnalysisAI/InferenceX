"""Build a new shared, offline client generation without allocating Slurm resources."""

from __future__ import annotations

import fcntl
import json
import os
import platform
import re
import shutil
import tomllib
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any

from infx.benchmarks.common import child_failed, read_json, run_child, write_json
from infx.benchmarks.identity import AGENTX_REVISION, LM_EVAL_REVISION
from infx.benchmarks.prepare import ClientSite, bind_file
from infx.srt_slurm.launch import RuntimeLock
from infx.srt_slurm.provision import inspect_assets, snapshot

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from infx.srt_slurm.provision import ProvisionConfig

NATIVE_LOCK = "benchmarks/multi_node/srt-slurm-recipes/configs/prepared-runtime-lock.json"
NATIVE_REPOSITORY = "https://github.com/SemiAnalysisAI/srt-slurm.git"
NVIDIA_REPOSITORY = "https://github.com/NVIDIA/srt-slurm.git"
NVIDIA_BASE = "984180e5b8755aef85e9995048b5a16cb5336bce"
CLIENT_REPOSITORIES = {
    "agentx": ("aiperf", "https://github.com/SemiAnalysisAI/aiperf.git", AGENTX_REVISION, "3.11"),
    "eval": (
        "lm-eval[api]",
        "https://github.com/EleutherAI/lm-evaluation-harness.git",
        LM_EVAL_REVISION,
        "3.12",
    ),
}


class ProvisionStepError(RuntimeError):
    """Only a stage identifier leaves the retained, credential-free child log."""


def installer_environment(generation: Path, ambient: Mapping[str, str]) -> dict[str, str]:
    """Public downloads receive no ambient credentials, Python injection, or user config."""
    allowed = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "SSL_CERT_FILE", "SSL_CERT_DIR")
    environment = {name: ambient[name] for name in allowed if name in ambient}
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "NETRC": str(generation / "empty-config/netrc"),
            "UV_KEYRING_PROVIDER": "disabled",
            "UV_PYTHON_INSTALL_DIR": str(generation / "python"),
            "UV_CACHE_DIR": str(generation / "uv-cache"),
            "UV_LINK_MODE": "copy",
            "XDG_CONFIG_HOME": str(generation / "empty-config"),
            "XDG_CACHE_HOME": str(generation / "cache"),
            "TMPDIR": str(generation / "temporary"),
            "HF_HOME": str(generation / "hf"),
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
        }
    )
    return environment


@dataclass
class Commands:
    generation: Path
    environment: dict[str, str]
    current_stage: str = "starting"
    number: int = 0

    def run(
        self,
        stage: str,
        argv: Sequence[str],
        *,
        cwd: Path,
        timeout: int,
        environment: Mapping[str, str] | None = None,
    ) -> str:
        self.current_stage = stage
        self.number += 1
        log = self.generation / "logs" / f"{self.number:02d}-{stage}.log"
        print(f"Provisioning {stage}; bounded to {timeout}s; log={log}", flush=True)
        write_json(self.generation / "progress.json", {"stage": stage, "log": str(log)})
        status = run_child(
            argv,
            env={**self.environment, **(environment or {})},
            cwd=cwd,
            log=log,
            timeout_seconds=timeout,
            terminate_grace_seconds=10,
        )
        write_json(log.with_suffix(".json"), {"stage": stage, **status})
        if child_failed(status):
            raise ProvisionStepError(
                f"Provisioning stage failed: {stage}; inspect its retained log"
            )
        return log.read_text().strip()


@contextmanager
def owned_generation(root: Path, namespace: str) -> Iterator[Path]:
    """One root lock, a never-reused generation, and retained incomplete work on failure."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", namespace) is None:
        raise ValueError("invalid provisioning namespace")
    if not root.is_absolute() or root.resolve() != root or root.is_relative_to("/workspace"):
        raise ValueError("generation root must be a canonical shared path outside /workspace")
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root / ".provision.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another provisioning operation owns this shared root") from error
        generations = root / "generations"
        generations.mkdir(exist_ok=True)
        if generations.resolve() != generations:
            raise ValueError("generation directory may not be a symlink")
        generation = generations / namespace
        generation.mkdir(exist_ok=False)
        for name in ("logs", "evidence", "envs", "projects", "temporary", "empty-config"):
            (generation / name).mkdir()
        (generation / "empty-config/netrc").touch(mode=0o600)
        write_json(
            generation / "state.json", {"state": "building", "qualification_complete": False}
        )
        try:
            yield generation
        except BaseException as error:
            write_json(
                generation / "state.json",
                {
                    "state": "failed",
                    "error_type": type(error).__name__,
                    "qualification_complete": False,
                },
            )
            raise


def private_snapshot(
    hub: Path, repository: str, revision: str, source: Path, *, dataset: bool
) -> Path:
    """Create only a private reference and a symlink to the unchanged canonical snapshot."""
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None or source.name != revision:
        raise ValueError("snapshot revision does not match the explicit source directory")
    if not source.is_dir() or source.resolve() != source:
        raise ValueError("source snapshot must be an existing canonical directory")
    prefix = "datasets--" if dataset else "models--"
    base = hub / (prefix + repository.replace("/", "--"))
    (base / "snapshots").mkdir(parents=True, exist_ok=False)
    target = base / "snapshots" / revision
    target.symlink_to(source, target_is_directory=True)
    (base / "refs").mkdir()
    reference = base / "refs/main"
    with reference.open("x") as stream:
        stream.write(revision + "\n")
    reference.chmod(0o444)
    return target


def _project(path: Path, dependencies: list[str], python: str, *, cpu_torch: bool = False) -> None:
    path.mkdir()
    content = (
        '[project]\nname = "infx-prepared-environment"\nversion = "0.0.0"\n'
        f'requires-python = "=={python}.*"\ndependencies = {json.dumps(dependencies)}\n'
        "[tool.uv]\npackage = false\n"
    )
    if cpu_torch:
        content += (
            '[tool.uv.sources]\ntorch = { index = "pytorch-cpu" }\n'
            '[[tool.uv.index]]\nname = "pytorch-cpu"\n'
            'url = "https://download.pytorch.org/whl/cpu"\nexplicit = true\n'
        )
    (path / "pyproject.toml").write_text(content)


def _sync(commands: Commands, uv: str, project: Path, name: str, python: str) -> Path:
    environment = commands.generation / "envs" / name
    commands.run(
        f"install-{name}",
        [
            uv,
            "sync",
            "--project",
            str(project),
            "--frozen",
            "--no-default-groups",
            "--no-install-project",
            "--no-editable",
            "--managed-python",
            "--python",
            python,
        ],
        cwd=project,
        timeout=1800,
        environment={"UV_PROJECT_ENVIRONMENT": str(environment)},
    )
    return environment / "bin/python"


def _native_source(commands: Commands, lock: RuntimeLock) -> Path:
    source = commands.generation / "native-source"
    source.mkdir()
    prefix = ["git", "-c", "credential.helper=", "-c", "core.askPass="]
    operations = [
        ("native-init", ["init"]),
        ("native-origin", ["remote", "add", "origin", lock.repository]),
        ("native-fetch", ["fetch", "--no-tags", "origin", lock.revision]),
        ("native-checkout", ["checkout", "--detach", lock.revision]),
        (
            "native-tag",
            ["fetch", "--no-tags", NVIDIA_REPOSITORY, "refs/tags/v2.2.1:refs/tags/v2.2.1"],
        ),
    ]
    for stage, arguments in operations:
        commands.run(stage, [*prefix, *arguments], cwd=source, timeout=600)
    for stage, reference, expected in (
        ("native-head", "HEAD", lock.revision),
        ("native-version-tag", "v2.2.1^{commit}", NVIDIA_BASE),
    ):
        if (
            commands.run(stage, [*prefix, "rev-parse", reference], cwd=source, timeout=30)
            != expected
        ):
            raise ValueError("native source or version tag differs from the reviewed pin")
    commands.run(
        "native-lineage",
        [*prefix, "merge-base", "--is-ancestor", NVIDIA_BASE, "HEAD"],
        cwd=source,
        timeout=30,
    )
    from infx.benchmarks.common import sha256_file

    if sha256_file(source / "uv.lock") != lock.uv_lock_sha256:
        raise ValueError("native dependency lock differs from the reviewed pin")
    return source


def _wheels(commands: Commands, uv: str, source: Path, checkout: Path) -> tuple[Path, Path]:
    requirements = sorted(
        {
            requirement
            for directory in (source, checkout)
            for requirement in tomllib.loads((directory / "pyproject.toml").read_text())[
                "build-system"
            ]["requires"]
        }
    )
    project = commands.generation / "projects/build"
    _project(project, requirements, "3.12")
    commands.run(
        "lock-build-tools",
        [uv, "lock", "--project", str(project), "--python", "3.12", "--managed-python"],
        cwd=project,
        timeout=600,
    )
    shutil.copyfile(project / "uv.lock", commands.generation / "evidence/build-uv.lock")
    python = _sync(commands, uv, project, "build", "3.12")
    constraints = commands.generation / "evidence/build-constraints.txt"
    commands.run(
        "export-build-constraints",
        [
            uv,
            "export",
            "--project",
            str(project),
            "--frozen",
            "--no-emit-project",
            "--output-file",
            str(constraints),
        ],
        cwd=project,
        timeout=60,
    )
    result = []
    for name, directory in (("native", source), ("wrapper", checkout)):
        output = commands.generation / "wheels" / name
        commands.run(
            f"build-{name}-wheel",
            [
                uv,
                "build",
                str(directory),
                "--wheel",
                "--no-build-isolation",
                "--force-pep517",
                "--python",
                str(python),
                "--out-dir",
                str(output),
            ],
            cwd=directory,
            timeout=600,
        )
        wheels = list(output.glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError("expected exactly one retained wheel per package")
        result.append(wheels[0])
    return result[0], result[1]


_GSM_SCRIPT = """import hashlib, json, os, pathlib, re, sys
from datasets import load_dataset
from huggingface_hub import snapshot_download
mode, expected_path, output = sys.argv[1:]
hub = pathlib.Path(os.environ["HF_HUB_CACHE"])
base = hub / "datasets--openai--gsm8k"
if mode == "online":
    snapshot = pathlib.Path(snapshot_download("openai/gsm8k", repo_type="dataset", revision="main", cache_dir=str(hub)))
    if not re.fullmatch("[0-9a-f]{40}", snapshot.name): raise ValueError("GSM8K revision is not immutable")
    (base / "refs").mkdir(exist_ok=True)
    (base / "refs/main").write_text(snapshot.name + "\\n")
revision = (base / "refs/main").read_text().strip()
dataset = load_dataset("openai/gsm8k", "main", cache_dir=os.environ["HF_DATASETS_CACHE"], **({"revision": revision} if mode == "online" else {}))
expected = json.loads(pathlib.Path(expected_path).read_text())
if len(dataset["test"]) != 1319 or set(expected) != {str(i) for i in range(1319)}:
    raise ValueError("GSM8K requires exactly the canonical 1319 test documents")
for index, document in enumerate(dataset["test"]):
    digest = hashlib.sha256(json.dumps(document, indent=2, ensure_ascii=False).encode()).hexdigest()
    if digest != expected[str(index)]: raise ValueError("GSM8K document differs from packaged identity")
if len(dataset["train"]) < 5: raise ValueError("GSM8K five-shot training split is incomplete")
pathlib.Path(output).write_text(json.dumps({"revision": revision, "test_documents": len(dataset["test"]), "train_documents": len(dataset["train"]), "offline": mode == "offline"}) + "\\n")
"""


def _clients(commands: Commands, uv: str, build_constraints: Path) -> dict[str, Path]:
    result = {}
    for kind, (name, repository, revision, minor) in CLIENT_REPOSITORIES.items():
        project = commands.generation / "projects" / kind
        dependencies = [f"{name} @ git+{repository}@{revision}"]
        if kind == "eval":
            # The pinned API backend imports models.utils, which imports both packages;
            # lm-eval[api] itself does not declare either dependency.
            dependencies.extend(["torch>=2,<3", "transformers>=4.56,<6"])
        _project(project, dependencies, minor, cpu_torch=kind == "eval")
        commands.run(
            f"lock-{kind}",
            [uv, "lock", "--project", str(project), "--python", minor, "--managed-python"],
            cwd=project,
            timeout=1800,
            environment={"UV_BUILD_CONSTRAINT": str(build_constraints)},
        )
        shutil.copyfile(project / "uv.lock", commands.generation / "evidence" / f"{kind}-uv.lock")
        result[kind] = _sync(commands, uv, project, kind, minor)
    return result


_CLIENT_PROBE_SCRIPT = """import importlib.metadata, json, pathlib, sys
kind, repository, snapshot, output = sys.argv[1:]
if kind == "agentx":
    from huggingface_hub import snapshot_download
    from aiperf.common.tokenizer import Tokenizer
    resolved = pathlib.Path(snapshot_download(repository, revision="main", local_files_only=True)).resolve()
    if resolved != pathlib.Path(snapshot).resolve():
        raise ValueError("nominal tokenizer cache resolves a different serving snapshot")
    tokenizer = Tokenizer.from_pretrained(repository, trust_remote_code=True)
    samples = ["InferenceX tokenizer preparation.", "你好，世界。"]
    tokens = [tokenizer.encode(sample) for sample in samples]
    if any(not row or any(type(token) is not int for token in row) for row in tokens):
        raise ValueError("prepared tokenizer returned invalid token IDs")
    decoded = [tokenizer.decode(row) for row in tokens]
    if any(not isinstance(text, str) or not text for text in decoded):
        raise ValueError("prepared tokenizer failed to decode")
    lengths = tokenizer.encode_lengths_batch(samples)
    if lengths != [len(row) for row in tokens]:
        raise ValueError("prepared tokenizer batch lengths differ from individual encoding")
    configuration = {}
    for filename in ("config.json", "tokenizer_config.json"):
        path = resolved / filename
        if path.is_file():
            raw = json.loads(path.read_text())
            configuration[filename] = {key: raw[key] for key in ("model_type", "tokenizer_class", "auto_map", "transformers_version") if key in raw}
    result = {"repository": repository, "snapshot": str(resolved), "tokens": tokens, "decoded": decoded, "batch_lengths": lengths, "configuration": configuration}
elif kind == "eval":
    from lm_eval.models.openai_completions import LocalChatCompletion
    model = LocalChatCompletion(model=repository, base_url="http://127.0.0.1:1/v1/chat/completions", tokenized_requests=False, max_length=16384, eos_string="</s>")
    messages = [{"role": "user", "content": "InferenceX client preparation."}]
    formatted = model.create_message([model.apply_chat_template(messages)])
    if formatted != messages:
        raise ValueError("prepared eval backend changed chat messages")
    result = {"backend": type(model).__name__, "messages": formatted, "tokenizer_backend": model.tokenizer_backend}
else:
    raise ValueError("unknown prepared client")
result["transformers_version"] = importlib.metadata.version("transformers")
pathlib.Path(output).write_text(json.dumps(result, ensure_ascii=False) + "\\n")
"""


def verify_clients(
    commands: Commands, config: ProvisionConfig, clients: dict[str, Path], env: dict[str, str]
) -> None:
    """Exercise the installed tokenizer and API chat backend offline before hashing assets."""
    script = commands.generation / "verify-client-behavior.py"
    script.write_text(_CLIENT_PROBE_SCRIPT)
    for kind, python in clients.items():
        commands.run(
            f"offline-{kind}-behavior",
            [
                str(python),
                "-I",
                str(script),
                kind,
                config.model_repository,
                str(snapshot(config, dataset=False)),
                str(commands.generation / "evidence" / f"{kind}-behavior.json"),
            ],
            cwd=commands.generation,
            timeout=600,
            environment=env,
        )


def _offline_env(generation: Path) -> dict[str, str]:
    paths = {
        "HF_HOME": generation / "hf",
        "HF_HUB_CACHE": generation / "hf/hub",
        "HF_DATASETS_CACHE": generation / "hf/datasets",
        "HF_MODULES_CACHE": generation / "hf/modules",
        "AIPERF_DATASET_MMAP_CACHE_DIR": generation / "mmap",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return {
        **{key: str(path) for key, path in paths.items()},
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }


def _sites(
    config: ProvisionConfig, clients: dict[str, Path], env: dict[str, str]
) -> dict[str, ClientSite]:
    model = snapshot(config, dataset=False)
    trace = snapshot(config, dataset=True)
    hub = Path(env["HF_HUB_CACHE"])
    model_view = private_snapshot(
        hub, config.model_repository, config.model_revision, model, dataset=False
    )
    trace_view = private_snapshot(
        hub, config.dataset_repository, config.dataset_revision, trace, dataset=True
    )
    roots = [
        str(model_view),
        str(trace_view),
        str(hub / "datasets--openai--gsm8k"),
        env["HF_DATASETS_CACHE"],
    ]
    refs = [
        str(model_view.parent.parent / "refs/main"),
        str(trace_view.parent.parent / "refs/main"),
    ]
    return {
        kind: ClientSite(
            python=str(python),
            distributions=["aiperf" if kind == "agentx" else "lm-eval"],
            env=env,
            env_unset=["PYTHONPATH", "PYTHONHOME", "BASH_ENV", "ENV", "VIRTUAL_ENV"],
            asset_roots=roots,
            asset_files=refs,
            model_path=str(model),
            timeout_seconds=14400,
            terminate_grace_seconds=60,
        )
        for kind, python in clients.items()
    }


_VERIFY_SCRIPT = """import json, pathlib, sys
from infx.benchmarks.common import write_json, verify_model_snapshot_assets, verify_snapshot_assets
from infx.benchmarks.identity import capture_identity, require_source_revision, AGENTX_REVISION, LM_EVAL_REVISION
from infx.benchmarks.prepare import ClientSite, bind_file, collect_assets
from infx.benchmarks.spec import RuntimeSpec
from infx.srt_slurm.launch import verify_wrapper_source
config = json.loads(pathlib.Path(sys.argv[1]).read_text())
root = pathlib.Path(config["generation"])
def require_shared_python(identity, minor):
    if not identity["python_version"].startswith(minor + "."):
        raise ValueError("prepared interpreter has the wrong Python minor")
    for value in identity["python_paths"].values():
        if not pathlib.Path(value).resolve().is_relative_to(root):
            raise ValueError("prepared interpreter or standard library escapes the shared generation")
wrapper = capture_identity(sys.executable, ["infx"], dataset_loader=None)
require_shared_python(wrapper, "3.12")
native = capture_identity(config["native_python"], ["srtctl"], dataset_loader=None)
require_shared_python(native, "3.12")
write_json(root / "evidence/native-identity.json", native)
import subprocess
capabilities = json.loads(subprocess.run([config["native_python"], "-I", "-m", "srtctl.cli.prepared", "capabilities", "--json"], capture_output=True, text=True, check=True, timeout=60).stdout)
if not set(config["capabilities"]) <= set(capabilities["capabilities"]):
    raise ValueError("installed native runtime lacks required capabilities")
write_json(root / "evidence/native-capabilities.json", capabilities)
verify_wrapper_source(wrapper, pathlib.Path(config["checkout"]))
write_json(root / "evidence/wrapper-identity.json", wrapper)
assets = collect_assets(ClientSite.model_validate(config["sites"]["agentx"]))
write_json(root / "evidence/assets.json", [asset.model_dump() for asset in assets])
write_json(root / "evidence/image.json", bind_file(pathlib.Path(config["image"])).model_dump())
for kind, raw_site in config["sites"].items():
    site = ClientSite.model_validate(raw_site)
    identity = capture_identity(site.python, site.distributions, dataset_loader="semianalysis_cc_traces_weka_062126" if kind == "agentx" else None, env={**__import__("os").environ, **site.env})
    require_shared_python(identity, "3.11" if kind == "agentx" else "3.12")
    require_source_revision(identity, "aiperf" if kind == "agentx" else "lm-eval", AGENTX_REVISION if kind == "agentx" else LM_EVAL_REVISION)
    if kind == "agentx" and identity.get("dataset_resolution", {}).get("metadata", {}).get("hf_dataset_name") != config["dataset_repository"]:
        raise ValueError("AgentX plugin resolves a different dataset")
    output = root / "evidence" / (kind + "-identity.json")
    write_json(output, identity)
    runtime = RuntimeSpec(python=site.python, identity=bind_file(output), distributions=site.distributions, env=site.env, env_unset=site.env_unset, assets=assets, timeout_seconds=site.timeout_seconds, terminate_grace_seconds=site.terminate_grace_seconds)
    verify_model_snapshot_assets(runtime, config["model_repository"], expected_revision=config["model_revision"], expected_snapshot=pathlib.Path(site.model_path))
    verify_snapshot_assets(runtime, config["dataset_repository"], expected_revision=config["dataset_revision"], only_snapshot=True)
    verify_snapshot_assets(runtime, "openai/gsm8k", expected_revision=None, only_snapshot=False)
    write_json(root / "evidence" / (kind + "-runtime.json"), runtime.model_dump())
"""


def require_clean_checkout(commands: Commands, checkout: Path, output: Path) -> str:
    """Allow only the caller's untracked artifact directory, never tracked/source changes."""
    exclusions = []
    if output.is_relative_to(checkout):
        relative = output.relative_to(checkout)
        if (
            relative == Path()
            or output.is_relative_to(checkout / "infx")
            or output.is_relative_to(checkout / ".git")
        ):
            raise ValueError("artifact output must be outside source and Git metadata paths")
        tracked = commands.run(
            "artifact-path-check",
            ["git", "ls-files", "--", str(relative)],
            cwd=checkout,
            timeout=30,
        )
        if tracked:
            raise ValueError("artifact output contains tracked repository files")
        exclusions = [f":(exclude,literal){relative}"]
    dirty = commands.run(
        "candidate-clean",
        ["git", "status", "--porcelain", "--untracked-files=all", "--", ".", *exclusions],
        cwd=checkout,
        timeout=30,
    )
    if dirty:
        raise ValueError("wrapper checkout must be clean before provisioning")
    head = commands.run("candidate-head", ["git", "rev-parse", "HEAD"], cwd=checkout, timeout=30)
    if re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise ValueError("candidate revision must be an immutable full commit")
    return head


def publish_evidence(generation: Path, output: Path) -> None:
    """Retain small review artifacts on success and interrupted/failed installations."""
    output.mkdir(parents=True, exist_ok=True)
    for name in ("evidence", "logs"):
        shutil.copytree(generation / name, output / name, dirs_exist_ok=True)


def provision(
    config: ProvisionConfig, checkout: Path, output: Path, namespace: str
) -> dict[str, Any]:
    """Provision a draft; deployed reader/collector identities are deliberately absent."""
    checkout, output = checkout.resolve(), output.resolve()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("H100 provisioning must run on the shared Linux x86_64 login runner")
    inventory = inspect_assets(config)
    if not inventory["assets_present"]:
        raise ValueError("required existing image, model, or trace assets are incomplete")
    root = Path(config.shared_root)
    if root.is_relative_to(Path(config.hub_cache)) or output.is_relative_to(Path(config.hub_cache)):
        raise ValueError("provisioning may not write into the existing Hugging Face cache")
    lock = RuntimeLock.model_validate(read_json(checkout / NATIVE_LOCK))
    if lock.repository != NATIVE_REPOSITORY:
        raise ValueError("native runtime repository is not allowlisted")
    uv = shutil.which("uv")
    if uv is None:
        raise ValueError("provisioning requires an explicitly installed uv executable")
    with owned_generation(root, namespace) as generation:
        commands = Commands(generation, installer_environment(generation, os.environ))
        try:
            head = require_clean_checkout(commands, checkout, output)
            with Path(config.image_path).open("rb") as image:
                if image.read(4) != b"hsqs":
                    raise ValueError("serving image is not a SquashFS image")
            unsquashfs = shutil.which("unsquashfs")
            if unsquashfs is not None:
                commands.run(
                    "image-superblock",
                    [unsquashfs, "-s", config.image_path],
                    cwd=generation,
                    timeout=60,
                )
            write_json(generation / "evidence/inventory.json", inventory)
            commands.run(
                "managed-python",
                [uv, "python", "install", "3.12", "3.11", "--no-bin"],
                cwd=generation,
                timeout=600,
            )
            source = _native_source(commands, lock)
            shutil.copyfile(source / "uv.lock", generation / "evidence/native-uv.lock")
            shutil.copyfile(checkout / "uv.lock", generation / "evidence/wrapper-uv.lock")
            native_wheel, wrapper_wheel = _wheels(commands, uv, source, checkout)
            wheels = [bind_file(path).model_dump() for path in (native_wheel, wrapper_wheel)]
            write_json(generation / "evidence/wheels.json", wheels)
            interpreters = {}
            for name, directory, wheel in (
                ("native", source, native_wheel),
                ("wrapper", checkout, wrapper_wheel),
            ):
                python = _sync(commands, uv, directory, name, "3.12")
                commands.run(
                    f"install-{name}-wheel",
                    [uv, "pip", "install", "--python", str(python), "--no-deps", str(wheel)],
                    cwd=generation,
                    timeout=300,
                )
                interpreters[name] = python
            commands.run(
                "verify-native-source",
                [
                    str(interpreters["native"]),
                    "-I",
                    "-c",
                    "import pathlib,sys; from srtctl.core.prepared import verify_loader_source; verify_loader_source(pathlib.Path(sys.argv[1]))",
                    str(source),
                ],
                cwd=generation,
                timeout=600,
            )
            if commands.run(
                "native-clean",
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=source,
                timeout=30,
            ):
                raise ValueError("native source changed during installation")
            commands.environment["UV_BUILD_CONSTRAINT"] = str(
                generation / "evidence/build-constraints.txt"
            )
            clients = _clients(commands, uv, generation / "evidence/build-constraints.txt")
            env = _offline_env(generation)
            sites = _sites(config, clients, env)
            verify_clients(commands, config, clients, env)
            script = generation / "materialize-gsm8k.py"
            script.write_text(_GSM_SCRIPT)
            expected = generation / "evidence/gsm8k-test-doc-hashes.json"
            expected.write_bytes(
                files("infx.benchmarks")
                .joinpath("resources/gsm8k-test-doc-hashes.json")
                .read_bytes()
            )
            for mode in ("online", "offline"):
                commands.run(
                    f"gsm8k-{mode}",
                    [
                        str(clients["eval"]),
                        "-I",
                        str(script),
                        mode,
                        str(expected),
                        str(generation / "evidence" / f"gsm8k-{mode}.json"),
                    ],
                    cwd=generation,
                    timeout=600,
                    environment={
                        **env,
                        "HF_HUB_OFFLINE": "0" if mode == "online" else "1",
                        "HF_DATASETS_OFFLINE": "0" if mode == "online" else "1",
                    },
                )
            modules = Path(env["HF_MODULES_CACHE"])
            if any(path.is_file() for path in modules.rglob("*")):
                sites = {
                    name: ClientSite.model_validate(
                        {**site.model_dump(), "asset_roots": [*site.asset_roots, str(modules)]}
                    )
                    for name, site in sites.items()
                }
            verification = generation / "verify-input.json"
            write_json(
                verification,
                {
                    "generation": str(generation),
                    "checkout": str(checkout),
                    "image": config.image_path,
                    "sites": {name: site.model_dump() for name, site in sites.items()},
                    "native_python": str(interpreters["native"]),
                    "capabilities": lock.capabilities,
                    "model_repository": config.model_repository,
                    "model_revision": config.model_revision,
                    "dataset_repository": config.dataset_repository,
                    "dataset_revision": config.dataset_revision,
                },
            )
            probe = generation / "verify-assets.py"
            probe.write_text(_VERIFY_SCRIPT)
            commands.run(
                "hash-and-verify-assets",
                [str(interpreters["wrapper"]), "-I", str(probe), str(verification)],
                cwd=generation,
                timeout=4800,
                environment=env,
            )
            client_sites = {}
            for name, site in sites.items():
                path = generation / f"{name}-site.json"
                write_json(path, site.model_dump())
                path.chmod(0o444)
                client_sites[name] = str(path)
            if require_clean_checkout(commands, checkout, output) != head:
                raise ValueError("candidate checkout changed while provisioning")
            draft = {
                "schema_version": 1,
                "cluster": "h100-dgxc",
                "native_python": str(interpreters["native"]),
                "native_source": str(source),
                "wrapper_python": str(interpreters["wrapper"]),
                "shared_root": str(generation),
                "model_snapshot": str(snapshot(config, dataset=False)),
                "model_revision": config.model_revision,
                "image": read_json(generation / "evidence/image.json"),
                "image_reference": config.image_reference,
                "client_sites": client_sites,
                "mounts": {str(path): str(path) for path in (generation, Path(config.hub_cache))},
            }
            write_json(generation / "site-draft.json", draft)
            report = {
                "schema_version": 1,
                "state": "prepared-draft",
                "generation": str(generation),
                "source_revision": head,
                "native_revision": lock.revision,
                "site_draft": str(generation / "site-draft.json"),
                "client_sites": client_sites,
                "qualification_complete": False,
                "deployment_pins_required": ["reader_revision", "collector_revision"],
            }
            write_json(generation / "state.json", report)
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / "site-draft.json", draft)
            write_json(output / "provisioning.json", report)
            publish_evidence(generation, output)
        except BaseException as error:
            with suppress(OSError):
                publish_evidence(generation, output)
            write_json(
                output / "provisioning.json",
                {
                    "schema_version": 1,
                    "state": "failed",
                    "generation": str(generation),
                    "stage": commands.current_stage,
                    "error_type": type(error).__name__,
                    "qualification_complete": False,
                },
            )
            raise
        return report
