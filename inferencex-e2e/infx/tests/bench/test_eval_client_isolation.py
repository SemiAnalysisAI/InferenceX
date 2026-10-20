"""The lm-eval client: which interpreter runs it, its locked environment and receipt, and its uv."""

import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path

import pytest

from infx.bench.env import BenchError
from infx.bench.eval import lm_eval
from infx.bench.eval.context import EvalContext
from infx.tests.bench.stubs import executable
from infx.tests.bench.test_eval_command import STUBS, UV, trace

HASH = " \\\n    --hash=sha256:"
TORCH_LOCK = f"torch==2.10.0+cpu{HASH}{'a' * 64}\n    # via -r req_torch.in\n"
CLIENT_LOCK = (
    f"huggingface-hub==1.33.0{HASH}{'b' * 64}{HASH}{'c' * 64}\n    # via\n    #   datasets\n"
    f"transformers==5.16.1{HASH}{'d' * 64}\n    # via -r req_pypi.in\n"
)
PINS = {"torch": "2.10.0+cpu", "huggingface-hub": "1.33.0", "transformers": "5.16.1"}
BUILD = ["--version", "venv", "pip", "pip", "pip"]
IMAGE_UV = "#!/bin/sh\necho 'uv 0.9.0 (x86_64-unknown-linux-gnu)'\n"
CURL = f"""#!{sys.executable}
import json, os, shutil, sys
args = sys.argv[1:]
with open(os.environ["STUB_CURL_TRACE"], "a") as trace:
    trace.write(json.dumps(args) + "\\n")
shutil.copyfile(os.environ["STUB_CURL_SERVES"], args[args.index("-o") + 1])
"""


@pytest.fixture
def tmp(tmp_path, monkeypatch):
    """The temporary directory that holds the client environment and its lock."""
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    return tmp


@pytest.fixture
def environ(tmp_path, tmp, monkeypatch):
    """An eval's environment: the recording uv alone on PATH, the lm-eval stubs, two small locks."""
    locks = tmp_path / "locks"
    locks.mkdir()
    (locks / "torch.lock").write_text(TORCH_LOCK)
    (locks / "client.lock").write_text(CLIENT_LOCK)
    monkeypatch.setattr(lm_eval, "CLIENT_LOCKS", (
        (str(locks / "torch.lock"), "https://download.pytorch.org/whl/cpu"),
        (str(locks / "client.lock"), "https://pypi.org/simple"),
    ))  # fmt: skip
    tools = tmp_path / "tools"
    tools.mkdir()
    executable(tools / "uv", UV)
    stubs = tmp_path / "stubs"
    for name, body in STUBS.items():
        (stubs / name).parent.mkdir(parents=True, exist_ok=True)
        (stubs / name).write_text(body)
    return {
        "PATH": str(tools),
        "HOME": str(tmp_path),
        "PYTHONPATH": str(stubs),
        "OPENAI_API_KEY": "EMPTY",
        "STUB_UV_TRACE": str(tmp_path / "uv.jsonl"),
        "STUB_LM_EVAL_TRACE": str(tmp_path / "lm_eval.jsonl"),
    }


@pytest.fixture
def release(environ, tmp_path):
    """``environ`` with the image's uv 0.9.0 on PATH and a curl serving ``served.tar.gz``."""
    tools = Path(environ["PATH"])
    executable(tools / "uv", IMAGE_UV)
    executable(tools / "curl", CURL)
    return {
        **environ,
        "STUB_CURL_TRACE": str(tmp_path / "curl.jsonl"),
        "STUB_CURL_SERVES": str(tmp_path / "served.tar.gz"),
    }


def distribution(site: Path, name: str, version: str, direct_url: dict | None = None) -> None:
    info = site / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    if direct_url:
        (info / "direct_url.json").write_text(json.dumps(direct_url))


def this_interpreter(tmp_path: Path, source: str, versions: dict, exit_code: int = 0) -> Path:
    """An interpreter with lm-eval from ``source`` over ``versions`` that exits ``exit_code``."""
    site = tmp_path / "this-site-packages"
    for name, version in versions.items():
        distribution(site, name, version)
    if source.startswith("https://"):
        direct_url = {"url": source, "archive_info": {}}
    else:
        direct_url = {"url": f"{lm_eval.REPOSITORY}.git", "vcs_info": {"vcs": "git", "commit_id": source}}
    distribution(site, "lm-eval", "0.4.9.2", direct_url)
    run = f'"{sys.executable}" "$@"\nexit {exit_code}' if exit_code else f'exec "{sys.executable}" "$@"'
    return executable(
        tmp_path / "this-python",
        f'#!/bin/sh\nexport STUB_CLIENT=this PYTHONPATH="{site}${{PYTHONPATH:+:$PYTHONPATH}}"\n{run}\n',
    )


def serve(tmp_path: Path, arch: str) -> str:
    """Serve a uv release archive for ``arch`` holding the recording uv; return its sha256."""
    archive = tmp_path / "served.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, text in (("uvx", "#!/bin/sh\n"), ("uv", UV)):
            data = text.encode()
            info = tarfile.TarInfo(f"uv-{arch}-unknown-linux-gnu/{name}")
            info.size, info.mode = len(data), 0o755
            tar.addfile(info, io.BytesIO(data))
    return hashlib.sha256(archive.read_bytes()).hexdigest()


def context(environ: dict, results: Path) -> EvalContext:
    return EvalContext(
        base_url="http://127.0.0.1:9", model="served-model", concurrency=1, context_length=8192,
        results_dir=results, suite=None, env=environ,
    )  # fmt: skip


def steps(tmp_path: Path) -> list[str]:
    """The uv subcommands run so far."""
    return [call["argv"][0] for call in trace(tmp_path / "uv.jsonl")]


def environments(tmp: Path) -> list[str]:
    """Client environments in the temporary directory, finished or not."""
    return sorted(path.name for path in tmp.iterdir() if path.is_dir())


def blocked_flocks() -> int:
    """This process's flock(2) requests waiting for another holder."""
    return sum(
        fields[1:3] == ["->", "FLOCK"] and fields[5] == str(os.getpid())
        for fields in (line.split() for line in Path("/proc/locks").read_text().splitlines())
    )


def wait_until(ready, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while not ready():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


@pytest.mark.parametrize(("exit_code", "source", "versions", "selected"), [
    (0, "abc1234", PINS, True),
    (0, f"{lm_eval.REPOSITORY}/archive/abc1234.tar.gz", PINS, True),
    (3, "abc1234", PINS, False),
    (0, "xabc1234y", PINS, False),
    (0, "abc1234", {**PINS, "transformers": "5.16.0"}, False),
])  # fmt: skip
def test_this_interpreter_runs_the_client_only_at_exactly_ref_over_exactly_the_locks(
    environ, tmp_path, monkeypatch, exit_code, source, versions, selected
):
    monkeypatch.setattr(lm_eval, "REF", "abc1234")
    this = this_interpreter(tmp_path, source, versions, exit_code)
    monkeypatch.setattr(sys, "executable", str(this))

    lm_eval.install(environ)

    receipt = json.loads(environ[lm_eval.CLIENT_RECEIPT])
    assert (environ[lm_eval.CLIENT_PYTHON] == str(this), receipt["isolated"]) == (selected, not selected)
    assert steps(tmp_path) == ([] if selected else BUILD)
    assert (receipt["lm_eval_ref"], receipt["versions"]) == ("abc1234", PINS)


@pytest.mark.parametrize("trouble", [
    {"STUB_UV_FAIL": "^venv "},
    {"STUB_UV_FAIL": r"client\.lock"},
    {"STUB_UV_FAIL": "lm-evaluation-harness"},
    {"STUB_UV_SKEW": "transformers==5.16.0"},
])  # fmt: skip
def test_a_failed_isolated_install_raises_and_run_does_not_fall_back(environ, tmp, tmp_path, trouble):
    environ = {**environ, **trouble, lm_eval.CLIENT_PYTHON: sys.executable}

    with pytest.raises(BenchError):
        lm_eval.install(environ)

    assert lm_eval.CLIENT_PYTHON not in environ
    assert environments(tmp) == []
    results = tmp_path / "results"
    results.mkdir()
    with pytest.raises(BenchError, match="no verified lm-eval client"):
        lm_eval.run(context(environ, results))
    assert trace(tmp_path / "lm_eval.jsonl") == []
    assert list(results.iterdir()) == []


@pytest.mark.parametrize("tamper", ["no receipt", "lock changed", "environment removed", "receipt replaced"])
def test_run_refuses_a_client_its_receipt_no_longer_verifies(environ, tmp_path, tamper):
    lm_eval.install(environ)
    root = Path(environ[lm_eval.CLIENT_PYTHON]).parent.parent
    if tamper == "no receipt":
        del environ[lm_eval.CLIENT_RECEIPT]
    elif tamper == "lock changed":
        Path(lm_eval.CLIENT_LOCKS[1][0]).write_text(CLIENT_LOCK.replace("5.16.1", "5.16.2"))
    elif tamper == "environment removed":
        shutil.rmtree(root)
    else:
        (root / lm_eval.RECEIPT).write_text(environ[lm_eval.CLIENT_RECEIPT].replace('"utc": "', '"utc": "1'))
    results = tmp_path / "results"
    results.mkdir()

    with pytest.raises(BenchError):
        lm_eval.run(context(environ, results))

    assert trace(tmp_path / "lm_eval.jsonl") == []
    assert list(results.iterdir()) == []


def test_run_records_the_receipt_and_runs_its_interpreter(environ, tmp_path, monkeypatch):
    this = this_interpreter(tmp_path, lm_eval.REF, PINS)
    monkeypatch.setattr(sys, "executable", str(this))
    lm_eval.install(environ)
    results = tmp_path / "results"
    results.mkdir()

    outcome = lm_eval.run(context({**environ, "STUB_LM_EVAL_FAIL_AT": "1"}, results))

    [ran] = trace(tmp_path / "lm_eval.jsonl")
    assert (outcome.returncode, ran["client"]) == (3, "this")
    client = json.loads((results / "eval_client.json").read_text())
    assert {key: client[key] for key in ("verified", "isolated", "python", "lm_eval_ref", "receipt")} == {
        "verified": True, "isolated": False, "python": str(this), "lm_eval_ref": lm_eval.REF, "receipt": None,
    }  # fmt: skip
    assert client["versions"] == PINS
    assert client["receipt_sha256"] == hashlib.sha256(environ[lm_eval.CLIENT_RECEIPT].encode()).hexdigest()


@pytest.mark.parametrize("damage", ["no receipt", "truncated receipt", "lm-eval removed"])
def test_an_environment_without_a_valid_receipt_is_rebuilt(environ, tmp_path, damage):
    first = dict(environ)
    lm_eval.install(first)
    root = Path(first[lm_eval.CLIENT_PYTHON]).parent.parent
    if damage == "no receipt":
        (root / lm_eval.RECEIPT).unlink()
    elif damage == "truncated receipt":
        (root / lm_eval.RECEIPT).write_text(first[lm_eval.CLIENT_RECEIPT][:40])
    else:
        shutil.rmtree(root / "site-packages" / "lm_eval-0.4.9.2.dist-info")
    (root / "bin" / "leftover").touch()
    built = len(steps(tmp_path))

    second = dict(environ)
    lm_eval.install(second)

    assert steps(tmp_path)[built:] == BUILD
    assert not (root / "bin" / "leftover").exists()
    assert second[lm_eval.CLIENT_PYTHON] == first[lm_eval.CLIENT_PYTHON]
    assert json.loads((root / lm_eval.RECEIPT).read_text())["verified"] is True


def test_a_verified_environment_is_reused_without_reinstalling(environ, tmp_path):
    first = dict(environ)
    lm_eval.install(first)
    built = trace(tmp_path / "uv.jsonl")

    second = dict(environ)
    lm_eval.install(second)

    assert [call["argv"][0] for call in built] == BUILD
    assert trace(tmp_path / "uv.jsonl") == built
    assert second == first


def test_concurrent_launchers_build_once_and_both_get_the_verified_environment(environ, tmp_path):
    hold = tmp_path / "hold"
    hold.mkdir()
    launchers = [{**environ, "STUB_UV_HOLD": str(hold)} for _ in range(2)]
    errors = []

    def launch(launched):
        try:
            lm_eval.install(launched)
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=launch, args=(launched,)) for launched in launchers]
    try:
        threads[0].start()
        assert wait_until((hold / "building").exists)
        threads[1].start()
        assert wait_until(lambda: blocked_flocks() > 0)
    finally:
        (hold / "release").touch()
        for thread in threads:
            thread.join(60)

    assert errors == []
    assert steps(tmp_path) == BUILD
    first, second = launchers
    assert first[lm_eval.CLIENT_PYTHON] == second[lm_eval.CLIENT_PYTHON]
    assert first[lm_eval.CLIENT_RECEIPT] == second[lm_eval.CLIENT_RECEIPT]
    assert json.loads(second[lm_eval.CLIENT_RECEIPT])["verified"] is True


def test_the_installer_ignores_uv_and_pip_settings_and_project_config(environ, tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    (project / "uv.toml").write_text('index-url = "http://127.0.0.1:9/simple"\n')
    (project / "pyproject.toml").write_text('[tool.uv]\nconstraint-dependencies = ["transformers<5"]\n')
    monkeypatch.chdir(project)
    polluted = {
        "UV_OVERRIDE": str(project / "overrides.txt"),
        "UV_CONSTRAINT": str(project / "constraints.txt"),
        "UV_CONFIG_FILE": str(project / "uv.toml"),
        "UV_INDEX_URL": "http://127.0.0.1:9/simple",
        "PIP_INDEX_URL": "http://127.0.0.1:9/simple",
        "PIP_CONSTRAINT": str(project / "constraints.txt"),
    }

    lm_eval.install({**environ, **polluted})

    calls = trace(tmp_path / "uv.jsonl")
    assert [call["argv"][0] for call in calls] == BUILD
    assert [key for call in calls for key in call["env"] if key.startswith(("UV_", "PIP_"))] == []
    assert [call["cwd"] for call in calls if call["cwd"] == str(project)] == []
    assert all("--no-config" in call["argv"] for call in calls[1:])


@pytest.mark.parametrize("arch", ["x86_64", "aarch64"])
def test_a_digest_verified_release_archive_supplies_the_client_uv(release, tmp_path, monkeypatch, arch):
    monkeypatch.setitem(lm_eval.CLIENT_UV_SHA256, arch, serve(tmp_path, arch))
    monkeypatch.setattr(lm_eval.platform, "machine", lambda: arch)

    lm_eval.install(release)

    assert [call[-1] for call in trace(tmp_path / "curl.jsonl")] == [
        f"https://github.com/astral-sh/uv/releases/download/{lm_eval.CLIENT_UV}/uv-{arch}-unknown-linux-gnu.tar.gz",
    ]
    assert steps(tmp_path) == BUILD
    assert json.loads(release[lm_eval.CLIENT_RECEIPT])["uv"] == lm_eval.CLIENT_UV


@pytest.mark.parametrize(("pinned", "version", "ran", "error"), [
    (False, lm_eval.CLIENT_UV, [], "has sha256"),
    (True, "0.12.4", ["--version"], "does not hold uv"),
])  # fmt: skip
def test_a_release_archive_is_refused_unless_its_digest_and_uv_match(
    release, tmp, tmp_path, monkeypatch, pinned, version, ran, error
):
    digest = serve(tmp_path, "x86_64")
    if pinned:
        monkeypatch.setitem(lm_eval.CLIENT_UV_SHA256, "x86_64", digest)
    monkeypatch.setattr(lm_eval.platform, "machine", lambda: "x86_64")

    with pytest.raises(BenchError, match=error):
        lm_eval.install({**release, "STUB_UV_VERSION": version})

    assert steps(tmp_path) == ran
    assert environments(tmp) == []


def test_a_machine_without_a_uv_release_archive_cannot_build_the_client(release, tmp, tmp_path, monkeypatch):
    monkeypatch.setattr(lm_eval.platform, "machine", lambda: "riscv64")

    with pytest.raises(BenchError, match="'riscv64'"):
        lm_eval.install(release)

    assert trace(tmp_path / "curl.jsonl") == []
    assert environments(tmp) == []
