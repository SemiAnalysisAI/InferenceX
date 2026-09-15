import hashlib
import json
import socket
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption("--fuzz-examples", type=int, default=100)
    parser.addoption("--fuzz-output", default=".fuzz")


def pytest_configure(config):
    if config.inipath != Path(__file__).with_name("pytest.ini"):
        return
    from hypothesis import settings
    from hypothesis.database import DirectoryBasedExampleDatabase

    count = config.getoption("--fuzz-examples")
    if count < 1:
        raise pytest.UsageError("--fuzz-examples must be positive")
    output = Path(config.getoption("--fuzz-output")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config.option.xmlpath = str(output / "results.xml")
    settings.register_profile("fuzz", max_examples=count, deadline=None, print_blob=True,
                              database=DirectoryBasedExampleDatabase(output / "examples"))
    settings.load_profile("fuzz")
    config._fuzz_output = output


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Fuzz tests must stub external network access")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def source_state():
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, check=True).stdout
    return {
        "commit": git("rev-parse", "HEAD").decode().strip(),
        "diff_sha256": hashlib.sha256(git("diff", "--binary", "HEAD")).hexdigest(),
        "fuzz_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in sorted(Path(__file__).parent.iterdir()) if path.suffix in {".py", ".ini"}},
    }


def pytest_sessionstart(session):
    if hasattr(session.config, "_fuzz_output"):
        session.config._fuzz_start = source_state()


def pytest_sessionfinish(session, exitstatus):
    if output := getattr(session.config, "_fuzz_output", None):
        end = source_state()
        (output / "run.json").write_text(json.dumps({
            "start": session.config._fuzz_start, "end": end,
            "source_changed": session.config._fuzz_start != end, "exit_code": int(exitstatus),
            "arguments": list(session.config.invocation_params.args),
            "python": sys.version, "hypothesis": version("hypothesis"), "pytest": version("pytest"),
        }, indent=2))
