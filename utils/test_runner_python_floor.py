"""Keep runner-invoked infx entrypoints importable on the runner's python3.

The GB200/GB300 srt-slurm launchers unset VIRTUAL_ENV and call these modules
through the login node's system interpreter, which lags the project's declared
3.12 floor. A name that only exists in a newer standard library raises at
import time, so the job dies after the benchmark has already run.
"""

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

# Shell entrypoints that reach infx through `python3 -m` rather than the venv.
RUNNER_ENTRYPOINTS = (
    "infx.results.agentic.power_adapter",
    "infx.srt_slurm.cluster_config",
    "infx.srt_slurm.synthetic_acceptance",
)

# Standard-library names the runner interpreter may be too old to provide.
TOO_NEW = {"datetime": {"UTC"}}


def _module_file(name):
    direct = ROOT / (name.replace(".", "/") + ".py")
    if direct.exists():
        return direct
    package = ROOT / name.replace(".", "/") / "__init__.py"
    return package if package.exists() else None


def _import_closure(entrypoint):
    """Walk infx-internal imports so a dependency cannot reintroduce the break."""
    pending, seen = [entrypoint], set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        path = _module_file(name)
        if path is None:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("infx"):
                pending.append(node.module)
                pending.extend(f"{node.module}.{alias.name}" for alias in node.names)
            elif isinstance(node, ast.Import):
                pending.extend(a.name for a in node.names if a.name.startswith("infx"))
        parts = name.split(".")
        pending.extend(".".join(parts[:index]) for index in range(1, len(parts)))
    return sorted(seen)


@pytest.mark.parametrize("entrypoint", RUNNER_ENTRYPOINTS)
def test_runner_entrypoints_avoid_names_the_runner_interpreter_lacks(entrypoint):
    offenders = []
    for name in _import_closure(entrypoint):
        path = _module_file(name)
        if path is None:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module in TOO_NEW:
                for alias in node.names:
                    if alias.name in TOO_NEW[node.module]:
                        offenders.append(
                            f"{path.relative_to(ROOT)}:{node.lineno} imports "
                            f"{node.module}.{alias.name}"
                        )

    assert not offenders, "\n".join(offenders)


def test_the_closure_reaches_transitive_infx_dependencies():
    closure = _import_closure("infx.results.agentic.power_adapter")

    assert "infx.results.power.single_node" in closure
