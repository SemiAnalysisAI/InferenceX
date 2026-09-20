"""Keep runner-invoked infx entrypoints importable on the runner's python3.

The srt-slurm launchers and the AMD multi-node scripts unset VIRTUAL_ENV and
call these modules through the host's system interpreter, which lags the
project's declared 3.12 floor. A name that only exists in a newer standard
library raises at import time, so the job dies after the benchmark has already
run and the allocation is already spent.

The entrypoint list is discovered from the shell scripts rather than written
out by hand: a hand-kept list silently goes stale the moment a script starts
invoking another module, which is exactly how
`infx.results.power.native_multinode` stayed broken after the first fix.
"""

import ast
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

# Directories holding the shell scripts that reach infx through `python3 -m`.
ENTRYPOINT_SEARCH_DIRS = ("runners", "benchmarks")

_MODULE_INVOCATION = re.compile(r"python3?\s+-m\s+(infx[A-Za-z0-9_.]*)")

# Standard-library names the runner interpreter may be too old to provide.
TOO_NEW = {"datetime": {"UTC"}}


def _discover_entrypoints():
    found = set()
    for directory in ENTRYPOINT_SEARCH_DIRS:
        for path in (ROOT / directory).rglob("*"):
            if not path.is_file():
                continue
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            found.update(_MODULE_INVOCATION.findall(text))
    return sorted(name for name in found if _module_file(name) is not None)


def _module_file(name):
    direct = ROOT / (name.replace(".", "/") + ".py")
    if direct.exists():
        return direct
    package = ROOT / name.replace(".", "/") / "__init__.py"
    return package if package.exists() else None


def _resolve_relative(current, node):
    """Turn `from . import x` / `from .mod import y` into absolute module names."""
    path = _module_file(current)
    parts = current.split(".")
    is_package = path is not None and path.name == "__init__.py"
    base = parts if is_package else parts[:-1]
    if node.level > 1:
        base = base[: len(base) - (node.level - 1)]
    if not base:
        return []
    prefix = ".".join(base)
    if node.module:
        prefix = f"{prefix}.{node.module}"
    return [prefix] + [f"{prefix}.{alias.name}" for alias in node.names]


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
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    pending.extend(_resolve_relative(name, node))
                elif (node.module or "").startswith("infx"):
                    pending.append(node.module)
                    pending.extend(f"{node.module}.{alias.name}" for alias in node.names)
            elif isinstance(node, ast.Import):
                pending.extend(a.name for a in node.names if a.name.startswith("infx"))
        parts = name.split(".")
        pending.extend(".".join(parts[:index]) for index in range(1, len(parts)))
    return sorted(seen)


RUNNER_ENTRYPOINTS = _discover_entrypoints()


def test_discovery_finds_the_known_runner_entrypoints():
    """A silent regex miss would make every check below vacuously pass."""
    assert len(RUNNER_ENTRYPOINTS) >= 5
    for expected in (
        "infx.results.agentic.power_adapter",
        "infx.results.power.native_multinode",
        "infx.srt_slurm.cluster_config",
    ):
        assert expected in RUNNER_ENTRYPOINTS


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
    assert not offenders, (
        f"{entrypoint} reaches a name the runner's system python3 may lack: "
        + "; ".join(offenders)
    )


def test_closure_reaches_transitive_dependencies():
    """Guard the walker itself: single_node is only reachable through an import."""
    closure = _import_closure("infx.results.agentic.power_adapter")
    assert "infx.results.power.single_node" in closure


def test_closure_follows_relative_imports():
    """native_multinode imports its siblings relatively; those must be walked too."""
    closure = _import_closure("infx.results.power.native_multinode")
    assert "infx.results.power.common" in closure
