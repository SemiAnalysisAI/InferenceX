"""Run a repository revision's own matrix generator or planner in a subprocess.

Only that revision's tooling interprets its configs, so config-format changes cannot break
consumers of older revisions. The tool imports only its own tree and inherits only
``INHERITED_ENV``. That keeps a caller's credentials out of its environment but does not
isolate it: callers holding credentials must run other revisions' tools where none are held.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from infx.config import MASTER_CONFIGS, git_repository_root, project_root


@dataclass(frozen=True)
class Tool:
    """A matrix entrypoint: its package module, or the script pre-package revisions shipped."""

    module: str
    legacy_script: str

    @property
    def module_path(self) -> str:
        return self.module.replace(".", "/") + ".py"


GENERATOR = Tool("infx.matrix.generate", "utils/matrix_logic/generate_sweep_configs.py")
PLANNER = Tool("infx.matrix.plan", "utils/process_changelog.py")
TOOLS = {"generate": GENERATOR, "plan": PLANNER}
# Fingerprints a revision's rows as its planner does; revisions without it hashed the row alone.
FINGERPRINTER = "infx.matrix.fingerprint"
CONFIG_DIRS = ("configs", ".github/configs")
NESTED_PROJECT = "inferencex-e2e/"
SNAPSHOT_PATHS = (
    "infx",
    "utils/matrix_logic",
    *CONFIG_DIRS,
    "benchmarks/single_node/srt-slurm-recipes",
    "benchmarks/multi_node/srt-slurm-recipes",
)
INHERITED_ENV = ("PATH", "HOME", "LANG", "TMPDIR")


@dataclass(frozen=True)
class Revision:
    """One revision's project tree: its tools' working directory, import root and data root."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())

    @property
    def master_configs(self) -> list[str]:
        return [str(self._configs / Path(path).name) for path in MASTER_CONFIGS]

    @property
    def runner_config(self) -> str:
        return str(self._configs / "runners.yaml")

    @property
    def _configs(self) -> Path:
        return next(
            (
                self.root / name
                for name in CONFIG_DIRS
                if (self.root / name / "runners.yaml").is_file()
            ),
            self.root / CONFIG_DIRS[0],
        )

    def invocation(self, tool: Tool, args: Sequence[str]) -> tuple[list[str], dict[str, str]]:
        """The command and environment that run this tree's own ``tool`` from ``root``."""
        script = self.root / tool.legacy_script
        if (self.root / tool.module_path).is_file():
            entrypoint, imports = ["-m", tool.module], [self.root]
        elif script.is_file():
            entrypoint, imports = [str(script)], [script.parent, self.root]
        else:
            raise ValueError(f"{self.root} has neither {tool.module_path} nor {tool.legacy_script}")
        return [sys.executable, *entrypoint, *args], self._environment(imports)

    def _environment(self, imports: Sequence[Path]) -> dict[str, str]:
        env = {name: os.environ[name] for name in INHERITED_ENV if name in os.environ}
        env["PYTHONPATH"] = os.pathsep.join(map(str, imports))
        env["INFERENCEX_REPOSITORY_ROOT"] = str(self.root)
        return env

    def generate(
        self,
        config_keys: Sequence[str],
        flags: Sequence[str],
        config_files: Sequence[str] | None = None,
    ) -> list[dict]:
        """Decode this revision's ``test-config`` matrix; raise CalledProcessError on failure.

        ``config_files`` default to the revision's master configs.
        """
        command, env = self.invocation(
            GENERATOR,
            [
                "test-config",
                "--config-keys",
                *config_keys,
                "--config-files",
                *(self.master_configs if config_files is None else config_files),
                "--runner-config",
                self.runner_config,
                *flags,
            ],
        )
        result = subprocess.run(
            command, cwd=self.root, env=env, capture_output=True, text=True, check=True
        )
        return json.loads(result.stdout)

    def fingerprints(self, rows: list[dict]) -> list[str]:
        """Each benchmark row's ``recipe-fingerprint`` as this revision's planner assigns it."""
        if not (self.root / f"{FINGERPRINTER.replace('.', '/')}.py").is_file():
            from infx.matrix.fingerprint import row_fingerprint

            return [row_fingerprint(row) for row in rows]
        result = subprocess.run(
            [sys.executable, "-m", FINGERPRINTER],
            cwd=self.root,
            env=self._environment([self.root]),
            input=json.dumps(rows),
            capture_output=True,
            text=True,
            check=True,
        )
        fingerprints = json.loads(result.stdout)
        if not isinstance(fingerprints, list) or len(fingerprints) != len(rows):
            raise ValueError(f"{FINGERPRINTER} did not print one fingerprint per row")
        return fingerprints


@contextmanager
def snapshot(ref: str) -> Iterator[Revision]:
    """Extract ``ref``'s committed generation inputs into a tree removed on exit.

    Blobs are read by the listed object IDs, so a moving ref cannot mix revisions; working-tree
    files never replace committed ones, and symlinks become plain files.
    """
    repository = git_repository_root()
    listing = subprocess.run(
        [
            "git",
            "ls-tree",
            "--full-tree",
            "-r",
            "-z",
            ref,
            "--",
            *SNAPSHOT_PATHS,
            *(NESTED_PROJECT + path for path in SNAPSHOT_PATHS),
        ],
        cwd=repository,
        capture_output=True,
        check=True,
    ).stdout
    objects = {}
    for entry in listing.split(b"\0")[:-1]:
        metadata, path = entry.split(b"\t", 1)
        objects[os.fsdecode(path)] = metadata.split()[2]
    if any(path.startswith(NESTED_PROJECT) for path in objects):
        objects = {
            path.removeprefix(NESTED_PROJECT): oid
            for path, oid in objects.items()
            if path.startswith(NESTED_PROJECT)
        }
    blobs = io.BytesIO(
        subprocess.run(
            ["git", "cat-file", "--batch"],
            input=b"".join(oid + b"\n" for oid in objects.values()),
            cwd=repository,
            capture_output=True,
            check=True,
        ).stdout
    )
    with tempfile.TemporaryDirectory(prefix="infx-revision-") as temp_dir:
        for path in objects:
            header = blobs.readline().split()
            if len(header) != 3 or header[1] != b"blob":
                raise ValueError(f"Could not read {path!r} at {ref!r}: {header!r}")
            content = blobs.read(int(header[2]))
            if blobs.read(1) != b"\n":
                raise ValueError(f"Incomplete Git blob for {path!r} at {ref!r}")
            destination = Path(temp_dir, path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        revision = Revision(Path(temp_dir))
        inputs = [*revision.master_configs, revision.runner_config]
        if missing := [
            Path(path).relative_to(revision.root).as_posix()
            for path in inputs
            if not Path(path).is_file()
        ]:
            raise ValueError(f"revision {ref!r} is missing generation inputs: {missing}")
        yield revision


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a checkout's own matrix generator or planner against its own configs."
    )
    parser.add_argument("tool", choices=TOOLS)
    parser.add_argument("checkout", type=Path, help="repository checkout of the revision")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="arguments for the tool")
    options = parser.parse_args(argv)
    revision = Revision(project_root(options.checkout))
    try:
        command, env = revision.invocation(TOOLS[options.tool], options.args)
    except ValueError as error:
        parser.error(str(error))
    raise SystemExit(subprocess.run(command, cwd=revision.root, env=env, check=False).returncode)


if __name__ == "__main__":
    main()
