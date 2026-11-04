from __future__ import annotations

import argparse
import io
import json
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from infx.bench.proc import status
from infx.config import MASTER_CONFIGS, git_repository_root, project_root


@dataclass(frozen=True)
class Tool:
    module: str
    legacy_script: str

    @property
    def module_path(self) -> str:
        """The module's file path relative to the revision root."""
        return self.module.replace(".", "/") + ".py"


GENERATOR = Tool("infx.matrix.generate", "utils/matrix_logic/generate_sweep_configs.py")
PLANNER = Tool("infx.matrix.plan", "utils/process_changelog.py")
TOOLS = {"generate": GENERATOR, "plan": PLANNER}
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
_FORK_REPLY = "infx-fork-exit "
_FORK_SERVER = """
import ast, importlib.util, json, os, runpy, sys, types

module, reply = sys.argv[1], sys.argv[2]
sys.path[0] = os.getcwd()
spec = importlib.util.find_spec(module)
spec.loader.get_code(module)
for node in ast.parse(spec.loader.get_source(module)).body:
    if isinstance(node, (ast.Import, ast.ImportFrom)) and getattr(node, "module", "") != "__future__":
        try:
            exec(compile(ast.Module([node], []), spec.origin, "exec"),
                 {"__name__": "__infx_preload__", "__package__": spec.parent})
        except BaseException:
            pass
for line in sys.stdin:
    arguments, output = json.loads(line)
    pid = os.fork()
    if pid == 0:
        null = os.open(os.devnull, os.O_RDWR)
        os.dup2(null, 0)
        os.dup2(null, 2)
        os.dup2(os.open(output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 1)
        sys.argv = [module, *arguments]
        sys.modules["__main__"] = types.ModuleType("__main__")
        runpy._run_module_as_main(module)
        raise SystemExit
    print(f"\\n{reply}{os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1])}", flush=True)
"""

GenerateRequest = tuple[Sequence[str], Sequence[str], Sequence[str] | None]


@dataclass(frozen=True)
class Revision:
    root: Path

    def __post_init__(self) -> None:
        """Resolve ``root`` to an absolute path."""
        object.__setattr__(self, "root", Path(self.root).resolve())

    @property
    def master_configs(self) -> list[str]:
        """Paths of the revision's master config files."""
        return [str(self._configs / Path(path).name) for path in MASTER_CONFIGS]

    @property
    def runner_config(self) -> str:
        """Path of the revision's runner config."""
        return str(self._configs / "runners.yaml")

    @property
    def _configs(self) -> Path:
        """The revision's config directory, the first that holds runners.yaml."""
        return next(
            (
                self.root / name
                for name in CONFIG_DIRS
                if (self.root / name / "runners.yaml").is_file()
            ),
            self.root / CONFIG_DIRS[0],
        )

    def invocation(self, tool: Tool, args: Sequence[str]) -> tuple[list[str], dict[str, str]]:
        """Return the command and environment that run ``tool`` with ``args`` in this revision.

        Prefer the module layout, else the legacy script. Raise ValueError when neither exists.
        """
        script = self.root / tool.legacy_script
        if (self.root / tool.module_path).is_file():
            entrypoint, imports = ["-m", tool.module], [self.root]
        elif script.is_file():
            entrypoint, imports = [str(script)], [script.parent, self.root]
        else:
            raise ValueError(f"{self.root} has neither {tool.module_path} nor {tool.legacy_script}")
        return [sys.executable, *entrypoint, *args], self._environment(imports)

    def _environment(self, imports: Sequence[Path]) -> dict[str, str]:
        """Return a minimal environment with ``imports`` on PYTHONPATH and the revision as root."""
        env = {name: os.environ[name] for name in INHERITED_ENV if name in os.environ}
        env["PYTHONPATH"] = os.pathsep.join(map(str, imports))
        env["INFERENCEX_REPOSITORY_ROOT"] = str(self.root)
        return env

    def _generator_arguments(
        self, config_keys: Sequence[str], flags: Sequence[str], config_files: Sequence[str] | None
    ) -> list[str]:
        """Return the generator's ``test-config`` arguments for these keys, flags, and configs."""
        return [
            "test-config",
            "--config-keys",
            *config_keys,
            "--config-files",
            *(self.master_configs if config_files is None else config_files),
            "--runner-config",
            self.runner_config,
            *flags,
        ]

    def generate(
        self,
        config_keys: Sequence[str],
        flags: Sequence[str],
        config_files: Sequence[str] | None = None,
    ) -> list[dict]:
        """Run the revision's generator for ``config_keys`` and return its rows.

        Raise CalledProcessError when the generator fails.
        """
        command, env = self.invocation(
            GENERATOR, self._generator_arguments(config_keys, flags, config_files)
        )
        result = subprocess.run(
            command, cwd=self.root, env=env, capture_output=True, text=True, check=True
        )
        return json.loads(result.stdout)

    def generate_each(self, requests: Sequence[GenerateRequest], workers: int) -> list[list[dict]]:
        """``generate(*request)`` for every request, in order.

        A module-layout revision runs each request in a fork of a process that already imported
        its generator's imports, so a request skips interpreter start-up and imports. A request
        that does not exit 0 there is run again as its own process, which raises what
        ``generate`` raises. A fork reads stdin from /dev/null and writes stdout to a file, and
        shares state its preloaded imports created; generators that depend on any of that need
        ``INFX_FORK_GENERATORS=0``, which runs every request as its own process.
        """
        if (
            os.environ.get("INFX_FORK_GENERATORS") == "0"
            or not hasattr(os, "fork")
            or not (self.root / GENERATOR.module_path).is_file()
        ):
            with ThreadPoolExecutor(workers) as pool:
                return list(pool.map(lambda request: self.generate(*request), requests))
        with tempfile.TemporaryDirectory(prefix="infx-generate-") as outputs, ExitStack() as stack:
            started = [
                stack.enter_context(_ForkServer(self, GENERATOR))
                for _ in range(max(1, min(workers, len(requests))))
            ]
            servers: queue.SimpleQueue[_ForkServer] = queue.SimpleQueue()
            for server in started:
                servers.put(server)
            stopping = threading.Event()

            def run(index: int, request: GenerateRequest) -> list[dict]:
                """Return request ``index``'s rows from a fork, or its own process on failure."""
                output = Path(outputs, f"{index}.json")
                server = servers.get()
                try:
                    code = server.run(self._generator_arguments(*request), output)
                finally:
                    servers.put(server)
                if stopping.is_set():
                    raise RuntimeError("generate_each was interrupted")
                if code != 0:
                    return self.generate(*request)
                return json.loads(output.read_text())

            pool = ThreadPoolExecutor(workers)
            try:
                return list(pool.map(run, range(len(requests)), requests))
            except BaseException:
                stopping.set()
                for server in started:
                    server.kill()
                raise
            finally:
                pool.shutdown(wait=True, cancel_futures=True)

    def fingerprints(self, rows: list[dict]) -> list[str]:
        """Return the revision's recipe fingerprint of each row, or row digests if it has none.

        Raise ValueError when the fingerprinter does not print one fingerprint per row.
        """
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


class _ForkServer:
    """A process in ``revision``'s environment that preloads ``tool``'s imports and forks.

    Each fork runs the tool as ``__main__`` with stdout in the request's output file.
    """

    def __init__(self, revision: Revision, tool: Tool) -> None:
        """Start the server in its own session, so a terminal Ctrl-C does not reach its forks.

        Requests are JSON ``[argv, output path]`` lines. Each exit status follows a marker, so
        output from importing the tool is never taken for a reply.
        """
        _, env = revision.invocation(tool, [])
        self._process = subprocess.Popen(
            [sys.executable, "-c", _FORK_SERVER, tool.module, _FORK_REPLY],
            cwd=revision.root,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("the fork server was started without pipes")
        self._requests, self._replies = self._process.stdin, self._process.stdout
        self._noisy = False

    def __enter__(self) -> Self:
        """Return the server itself."""
        return self

    def kill(self) -> None:
        """Kill the server and its forks, ignoring a server that already exited."""
        with suppress(ProcessLookupError):
            os.killpg(self._process.pid, signal.SIGKILL)

    def __exit__(self, error_type: type[BaseException] | None, *_: object) -> None:
        """Close the server's stdin and wait for it to exit, killing it first on an error."""
        if error_type is not None:
            self.kill()
        with suppress(OSError):
            self._requests.close()
        self._process.wait()
        self._replies.close()

    def run(self, arguments: Sequence[str], output: Path) -> int | None:
        """The fork's exit status, or None when the server is gone, answered garbage, or its
        imports wrote to stdout, which a fresh process would put before every request's rows."""
        try:
            request = [[os.fsdecode(argument) for argument in arguments], str(output)]
            self._requests.write(json.dumps(request) + "\n")
            self._requests.flush()
            for line in self._replies:
                if line.startswith(_FORK_REPLY):
                    return None if self._noisy else int(line.removeprefix(_FORK_REPLY))
                self._noisy = self._noisy or line != "\n"
        except (OSError, ValueError):
            return None
        return None


@contextmanager
def snapshot(ref: str) -> Iterator[Revision]:
    """Extract the matrix inputs at git ``ref`` into a temporary directory and yield its Revision.

    Raise ValueError when a blob cannot be read or the configs needed to generate are missing.
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
    """Run a checkout's generator or planner with its own configs and exit with its status."""
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
    raise SystemExit(
        status(subprocess.run(command, cwd=revision.root, env=env, check=False).returncode)
    )


if __name__ == "__main__":
    main()
