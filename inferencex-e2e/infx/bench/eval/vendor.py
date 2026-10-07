"""Vendor verifier evals: one runner drives each provider's pinned adapter in ``infx/evals/``."""

from __future__ import annotations

import gzip
import os
import shutil
import sys
import tarfile
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from infx.bench import env, proc, uv
from infx.bench.eval.context import EvalContext, EvalOutcome

EVALS = proc.REPO_ROOT / "infx" / "evals"
KIMI_VERIFIER_REPO = "https://github.com/MoonshotAI/Kimi-Vendor-Verifier.git"
KIMI_VERIFIER_REF = "3dad65a760a8867cda72f6dd8848d876a4e851b4"
KIMI_VERIFIER_ARCHIVE_SHA256 = "ede9ea300c72ccfde9d8975ea4b1b54e423c7625690f6631ab1e65a715821e01"
KIMI_REQUIREMENTS = (
    "httpx[http2]==0.28.1",
    "openai==2.14.0",
    "jsonschema==4.25.1",
    "pytest==8.4.2",
)
MINIMAX_REQUIREMENTS = (
    "jsonschema==4.25.1",
    "loguru==0.7.3",
    "megfile==4.2.5",
    "numpy==2.3.4",
    "openai==2.7.1",
    "tqdm==4.67.1",
)
BFCL_INSTALL_TIMEOUT_S = 600
BFCL_ARCHIVE = "bfcl_upstream_artifacts.tar.gz"

_PROVISION = "Python runtime preparation"
_VERSION_CHECK = "import sys; raise SystemExit(sys.version_info < (3, int(sys.argv[1])))"


class StepError(Exception):
    """A setup step failed; ``returncode`` becomes the eval's exit code."""

    def __init__(self, message: str, returncode: int) -> None:
        super().__init__(message)
        self.returncode = returncode


@dataclass(frozen=True)
class Suite:
    """One selectable suite of a provider."""

    label: str
    """Names the verifier in logs and integration-error messages."""
    adapter: str
    """Script in ``infx/evals/``."""
    results: tuple[str, ...]
    """Globs of the adapter's own report; all matching means it recorded the outcome itself."""
    requirements: tuple[str, ...] = ()
    timeout_s: int | None = None
    """Adapter deadline; expiry kills it with exit code 124."""
    finalize: Callable[[Job], int] | None = None
    """Runs after the adapter, even a failed one; nonzero fails a passing run."""


@dataclass(frozen=True)
class Provider:
    """How to provision, prepare, and drive one provider's adapter."""

    default_suite: str
    suites: Mapping[str, Suite]
    python_minor: int
    """The verifier interpreter must be Python ``3.<python_minor>`` or newer."""
    prepare: Callable[[Job], tuple[list[str], dict[str, str]]]
    """Installs the pinned runtime; returns extra adapter arguments and environment."""
    system_site_packages: bool = False
    """Run the adapter in a venv that also sees the image's site-packages."""
    suite_flag: str | None = None
    """Adapter flag naming the suite; ``None`` when each suite has its own adapter."""
    run_command: tuple[str, ...] = ()
    failure_command: tuple[str, ...] = ()
    """Leading adapter arguments that write an integration-error result."""
    message_flag: str = "--integration-error"


@dataclass
class Job:
    """One suite run, shared by the runner and the provider hooks."""

    provider: Provider
    ctx: EvalContext
    suite: str
    spec: Suite
    scratch: Path
    """Private directory for runtimes and checkouts; deleted after the run."""
    python: str = "python3"
    """The verifier interpreter; the image python3 (3.10 floor) until provisioning replaces it."""

    def step(
        self,
        stage: str,
        argv: list[str],
        *,
        environ: Mapping[str, str] | None = None,
        timeout_s: int | None = None,
    ) -> None:
        """Run one setup command; failure ends the run with an integration error."""
        rc = proc.call(argv, self.ctx.env if environ is None else environ, timeout=timeout_s)
        if rc:
            raise StepError(f"{self.spec.label} {stage} failed with exit code {rc}", rc)

    def adapter_argv(self, command: tuple[str, ...], *args: str) -> list[str]:
        suite = [self.provider.suite_flag, self.suite] if self.provider.suite_flag else []
        return [
            self.python, str(EVALS / self.spec.adapter), *command, *args,
            "--model", self.ctx.model, "--output-dir", str(self.ctx.results_dir), *suite,
        ]  # fmt: skip

    def published(self) -> bool:
        results = self.ctx.results_dir
        return all(any(p.is_file() for p in results.glob(glob)) for glob in self.spec.results)


def run(provider: Provider, ctx: EvalContext) -> EvalOutcome:
    """Run ``ctx.suite``, else the provider's default suite."""
    suite = ctx.suite or provider.default_suite
    spec = provider.suites.get(suite)
    if spec is None:
        title = provider.suites[provider.default_suite].label
        raise env.InputError(f"unsupported {title} suite {suite!r}")
    with tempfile.TemporaryDirectory(
        prefix="infx-vendor-eval-", ignore_cleanup_errors=True
    ) as scratch:
        job = Job(provider, ctx, suite, spec, Path(scratch))
        try:
            job.python = _provision(job)
            args, adapter_env = provider.prepare(job)
        except StepError as failure:
            _publish_failure(job, str(failure))
            return EvalOutcome(failure.returncode, suite)
        rc = proc.call(
            job.adapter_argv(provider.run_command, *args, "--base-url", f"{ctx.base_url}/v1"),
            {**ctx.env, **adapter_env},
            timeout=spec.timeout_s,
        )
        finalized = spec.finalize(job) if spec.finalize else 0
        if rc and not job.published():
            _publish_failure(job, f"{spec.label} evaluation failed with exit code {rc}")
    return EvalOutcome(rc or finalized, suite)


def _publish_failure(job: Job, message: str) -> None:
    """Have the adapter record ``message`` as a zero-score integration-error result."""
    print(f"ERROR: {message}", file=sys.stderr)
    provider = job.provider
    argv = [*job.adapter_argv(provider.failure_command), provider.message_flag, message]
    rc = proc.call(argv, job.ctx.env)
    if not job.published():
        print(
            f"ERROR: failed to write the {job.spec.label} failure artifact (exit code {rc})",
            file=sys.stderr,
        )


def _provision(job: Job) -> str:
    """Return the verifier interpreter, building a uv venv when the image python3 will not do."""
    try:
        installer = uv.find()
    except env.BenchError as error:
        raise _provision_failed(job, str(error)) from error
    minor, site_packages = job.provider.python_minor, job.provider.system_site_packages
    # uv reads a bare "python3" as any Python 3, so name the image's own interpreter.
    python3 = shutil.which(job.python, path=job.ctx.env.get("PATH")) or job.python
    new_enough = proc.call([python3, "-c", _VERSION_CHECK, str(minor)], job.ctx.env) == 0
    if new_enough and not site_packages:
        return python3
    root = job.scratch / "python"
    venv = root / "venv"
    site = ["--system-site-packages"] if site_packages else []
    uv_env = {
        **job.ctx.env,
        "UV_CACHE_DIR": str(root / "uv-cache"),
        "UV_PYTHON_INSTALL_DIR": str(root / "python"),
    }
    base = python3 if new_enough else f"3.{minor}"
    job.step(_PROVISION, [installer, "venv", "--python", base, *site, str(venv)], environ=uv_env)
    python = venv / "bin" / "python"
    if not os.access(python, os.X_OK):
        raise _provision_failed(job, f"{_PROVISION} did not create {python}")
    return str(python)


def _provision_failed(job: Job, detail: str) -> StepError:
    print(f"ERROR: {detail}", file=sys.stderr)
    return StepError(f"{job.spec.label} {_PROVISION} failed with exit code 1", 1)


def _pip_target(job: Job, target: Path) -> list[str]:
    """Install the suite's pinned requirements into ``target``, outside the interpreter."""
    return uv.pip(
        "install", "-q", "--no-cache", "--target", str(target), *job.spec.requirements,
        python=job.python,
    )  # fmt: skip


def _prepare_kimi(job: Job) -> tuple[list[str], dict[str, str]]:
    runtime = job.scratch / "kimi-runtime"
    job.step("dependency installation", _pip_target(job, runtime))
    checkout = job.scratch / "kimi-verifier"
    checkout.mkdir()
    job.step("checkout", [
        job.python, str(EVALS / "_kimi_verifier_archive.py"),
        KIMI_VERIFIER_REPO, KIMI_VERIFIER_REF, KIMI_VERIFIER_ARCHIVE_SHA256, str(checkout),
    ])  # fmt: skip
    args = ["--verifier-dir", str(checkout)]
    if model_prefix := env.optional("MODEL_PREFIX", job.ctx.env):
        args += ["--model-prefix", model_prefix]
    return args, {"PYTHONPATH": os.pathsep.join([str(runtime), proc.pythonpath(job.ctx.env)])}


def _prepare_minimax(job: Job) -> tuple[list[str], dict[str, str]]:
    source = job.scratch / "minimax-source"
    dependencies = job.scratch / "minimax-deps"
    stage = "pinned runtime preparation"
    job.step(stage, [
        job.python, str(EVALS / "minimax_m3_full_eval.py"),
        "prepare-source", "--source-dir", str(source),
    ])  # fmt: skip
    job.step(stage, _pip_target(job, dependencies))
    return [
        "--python", job.python, "--source-dir", str(source), "--dependency-dir", str(dependencies),
    ], {}  # fmt: skip


def _bfcl_project(job: Job) -> Path:
    return job.scratch / "bfcl-project"


def _prepare_bfcl(job: Job) -> tuple[list[str], dict[str, str]]:
    job.step("dependency installation", [
        job.python, str(EVALS / job.spec.adapter),
        "--install-runtime", str(job.scratch / "bfcl-wheel"), "--uv", uv.find(),
    ], timeout_s=BFCL_INSTALL_TIMEOUT_S)  # fmt: skip
    project = _bfcl_project(job)
    project.mkdir()
    return ["--bfcl-project-root", str(project)], {}


def _archive_bfcl_project(job: Job) -> int:
    """Keep BFCL's raw generations and scores next to the projected results."""
    try:
        archive_tree(_bfcl_project(job), job.ctx.results_dir / BFCL_ARCHIVE)
    except (OSError, ValueError) as error:
        print(f"ERROR: failed to archive BFCL upstream artifacts: {error}", file=sys.stderr)
        return 1
    return 0


def archive_tree(root: Path, archive: Path) -> None:
    """Write ``root`` as a byte-reproducible ``.tar.gz``; refuse symlinks and special files."""
    temporary = archive.with_name(f".{archive.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with (
            temporary.open("xb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
            tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar,
        ):
            for path in sorted(
                root.rglob("*"), key=lambda entry: entry.relative_to(root).as_posix()
            ):
                name = path.relative_to(root).as_posix()
                if path.is_symlink():
                    raise ValueError(f"refusing to archive symbolic link: {name}")
                info = tar.gettarinfo(str(path), arcname=name)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                if info.isdir():
                    tar.addfile(info)
                elif info.isfile():
                    with path.open("rb") as source:
                        tar.addfile(info, source)
                else:
                    raise ValueError(f"refusing to archive special file: {name}")
        temporary.replace(archive)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


_KIMI = Suite(
    label="Kimi Vendor Verifier",
    adapter="kimi_vendor_eval.py",
    results=("results_kimi_vendor_*.json",),
    requirements=KIMI_REQUIREMENTS,
)
_BFCL = Suite(
    label="BFCL",
    adapter="bfcl_adapter.py",
    results=("bfcl_report.json", "results_bfcl.json"),
)

PROVIDERS: dict[str, Provider] = {
    "kimi-vendor": Provider(
        default_suite="kimi_tool_call_schema",
        suites={
            "kimi_tool_call_schema": _KIMI,
            "kimi_tool_call_schema_full": replace(
                _KIMI, requirements=(*KIMI_REQUIREMENTS, "pytest-xdist==3.8.0")
            ),
        },
        python_minor=12,
        prepare=_prepare_kimi,
        suite_flag="--task-name",
    ),
    "minimax-vendor": Provider(
        default_suite="minimax_m3_smoke",
        suites={
            "minimax_m3_smoke": Suite(
                label="MiniMax Provider Verifier",
                adapter="minimax_provider_eval.py",
                results=("results_minimax_vendor_*.json",),
                requirements=MINIMAX_REQUIREMENTS,
            ),
            "minimax_m3_full": Suite(
                label="MiniMax M3 full",
                adapter="minimax_m3_full_eval.py",
                results=("results_minimax_vendor_full_*.json",),
                requirements=MINIMAX_REQUIREMENTS,
            ),
        },
        python_minor=12,
        prepare=_prepare_minimax,
        run_command=("run",),
        failure_command=("failure",),
        message_flag="--message",
    ),
    # BFCL supports Python 3.10 and reuses the image's installed stack.
    "bfcl": Provider(
        default_suite="bfcl_smoke",
        suites={
            "bfcl_smoke": replace(_BFCL, timeout_s=900),
            "bfcl_vllm_minimax_m3": replace(_BFCL, timeout_s=7200, finalize=_archive_bfcl_project),
            "bfcl_vllm_kimi": replace(_BFCL, timeout_s=14400, finalize=_archive_bfcl_project),
        },
        python_minor=10,
        system_site_packages=True,
        prepare=_prepare_bfcl,
        suite_flag="--suite",
    ),
}
"""Vendor providers by ``EVAL_FRAMEWORK`` name."""
