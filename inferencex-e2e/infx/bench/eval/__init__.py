"""Evaluate a ready server with one eval framework, then stage and record the eval."""

from __future__ import annotations

import argparse
import functools
import os
import re
import sys
import tempfile
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from infx.bench import env, server
from infx.bench.eval import lm_eval, meta, stage, vendor
from infx.bench.eval.context import EvalContext, EvalOutcome

Framework = Callable[[EvalContext], EvalOutcome]
FRAMEWORKS: dict[str, Framework] = {
    "lm-eval": lm_eval.run,
    **{name: functools.partial(vendor.run, p) for name, p in vendor.PROVIDERS.items()},
}
_SUITE = re.compile(r"[A-Za-z0-9_.-]+")


def main(argv: list[str]) -> int:
    """Run the ``eval`` command."""
    parser = argparse.ArgumentParser(
        prog="python3 -m infx.bench eval",
        description=__doc__,
        epilog="Exits with the framework's code, else 1 when staging or meta_env.json failed.",
    )
    parser.add_argument("--endpoint", required=True, help="server root, e.g. http://localhost:8000")
    parser.add_argument(
        "--concurrency",
        required=True,
        help='concurrent requests; a list ("1 4 8") runs lm-eval once per value, suffixes the '
        "artifacts _conc<N>, and exits 0, leaving failed values to score validation",
    )
    parser.add_argument(
        "--stage-to",
        required=True,
        type=Path,
        help="directory receiving the allow-listed artifacts and meta_env.json, even on failure",
    )
    parser.add_argument(
        "--framework",
        help=f"one of {', '.join(FRAMEWORKS)}; EVAL_FRAMEWORK overrides it (default: lm-eval)",
    )
    args = parser.parse_args(argv)
    return evaluate(args.endpoint, args.concurrency, args.stage_to, args.framework)


def _server_root(endpoint: str) -> str:
    root = endpoint.rstrip("/")
    try:
        parts = urllib.parse.urlsplit(root)
        valid = parts.scheme in {"http", "https"} and bool(parts.hostname) and parts.port != 0
    except ValueError:
        valid = False
    if not valid:
        raise env.InputError(f"--endpoint must be an http(s) server root, got {endpoint!r}")
    return root


def _concurrencies(values: str) -> list[int]:
    concurrencies = [env.parse_positive_int("--concurrency", value) for value in values.split()]
    if not concurrencies:
        raise env.InputError("--concurrency must name at least one concurrency")
    return concurrencies


def evaluate(
    endpoint: str,
    concurrency: str,
    destination: Path,
    framework: str | None = None,
    environ: Mapping[str, str] = os.environ,
) -> int:
    """Run, stage, and record one eval; return its exit code."""
    environ = dict(environ)
    name = environ.get("EVAL_FRAMEWORK") or framework or "lm-eval"
    if name not in FRAMEWORKS:
        raise env.InputError(
            f"unknown eval framework {name!r}; expected one of {', '.join(sorted(FRAMEWORKS))}"
        )
    suite = env.optional("EVAL_SUITE", env=environ)
    if suite is not None and not _SUITE.fullmatch(suite):
        raise env.InputError("EVAL_SUITE may contain only letters, digits, '.', '_', and '-'")
    if suite is not None and name not in vendor.PROVIDERS:
        raise env.InputError(
            f"EVAL_SUITE is only supported with {', '.join(sorted(vendor.PROVIDERS))}"
        )
    concurrencies = _concurrencies(concurrency)
    if len(concurrencies) > 1 and name != "lm-eval":
        raise env.InputError("batched eval concurrency is only supported for lm-eval")
    eval_only = env.flag("EVAL_ONLY", env=environ)
    checkpoint = env.require("MODEL", env=environ)["MODEL"]
    model = environ.get("MODEL_NAME") or checkpoint
    base_url = _server_root(endpoint)
    # Reject malformed metadata inputs before the eval spends hours.
    meta.build(environ, conc=concurrencies[0], suite="")
    if eval_only and name in vendor.PROVIDERS:
        server.wait_chat_route(base_url, model, server.chat_route_budget(environ))
    context = functools.partial(
        EvalContext,
        base_url=base_url,
        model=model,
        context_length=lm_eval.prepare(environ) if name == "lm-eval" else 0,
        suite=suite,
        env=environ,
    )
    if len(concurrencies) > 1:
        return _batched(FRAMEWORKS[name], context, concurrencies, destination, environ)
    return _single(FRAMEWORKS[name], context, concurrencies[0], destination, environ)


def _run(
    framework: Framework,
    context: Callable[..., EvalContext],
    conc: int,
    destination: Path,
    suffix: str,
) -> tuple[EvalOutcome, list[Path] | None]:
    """Run ``framework`` in a fresh results directory; stage what it wrote (``None``: failed)."""
    with tempfile.TemporaryDirectory(
        prefix=f"eval_out-conc{conc}-", ignore_cleanup_errors=True
    ) as results:
        outcome = framework(context(concurrency=conc, results_dir=Path(results)))
        try:
            staged = stage.copy(Path(results), destination, suffix=suffix)
        except OSError as error:
            print(
                f"ERROR: failed to stage eval artifacts in {destination}: {error}", file=sys.stderr
            )
            staged = None
    return outcome, staged


def _record(
    destination: Path,
    environ: Mapping[str, str],
    conc: int,
    suite: str,
    batch: Mapping[str, Sequence[int]] | None = None,
) -> bool:
    path = destination / "meta_env.json"
    try:
        destination.mkdir(parents=True, exist_ok=True)
        meta.write(path, meta.build(environ, conc=conc, suite=suite, batch=batch))
    except OSError as error:
        print(f"ERROR: failed to write {path}: {error}", file=sys.stderr)
        return False
    return True


def _single(
    framework: Framework,
    context: Callable[..., EvalContext],
    conc: int,
    destination: Path,
    environ: Mapping[str, str],
) -> int:
    outcome, staged = _run(framework, context, conc, destination, "")
    recorded = _record(destination, environ, conc, outcome.suite)
    if outcome.returncode:
        print(f"ERROR: eval failed with exit code {outcome.returncode}", file=sys.stderr)
        return outcome.returncode
    if staged is None or not recorded:
        return 1
    print(f"Staged eval artifacts in: {destination}")
    return 0


def _batched(
    framework: Framework,
    context: Callable[..., EvalContext],
    concurrencies: list[int],
    destination: Path,
    environ: Mapping[str, str],
) -> int:
    completed: list[int] = []
    failed: list[int] = []
    suite = ""
    for conc in concurrencies:
        print(f"Running lm-eval at concurrency {conc}")
        outcome, staged = _run(framework, context, conc, destination, f"_conc{conc}")
        suite = outcome.suite
        if staged == []:
            print(f"WARN: no eval artifacts were produced for concurrency {conc}", file=sys.stderr)
        if outcome.returncode == 0 and staged:
            completed.append(conc)
        else:
            print(f"ERROR: lm-eval failed at concurrency {conc}", file=sys.stderr)
            failed.append(conc)
    if failed:
        print(
            f"ERROR: batched eval failed for concurrency: {' '.join(map(str, failed))}; "
            "score validation fails the job after upload",
            file=sys.stderr,
        )
    batch = {
        "eval_concs": concurrencies,
        "completed_eval_concs": completed,
        "failed_eval_concs": failed,
    }
    if not _record(destination, environ, concurrencies[0], suite, batch):
        return 1
    print(f"Prepared batched eval artifacts in: {destination}")
    return 0
