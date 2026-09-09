"""Controlled H3 CI, full-stream comparison, and portable evidence reports."""

from __future__ import annotations

import argparse
import json
import os
import signal
import stat
import sys
from pathlib import Path
from typing import Any


def _load_mvp_object(path: Path) -> dict[str, Any]:
    """Read a bounded regular configuration file, never a FIFO or device."""
    limit = 4 * 1024 * 1024
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"MVP configuration must be a regular file: {path}")
        if metadata.st_size > limit:
            raise ValueError(f"MVP configuration exceeds 4 MiB: {path}")
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"MVP configuration exceeds 4 MiB while being read: {path}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value {value!r} in {path}")

    result = json.loads(
        data,
        object_pairs_hook=unique_object,
        parse_constant=invalid_constant,
    )
    if not isinstance(result, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return result


def _mvp_command(arguments: argparse.Namespace) -> int:
    try:
        if arguments.command == "gpu-manifest":
            from .mvp_gpu_manifest import write_gpu_manifest

            result = write_gpu_manifest(
                arguments.kind, arguments.directory, arguments.output,
                model_revision=arguments.model_revision, timeout_seconds=arguments.timeout_seconds,
            )
            print(json.dumps({key: value for key, value in result.items() if key != "files"},
                             indent=2, sort_keys=True, allow_nan=False))
            return 0

        if arguments.command == "gpu-job":
            from .mvp_gpu_job import preview_gpu_job, run_gpu_job

            spec = _load_mvp_object(arguments.spec)
            if not arguments.execute:
                result = preview_gpu_job(spec)
                print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
                return 0
            if arguments.output is None:
                raise ValueError("gpu-job --execute requires --output")
            result = run_gpu_job(spec, arguments.output)
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
            if (result.get("ci_accepted") is True and result.get("status") == "complete"
                    and result.get("measurement_status") == "complete"
                    and result.get("regression_status") == "pass" and result.get("cleanup_status") == "clean"):
                return 0
            if (result.get("regression_status") == "fail" or result.get("status") in {"failed", "aborted"}
                    or result.get("cleanup_status") == "failed"):
                return 1
            return 2

        if arguments.command == "gpu-report":
            from .mvp_gpu_report import write_gpu_report

            result = write_gpu_report(arguments.job, arguments.output)
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
            # Successfully rendering an incomplete/failed run is not acceptance
            # of that run. The GPU job's separate exit code remains authoritative.
            return 0

        if arguments.command == "run":
            from .mvp_runner import preview_plan, run_plan

            plan = _load_mvp_object(arguments.plan)
            from .mvp_serving import settings
            serving = settings(arguments.serving_concurrency, arguments.delivery_deadline_seconds)
            if not arguments.execute:
                result = preview_plan(plan, runtime=arguments.runtime)
                if serving:
                    result["serving"] = serving
                print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
                return 0
            required = ("endpoint", "runtime_revision", "hardware_label", "model_revision", "output")
            missing = [name.replace("_", "-") for name in required if not getattr(arguments, name)]
            if missing:
                raise ValueError("--execute also requires " + ", ".join("--" + name for name in missing))
            def interrupt_run(_signum, _frame):
                raise KeyboardInterrupt

            previous_sigterm = signal.signal(signal.SIGTERM, interrupt_run)
            try:
                result = run_plan(
                    plan,
                    arguments.output,
                    endpoint=arguments.endpoint,
                    runtime=arguments.runtime,
                    runtime_revision=arguments.runtime_revision,
                    hardware_label=arguments.hardware_label,
                    model_revision=arguments.model_revision,
                    timeout_seconds=arguments.timeout_seconds,
                    api_key_env=arguments.api_key_env,
                    serving_concurrency=arguments.serving_concurrency,
                    delivery_deadline_seconds=arguments.delivery_deadline_seconds,
                )
            finally:
                signal.signal(signal.SIGTERM, previous_sigterm)
            receipt = {
                key: result.get(key)
                for key in ("run_id", "evidence_kind", "status", "summary")
            }
            receipt["run_json"] = str((arguments.output / "run.json").resolve())
            print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
            return 0 if result.get("status") == "complete" else 1

        if arguments.command == "compare":
            from .mvp_compare import compare_runs
            from .mvp_report import write_report

            result = compare_runs(
                arguments.baseline, arguments.candidate,
                policy=_load_mvp_object(arguments.policy),
            )
            if arguments.report is not None:
                write_report(result, arguments.report)
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
            return {"pass": 0, "fail": 1, "inconclusive": 2}[result["overall_status"]]

        if arguments.command == "inspect-media":
            from .mvp_media import analyze_media

            expected = _load_mvp_object(arguments.expected) if arguments.expected else None
            result = analyze_media(arguments.media, expected=expected)
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
            return 0 if result["valid"] else 1
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        message = str(exc)
        if isinstance(exc, ImportError):
            message += "; install media dependencies with: uv sync"
        print(json.dumps({"status": "error", "error": message}, allow_nan=False), file=sys.stderr)
        return 2
    raise AssertionError(arguments.command)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vgbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gpu_manifest = subparsers.add_parser("gpu-manifest", help="hash staged runtime/model files; no GPU work or downloads")
    gpu_manifest.add_argument("kind", choices=("runtime", "model"))
    gpu_manifest.add_argument("directory", type=Path)
    gpu_manifest.add_argument("--model-revision", help="immutable model revision, required for a model inventory")
    gpu_manifest.add_argument("--timeout-seconds", type=float, default=600)
    gpu_manifest.add_argument("--output", required=True, type=Path, help="new manifest file outside the inventoried tree")

    gpu_job = subparsers.add_parser("gpu-job", help="preview a bounded controlled GPU job; --execute requires authorization")
    gpu_job.add_argument("spec", type=Path)
    gpu_job.add_argument("--execute", action="store_true")
    gpu_job.add_argument("--output", type=Path, help="new evidence directory; no overwrites")

    gpu_report = subparsers.add_parser("gpu-report", help="render recorded GPU job evidence, including incomplete runs")
    gpu_report.add_argument("job", type=Path)
    gpu_report.add_argument("--output", required=True, type=Path, help="new report directory")

    run = subparsers.add_parser("run", help="preview an H3 plan; --execute explicitly submits it")
    run.add_argument("plan", type=Path)
    run.add_argument("--runtime", choices=("sglang", "vllm-omni"), default="sglang")
    run.add_argument("--execute", action="store_true", help="submit real requests to your H3 server")
    run.add_argument("--endpoint", help="self-hosted server base URL, without /v1/videos")
    run.add_argument("--runtime-revision", help="operator-declared exact server code/container revision")
    run.add_argument("--hardware-label", help="operator-declared hardware, precision, and topology label")
    run.add_argument("--model-revision", help="operator-declared exact checkpoint revision matching the plan")
    run.add_argument("--output", type=Path, help="new output directory; existing paths are never overwritten")
    run.add_argument("--timeout-seconds", type=float, default=3600)
    run.add_argument("--api-key-env", help="name of an environment variable containing a bearer token")
    run.add_argument("--serving-concurrency", type=int, help="opt into closed-loop delivery load with 1-32 concurrent requests; validation is separate")
    run.add_argument("--delivery-deadline-seconds", type=float, help="optional submit-to-downloaded-media deadline for technical goodput; not an attempt timeout")

    compare = subparsers.add_parser("compare", help="compare paired run bundles and return a CI exit code")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--policy", required=True, type=Path)
    compare.add_argument("--report", type=Path, help="new portable HTML report plus a local media-assets directory")

    inspect = subparsers.add_parser("inspect-media", help="fully decode a file and measure technical media integrity")
    inspect.add_argument("media", type=Path)
    inspect.add_argument("--expected", type=Path, help="optional JSON media expectations")
    return parser


def main(argv: list[str] | None = None) -> int:
    return _mvp_command(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
