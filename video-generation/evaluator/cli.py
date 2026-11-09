"""Controlled video-generation CI, full-stream comparison, and portable evidence reports."""

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
        if arguments.command == "srt-finalize":
            from .srt_bundle import build_bundle

            result = build_bundle(
                video_dir=arguments.video_dir,
                power_dir=arguments.power_dir,
                workspace=arguments.workspace,
                result_filename=arguments.result_filename,
                concurrencies=arguments.concurrencies,
                output=arguments.output,
            )
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
            return 0 if result["status"] == "complete" else 1

        if arguments.command == "run":
            from .runner import preview_plan, run_plan

            plan = _load_mvp_object(arguments.plan)
            from .serving import settings

            serving = settings(arguments.serving_concurrency, arguments.delivery_deadline_seconds)
            if not arguments.execute:
                result = preview_plan(plan, runtime=arguments.runtime)
                if serving:
                    result["serving"] = serving
                print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
                return 0
            required = (
                "endpoint",
                "runtime_revision",
                "hardware_label",
                "output",
            )
            missing = [name.replace("_", "-") for name in required if not getattr(arguments, name)]
            if missing:
                raise ValueError(
                    "--execute also requires " + ", ".join("--" + name for name in missing)
                )

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
                    timeout_seconds=arguments.timeout_seconds,
                    api_key_env=arguments.api_key_env,
                    serving_concurrency=arguments.serving_concurrency,
                    delivery_deadline_seconds=arguments.delivery_deadline_seconds,
                    numerical_identity=(
                        _load_mvp_object(arguments.numerical_identity)
                        if arguments.numerical_identity
                        else None
                    ),
                )
            finally:
                signal.signal(signal.SIGTERM, previous_sigterm)
            receipt = {
                key: result.get(key) for key in ("run_id", "evidence_kind", "status", "summary")
            }
            receipt["run_json"] = str((arguments.output / "run.json").resolve())
            print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
            return 0 if result.get("status") == "complete" else 1

        if arguments.command == "compare":
            from .compare import compare_runs
            from .report import write_report

            result = compare_runs(
                arguments.baseline,
                arguments.candidate,
                policy=_load_mvp_object(arguments.policy),
            )
            if arguments.report is not None:
                write_report(result, arguments.report)
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
            return {"pass": 0, "fail": 1, "inconclusive": 2}[result["overall_status"]]

        if arguments.command == "score-quality":
            from .quality import score_paired, score_quality

            bundles = dict(arguments.bundles)
            baselines = dict(arguments.baseline)
            if len(bundles) != len(arguments.bundles) or len(baselines) != len(arguments.baseline):
                raise ValueError("each artifact ID may appear only once")
            references = (arguments.descriptor, arguments.references)
            if baselines:
                if arguments.baseline_run_id is None or references != (None, None):
                    raise ValueError(
                        "paired scoring takes --baseline-run-id and no --descriptor or --references"
                    )
                result = score_paired(
                    bundles=bundles,
                    baselines=baselines,
                    run_id=arguments.run_id,
                    baseline_run_id=arguments.baseline_run_id,
                    source_revision=arguments.source_revision,
                    output=arguments.output,
                )
            elif None in references or arguments.baseline_run_id is not None:
                raise ValueError("reference scoring takes --descriptor and --references")
            else:
                result = score_quality(
                    bundles=bundles,
                    references=arguments.references,
                    descriptor=arguments.descriptor,
                    run_id=arguments.run_id,
                    source_revision=arguments.source_revision,
                    output=arguments.output,
                )
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
            return 0

        if arguments.command == "inspect-media":
            from .media import analyze_media

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


def _bundle_argument(value: str) -> tuple[int, Path]:
    artifact, separator, path = value.partition("=")
    if not separator or not artifact.isdigit() or not path:
        raise argparse.ArgumentTypeError("expected ARTIFACT_ID=PATH")
    return int(artifact), Path(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vgbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    srt_finalize = subparsers.add_parser(
        "srt-finalize",
        help="assemble the app bundle from an srt-slurm video job and its power validation",
    )
    srt_finalize.add_argument("--video-dir", required=True, type=Path)
    srt_finalize.add_argument("--power-dir", required=True, type=Path)
    srt_finalize.add_argument("--workspace", required=True, type=Path)
    srt_finalize.add_argument("--result-filename", required=True)
    srt_finalize.add_argument("--concurrencies", nargs="+", type=int, required=True)
    srt_finalize.add_argument("--output", required=True, type=Path)

    run = subparsers.add_parser("run", help="preview a video plan; --execute explicitly submits it")
    run.add_argument("plan", type=Path)
    run.add_argument("--runtime", choices=("sglang", "vllm-omni"), default="sglang")
    run.add_argument(
        "--execute", action="store_true", help="submit real requests to your video server"
    )
    run.add_argument("--endpoint", help="self-hosted server base URL, without /v1/videos")
    run.add_argument(
        "--runtime-revision", help="operator-declared exact server code/container revision"
    )
    run.add_argument(
        "--hardware-label", help="operator-declared hardware, precision, and topology label"
    )
    run.add_argument(
        "--output", type=Path, help="new output directory; existing paths are never overwritten"
    )
    run.add_argument("--timeout-seconds", type=float, default=3600)
    run.add_argument(
        "--api-key-env", help="name of an environment variable containing a bearer token"
    )
    run.add_argument(
        "--serving-concurrency",
        type=int,
        help="opt into closed-loop delivery load with 1-32 concurrent requests; validation is separate",
    )
    run.add_argument(
        "--delivery-deadline-seconds",
        type=float,
        help="optional submit-to-downloaded-media deadline for technical goodput; not an attempt timeout",
    )
    run.add_argument(
        "--numerical-identity",
        type=Path,
        help="JSON server identity (image digest, argv, environment, GPU topology) that lets compare gate same-configuration fidelity",
    )

    compare = subparsers.add_parser(
        "compare", help="compare paired run bundles and return a CI exit code"
    )
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--policy", required=True, type=Path)
    compare.add_argument(
        "--report", type=Path, help="new portable HTML report plus a local media-assets directory"
    )

    score = subparsers.add_parser(
        "score-quality",
        help="score stored h3-results clips against the public Diffusers reference release, "
        "or with --baseline against the matched lossless deployment",
    )
    score.add_argument(
        "bundles",
        nargs="+",
        type=_bundle_argument,
        metavar="ARTIFACT_ID=PATH",
        help="h3-results bundle, as a zip or directory, and its GitHub artifact ID",
    )
    score.add_argument("--descriptor", type=Path)
    score.add_argument("--references", type=Path, help="directory of release MP4s")
    score.add_argument(
        "--baseline",
        action="append",
        default=[],
        type=_bundle_argument,
        metavar="ARTIFACT_ID=PATH",
        help="h3-results bundle of the lossless run; repeat for each deployment",
    )
    score.add_argument("--baseline-run-id", help="GitHub run that stores the baseline bundles")
    score.add_argument("--run-id", required=True, help="GitHub run that stores the bundles")
    score.add_argument("--source-revision", required=True, help="git commit of this scorer")
    score.add_argument("--output", required=True, type=Path, help="new attachment directory")

    inspect = subparsers.add_parser(
        "inspect-media", help="fully decode a file and measure technical media integrity"
    )
    inspect.add_argument("media", type=Path)
    inspect.add_argument("--expected", type=Path, help="optional JSON media expectations")
    return parser


def main(argv: list[str] | None = None) -> int:
    return _mvp_command(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
