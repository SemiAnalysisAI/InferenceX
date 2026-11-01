"""Opt-in host-restore qualification before an otherwise unchanged evaluation."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from infx.bench.env import InputError, require

FLAG = "EVAL_HICACHE_RESTORE_PROBE"
PINS = (
    "EVAL_HICACHE_EXPECTED_IMAGE",
    "EVAL_HICACHE_EXPECTED_REVISION",
    "EVAL_HICACHE_EXPECTED_SOURCE",
)
LOG_DIR = Path("/logs")
REPORT_NAME = "hicache_restore_report.json"


def enabled(values: Mapping[str, str]) -> bool:
    """Missing is off; reject ambiguous values and orphaned provenance pins."""
    value = values.get(FLAG)
    if value is None:
        if any(name in values for name in PINS):
            raise InputError(f"{FLAG} is required when HiCache provenance pins are set")
        return False
    if value not in {"1", "true", "0", "false"}:
        raise InputError(f"{FLAG} must be 1, true, 0, or false, got {value!r}")
    active = value in {"1", "true"}
    if not active and any(name in values for name in PINS):
        raise InputError("HiCache provenance pins require an enabled restore probe")
    return active


def recipe_environment(values: Mapping[str, object], *, eval_only: bool) -> dict[str, str]:
    """Forward only the reviewed flag and pins, never arbitrary recipe env."""
    selected = {name: str(values[name]) for name in (FLAG, *PINS) if name in values}
    if not enabled(selected):
        return {}
    if not eval_only:
        raise InputError(f"{FLAG} is allowed only for EVAL_ONLY requests")
    pins = require(*PINS, env=selected)
    if not re.fullmatch(r".+@sha256:[0-9a-f]{64}", pins[PINS[0]]):
        raise InputError(f"{PINS[0]} must pin an immutable sha256 image")
    for name in PINS[1:]:
        if not re.fullmatch(r"[0-9a-f]{40}", pins[name]):
            raise InputError(f"{name} must be a full 40-character commit revision")
    return selected


def flush_cache(endpoint: str) -> None:
    """The pinned server returns HTTP 200 and a 'Cache flushed.' success body."""
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise InputError("cache flush requires an HTTP(S) server endpoint")
    request = Request(  # noqa: S310 -- HTTP(S) endpoint checked above.
        f"{endpoint.rstrip('/')}/flush_cache?timeout=30",
        data=b"",
        method="POST",
    )
    with urlopen(request, timeout=40) as response:  # noqa: S310 -- checked above.
        body = response.read(4096).decode("utf-8", errors="replace")
        if response.status != 200 or not body.startswith("Cache flushed."):
            raise RuntimeError("server did not confirm a successful cache flush")


def _write_report(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def run(endpoint: str, destination: Path, environ: Mapping[str, str]) -> int:
    """Fail closed before canonical evaluation; never mutate evaluator inputs."""
    if not enabled(environ):
        return 0
    # The native single-node collector bundles /logs even on failed post_eval.
    report_path = LOG_DIR / "eval_results" / REPORT_NAME
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        selected = recipe_environment(environ, eval_only=environ.get("EVAL_ONLY") == "true")
        model = require("MODEL", "MODEL_PATH", env=environ)
        fingerprint = LOG_DIR / "fingerprint_agg_w0.json"
        boot_logs = sorted(LOG_DIR.glob("*_agg_w0.out"))
        if not fingerprint.is_file() or len(boot_logs) != 1:
            raise InputError(
                "HiCache prelude requires /logs/fingerprint_agg_w0.json "
                "and exactly one /logs/*_agg_w0.out from this allocation"
            )
        from infx.bench.eval import host_restore_probe

        argv = [
            "--url",
            endpoint,
            "--model",
            environ.get("MODEL_NAME") or model["MODEL"],
            "--expected-model",
            model["MODEL_PATH"],
            "--expected-image",
            selected[PINS[0]],
            "--expected-revision",
            selected[PINS[1]],
            "--expected-source",
            selected[PINS[2]],
            "--expected-host-fraction",
            "0.70",
            "--server-fingerprint",
            str(fingerprint),
            "--server-boot-log",
            str(boot_logs[0]),
            "--metric-label",
            "tp_rank=0",
            "--metric-label",
            "pp_rank=0",
            "--exclusive-server",
            "--max-seconds",
            "1800",
            "--execute",
            "--output",
            str(report_path),
        ]
        code = host_restore_probe.main(argv)
        report = json.loads(report_path.read_text())
        if not isinstance(report, dict):
            raise ValueError("HiCache probe report must be a JSON object")
        if code != 0 or report.get("status") != "DIAGNOSTIC_PASS":
            print(f"ERROR: HiCache restore qualification did not pass; see {report_path}")
            return code if isinstance(code, int) and code > 0 else 1
        try:
            flush_cache(endpoint)
        except (OSError, RuntimeError, ValueError, InputError) as exc:
            report["status"] = "FLUSH_FAILED"
            report["canonical_cache_flush"] = {"passed": False, "error": str(exc)}
            _write_report(report_path, report)
            print(f"ERROR: HiCache diagnostic cache flush failed: {exc}")
            return 1
        report["canonical_cache_flush"] = {"passed": True}
        _write_report(report_path, report)
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, InputError) as exc:
        _write_report(
            report_path,
            {"status": "PRELUDE_FAILED", "error": str(exc)},
        )
        print(f"ERROR: HiCache evaluation prelude failed: {exc}")
        return 1
    finally:
        if report_path.is_file():
            destination.mkdir(parents=True, exist_ok=True)
            target = destination / REPORT_NAME
            if target != report_path:
                shutil.copyfile(report_path, target)
