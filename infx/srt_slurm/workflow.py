"""GitHub Actions boundary for the isolated native H100 pilot."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

from pydantic import ValidationError

from infx.benchmarks.common import write_json
from infx.srt_slurm.job import parse_job
from infx.srt_slurm.launch import execute, prepare
from infx.srt_slurm.qualification import PURPOSE, require_pr_event
from infx.srt_slurm.render import PilotSite, PreparedSite


def load_site(environment: Mapping[str, str]) -> PreparedSite:
    """Explain missing deployment configuration before touching preparation or Slurm."""
    purpose = environment.get("NATIVE_PURPOSE", "publication")
    if purpose == PURPOSE:
        require_pr_event(environment)
        raw = environment.get("NATIVE_PREPARED_SITE_JSON", "")
        if not raw.strip():
            raise ValueError(
                "Nonpublishing qualification requires INFX_H100_PHASE1_PREPARED_SITE_JSON"
            )
        try:
            return PreparedSite.model_validate_json(raw)
        except ValidationError:
            raise ValueError(
                "INFX_H100_PHASE1_PREPARED_SITE_JSON must contain valid deployment-free "
                "PreparedSite JSON"
            ) from None
    if purpose != "publication":
        raise ValueError("unsupported native execution purpose")
    variables = {
        "NATIVE_SITE_JSON": "INFX_H100_PHASE1_SITE_JSON",
        "NATIVE_READER_REVISION": "INFX_PHASE1_READER_REVISION",
        "NATIVE_COLLECTOR_REVISION": "INFX_PHASE1_COLLECTOR_REVISION",
    }
    missing = [
        repository_name
        for name, repository_name in variables.items()
        if not environment.get(name, "").strip()
    ]
    if missing:
        raise ValueError("Native H100 pilot requires repository variables: " + ", ".join(missing))
    try:
        site = PilotSite.model_validate_json(environment["NATIVE_SITE_JSON"])
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in item['loc']) or 'JSON'}: {item['msg']}"
            for item in error.errors(include_input=False, include_context=False, include_url=False)
        )
        raise ValueError(
            "Repository variable INFX_H100_PHASE1_SITE_JSON must contain valid PilotSite JSON: "
            + problems
        ) from None
    mismatched = [
        variables[name]
        for name, expected in (
            ("NATIVE_READER_REVISION", site.reader_revision),
            ("NATIVE_COLLECTOR_REVISION", site.collector_revision),
        )
        if environment[name] != expected
    ]
    if mismatched:
        raise ValueError(
            "Repository variables "
            + ", ".join(mismatched)
            + " must match the deployed revisions recorded in INFX_H100_PHASE1_SITE_JSON"
        )
    return site


def main() -> int:
    site = load_site(os.environ)
    root = Path(os.environ.get("NATIVE_CHECKOUT_ROOT", os.environ["GITHUB_WORKSPACE"])).resolve()
    raw = json.loads(os.environ["NATIVE_CONFIG_JSON"])
    if os.environ["NATIVE_AGENTX_FAST"] != "false" or os.environ["NATIVE_EVAL_LIMIT"] not in (
        "",
        "full",
    ):
        raise ValueError("Phase 1 requires the full-duration/full-dataset qualification policy")
    if os.environ["NATIVE_REQUIRE_POWER"] != "false":
        raise ValueError("the Phase 1 telemetry exception cannot satisfy require-power")
    for key, variable in (("run-eval", "NATIVE_RUN_EVAL"), ("eval-only", "NATIVE_EVAL_ONLY")):
        value = json.loads(os.environ[variable])
        if key in raw and raw[key] != value:
            raise ValueError(f"workflow and matrix disagree on {key}")
        raw[key] = value
    if raw["eval-only"]:
        raw["eval-framework"] = os.environ["NATIVE_EVAL_FRAMEWORK"]
        raw["eval-suite"] = os.environ["NATIVE_EVAL_SUITE"]
    job = parse_job(
        raw,
        root,
        {
            "priority": os.environ["NATIVE_PRIORITY"],
            "queue-token": os.environ["NATIVE_QUEUE_TOKEN"],
            "node-count": 1,
        },
    )
    source = {
        "repository": os.environ["GITHUB_REPOSITORY"],
        "run_id": int(os.environ["GITHUB_RUN_ID"]),
        "attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
        "head_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
        ).stdout.strip(),
    }
    qualification = os.environ.get("NATIVE_PURPOSE") == PURPOSE
    if qualification:
        event = require_pr_event(os.environ)
        if source["head_sha"] != os.environ["GITHUB_SHA"]:
            raise ValueError("qualification checkout must match the actual PR workflow commit")
        source.update(purpose=PURPOSE, event="pull_request", pull_request=event["number"])
    output = root / "native-qualification" if qualification else root
    output.mkdir(exist_ok=True)
    if qualification:
        # Preparation may fail before an effective identity exists. Retain that failure
        # under the requested point; the summary still requires a complete bound bundle.
        with Path(os.environ["GITHUB_ENV"]).open("a") as stream:
            stream.write(f"NATIVE_POINT_ID={job.point_id}\n")
        write_json(
            output / "preparation-request.json",
            {
                "requested_point_id": job.point_id,
                "source": source,
            },
        )
    try:
        bundle = prepare(job, site, root, source)
    except Exception as error:
        if qualification:
            write_json(
                output / "preparation-failure.json",
                {
                    "requested_point_id": job.point_id,
                    "source": source,
                    "error_type": type(error).__name__,
                    "error": str(error)[:2000],
                },
            )
        raise
    with Path(os.environ["GITHUB_ENV"]).open("a") as stream:
        stream.write(f"NATIVE_POINT_ID={bundle['point_id']}\nGPU_COUNT=8\n")
        if not qualification:
            stream.write(f"RESULT_FILENAME={bundle['point_id']}\n")
    diagnostics = output / "native-execution"
    diagnostics.mkdir(exist_ok=True)
    write_json(
        diagnostics / "prepared.json",
        {
            "schema_version": 1,
            "point_id": bundle["point_id"],
            "execution_id": bundle["execution_id"],
            "bundle_digest": bundle["bundle_digest"],
            "native_manifest_sha256": bundle["prepared"]["manifest_sha256"],
            "source": source,
        },
    )
    execute(bundle, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
