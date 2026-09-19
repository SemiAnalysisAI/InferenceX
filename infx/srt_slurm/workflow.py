"""GitHub Actions boundary for the isolated native H100 pilot."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from infx.benchmarks.common import write_json
from infx.srt_slurm.job import parse_job
from infx.srt_slurm.launch import execute, prepare
from infx.srt_slurm.render import PilotSite


def main() -> int:
    root = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
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
    site = PilotSite.model_validate_json(os.environ["NATIVE_SITE_JSON"])
    if (
        os.environ["NATIVE_READER_REVISION"] != site.reader_revision
        or os.environ["NATIVE_COLLECTOR_REVISION"] != site.collector_revision
    ):
        raise ValueError("reader-first deployment and trusted collector pins are not enabled")
    source = {
        "repository": os.environ["GITHUB_REPOSITORY"],
        "run_id": int(os.environ["GITHUB_RUN_ID"]),
        "attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
        "head_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
        ).stdout.strip(),
    }
    bundle = prepare(job, site, root, source)
    with Path(os.environ["GITHUB_ENV"]).open("a") as stream:
        stream.write(
            f"RESULT_FILENAME={bundle['point_id']}\nNATIVE_POINT_ID={bundle['point_id']}\nGPU_COUNT=8\n"
        )
    diagnostics = root / "native-execution"
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
    execute(bundle, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
