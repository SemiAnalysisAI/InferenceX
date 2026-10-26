"""Deterministic facts and math for the CODEOWNER sign-off verifier.

Every subcommand runs trusted default-branch code. PR content is read only as data
(git objects, YAML, JSON, GitHub API responses); no PR code is executed.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from infx import github
from infx.golden_al_distribution import curves
from infx.klaud import api

SERIES = ("model", "precision", "framework", "hardware", "spec_method")
TOPOLOGY = (
    "disagg",
    "is_multinode",
    "prefill_tp",
    "prefill_ep",
    "prefill_dp_attention",
    "prefill_num_workers",
    "num_prefill_gpu",
    "decode_tp",
    "decode_ep",
    "decode_dp_attention",
    "decode_num_workers",
    "num_decode_gpu",
)
EVAL_METRICS = ("em_strict", "em_strict_se", "n_eff")
POINT_FIELDS = ("benchmark_type", "isl", "osl", "image", "recipe_fingerprint")
POINT_METRICS = ("tput_per_gpu", "mean_intvty", "median_e2el", "p90_e2el")
BENCH_JOB = re.compile(r"^(?:(?:single|multi)-node )?(?:agentic|canary|\d+k\d+k) /")
EVAL_JOB = re.compile(r"^(?:(?:single|multi)-node )?(?:agentic )?eval /")
RUN_ID = re.compile(r"/actions/runs/([0-9]+)")
IMAGE_REF = re.compile(r"""\b(?:image|container):\s*["']?([A-Za-z0-9][\w./#:@+-]*)""")
DATA_PATHS = (
    "inferencex-e2e/configs",
    "inferencex-e2e/benchmarks/single_node/srt-slurm-recipes",
    "inferencex-e2e/benchmarks/multi_node/srt-slurm-recipes",
)
SIGMAS = 2


# --- public InferenceX evidence -------------------------------------------------------


def _summary(row: dict[str, Any], fields: tuple[str, ...], metrics: tuple[str, ...]) -> dict:
    values = row.get("metrics") or {}
    return {
        **{field: row.get(field) for field in (*fields, "date", "conc", *TOPOLOGY)},
        **{metric: values.get(metric) for metric in metrics},
        "run_url": row.get("run_url"),
    }


def _newest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (str(row["date"]), str(row["run_url"])), reverse=True)


def eval_cohorts(rows: list[Any], target: dict[str, str], limit: int) -> dict[str, Any]:
    """Split ingested evaluations into the exact cohort and same model/precision/task peers."""
    exact, wider = [], []
    for row in rows:
        if not isinstance(row, dict) or any(
            row.get(field) != target[field] for field in ("model", "precision", "task")
        ):
            continue
        same = all(row.get(field) == target[field] for field in target)
        (exact if same else wider).append(_summary(row, (*SERIES, "task"), EVAL_METRICS))
    exact, wider = _newest(exact), _newest(wider)
    return {
        "target": target,
        "exact": exact[:limit],
        "exact_total": len(exact),
        "same_model_precision_task": wider[:limit],
        "same_model_precision_task_total": len(wider),
    }


def series_points(rows: list[Any], target: dict[str, str], limit: int) -> dict[str, Any]:
    """Return the ingested benchmark points of one model/precision/framework/hardware/spec series."""
    points = _newest(
        [
            _summary(row, (*SERIES, *POINT_FIELDS), POINT_METRICS)
            for row in rows
            if isinstance(row, dict) and all(row.get(field) == target[field] for field in target)
        ]
    )
    return {"target": target, "points": points[:limit], "points_total": len(points)}


# --- Check 1(a): sweep and eval check runs per in-PR commit ---------------------------


def commit_validation(check_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the newest attempt of every benchmark and eval check run on one commit."""
    newest: dict[str, dict[str, Any]] = {}
    for run in check_runs:
        name = str(run.get("name", ""))
        if (BENCH_JOB.match(name) or EVAL_JOB.match(name)) and (
            name not in newest or int(run.get("id", 0)) > int(newest[name].get("id", 0))
        ):
            newest[name] = run
    report: dict[str, Any] = {"benchmark": {}, "eval": {}, "run_ids": set()}
    for name, run in sorted(newest.items()):
        kind = "eval" if EVAL_JOB.match(name) else "benchmark"
        conclusion = run.get("conclusion") or run.get("status") or "pending"
        report[kind].setdefault(conclusion, []).append(name)
        if conclusion == "success" and (match := RUN_ID.search(str(run.get("details_url", "")))):
            report["run_ids"].add(int(match[1]))
    executed_bench = [
        name
        for name in report["benchmark"].get("success", [])
        if not name.startswith(("canary", "multi-node canary"))
    ]
    blocking = {
        conclusion
        for kind in ("benchmark", "eval")
        for conclusion in report[kind]
        if conclusion not in ("success", "skipped", "neutral")
    }
    report["satisfied"] = bool(executed_bench and report["eval"].get("success")) and not blocking
    report["run_ids"] = sorted(report["run_ids"])
    return report


def sweep_checks(repo: str, pr: int, fetch: Callable[..., Any] = github.paginate) -> list[dict]:
    commits = fetch(repo, f"pulls/{pr}/commits")
    return [
        {
            "sha": commit["sha"],
            **commit_validation(
                fetch(repo, f"commits/{commit['sha']}/check-runs", item_key="check_runs")
            ),
        }
        for commit in commits
    ]


# --- Check 2(b): accuracy gap against siblings and the ingested cohort ----------------


def hardware_of(hw: str) -> str:
    """Map an eval artifact runner label such as ``CLUSTER:MI355X-AMDS`` to ``mi355x``."""
    return hw.rpartition(":")[2].split("-")[0].lower()


def _standard_error(row: dict) -> tuple[float | None, bool]:
    """Reported SE, or the binomial SE from ``n_eff`` when the reported one is zero or absent."""
    se, score, n = row.get("em_strict_se"), row.get("em_strict"), row.get("n_eff")
    if isinstance(se, int | float) and se > 0:
        return se, False
    if isinstance(score, int | float) and isinstance(n, int | float) and n > 0 and 0 <= score <= 1:
        return math.sqrt(score * (1 - score) / n), True
    return None, False


def _gap(candidate: dict, reference: dict | None) -> dict[str, Any] | None:
    if reference is None:
        return None
    score, ref = candidate.get("em_strict"), reference.get("em_strict")
    (se, se_estimated), (ref_se, ref_estimated) = (
        _standard_error(candidate),
        _standard_error(reference),
    )
    result: dict[str, Any] = {"reference": reference}
    if not all(isinstance(value, int | float) for value in (score, ref, se, ref_se)):
        return result | {"verdict": "unverifiable", "reason": "missing score or standard error"}
    if se_estimated or ref_estimated:
        result["standard_error_estimated_from_n_eff"] = True
    gap = ref - score
    threshold = SIGMAS * math.sqrt(se**2 + ref_se**2)
    return result | {
        "gap": round(gap, 6),
        "threshold": round(threshold, 6),
        "verdict": "materially-weaker" if gap > threshold else "ok",
    }


def _topology(row: dict) -> tuple:
    def text(value: Any) -> str:
        return str(value).lower()

    return tuple(
        text(row.get(field)) for field in ("decode_tp", "decode_ep", "decode_dp_attention")
    )


def eval_gaps(
    candidates: list[dict], feed: list[Any], run_id: int | None, hardware: str | None
) -> list[dict]:
    """Compare each candidate eval with its best sibling and its closest ingested peer."""
    report = []
    for candidate in candidates:
        target = {
            "model": candidate.get("model_prefix"),
            "precision": candidate.get("precision"),
            "framework": candidate.get("framework"),
            "hardware": hardware or hardware_of(str(candidate.get("hw", ""))),
            "spec_method": candidate.get("spec_decoding"),
            "task": candidate.get("task"),
        }
        siblings = [
            row
            for row in candidates
            if row is not candidate
            and row.get("task") == candidate.get("task")
            and isinstance(row.get("em_strict"), int | float)
        ]
        best_sibling = max(siblings, key=lambda row: row["em_strict"], default=None)
        peers = [
            row
            for row in feed
            if isinstance(row, dict)
            and all(row.get(field) == value for field, value in target.items())
            and not (run_id and str(row.get("run_url", "")).endswith(f"/{run_id}"))
        ]
        scope = "exact"
        if not peers:
            scope = "same_model_precision_task"
            peers = [
                row
                for row in feed
                if isinstance(row, dict)
                and all(row.get(field) == target[field] for field in ("model", "precision", "task"))
                and not (run_id and str(row.get("run_url", "")).endswith(f"/{run_id}"))
            ]
        topology = _topology(candidate)
        same_shape = [row for row in peers if _topology(row) == topology]
        newest = _newest(
            [_summary(row, (*SERIES, "task"), EVAL_METRICS) for row in same_shape or peers]
        )
        report.append(
            {
                "candidate": {
                    **target,
                    "decode_tp": candidate.get("decode_tp"),
                    "decode_ep": candidate.get("decode_ep"),
                    "decode_dp_attention": candidate.get("decode_dp_attention"),
                    "conc": candidate.get("conc"),
                    "em_strict": candidate.get("em_strict"),
                    "em_strict_se": candidate.get("em_strict_se"),
                    "n_eff": candidate.get("n_eff"),
                },
                "sibling": _gap(
                    candidate,
                    best_sibling
                    and {
                        key: best_sibling.get(key)
                        for key in (
                            "decode_tp",
                            "decode_ep",
                            "decode_dp_attention",
                            "conc",
                            "em_strict",
                            "em_strict_se",
                            "n_eff",
                        )
                    },
                ),
                "cohort_scope": scope if newest else "none",
                "cohort_same_topology": bool(same_shape),
                "cohort": _gap(candidate, newest[0] if newest else None),
            }
        )
    return report


# --- Check 18(g,h): removed points, dominance, and deltas -----------------------------


def _dominates(a: dict, b: dict) -> bool:
    x, y = "tput_per_gpu", "mean_intvty"
    return a[x] >= b[x] and a[y] >= b[y] and (a[x] > b[x] or a[y] > b[y])


def dominance(before: list[dict], after: list[dict]) -> dict[str, Any]:
    """Removed points must be dominated by an after point; shared points report deltas."""
    for point in (*before, *after):
        if not isinstance(point.get("label"), str) or not all(
            isinstance(point.get(metric), int | float)
            and math.isfinite(point[metric])
            and point[metric] > 0
            for metric in ("tput_per_gpu", "mean_intvty")
        ):
            raise ValueError(
                f"Each point needs a label and positive tput_per_gpu and mean_intvty: {point}"
            )
    after_by_label = {point["label"]: point for point in after}
    removed = []
    for point in before:
        if point["label"] in after_by_label:
            continue
        cover = next((other for other in after if _dominates(other, point)), None)
        removed.append({"label": point["label"], "dominated_by": cover and cover["label"]})
    deltas = [
        {
            "label": point["label"],
            "tput_per_gpu_pct": round(
                100 * (after_by_label[point["label"]]["tput_per_gpu"] / point["tput_per_gpu"] - 1),
                2,
            ),
            "mean_intvty_pct": round(
                100 * (after_by_label[point["label"]]["mean_intvty"] / point["mean_intvty"] - 1), 2
            ),
        }
        for point in before
        if point["label"] in after_by_label
    ]
    return {
        "removed": removed,
        "undominated_removed": [item["label"] for item in removed if item["dominated_by"] is None],
        "deltas": deltas,
    }


# --- git data at a PR commit ----------------------------------------------------------


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return subprocess.run(["git", "-C", root, *args], capture_output=True, check=check)


def merge_base(base_ref: str, sha: str) -> str:
    return _git("merge-base", base_ref, sha).stdout.decode().strip()


def _extract(rev: str, target: Path) -> Path:
    with tarfile.open(fileobj=io.BytesIO(_git("archive", rev, "--", *DATA_PATHS).stdout)) as tar:
        tar.extractall(target, filter="data")
    return target / "inferencex-e2e"


def matrix_at(rev: str, workdir: Path) -> tuple[dict, dict[str, str]]:
    """Generate every config key's points from a revision's config data with trusted code."""
    from infx.matrix.changed import _points
    from infx.matrix.generate import generate_config_matrix
    from infx.matrix.validation import load_config_files, load_runner_file

    project = _extract(rev, workdir)
    configs = [
        str(project / "configs" / name) for name in ("amd-master.yaml", "nvidia-master.yaml")
    ]
    master = load_config_files([path for path in configs if Path(path).is_file()], validate=False)
    runners = load_runner_file(str(project / "configs" / "runners.yaml"), validate=False)
    matrix, errors = {}, {}
    for key in master:
        try:
            rows = generate_config_matrix([key], master, runners, eval_mode="none", root=project)
            matrix[key] = _points(rows, project)
        except Exception as error:  # noqa: BLE001 - one malformed key must not hide the others
            errors[key] = str(error)[:300]
    return matrix, errors


def public_model_name(point: dict) -> str:
    return str(point.get("model", "")).rpartition("/")[2]


def _point(point: dict) -> dict:
    shape = (
        str(point.get("srt-recipe", "")).rpartition("/")[2]
        if "prefill" in point
        else f"tp{point.get('tp')} ep{point.get('ep')} dpa{point.get('dp-attn')}"
    )
    return {
        "label": f"{point.get('exp-name')} {shape} conc {point.get('conc')}",
        "conc": point.get("conc"),
        "image": point.get("image"),
        "recipe_fingerprint": point.get("recipe-fingerprint"),
    }


def _measurement(row: dict) -> dict | None:
    metrics = row.get("metrics") or {}
    values = {metric: metrics.get(metric) for metric in ("tput_per_gpu", "mean_intvty")}
    if not all(isinstance(value, int | float) and value > 0 for value in values.values()):
        return None
    return values | {"date": row.get("date"), "run_url": row.get("run_url")}


def unofficial_run(run_id: int) -> list[dict]:
    """Benchmark rows of one PR sweep from the public unofficial-run API."""
    url = f"{api.PUBLIC}/api/unofficial-run?runId={int(run_id)}"
    request = urllib.request.Request(  # noqa: S310 - fixed https host
        url, headers={"Accept": "application/json", "User-Agent": api.USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed https host
        payload = json.loads(response.read(api.MAX_BYTES + 1))
    rows = payload.get("benchmarks") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"unofficial-run {run_id} returned no benchmark list")
    return rows


def _latest(rows: list[Any]) -> dict[tuple, dict]:
    latest: dict[tuple, dict] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("recipe_fingerprint") and _measurement(row):
            key = (row["recipe_fingerprint"], row.get("conc"))
            if key not in latest or str(row.get("date")) > str(latest[key].get("date")):
                latest[key] = row
    return latest


def points_diff(base: dict, head: dict, ingested: list[Any], new_run: list[Any]) -> list[dict]:
    """Added and removed generated points per config key, with measurements and dominance.

    Points match measurements by recipe fingerprint and concurrency. Base points use the
    ingested curve; head points prefer the PR's validated sweep, then the ingested curve.
    A removed point passes only when a measured head point dominates it.
    """
    from infx.matrix.changed import compare

    before, after = _latest(ingested), {**_latest(ingested), **_latest(new_run)}

    def measured(point: dict, table: dict[tuple, dict]) -> dict:
        item = _point(point)
        row = table.get((item["recipe_fingerprint"], item["conc"]))
        return item | {"measured": row and _measurement(row)}

    report = []
    for change in compare(base, head):
        head_points = [
            measured(point, after) for point in head.get(change["config-key"], ({}, []))[0].values()
        ]
        added = [measured(point, after) for point in change["added"]]
        removed = []
        for point in change["removed"]:
            item = measured(point, before)
            if item["measured"]:
                item["dominated_by"] = next(
                    (
                        other["label"]
                        for other in head_points
                        if other["measured"] and _dominates(other["measured"], item["measured"])
                    ),
                    None,
                )
            removed.append(item)
        added_labels = {item["label"] for item in added}
        replaced = {item["label"]: item for item in added if item["measured"]}
        dropped = [item for item in removed if item["label"] not in added_labels]
        deltas = [
            {
                "label": item["label"],
                "tput_per_gpu_pct": round(
                    100
                    * (
                        replaced[item["label"]]["measured"]["tput_per_gpu"]
                        / item["measured"]["tput_per_gpu"]
                        - 1
                    ),
                    2,
                ),
                "mean_intvty_pct": round(
                    100
                    * (
                        replaced[item["label"]]["measured"]["mean_intvty"]
                        / item["measured"]["mean_intvty"]
                        - 1
                    ),
                    2,
                ),
            }
            for item in removed
            if item["measured"] and item["label"] in replaced
        ]
        report.append(
            {
                "config-key": change["config-key"],
                "base-points": change["base-points"],
                "head-points": change["head-points"],
                "added": added,
                "removed": removed,
                "undominated_removed": [
                    item["label"]
                    for item in dropped
                    if item["measured"] and not item.get("dominated_by")
                ],
                "unmeasured_removed": [item["label"] for item in dropped if not item["measured"]],
                "retuned_deltas": deltas,
            }
        )
    return report


def changed_models(base: dict, head: dict) -> list[tuple[str, str]]:
    """(model prefix, checkpoint name) for every config key whose points changed."""
    from infx.matrix.changed import compare

    models = set()
    for change in compare(base, head):
        for matrix in (base, head):
            for point in matrix.get(change["config-key"], ({}, []))[0].values():
                models.add((str(point.get("model-prefix", "")), public_model_name(point)))
    return sorted(model for model in models if all(model))


def ingested_points(prefix: str, checkpoint: str, as_of: str | None) -> tuple[str, list[dict]]:
    """Ingested rows for a model prefix.

    The public API keys models by display name, which is usually the checkpoint name
    without quantization or size suffixes (``MiniMax-M3-NVFP4`` is ``MiniMax-M3``), so
    trailing tokens are dropped until a name returns rows for this prefix.
    """
    tokens = checkpoint.split("-")
    for end in range(len(tokens), 0, -1):
        name = "-".join(tokens[:end])
        try:
            rows = api.fetch("benchmarks", model=name, date=as_of, exact=False).payload
        except api.ReadError:
            continue
        matching = [row for row in rows if isinstance(row, dict) and row.get("model") == prefix]
        if matching:
            return name, matching
    raise LookupError(f"no public benchmark name for {prefix} ({checkpoint})")


# --- Check 18(c): newly referenced images ---------------------------------------------


def image_refs(text: str) -> set[str]:
    return {match.replace("#", "/") for match in IMAGE_REF.findall(text)}


def new_images(base: str, sha: str) -> list[str]:
    files = [
        path
        for path in _git("diff", "--name-only", base, sha).stdout.decode().splitlines()
        if path.startswith("inferencex-e2e/") and path.endswith((".yaml", ".yml"))
    ]
    introduced: set[str] = set()
    for path in files:
        after = _git("show", f"{sha}:{path}", check=False)
        before = _git("show", f"{base}:{path}", check=False)
        old = before.stdout.decode() if before.returncode == 0 else ""
        introduced |= image_refs(after.stdout.decode()) - image_refs(old)
    return sorted(introduced)


def inspect_image(image: str) -> dict[str, Any]:
    """Resolve a reference; a ``tag@digest`` pin resolves by its digest, which is what runs."""
    if "@" in image:
        name, _, digest = image.partition("@")
        repository = name.rsplit(":", 1)[0] if ":" in name.rpartition("/")[2] else name
        return {"image": image, **_inspect(f"{repository}@{digest}")}
    return {"image": image, **_inspect(image)}


def _inspect(reference: str) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["docker", "manifest", "inspect", reference],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"inspected": reference, "resolves": None, "error": type(error).__name__}
    if result.returncode == 0:
        return {"inspected": reference, "resolves": True}
    return {"inspected": reference, "resolves": False, "error": result.stderr.strip()[-300:]}


# --- Check 11(b): golden acceptance length --------------------------------------------


def golden_al(
    model: str, spec: dict[str, Any], thinking: str, framework: str, pinned: float | None
) -> dict[str, Any]:
    name = curves.curve_name(model, spec)
    expected = curves.golden_length(model, spec, thinking)
    trt = "trt" in framework.lower()
    setting = expected - 1 if trt else expected
    report = {
        "curve": f"{name}.yaml",
        "golden_al": expected,
        "expected_setting": round(setting, 6),
        "setting": "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS (AL - 1)"
        if trt
        else "SGLANG_SIMULATE_ACC_LEN or vLLM synthetic_acceptance_length",
    }
    if pinned is not None:
        report["pinned"] = pinned
        report["matches"] = math.isclose(pinned, setting, abs_tol=1e-6)
    return report


# --- facts report ---------------------------------------------------------------------


def _section(build: Callable[[], Any]) -> Any:
    try:
        return build()
    except Exception as error:  # noqa: BLE001 - a failed section is reported, never fatal
        return {"error": f"{type(error).__name__}: {str(error)[:500]}"}


def later_commits(since: str, sha: str) -> list[dict[str, Any]]:
    """Commits after the validated commit up to the pinned SHA, with parents and files."""
    log = (
        _git("log", "--first-parent", "--format=%H %P", f"{since}..{sha}")
        .stdout.decode()
        .splitlines()
    )
    commits = []
    for line in reversed(log):
        commit, *parents = line.split()
        files = _git(
            "diff-tree", "--no-commit-id", "--name-only", "-r", "-m", "--first-parent", commit
        )
        commits.append(
            {
                "sha": commit,
                "merge": len(parents) > 1,
                "subject": _git("log", "-1", "--format=%s", commit).stdout.decode().strip(),
                "files": sorted(set(files.stdout.decode().split())),
            }
        )
    return commits


def _eval_rows(repo: str, run_id: int, workdir: Path) -> list[dict]:
    target = workdir / str(run_id)
    for pattern in ("eval_results_*", "eval_*"):
        result = subprocess.run(
            [
                "gh",
                "run",
                "download",
                str(run_id),
                "--repo",
                repo,
                "-p",
                pattern,
                "-D",
                str(target),
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            break
    rows = []
    for path in sorted(target.rglob("agg_eval_all.json")):
        rows.extend(row for row in json.loads(path.read_text()) if isinstance(row, dict))
    return rows


def build_report(
    repo: str, pr: int, sha: str, base_ref: str, as_of: str | None = None
) -> dict[str, Any]:
    report: dict[str, Any] = {"repo": repo, "pr": pr, "pinned_sha": sha}
    base = merge_base(base_ref, sha)
    report["merge_base"] = base
    checks = _section(lambda: sweep_checks(repo, pr))
    report["check_1_commits"] = checks
    validated = (
        next(
            (
                commit
                for commit in reversed(checks)
                if commit["satisfied"]
                and _git("merge-base", "--is-ancestor", commit["sha"], sha, check=False).returncode
                == 0
            ),
            None,
        )
        if isinstance(checks, list)
        else None
    )
    report["check_1_latest_validated"] = validated and {
        "sha": validated["sha"],
        "run_ids": validated["run_ids"],
    }
    if validated:
        report["check_1_later_commits"] = _section(lambda: later_commits(validated["sha"], sha))
    with tempfile.TemporaryDirectory() as scratch:
        workdir = Path(scratch)
        if validated:

            def gaps() -> dict[str, Any]:
                feed = api.fetch("evaluations").payload
                return {
                    str(run_id): eval_gaps(_eval_rows(repo, run_id, workdir), feed, run_id, None)
                    for run_id in validated["run_ids"]
                }

            report["check_2_eval_gaps"] = _section(gaps)

        def diff() -> dict[str, Any]:
            before, before_errors = matrix_at(base, workdir / "base")
            after, after_errors = matrix_at(sha, workdir / "head")
            ingested, unavailable = [], {}
            for prefix, checkpoint in changed_models(before, after):
                try:
                    ingested += ingested_points(prefix, checkpoint, as_of)[1]
                except LookupError as error:
                    unavailable[prefix] = str(error)
            new_run = [
                row
                for run_id in (validated or {}).get("run_ids", [])
                for row in unofficial_run(run_id)
            ]
            return {
                "changes": points_diff(before, after, ingested, new_run),
                "ingested_models_unavailable": unavailable,
                "generation_errors": {
                    "base": before_errors,
                    "head": {
                        key: error
                        for key, error in after_errors.items()
                        if key not in before_errors
                    },
                },
            }

        report["check_18g_points"] = _section(diff)
    report["check_18c_images"] = _section(
        lambda: [inspect_image(image) for image in new_images(base, sha)]
    )
    return report


# --- command line ---------------------------------------------------------------------


def _fetch(resource: str, parser: argparse.ArgumentParser, **kwargs: Any) -> api.Feed:
    try:
        feed = api.fetch(resource, **kwargs)
    except api.ReadError as error:
        parser.exit(1, f"{resource} feed unavailable: {error}\n")
    if not isinstance(feed.payload, list):
        parser.exit(1, f"{resource} feed is not a list\n")
    return feed


def _series_args(command: argparse.ArgumentParser) -> None:
    for field in SERIES:
        command.add_argument("--" + field.replace("_", "-"), required=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    facts = commands.add_parser("report", help="all deterministic facts for one pinned PR commit")
    facts.add_argument("--repo", required=True)
    facts.add_argument("--pr", type=int, required=True)
    facts.add_argument("--sha", required=True)
    facts.add_argument("--base-ref", default="origin/main")
    facts.add_argument("--out", type=Path, required=True)
    facts.add_argument("--as-of", help="read ingested benchmarks as of this date (re-verification)")

    evals = commands.add_parser("evals", help="Check 2: ingested peer evaluations")
    _series_args(evals)
    evals.add_argument("--task", required=True)
    evals.add_argument("--limit", type=int, default=25)

    points = commands.add_parser("points", help="Check 18: ingested benchmark points of a series")
    points.add_argument(
        "--model-name", required=True, help="public model name, e.g. DeepSeek-V4.1-Flash"
    )
    _series_args(points)
    points.add_argument("--limit", type=int, default=25)

    dom = commands.add_parser("dominance", help="Check 18(g,h): removed-point dominance and deltas")
    dom.add_argument("--input", type=Path, required=True, help='{"before": [...], "after": [...]}')

    al = commands.add_parser("golden-al", help="Check 11(b): expected simulated acceptance length")
    al.add_argument("--model", required=True, help="model prefix, e.g. qwen3.5")
    al.add_argument("--method", required=True, help="speculative method, e.g. mtp, eagle3, dspark")
    al.add_argument("--tokens", type=int, required=True, help="num_speculative_tokens")
    al.add_argument("--thinking", choices=curves.THINKING_MODES, required=True)
    al.add_argument("--framework", required=True)
    al.add_argument("--draft-model", default="")
    al.add_argument("--draft-sample-method")
    al.add_argument("--pinned", type=float, help="value set in the recipe")

    args = parser.parse_args(argv)
    if args.command == "report":
        report: Any = build_report(args.repo, args.pr, args.sha, args.base_ref, args.as_of)
        args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
        failed = [
            key for key, value in report.items() if isinstance(value, dict) and "error" in value
        ]
        print(f"Wrote {args.out}; failed sections: {', '.join(failed) or 'none'}")
        return
    if args.command == "evals":
        feed = _fetch("evaluations", parser)
        target = {field: getattr(args, field) for field in SERIES} | {"task": args.task}
        report = {"source": feed.url, "retrieved_at": feed.retrieved_at}
        report |= eval_cohorts(feed.payload, target, args.limit)
    elif args.command == "points":
        feed = _fetch("benchmarks", parser, model=args.model_name)
        target = {field: getattr(args, field) for field in SERIES}
        report = {"source": feed.url, "retrieved_at": feed.retrieved_at}
        report |= series_points(feed.payload, target, args.limit)
    elif args.command == "dominance":
        data = json.loads(args.input.read_text())
        try:
            report = dominance(data["before"], data["after"])
        except (KeyError, TypeError, ValueError) as error:
            parser.exit(1, f"invalid dominance input: {error}\n")
    else:
        spec = {
            "method": args.method,
            "num_speculative_tokens": args.tokens,
            "model": args.draft_model,
            "draft_sample_method": args.draft_sample_method,
        }
        try:
            report = golden_al(args.model, spec, args.thinking, args.framework, args.pinned)
        except ValueError as error:
            parser.exit(1, f"no golden AL: {error}\n")
    json.dump(report, sys.stdout, separators=(",", ":"), default=str)
    print()


if __name__ == "__main__":
    main()
