"""Snapshot a model's published dashboard rows with the recipe variant behind each point.

A field missing from an optional part (prices, evaluations, a row's metrics) leaves that part
null and says why in ``warnings``. Exits 2 on bad arguments or an unreachable API, 3 when an
/api/v1 response lacks a field the core path needs (the model families, or a row's identity),
and 4 when the requested curve has rows but none binds; no published rows is not a failure.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from infx.config import MASTER_CONFIGS, repository_root
from infx.klaud.api import ReadError, fetch
from infx.matrix.revision import Revision, snapshot
from infx.srt_slurm.generate import bound_variant, point_environment
from infx.srt_slurm.variants import expand_variants

INFERENCEX = "SemiAnalysisAI/InferenceX"
RUN_URL = re.compile(r"https://github\.com/([\w.-]+/[\w.-]+)/actions/runs/(\d+)")
PULL_URL = re.compile(r"/pull/(\d+)")
BIND_ERRORS = (OSError, KeyError, TypeError, ValueError, yaml.YAMLError)
ENGINES = {
    "sglang": "sgl-project/sglang",
    "vllm": "vllm-project/vllm",
    "trt": "NVIDIA/TensorRT-LLM",
}
IMAGE_COMMIT = re.compile(r"-([0-9a-f]{7,40})(?:@|$)")
PRICE_KEY = "h"
IDENTITY = ("id", "model", "hardware", "framework", "precision", "conc", "benchmark_type")
DERIVED = ("tput_per_physical_gpu", "tok_per_usd")
# The frontier axes and the direction that is better: the metrics the snapshot reads or derives
BETTER = {
    **{f"{percentile}_intvty": "max" for percentile in ("median", "p75", "p90")},
    **{f"{p}_{latency}": "min" for p in ("median", "p75", "p90") for latency in ("ttft", "e2el")},
    **dict.fromkeys(("tput_per_gpu", "input_tput_per_gpu", "output_tput_per_gpu", *DERIVED), "max"),
}
DEFAULT_FRONTIER = ("p90_intvty", "tput_per_gpu")


def _spelling(name: str) -> str:
    return re.sub(r"[^0-9a-z]", "", name.casefold())


def family(models: list[dict], name: str) -> dict[str, Any]:
    """The dashboard model that shows ``name`` (a display name or a raw key in any spelling), the
    raw keys it pools, and ``name``'s own key, None when a display name pools several."""
    for entry in models:
        keys = entry["dbKeys"]
        named = [key for key in keys if _spelling(key) == _spelling(name)]
        if entry["name"] == name or named:
            model = named[0] if named else keys[0] if len(keys) == 1 else None
            return {
                "display_model": entry["name"],
                "pooled_models": keys,
                "model_key": model,
                "category": entry.get("category"),
            }
    raise ValueError(f"no dashboard model shows {name!r}")


def model_key(configs: list[Path], hf_model: str) -> str | None:
    """The model-prefix the matrix ``configs`` run ``hf_model`` under, if exactly one."""
    keys = {
        str(config.get("model-prefix"))
        for path in configs
        for config in (yaml.safe_load(path.read_text()) or {}).values()
        if isinstance(config, dict) and str(config.get("model")).casefold() == hf_model.casefold()
    }
    return keys.pop() if len(keys) == 1 else None


def curve_of(row: Mapping[str, Any]) -> dict[str, Any]:
    """The curve a row is drawn on: the API replaces a curve's points together."""
    fields = ("hardware", "framework", "precision", "benchmark_type", "isl", "osl")
    return {field: row.get(field) for field in fields}


def line_of(row: Mapping[str, Any]) -> list[Any]:
    """The line whose frontier the dashboard marks a row on: its curve on the curve's date, and
    for a fixed sequence its spec method too. Pooled models' curves share a line on one date."""
    spec = None if row.get("benchmark_type") == "agentic_traces" else row.get("spec_method")
    return [*curve_of(row).values(), spec, row.get("curve_date") or row.get("date")]


def family_rows(rows: list[dict], models: list[str], agentic: bool) -> list[dict]:
    """The rows of the pooled ``models``, AgentX only with ``agentic``, by curve and concurrency.

    The API returns only each curve's latest snapshot, so these are the rows the dashboard shows.
    Raise KeyError naming the first identity field a row lacks.
    """
    for row in rows:
        if missing := [field for field in IDENTITY if field not in row]:
            raise KeyError(missing[0])
    selected = [
        row
        for row in rows
        if row["model"] in models and (not agentic or row["benchmark_type"] == "agentic_traces")
    ]
    return sorted(
        selected,
        key=lambda row: (*map(str, curve_of(row).values()), int(row["conc"]), str(row["id"])),
    )


def interactivity(row: Mapping[str, Any], percentile: str = "p90") -> float | None:
    """Interactivity at ``percentile`` as the dashboard plots it: for AgentX, 1 / ITL,
    full-response where the row has full-response ITLs; for a fixed sequence, the published
    ``<percentile>_intvty``."""
    metrics = row.get("metrics") or {}
    if row.get("benchmark_type") != "agentic_traces":
        published = metrics.get(f"{percentile}_intvty")
        return published if isinstance(published, int | float) and published > 0 else None
    full = any(k.endswith("_full_response_itl") and k != "std_full_response_itl" for k in metrics)
    itl = metrics.get(f"{percentile}_full_response_itl")
    if not (isinstance(itl, int | float) and itl > 0):
        itl = None if full else metrics.get(f"{percentile}_itl")
    return 1 / itl if isinstance(itl, int | float) and itl > 0 else None


def metric(row: Mapping[str, Any], name: str) -> float | None:
    """``row``'s value of the frontier axis ``name``, or None: interactivity as the dashboard
    plots it, a field the snapshot derives, else the row's published metric."""
    if name.endswith("_intvty"):
        return interactivity(row, name.removesuffix("_intvty"))
    value = row.get(name) if name in DERIVED else (row.get("metrics") or {}).get(name)
    return value if isinstance(value, int | float) else None


def frontier(rows: list[dict], axes: tuple[str, ...] = DEFAULT_FRONTIER) -> list[dict]:
    """The rows of one line on its Pareto frontier of the two ``axes``, each toward its
    ``BETTER`` direction; by default P90 interactivity and tok/s/GPU, as the dashboard's
    interactivity chart draws it (paretoFrontUpperLeft)."""
    points = []
    for row in rows:
        values = [metric(row, name) for name in axes]
        if None not in values:
            signs = [1 if BETTER[name] == "max" else -1 for name in axes]
            points.append((signs[0] * values[0], signs[1] * values[1], row))
    front: list[tuple[float, float, dict]] = []
    for x, y, row in sorted(points, key=lambda point: (point[0], -point[1])):
        if front and x == front[-1][0]:
            if y > front[-1][1]:
                front[-1] = (x, y, row)
            continue
        while front and y >= front[-1][1]:
            front.pop()
        front.append((x, y, row))
    return [row for _, _, row in front]


def plotted(row: Mapping[str, Any], axes: tuple[str, ...] = DEFAULT_FRONTIER) -> bool:
    """Whether ``row`` has both ``axes``, by default those of the dashboard's interactivity chart."""
    return all(metric(row, name) is not None for name in axes)


def frontier_axes(text: str) -> tuple[str, ...]:
    """``X,Y`` as two frontier axes; raise ArgumentTypeError naming the valid ones."""
    axes = tuple(name.strip() for name in text.split(","))
    if len(axes) != 2 or not set(axes) <= BETTER.keys():
        raise argparse.ArgumentTypeError(f"{text!r} is not X,Y of {', '.join(BETTER)}")
    return axes


def physical_gpus(row: Mapping[str, Any]) -> int | None:
    """The GPUs a row's deployment holds; the dashboard reports an aggregated row's TP x EP."""
    prefill, decode = row.get("num_prefill_gpu"), row.get("num_decode_gpu")
    if row.get("disagg"):
        return prefill + decode if isinstance(prefill, int) and isinstance(decode, int) else None
    tp, ep = row.get("prefill_tp"), row.get("prefill_ep")
    if isinstance(tp, int) and isinstance(ep, int) and tp > 1 and ep > 1 and prefill == tp * ep:
        return max(tp, ep)
    return prefill if isinstance(prefill, int) else None


def engine(framework: str) -> str:
    return str(framework).removeprefix("dynamo-")


def engine_source(row: Mapping[str, Any]) -> dict[str, str | None]:
    """The engine repository a row ran and the commit its image tag names; a date is no commit."""
    tag = IMAGE_COMMIT.search(str(row.get("image") or ""))
    commit = tag[1] if tag and not re.fullmatch(r"20\d{6}", tag[1]) else None
    return {"repository": ENGINES.get(engine(row.get("framework", ""))), "commit": commit}


def published(row: dict, prices: Mapping[str, Any]) -> dict[str, Any]:
    """A dashboard row with its curve, physical GPUs, tok/$ and engine."""
    metrics = row.get("metrics") or {}
    gpus, total = physical_gpus(row), metrics.get("total_tput_tps")
    price = (prices.get(row.get("hardware")) or {}).get(PRICE_KEY)
    tput = metrics.get("tput_per_gpu")
    return {
        **row,
        "curve": curve_of(row),
        "physical_gpus": gpus,
        "tput_per_physical_gpu": total / gpus if gpus and isinstance(total, int | float) else None,
        "tok_per_usd": tput * 3600 / price if price and isinstance(tput, int | float) else None,
        "price_key": PRICE_KEY if price else None,
        "engine": engine_source(row),
    }


def run_dates(row: Mapping[str, Any]) -> list[str]:
    """The days whose workflow-info may list ``row``'s run: its start day, then its own date."""
    days = (str(row.get("run_started_at") or "")[:10], str(row.get("date") or ""))
    return [day for day in dict.fromkeys(days) if day]


def producer(
    row: Mapping[str, Any], infos: Mapping[str, dict], repository: str = INFERENCEX
) -> dict[str, Any]:
    """The repository, run, head commit and pull request that produced ``row``, from ``infos``
    (each of ``run_dates(row)``'s workflow-info), or why there is none in ``unbound``."""
    run = RUN_URL.match(str(row.get("run_url") or ""))
    if run is None:
        return {"unbound": "no GitHub Actions run URL"}
    found = {"repository": run[1], "run_id": run[2]}
    if run[1] != repository:
        return {**found, "unbound": f"its run is in {run[1]}, not {repository}"}
    for day in run_dates(row):
        info = infos.get(day) or {}
        heads = {
            entry["head_sha"]
            for entry in info.get("runConfigs", [])
            if entry.get("github_run_id") == run[2]
            and str(entry.get("html_url")).startswith(f"https://github.com/{repository}/")
            and entry.get("head_sha")
        }
        if len(heads) == 1:
            pulls = [
                match[1]
                for entry in info.get("changelogs", [])
                if entry.get("workflow_run_id") == run[2]
                and (match := PULL_URL.search(str(entry.get("pr_link"))))
            ]
            return {**found, "commit": heads.pop(), "pull": pulls[0] if pulls else None}
    return {**found, "unbound": "workflow-info lists no single head commit for the run"}


def fetch_commit(clone: Path, repository: str, commit: str, pull: str | None) -> str | None:
    """Fetch ``commit`` into ``clone``, through its pull request head (which may be a fork's) if
    GitHub will not serve it by id; None once fetched, else git's reason."""
    url = f"https://github.com/{repository}.git"
    error = "not fetchable"
    for ref in (commit, f"refs/pull/{pull}/head") if pull else (commit,):
        done = subprocess.run(
            ["git", "-C", str(clone), "fetch", "-q", "--depth", "1", url, ref],
            capture_output=True,
            text=True,
            check=False,
        )
        present = ["git", "-C", str(clone), "cat-file", "-e", f"{commit}^{{commit}}"]
        if subprocess.run(present, capture_output=True, check=False).returncode == 0:
            return None
        error = " ".join(done.stderr.split())[:300] or error
    return error


def points(
    revision: Revision, models: list[str], precision: str, hardware: str | None = None
) -> list[dict]:
    """The revision's matrix points of ``models`` in ``precision``, one per concurrency, as its
    own generator writes them with their recipe fingerprints. Only ``hardware``'s configs are
    generated when it has any, so another GPU's config the generator refuses fails none."""
    configs = [
        (key, config)
        for path in revision.master_configs
        for key, config in (yaml.safe_load(Path(path).read_text()) or {}).items()
        if isinstance(config, dict)
        and config.get("model-prefix") in models
        and str(config.get("precision")).lower() == precision
    ]
    keys = [key for key, config in configs if hardware and runs_on(config, hardware)] or [
        key for key, _ in configs
    ]
    rows = revision.generate(keys, []) if keys else []
    return [
        {**row, "conc": conc}
        for row in rows
        for conc in (row["conc"] if isinstance(row["conc"], list) else [row["conc"]])
    ]


def bind(rows: list[dict], matrix: list[dict]) -> tuple[list[tuple[dict, list[str]]], dict]:
    """Each matrix point with the ids of the ``rows`` it produced, matched by recipe fingerprint
    and concurrency as reporting joins them, and the reason each other row has none."""
    index: dict[tuple, dict] = {}
    for point in matrix:
        index.setdefault((point.get("recipe-fingerprint"), point["conc"]), point)
    matched: dict[int, tuple[dict, list[str]]] = {}
    unbound = {}
    missing = (
        "no matrix point has its fingerprint and concurrency"
        if any(fingerprint for fingerprint, _ in index)
        else "its commit's matrix points carry no recipe fingerprints"
    )
    for row in rows:
        point = index.get((row.get("recipe_fingerprint"), int(row["conc"])))
        if not row.get("recipe_fingerprint"):
            unbound[str(row["id"])] = "the row has no recipe fingerprint"
        elif point is None:
            unbound[str(row["id"])] = missing
        else:
            matched.setdefault(id(point), (point, []))[1].append(str(row["id"]))
    return list(matched.values()), unbound


def variant(revision: Revision, point: dict) -> dict[str, Any]:
    """``point``'s recipe file, the variant the launcher selects for it, and that variant as
    srtctl expands it. Raise one of BIND_ERRORS when the point has no bindable recipe."""
    if not point.get("srt-recipe"):
        raise ValueError("the point has no srt recipe")
    run = {**point, "conc": [point["conc"]]} if "prefill" in point else point
    selector, _ = bound_variant(run, point_environment(run), revision.root, expand=expand_variants)
    path = point["srt-recipe"].partition(":")[0]
    text = (revision.root / path).read_text()
    expanded = expand_variants(yaml.safe_load(text), selector)
    if len(expanded) != 1:
        raise ValueError(f"{path}:{selector} expands to {len(expanded)} variants, not one")
    return {"path": path, "selector": selector, "text": text, "recipe": expanded[0][1]}


def write_variant(out: Path, commit: str, point: dict, found: dict, row_ids: list[str]) -> dict:
    """Write ``found``'s recipe files under ``out`` and return its snapshot entry."""
    identity = json.dumps([commit, found["path"], found["selector"]], separators=(",", ":"))
    name = (found["selector"] or Path(found["path"]).stem).replace("[", "-").replace("]", "")
    file = f"recipes/{name}-{hashlib.sha256(identity.encode()).hexdigest()[:8]}.yaml"
    source = f"sources/{commit}/{found['path']}"
    for path, text in (
        (file, yaml.safe_dump(found["recipe"], sort_keys=False)),
        (source, found["text"]),
    ):
        (out / path).parent.mkdir(parents=True, exist_ok=True)
        (out / path).write_text(text)
    return {
        "file": file,
        "source_file": source,
        "commit": commit,
        "path": found["path"],
        "selector": found["selector"],
        "blob_sha256": hashlib.sha256(found["text"].encode()).hexdigest(),
        "recipe_fingerprint": point.get("recipe-fingerprint"),
        "conc": point["conc"],
        "image": point.get("image"),
        "model": point.get("model"),
        "model_prefix": point.get("model-prefix"),
        "matrix": point,
        "row_ids": row_ids,
    }


def evaluated(rows: list[dict], target: Mapping[str, Any], precision: str) -> list[dict]:
    """The evaluations of the target's own model (each pooled one without a key) in ``precision``:
    accuracy stays per model."""
    models = [target["model_key"]] if target["model_key"] else target["pooled_models"]
    return [row for row in rows if row.get("model") in models and row.get("precision") == precision]


def runs_on(point: Mapping[str, Any], hardware: str) -> bool:
    """Whether a matrix point's runner (``cluster:gb300-nv``) is named for ``hardware``."""
    runner = str(point.get("runner")).removeprefix("cluster:").lower()
    return runner == hardware or runner.startswith(hardware + "-")


def hints(clone: Path, commit: str, matrix: list[dict], row: Mapping[str, Any]) -> dict:
    """Where to look at ``commit`` for an unbound row's recipe: the matrix points of its engine
    on its GPU at its concurrency (at any, with none at its own), and the single-node scripts
    named for its model, precision and GPU."""
    hardware = str(row.get("hardware"))
    lane = [
        point
        for point in matrix
        if engine(point.get("framework", "")) == engine(row.get("framework", ""))
        and runs_on(point, hardware)
    ]
    near = [point for point in lane if point["conc"] == int(row["conc"])] or lane
    tree = ["git", "-C", str(clone), "ls-tree", "-r", "--name-only", commit]
    paths = subprocess.run(tree, capture_output=True, text=True, check=False).stdout.split()
    prefix = f"{row.get('model')}_{row.get('precision')}_{hardware}"
    return {
        "points": [
            {"exp_name": name, "recipe": recipe}
            for name, recipe in sorted(
                {(point.get("exp-name"), point.get("srt-recipe")) for point in near}, key=str
            )
        ],
        "scripts": [
            path
            for path in paths
            if "single_node/" in path
            and Path(path).name.startswith(prefix)
            and path.endswith(".sh")
        ],
    }


def bind_commits(
    commits: dict[str, list[dict]],
    producers: dict,
    models: list[str],
    precision: str,
    hardware: str,
    out: Path,
) -> tuple[list[dict], dict, dict]:
    """Bind each commit's rows of ``hardware``; return the variants written under ``out``, why
    each other row is unbound, and hints for those whose commit could be read."""
    variants: list[dict] = []
    unbound: dict[str, str] = {}
    found_hints: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="infx-published-") as clone:
        subprocess.run(["git", "init", "-q", clone], check=True)
        for commit, group in commits.items():
            ids = [str(row["id"]) for row in group]
            found = producers[ids[0]]
            if error := fetch_commit(Path(clone), found["repository"], commit, found["pull"]):
                unbound.update(dict.fromkeys(ids, f"commit unavailable: {error}"))
                continue
            try:
                with contextlib.chdir(clone), snapshot(commit) as revision:
                    matrix = points(revision, models, precision, hardware)
                    matched, missing = bind(group, matrix)
                    for point, row_ids in matched:
                        try:
                            bound = variant(revision, point)
                        except BIND_ERRORS as error:
                            missing.update(
                                dict.fromkeys(row_ids, f"{type(error).__name__}: {error}")
                            )
                            continue
                        variants.append(write_variant(out, commit, point, bound, row_ids))
            except (subprocess.CalledProcessError, ValueError) as error:
                said = (getattr(error, "stderr", None) or "").strip().splitlines()
                reason = f"the commit's matrix cannot be generated: {said[-1] if said else error}"
                unbound.update(dict.fromkeys(ids, reason[:400]))
                continue
            unbound.update(missing)
            for row in group:
                if str(row["id"]) in missing:
                    found_hints[str(row["id"])] = hints(Path(clone), commit, matrix, row)
    return variants, unbound, found_hints


def main(argv: list[str] | None = None) -> int:
    """Write the snapshot named on the command line and return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="dashboard model or raw model key")
    parser.add_argument("--hf-model", help="checkpoint whose matrix model-prefix is the model key")
    parser.add_argument("--hardware", type=str.lower, required=True, help="e.g. gb300")
    parser.add_argument("--framework", type=str.lower, required=True, help="e.g. dynamo-sglang")
    parser.add_argument("--precision", type=str.lower, required=True, help="e.g. fp4")
    parser.add_argument("--agentic", action="store_true", help="only AgentX (agentic_traces) rows")
    parser.add_argument("--output-dir", type=Path, required=True, help="new or empty directory")
    parser.add_argument("--repository", default=INFERENCEX, help="the only repository bound")
    parser.add_argument(
        "--frontier",
        type=frontier_axes,
        action="append",
        metavar="X,Y",
        help=f"also mark each row's frontier on these axes (repeatable): {', '.join(BETTER)}",
    )
    args = parser.parse_args(argv)
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} is not empty")
    try:
        options, warnings = fetch("options"), []
    except ReadError as error:
        parser.error(str(error))
    try:
        target = family(options.payload["models"], args.model)
    except (KeyError, TypeError) as error:
        return lacks("/api/v1/views/options", error)
    except ValueError as error:
        parser.error(str(error))
    configs = [repository_root() / path for path in MASTER_CONFIGS]
    if args.hf_model and (key := model_key(configs, args.hf_model)):
        if key not in target["pooled_models"]:
            parser.error(f"{args.hf_model} runs as {key}, not a {target['display_model']}")
        target["model_key"] = key
    try:
        benchmarks = fetch("benchmarks", model=target["display_model"])
    except ReadError as error:
        parser.error(str(error))
    requested = (args.hardware, args.framework, args.precision)
    try:
        rows = family_rows(benchmarks.payload, target["pooled_models"], args.agentic)
    except (KeyError, TypeError, ValueError) as error:
        return lacks("/api/v1/benchmarks", error)
    mine = [
        row for row in rows if (row["hardware"], row["framework"], row["precision"]) == requested
    ]
    infos, evaluations = {}, None
    scope = "agentic_traces" if args.agentic else None
    for day in sorted({day for row in mine for day in run_dates(row)}):
        try:
            infos[day] = fetch("workflow-info", date=day, benchmark_type=scope)
        except ReadError as error:
            warnings.append(
                f"/api/v1/workflow-info for {day} unavailable ({error}): its runs unbound"
            )
    try:
        evaluations = fetch("evaluations")
    except ReadError as error:
        warnings.append(f"/api/v1/evaluations unavailable ({error}): no evaluations")
    payloads = {day: feed.payload for day, feed in infos.items()}
    prices = {
        entry.get("key"): entry.get("costPerHour")
        for entry in options.payload.get("hardware") or []
    }
    if unpriced := sorted(
        {row["hardware"] for row in rows if not (prices.get(row["hardware"]) or {}).get(PRICE_KEY)}
    ):
        warnings.append(
            f"/api/v1/views/options has no costPerHour.{PRICE_KEY} for {unpriced}: tok_per_usd null"
        )
    if unplotted := len([row for row in rows if not plotted(row)]):
        warnings.append(f"{unplotted} rows lack P90 ITL or tput_per_gpu: on_frontier null")
    producers = {str(row["id"]): producer(row, payloads, args.repository) for row in mine}
    out.mkdir(parents=True, exist_ok=True)
    commits: dict[str, list[dict]] = {}
    for row in mine:
        commits.setdefault(producers[str(row["id"])].get("commit"), []).append(row)
    commits.pop(None, None)
    variants, unbound, found_hints = bind_commits(
        commits, producers, target["pooled_models"], args.precision, args.hardware, out
    )
    unbound |= {key: found["unbound"] for key, found in producers.items() if "unbound" in found}
    files = {row_id: item["file"] for item in variants for row_id in item["row_ids"]}
    entries = [published(row, prices) for row in rows]
    lines: dict[str, list[dict]] = {}
    for entry in entries:
        lines.setdefault(json.dumps(line_of(entry)), []).append(entry)
    marked = {
        axes: {id(entry) for group in lines.values() for entry in frontier(group, axes)}
        for axes in (DEFAULT_FRONTIER, *(args.frontier or []))
    }
    snapshot_rows = []
    for row, entry in zip(rows, entries, strict=True):
        entry["on_frontier"] = id(entry) in marked[DEFAULT_FRONTIER] if plotted(entry) else None
        if args.frontier:
            entry["frontiers"] = {
                ",".join(axes): id(entry) in marked[axes] if plotted(entry, axes) else None
                for axes in args.frontier
            }
        if str(row["id"]) in producers:
            entry |= {
                "requested": True,
                "inferencex_commit": producers[str(row["id"])].get("commit"),
                "variant": files.get(str(row["id"])),
                "unbound": unbound.get(str(row["id"])),
                "hints": found_hints.get(str(row["id"])),
            }
        snapshot_rows.append(entry)
    document = {
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "target": {
            **vars(args),
            "output_dir": str(out),
            **target,
            "cost_per_hour": prices.get(args.hardware),
        },
        "sources": {
            feed.url: feed.sha256
            for feed in (options, benchmarks, evaluations, *infos.values())
            if feed
        },
        "rows": snapshot_rows,
        "variants": variants,
        "evaluations": evaluated(evaluations.payload, target, args.precision)
        if evaluations
        else [],
        "warnings": warnings,
    }
    if not args.frontier:
        del document["target"]["frontier"]
    (out / "snapshot.json").write_text(json.dumps(document, indent=1) + "\n")
    print(
        f"{len(rows)} published rows, {len(mine)} on the requested curve: {len(files)} bound to "
        f"{len(variants)} variants at {len(commits)} commits, {len(unbound)} unbound; wrote {out}"
    )
    if mine and not files:
        print(f"none of the {len(mine)} rows of the requested curve bound", file=sys.stderr)
        return 4
    return 0


def lacks(endpoint: str, error: Exception) -> int:
    """Say that ``endpoint``'s response lacks a field the core path needs; return the exit status."""
    reason = f"{type(error).__name__}: {error}"
    print(f"{endpoint} lacks a field the snapshot needs ({reason})", file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
