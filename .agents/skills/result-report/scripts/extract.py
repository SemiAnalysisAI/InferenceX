"""Turn benchmark files into report points: throughput, latency, cache tiers, search summary."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

PCTS = ("avg", "p50", "p90", "p95", "p99")
GRID_RE = re.compile(r"^(sweep|fixed)-isl(\d+k?)-osl(\d+k?)-cache(\d+)$")
TO_MS = {"ms": 1.0, "sec": 1000.0, "s": 1000.0, "us": 1e-3, "ns": 1e-6}


def tokens(label: str) -> int:
    return int(label[:-1]) * 1024 if label.endswith("k") else int(label)


def parse_grid(scenario: str | None) -> dict | None:
    # source: utils/serving/collect_sweep_table.py:11 (widened to fixed-*; no raise on other names)
    match = GRID_RE.match(scenario or "")
    if not match:
        return None
    kind, isl, osl, cache = match.groups()
    return {"kind": "search" if kind == "sweep" else "fixed", "isl": isl, "osl": osl, "cache": int(cache)}


def ms_stats(entry: object) -> dict:
    if not isinstance(entry, dict):
        return dict.fromkeys(PCTS)
    scale = TO_MS.get(entry.get("unit", "ms"), 1.0)
    return {k: entry[k] * scale if entry.get(k) is not None else None for k in PCTS}


def _avg(export: dict, tag: str) -> float | None:
    entry = export.get(tag)
    return entry.get("avg") if isinstance(entry, dict) else None


def _empty_metrics(source: str) -> dict:
    return {
        "throughput": dict.fromkeys(("rps", "input_tps", "output_tps", "total_tps", "output_tps_per_gpu")),
        "ttft": dict.fromkeys(PCTS), "itl": dict.fromkeys(PCTS), "e2e": dict.fromkeys(PCTS),
        "tps_per_user": None, "error_rate_pct": None, "errors": None, "client_cache_hit": None,
        "source": source,
    }


def from_aiperf(export: dict) -> dict:
    """profile_export_aiperf.json: latency in ms with avg..p99, throughputs carry avg only."""
    m = _empty_metrics("aiperf")
    m["throughput"].update(
        rps=_avg(export, "request_throughput"),
        input_tps=_avg(export, "input_token_throughput"),
        output_tps=_avg(export, "output_token_throughput"),
        total_tps=_avg(export, "total_token_throughput"),
    )
    m["ttft"] = ms_stats(export.get("time_to_first_token"))
    m["itl"] = ms_stats(export.get("inter_token_latency"))
    m["e2e"] = ms_stats(export.get("request_latency"))
    m["tps_per_user"] = _avg(export, "output_token_throughput_per_user")
    rate, errors = _avg(export, "request_error_rate"), _avg(export, "error_request_count")
    if rate is None and errors is None and _avg(export, "request_count") is not None:
        rate, errors = 0.0, 0  # aiperf omits the error metrics when no request failed
    m["error_rate_pct"] = rate
    m["errors"] = int(errors) if errors is not None else None
    hit = _avg(export, "overall_usage_prompt_cache_read_pct")
    m["client_cache_hit"] = hit / 100 if hit is not None else None
    return m


def _ms(seconds: float | None) -> float | None:
    return seconds * 1000.0 if seconds is not None else None


def from_aggregate(agg: dict) -> dict:
    """Aggregate JSON fallback: latency in seconds with mean..p95 (no p99)."""
    m = _empty_metrics("aggregate")
    rm = agg.get("request_metrics") or {}
    latency, thr = rm.get("latency") or {}, rm.get("throughput") or {}
    for key, src in (("ttft", "ttft"), ("itl", "itl"), ("e2e", "e2el")):
        d = latency.get(src) or {}
        m[key] = {"avg": _ms(d.get("mean")), "p50": _ms(d.get("p50")), "p90": _ms(d.get("p90")),
                  "p95": _ms(d.get("p95")), "p99": None}
    duration, ok = thr.get("duration_seconds"), agg.get("num_requests_successful")
    m["throughput"].update(
        rps=ok / duration if ok and duration else None,
        input_tps=(thr.get("input") or {}).get("tokens_per_second"),
        output_tps=(thr.get("output") or {}).get("tokens_per_second"),
        total_tps=(thr.get("total") or {}).get("tokens_per_second"),
    )
    return m


def from_search_aggregate(agg: dict) -> dict:
    """Search-mode aggregate (infx/results/sweep.py): winning probe's rps/tps and ttft/itl avg+p90 in ms."""
    m = _empty_metrics("search_aggregate")
    m["throughput"].update(rps=agg.get("rps"), output_tps=agg.get("output_tps"))
    m["ttft"].update(avg=agg.get("ttft_avg_ms"), p90=agg.get("ttft_p90_ms"))
    m["itl"].update(avg=agg.get("itl_avg_ms"), p90=agg.get("itl_p90_ms"))
    return m


def _cache(source: str, engine: str | None) -> dict:
    return {"overall": None, "gpu": None, "cpu": None, "external": None, "router": None,
            "tokens_by_tier": {}, "engine": engine, "source": source}


def aggregate_tiers(agg: dict) -> dict | None:
    sm = agg.get("server_metrics") or {}
    if not sm.get("present"):
        return None
    cache = sm.get("cache") or {}
    out = _cache("aggregate", sm.get("adapter"))
    tiers = dict(cache.get("cached_tokens_by_source") or {})
    overall = cache.get("overall_cache_hit_rate")
    if "total" in tiers:  # the repo adapter sums a cache_source=total series into overall
        tiers.pop("total")
        prompt_total = (sm.get("tokens") or {}).get("prompt_total")
        overall = sum(tiers.values()) / prompt_total if prompt_total else None
    router = cache.get("router_kv_hit_rate")
    out.update(
        overall=overall,
        gpu=cache.get("gpu_cache_hit_rate"),
        # vLLM copies its external (offload) rate into cpu: the tiers are not separable
        cpu=None if out["engine"] == "vllm" else cache.get("cpu_cache_hit_rate"),
        external=cache.get("external_cache_hit_rate"),
        router=router if router is not None else cache.get("frontend_cache_hit_rate"),
        tokens_by_tier=tiers,
    )
    return out


def _series(metrics: dict, name: str) -> list[dict]:
    entry = metrics.get(name)
    return [s for s in entry.get("series", []) if isinstance(s, dict)] if isinstance(entry, dict) else []


def _value(series: dict) -> float | None:
    # source: infx/results/agentic/common.py:114 (series_stat, preferred total/sum/max/avg)
    stats = series.get("stats") or {}
    for key in ("total", "sum", "max", "avg"):
        if stats.get(key) is not None:
            return float(stats[key])
    return None


def _sum(metrics: dict, name: str) -> float | None:
    values = [v for s in _series(metrics, name) if (v := _value(s)) is not None]
    return sum(values) if values else None


def _by_label(metrics: dict, name: str, label: str) -> dict[str, float]:
    # source: infx/results/agentic/common.py:186 (sum_by_label)
    out: dict[str, float] = {}
    for s in _series(metrics, name):
        key, value = (s.get("labels") or {}).get(label), _value(s)
        if key is not None and value is not None:
            out[str(key)] = out.get(str(key), 0.0) + value
    return out


def _rate(num: float | None, den: float | None) -> float | None:
    return num / den if num is not None and den else None


def tiers_from_server_metrics(export: dict) -> dict:
    """Cache tiers from a (small, per-probe) server_metrics_export.json."""
    metrics = export.get("metrics") if isinstance(export.get("metrics"), dict) else {}
    names = set(metrics)
    if any(n.startswith("sglang:") for n in names):
        # source: infx/results/agentic/backends/sglang.py:53-81, minus the cache_source=total double count
        out = _cache("server_metrics", "sglang")
        prompt = _sum(metrics, "sglang:prompt_tokens")
        by = _by_label(metrics, "sglang:cached_tokens", "cache_source")
        by.pop("total", None)
        storage = [v for k, v in by.items() if k.startswith("storage_")]
        out.update(
            gpu=_rate(by.get("device"), prompt),
            cpu=_rate(by.get("host"), prompt),
            external=_rate(sum(storage), prompt) if storage else None,
            overall=_rate(sum(by.values()), prompt) if by else None,
            tokens_by_tier=by,
        )
    elif any(n.startswith("vllm:") for n in names):
        # source: infx/results/agentic/backends/vllm.py:44-95; cpu stays None (not separable from L3)
        out = _cache("server_metrics", "vllm")
        by = _by_label(metrics, "vllm:prompt_tokens_by_source", "source")
        total = sum(by.values())
        if total > 0:
            local, external = by.get("local_cache_hit"), by.get("external_kv_transfer")
            out.update(
                gpu=_rate(local, total),
                external=_rate(external, total),
                overall=((local or 0.0) + (external or 0.0)) / total,
                tokens_by_tier={k: by[k] for k in ("local_cache_hit", "external_kv_transfer") if k in by},
            )
        else:
            gpu = _rate(_sum(metrics, "vllm:prefix_cache_hits"), _sum(metrics, "vllm:prefix_cache_queries"))
            out.update(gpu=gpu, overall=gpu, external=_rate(
                _sum(metrics, "vllm:external_prefix_cache_hits"), _sum(metrics, "vllm:external_prefix_cache_queries")))
    else:
        out = _cache("server_metrics", None)
    hits = _by_label(metrics, "smg_cache_aware_hit", "result")
    if hits:
        out["router"] = _rate(hits.get("hit"), sum(hits.values()))
    return out


SCENARIO_CMD_RE = re.compile(r"--config\s+\S*?scenarios/([\w.-]+)\.yaml")
CONC_CMD_RE = re.compile(r"--concurrency\s+(\d+)")
DIR_CONC_RE = re.compile(r"(?:ccu|conc|_c)(\d+)")
SERVING_SCENARIO_RE = re.compile(r"^scenario:\s*([\w.-]+)\s*$", re.M)
SERVING_NAME_RE = re.compile(r"^name:\s*(\S+)", re.M)
GRID_SUFFIX_RE = re.compile(r"-(?:(?:sweep|fixed)-)?isl\d+k?-osl\d+k?-cache\d+$")
RUN_SUFFIX_RE = re.compile(r"(?:_c\d+|_search)$")
COMMENT_RE = re.compile(r"^\s*#.*\n", re.M)
PER_RUN_KEYS_RE = re.compile(r"^(?:name|scenario|conc):.*\n?(?:\s*-\s.*\n?)*", re.M)
IDENTITY_KEYS = ("model", "image", "framework", "precision", "tp", "ep", "dp_attention",
                 "spec_decoding", "num_gpus", "hw")
CONFIG_FILES = ("serving.yaml", "server_command.txt", "router_command.txt", "benchmark_command.txt")
SLA_NAMES = {"time_to_first_token": "TTFT", "inter_token_latency": "ITL", "request_latency": "E2E"}
SLA_OPS = {"lt": "<", "le": "≤", "gt": ">", "ge": "≥"}
MAX_MEMBER_MB = 40  # largest compressed server_metrics_export.json read (real probes run 25-35 MB each)
PROBE_DIR = "aiperf_artifacts/search_iter_{:04d}/profile_runs/run_0001/"


def summarize_search(history: dict) -> dict:
    # source: infx/results/sweep.py:32 (copied: that module imports env-bound identity helpers)
    boundary = history.get("boundary_summary")
    reason = history.get("convergence_reason")
    base = {"convergence_reason": reason, "iterations": len(history.get("iterations") or []),
            "max_conc": None, "infeasible_min": None, "binding_constraint": None, "boundary_type": None}
    if boundary is None:
        return {**base, "state": "no_data"}
    passed, failed = boundary.get("feasible_max"), boundary.get("infeasible_min")
    base["max_conc"] = passed["value"] if passed else None
    base["infeasible_min"] = failed["value"] if failed else None
    base["binding_constraint"] = boundary.get("binding_constraint")
    base["boundary_type"] = boundary.get("boundary_type")
    if passed is None:
        state = "no_pass_in_range"
    elif failed is None:
        state = "no_failure_in_range"
    elif boundary.get("boundary_type") == "cliff":
        state = "cliff"
    elif reason == "max_iterations":
        state = "max_iterations"
    else:
        state = "ok"
    return {**base, "state": state}


def sla_text(f: dict) -> str:
    tag = f.get("metric_tag") or "—"
    unit = " ms" if tag in SLA_NAMES else ""
    thr = f.get("threshold")
    op = f.get("op")
    return (f"{SLA_NAMES.get(tag, tag)} {f.get('stat') or '—'} "
            f"{SLA_OPS.get(op, op) or '—'} {'—' if thr is None else format(thr, 'g')}{unit if thr is not None else ''}")


def _read_json(files, *paths: str):
    for path in paths:
        data = files.read(path) if files else None
        if data is not None:
            return json.loads(data)
    return None


def _read_text(files, path: str) -> str | None:
    data = files.read(path) if files else None
    return data.decode("utf-8", "replace") if data is not None else None


def _match(pattern: re.Pattern, text: str | None) -> str | None:
    found = pattern.search(text or "")
    return found.group(1) if found else None


def _observed(export: dict | None, f: dict) -> float | None:
    if not export or f.get("stat") not in PCTS:
        return None
    return ms_stats(export.get(f.get("metric_tag")))[f["stat"]]


def search_summary(history: dict, files, notes: list[str] | None = None,
                   max_member_mb: float = MAX_MEMBER_MB) -> tuple[dict, dict | None, dict | None]:
    """(search block, winning probe metrics, winning probe cache tiers).

    Per-probe numbers come from the small aiperf export. server_metrics_export.json is hundreds of MB
    uncompressed, so it is read only for the winning probe and only if its compressed size is <= the cap."""
    summary = summarize_search(history)
    filters = (history.get("config") or {}).get("sla_filters") or []
    boundary = history.get("boundary_summary") or {}
    winner = (boundary.get("feasible_max") or {}).get("iteration_idx")
    best = ((history.get("best_trials") or [{}])[0] or {}).get("iteration_idx")
    breach = (boundary.get("infeasible_min") or {}).get("first_breach")
    probes, win_metrics, win_cache = [], None, None
    for it in history.get("iterations") or []:
        idx = it["iteration_idx"]
        export = _read_json(files, PROBE_DIR.format(idx) + "profile_export_aiperf.json")
        metrics = from_aiperf(export) if export else None
        if idx == winner:
            win_metrics = metrics
            sm_path = PROBE_DIR.format(idx) + "server_metrics_export.json"
            size = files.size(sm_path) if files else None
            if size is None:
                pass  # absent, or size unknown: never read blind (a missing member has nothing to read)
            elif size > max_member_mb * 1e6:
                if notes is not None:
                    kind = getattr(files, "size_kind", "compressed")
                    notes.append(f"cache tiers unavailable for winner probe: server_metrics_export.json is "
                                 f"{size / 1e6:.0f} MB {kind} (cap {max_member_mb:g} MB)")
            else:
                sm = _read_json(files, sm_path)
                win_cache = tiers_from_server_metrics(sm) if sm else None
        probes.append({
            "iter": idx,
            "conc": next(iter((it.get("variation_values") or {}).values()), None),
            "feasible": it.get("feasible"),
            "objective": (it.get("objective_values") or [None])[0],
            "observed": {f"{f['metric_tag']}.{f['stat']}": _observed(export, f) for f in filters},
            "output_tps": metrics["throughput"]["output_tps"] if metrics else None,
            "cache_overall": metrics["client_cache_hit"] if metrics else None,
            "winner": idx == winner,
            "best_objective": idx == best,
        })
    search = {
        **summary,
        "sla": [sla_text(f) for f in filters],
        "sla_filters": filters,
        "binding": sla_text(breach) if breach else None,
        "breach_observed": breach.get("observed") if breach else None,
        "best_trials_conc": next((p["conc"] for p in probes if p["best_objective"]), None),
        "probes": probes,
    }
    return search, win_metrics, win_cache


def scenario_of(agg: dict, config: dict) -> str | None:
    """Scenario label: the aggregate's, else the scenario file in the aiperf command, else serving.yaml's."""
    return (agg.get("scenario") or _match(SCENARIO_CMD_RE, config["benchmark_command.txt"])
            or _match(SERVING_SCENARIO_RE, config["serving.yaml"]))


def build_point(cfg, max_member_mb: float = MAX_MEMBER_MB) -> dict:
    agg, files = cfg.agg or {}, cfg.files
    config = {name: _read_text(files, name) for name in CONFIG_FILES}
    serving, bench_cmd = config["serving.yaml"], config["benchmark_command.txt"]
    history = _read_json(files, "aiperf_artifacts/search_history.json", "search_history.json")
    scenario = scenario_of(agg, config)
    notes: list[str] = []
    point = {"rf": cfg.rf, "source": cfg.source.link, "scenario": scenario, "grid": parse_grid(scenario),
             "identity": {k: agg.get(k) for k in IDENTITY_KEYS}, "config": config, "notes": notes,
             "search": None}
    if history:
        search, metrics, cache = search_summary(history, files, notes, max_member_mb)
        point.update(mode="search", conc=search["max_conc"], search=search)
        if metrics is None and agg.get("output_tps") is not None:
            metrics = from_search_aggregate(agg)
            notes.append("winning probe export missing: metrics from the search aggregate (avg/p90 only)")
        elif metrics is None:
            notes.append("winning probe export missing: no metrics for the search point")
        if cache is None and agg:
            cache = aggregate_tiers(agg)  # search aggregate may carry server_metrics.cache
    elif agg.get("state"):  # search aggregate without its raw tree
        point.update(mode="search", conc=agg.get("max_conc"), search=None)
        metrics, cache = from_search_aggregate(agg), None
        notes.append("search raw tree missing: no probe table; metrics from the search aggregate (avg/p90 only)")
    else:
        export = _read_json(files, "aiperf_artifacts/profile_export_aiperf.json", "profile_export_aiperf.json")
        point["mode"] = "fixed"
        if export:
            metrics = from_aiperf(export)
        elif agg.get("request_metrics"):
            metrics = from_aggregate(agg)
            notes.append("p99 unavailable: aggregate-only (no aiperf export); latency converted from seconds")
        else:
            metrics = None
        cache = aggregate_tiers(agg) if agg else None
        conc = (agg.get("conc") or _match(CONC_CMD_RE, bench_cmd)
                or (((export or {}).get("input_config") or {}).get("loadgen") or {}).get("concurrency")
                or _match(DIR_CONC_RE, cfg.rf))
        point["conc"] = int(conc) if conc is not None else None
    if cache is None and metrics and metrics.get("client_cache_hit") is not None:
        cache = {**_cache("client", None), "overall": metrics["client_cache_hit"]}
    gpus = agg.get("num_gpus") or serving_identity(serving).get("num_gpus")
    if metrics and gpus and metrics["throughput"]["output_tps"] is not None:
        metrics["throughput"]["output_tps_per_gpu"] = metrics["throughput"]["output_tps"] / gpus
    validator = (agg.get("request_accounting") or {}).get("records_error_dropped")
    errors = metrics.get("errors") if metrics else None
    if validator is not None and errors is not None and validator - errors > max(10, errors):
        notes.append(f"validator dropped {validator} error records vs {errors} aiperf errors")
    point["metrics"], point["cache"] = metrics, cache
    set_cache_check(point)
    return point


def set_cache_check(point: dict) -> None:
    if point["grid"]:
        nominal = point["grid"]["cache"] / 100
        measured = point["cache"]["overall"] if point["cache"] else None
        point["cache_check"] = {"nominal": nominal, "measured": measured,
                                "mismatch": None if measured is None else abs(measured - nominal) > 0.10}
    else:
        point["cache_check"] = None


SERVING_FIELDS = {"model": "model", "image": "image", "precision": "precision", "num_gpus": "gpus", "hw": "runner"}


def serving_identity(serving: str | None) -> dict:
    """Identity fields from top-level `key: value` lines of serving.yaml (no yaml parser needed)."""
    out: dict = {}
    for ident, key in SERVING_FIELDS.items():
        found = re.search(rf"^{key}:[ \t]*(\S.*?)(?:[ \t]+#.*)?\s*$", COMMENT_RE.sub("", serving or ""), re.M)
        if found:
            value = found.group(1).strip("'\"")
            out[ident] = int(value) if ident == "num_gpus" and value.isdigit() else value
    return out


def _strip_run_suffix(name: str) -> str:
    return GRID_SUFFIX_RE.sub("", RUN_SUFFIX_RE.sub("", name))


def arm_of(point: dict, cfg) -> tuple[str, str]:
    """Arm = same serving config. name/scenario/conc and comments do not split an arm."""
    serving = point["config"]["serving.yaml"]
    if serving:
        body = PER_RUN_KEYS_RE.sub("", COMMENT_RE.sub("", serving))
        return ("yaml:" + hashlib.sha1(body.encode()).hexdigest()[:12],
                _strip_run_suffix(_match(SERVING_NAME_RE, serving) or cfg.rf))
    if cfg.agg:
        return ("id:" + json.dumps(point["identity"], sort_keys=True),
                _strip_run_suffix(cfg.agg.get("infmax_model_prefix") or cfg.rf))
    return "src:" + cfg.source.link, Path(cfg.source.link).name or cfg.source.link


def build_data(configs, notes, baseline=None, title=None, inputs=(), generated="",
               max_member_mb: float = MAX_MEMBER_MB) -> dict:
    points = []
    built = []
    for cfg in configs:
        point = build_point(cfg, max_member_mb)
        built.append((cfg, point, *arm_of(point, cfg)))
    # an aggregate-only config (no serving.yaml) joins a yaml-backed arm only if exactly one has its identity
    ident = lambda pt: json.dumps(pt["identity"], sort_keys=True)
    yaml_arms: dict[str, set[str]] = {}
    for _, pt, k, _ in built:
        if k.startswith("yaml:"):
            yaml_arms.setdefault(ident(pt), set()).add(k)
    notes = list(notes)
    resolved = []  # (point, arm key, label, merged)
    for cfg, point, key, label in built:
        merged = False
        if key.startswith("id:") and ident(point) in yaml_arms:
            matches = yaml_arms[ident(point)]
            if len(matches) == 1:
                key, merged = next(iter(matches)), True
            else:
                notes.append(f"aggregate-only c{point['conc']}: identity matches {len(matches)} serving.yaml arms, kept separate")
        resolved.append((point, key, label, merged))
    arms: dict[str, dict] = {}
    for point, key, label, merged in resolved:  # arms keep discovery order
        if merged or key in arms:
            continue
        taken, final, n = {a["label"] for a in arms.values()}, label, 2
        while final in taken:
            final, n = f"{label} #{n}", n + 1
        identity = {k: v if v is not None else serving_identity(point["config"]["serving.yaml"]).get(k)
                    for k, v in point["identity"].items()}  # serving.yaml fills what the aggregate lacks
        arms[key] = {"label": final, "identity": identity, "sources": [], "config": point["config"]}
    for point, key, label, merged in resolved:
        point.pop("config")
        arm = arms[key]
        if point["source"] not in arm["sources"]:
            arm["sources"].append(point["source"])
        point["arm"] = arm["label"]
        points.append(point)
    for point, key, label, merged in resolved:
        if merged and point["scenario"] is None:
            seen = {p["scenario"] for p in points if p["arm"] == arms[key]["label"] and p["scenario"]}
            if len(seen) == 1:
                point["scenario"] = next(iter(seen))
                point["grid"] = parse_grid(point["scenario"])
                set_cache_check(point)
    labels = [a["label"] for a in arms.values()]
    if baseline is not None and baseline not in labels:
        raise ValueError(f"--baseline {baseline!r} is not an arm; arms: {labels}")
    point_notes = [f"{p['arm']} · {p['scenario'] or '—'} · c{'—' if p['conc'] is None else p['conc']}: {n}" for p in points for n in p["notes"]]
    return {
        "title": title or (labels[0] if labels else "benchmark report"),
        "generated": generated,
        "inputs": list(inputs),
        "baseline": baseline or (labels[0] if labels else None),
        "arms": list(arms.values()),
        "points": points,
        "notes": notes + point_notes,
        "bytes_fetched": 0,
        "findings": None,
    }
