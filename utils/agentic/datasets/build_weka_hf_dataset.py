#!/usr/bin/env python3
"""Build + upload a weka-with-subagents HuggingFace dataset (and optional 256k variant).

End-to-end pipeline:
  1. sample_proxy_traces.py -> per-session proxy JSONLs
  2. proxy_to_weka.py       -> per-trace weka JSONs (dedup runs here)
  3. concat                 -> traces.jsonl (one trace per line)
  4. plots + stats + README -> dataset card payload
  5. huggingface_hub.upload_folder

If --repo-256k is given, additionally produces a 256k-capped variant where
each request with input + output > 256_000 tokens is dropped and the
surviving timeline keeps its original relative timestamps.

Authentication:
  --db-url   or env AGENTIC_PROXY_DB_URL
  --hf-token or env HF_TOKEN     (huggingface_hub also accepts ~/.cache/huggingface/token)

Example:
    python utils/agentic/datasets/build_weka_hf_dataset.py \\
        --repo-base semianalysisai/cc-traces-weka-with-subagents-060226 \\
        --repo-256k semianalysisai/cc-traces-weka-with-subagents-060226-256k \\
        --min-trace-version 6 --max-trace-version 6 \\
        --min-main-turns 20 --require-cli-min 2.1.139 \\
        --max-parallel-subagents 5 \\
        --work-dir /tmp/weka_build_060226
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CAP_TOKENS = 256_000
HERE = Path(__file__).resolve().parent
SAMPLER = HERE / "sample_proxy_traces.py"
CONVERTER = HERE / "proxy_to_weka.py"
PLOT_WEKA = HERE / "plot_weka_distributions.py"
PLOT_SUBAGENT = HERE / "plot_subagent_distributions.py"


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--repo-base", required=True,
                   help="HF dataset repo id for the unfiltered build.")
    p.add_argument("--repo-256k", default=None,
                   help="HF dataset repo id for the 256k-capped variant. "
                        "Omit to skip the 256k build.")
    p.add_argument("--base-max-isl", type=int, default=None,
                   help="Per-request ISL cap applied to the BASE build only: "
                        "drop any request whose weka `in` (hash-block count × "
                        "64) exceeds this while preserving the surviving timeline. "
                        "Use to remove the >~1M hash-block overcount artifacts "
                        "(e.g. 990016 = closest 64-multiple to 990k). The 256k "
                        "variant already excludes these via its own cap.")
    p.add_argument("--work-dir", type=Path, required=True,
                   help="Cache directory for sample/convert/upload payload.")

    # sampler pass-through
    p.add_argument("--min-trace-version", type=int, default=None)
    p.add_argument("--max-trace-version", type=int, default=None)
    p.add_argument("--min-requests", type=int, default=None,
                   help="Drop sessions with fewer than this many Anthropic requests.")
    p.add_argument("--max-requests", type=int, default=None,
                   help="Drop sessions with more than this many Anthropic requests.")
    p.add_argument("--min-main-turns", type=int, default=None)
    p.add_argument("--require-cli-min", type=str, default=None)
    p.add_argument("--max-parallel-subagents", type=int, default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap session count (for smoke tests). Implies --sampling top.")
    p.add_argument("--sampling", choices=("top", "recent", "random"), default="top")
    p.add_argument("--exclude-dynamic-workflow-bug", action="store_true",
                   help="Drop sessions hit by the Claude Code CLI<2.1.174 "
                        "dynamic-workflow bug (interleaved unlabeled subagents).")
    p.add_argument("--dwbug-min-peak", type=int, default=None,
                   help="Peak concurrent unlabeled-trajectory threshold for the "
                        "dynamic-workflow-bug filter (sampler default 3).")

    # auth
    p.add_argument("--db-url", default=None,
                   help="Postgres URL (else $AGENTIC_PROXY_DB_URL).")
    p.add_argument("--hf-token", default=None,
                   help="HF write token (else $HF_TOKEN or cached login).")

    # idempotency
    p.add_argument("--skip-sample", action="store_true",
                   help="Reuse work-dir/proxy/ if present.")
    p.add_argument("--skip-convert", action="store_true",
                   help="Reuse work-dir/per_trace/ if present.")
    p.add_argument("--skip-upload", action="store_true",
                   help="Build payloads but don't push to HF.")

    return p.parse_args()


def _run(cmd: list, **kw) -> None:
    print(f"$ {' '.join(shlex.quote(str(c)) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kw)


def stage_sample(args, work_dir: Path) -> Path:
    proxy_dir = work_dir / "proxy"
    if args.skip_sample and proxy_dir.exists() and any(proxy_dir.glob("*.jsonl")):
        n = sum(1 for _ in proxy_dir.glob("*.jsonl"))
        print(f"[sample] reusing {n} cached session JSONLs in {proxy_dir}")
        return proxy_dir
    proxy_dir.mkdir(parents=True, exist_ok=True)
    cmd: list = [
        sys.executable, str(SAMPLER), "--out", str(proxy_dir),
        "--sampling", args.sampling,
    ]
    for flag, val in [
        ("--min-trace-version", args.min_trace_version),
        ("--max-trace-version", args.max_trace_version),
        ("--min-requests", args.min_requests),
        ("--max-requests", args.max_requests),
        ("--min-main-turns", args.min_main_turns),
        ("--require-cli-min", args.require_cli_min),
        ("--max-parallel-subagents", args.max_parallel_subagents),
        ("--limit", args.limit),
    ]:
        if val is not None:
            cmd += [flag, str(val)]
    if args.exclude_dynamic_workflow_bug:
        cmd.append("--exclude-dynamic-workflow-bug")
        if args.dwbug_min_peak is not None:
            cmd += ["--dwbug-min-peak", str(args.dwbug_min_peak)]
    if args.db_url:
        cmd += ["--db-url", args.db_url]
    _run(cmd)
    return proxy_dir


def stage_convert(args, proxy_dir: Path, work_dir: Path) -> Path:
    per_trace = work_dir / "per_trace"
    if args.skip_convert and per_trace.exists() and any(per_trace.glob("*.json")):
        n = sum(1 for _ in per_trace.glob("*.json"))
        print(f"[convert] reusing {n} cached per-trace JSONs in {per_trace}")
        return per_trace
    per_trace.mkdir(parents=True, exist_ok=True)
    _run([
        sys.executable, str(CONVERTER),
        "-i", str(proxy_dir), "-o", str(per_trace),
    ])
    return per_trace


# ---------------------------------------------------------------------------
# Payload assembly
# ---------------------------------------------------------------------------


def _concat_traces_jsonl(per_trace_dir: Path, out_path: Path) -> int:
    """Concat each per-trace .json (multi-line pretty-printed) into a single
    one-line-per-trace traces.jsonl. Re-serializes via json.dumps to guarantee
    JSONL invariant (avoids the v1-052726 broken-JSONL bug where pretty-printed
    content was concatenated raw and pyarrow.read_json choked)."""
    n = 0
    with out_path.open("w") as out:
        for p in sorted(per_trace_dir.glob("*.json")):
            try:
                trace = json.loads(p.read_text())
            except json.JSONDecodeError as e:
                print(f"  WARN skipped malformed {p.name}: {e}", file=sys.stderr)
                continue
            out.write(json.dumps(trace, separators=(",", ":")) + "\n")
            n += 1
    return n


def _compute_stats(per_trace_dir: Path) -> dict:
    """Aggregate counts and token totals across every weka trace."""
    n_traces = 0
    n_main = 0
    n_groups = 0
    n_inners = 0
    tot_in = 0
    tot_out = 0
    for p in sorted(per_trace_dir.glob("*.json")):
        try:
            trace = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        n_traces += 1
        for req in trace.get("requests", []):
            if req.get("type") == "subagent":
                n_groups += 1
                for r in req.get("requests", []):
                    n_inners += 1
                    tot_in += r.get("in", 0) or 0
                    tot_out += r.get("out", 0) or 0
            else:
                n_main += 1
                tot_in += req.get("in", 0) or 0
                tot_out += req.get("out", 0) or 0
    return {
        "traces": n_traces,
        "main_turns": n_main,
        "subagent_groups": n_groups,
        "subagent_inner_requests": n_inners,
        "total_model_requests": n_main + n_inners,
        "total_input_tokens": tot_in,
        "total_output_tokens": tot_out,
    }


def _format_stats(stats: dict) -> str:
    lines = []
    for k, v in stats.items():
        lines.append(f"{k}: {v:>15,}")
    return "\n".join(lines) + "\n"


def _version_label(min_v: int | None, max_v: int | None) -> str:
    """Concise human label for the trace-version filter (e.g. 'v7 only')."""
    if min_v is not None and max_v is not None:
        return f"v{min_v} only" if min_v == max_v else f"v{min_v}-v{max_v}"
    if min_v is not None:
        return f"v{min_v}+"
    if max_v is not None:
        return f"≤v{max_v}"
    return "all versions"


def _build_readme(
    repo_id: str,
    stats: dict,
    sampler_cmd: list,
    filters_block: str,
    is_256k: bool,
    version_label: str,
    parent_repo_id: str | None = None,
    pretty_date: str | None = None,
    isl_cap: int | None = None,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    pretty_label = (
        f"CC Traces — Weka, With Subagents, "
        f"{'256k cap, ' if is_256k else ''}"
        f"{version_label} ({pretty_date or datetime.now(timezone.utc).strftime('%b %d %Y')})"
    )
    plugin_key = (
        "semianalysis_cc_traces_weka_with_subagents_256k"
        if is_256k
        else "semianalysis_cc_traces_weka_with_subagents"
    )

    # HF YAML front-matter so the dataset card renders metadata and the
    # data viewer picks up traces.jsonl as the default split.
    front_matter = (
        "---\n"
        "license: apache-2.0\n"
        f"pretty_name: {pretty_label}\n"
        "task_categories:\n  - text-generation\n"
        "tags:\n"
        "  - llm\n  - inference\n  - benchmarking\n  - kv-cache\n"
        "  - agentic\n  - multi-turn\n  - claude\n  - subagents\n"
        "size_categories:\n  - n<1K\n"
        "configs:\n"
        "  - config_name: default\n"
        "    data_files:\n"
        "      - split: train\n        path: traces.jsonl\n"
        "---\n\n"
    )

    parent_block = ""
    if is_256k and parent_repo_id:
        parent_block = (
            f"\nDerived from [{parent_repo_id}](https://huggingface.co/datasets/"
            f"{parent_repo_id}) by applying the 256k per-request cap and "
            "preserving the surviving requests' relative timestamps.\n"
        )

    bullet_filter = (
        "\n## 256k filter rule\n\n"
        f"- Per-request `input + output ≤ {CAP_TOKENS:,}` tokens. Applied at "
        "request granularity, not conversation. Main-agent and sub-agent "
        "inner requests are evaluated independently.\n"
        "- Sub-agent groups where *every* inner request was filtered → "
        "entire group dropped.\n"
        "- Sub-agent groups with only *some* inners filtered → partial group "
        "kept (surviving inners retained).\n"
        "- Timeline preservation: surviving request `t` values keep their "
        "original relative offsets, including sub-agent overlap. If the first "
        "request was filtered, all surviving timestamps are shifted by one "
        "uniform offset so the earliest survivor starts at `t = 0`.\n"
        if is_256k else ""
    )

    isl_note_block = (
        "\n## ISL cap — KV-cache-block overcount note\n\n"
        f"Requests whose input length (`in`) exceeded **{isl_cap:,} tokens** "
        "(the closest multiple of 64 to 990k) were dropped from this build, "
        "while preserving the surviving timeline's relative timestamps.\n\n"
        "**Why.** The weka `in` field is derived from the proxy's "
        "`hash_token_count`, which is recorded as *(number of 64-token "
        "KV-cache prefix blocks) × 64* — it is always an exact multiple of "
        "64, not a true tokenizer count. For very large agentic contexts "
        "with heavy prompt caching this block count drifts **above** the "
        "real prompt size: cross-checking against the billed token columns "
        "(`input + cache_read + cache_write`) shows it tracks ~1.00× on "
        "average but overcounts by as much as ~260k tokens in the "
        "heavy-cache-write tail. That tail pushed a small number of requests "
        "past 1M `in` even though **no request's real prompt actually "
        "exceeds 1M**. These inflated rows are the artifacts removed here.\n\n"
        "Applied at request granularity (main-agent turns and sub-agent "
        "inner requests evaluated independently); a sub-agent group is "
        "dropped only if *every* inner request was over the cap.\n"
        if (isl_cap is not None and not is_256k) else ""
    )

    plots_block = (
        "\n## Distribution plots\n\n"
        "### Main-agent stream\n\n"
        "![Main-stream distributions — log x](plots/distributions_log.png)\n\n"
        "![Main-stream distributions — linear x](plots/distributions_linear.png)\n\n"
        "### Sub-agent fan-out\n\n"
        "![Sub-agent distributions — log x](plots/subagent_distributions_log.png)\n\n"
        "![Sub-agent distributions — linear x](plots/subagent_distributions_linear.png)\n"
    )

    return (
        f"{front_matter}"
        f"# {repo_id}\n\n"
        f"WekaTrace corpus derived from SemiAnalysis Claude Code proxy traces. "
        f"Built {now} via `utils/agentic/datasets/build_weka_hf_dataset.py`.\n"
        f"{parent_block}\n"
        f"## Filters\n\n{filters_block}\n"
        f"{bullet_filter}"
        f"{isl_note_block}\n"
        f"## Stats\n\n```\n{_format_stats(stats)}```\n"
        f"{plots_block}\n"
        f"## Source script\n\n```\n{' '.join(shlex.quote(str(c)) for c in sampler_cmd)}\n```\n\n"
        f"## Loader plugin\n\nLoad in aiperf via:\n\n"
        f"```\n--public-dataset {plugin_key}\n```\n"
    )


def _filters_block(args, *, cap_256k: bool = False) -> str:
    bits = []
    if args.min_trace_version is not None and args.max_trace_version is not None \
            and args.min_trace_version == args.max_trace_version:
        bits.append(f"- Trace version: exactly v{args.min_trace_version}")
    else:
        if args.min_trace_version is not None:
            bits.append(f"- min trace version: v{args.min_trace_version}")
        if args.max_trace_version is not None:
            bits.append(f"- max trace version: v{args.max_trace_version}")
    if args.min_requests is not None:
        bits.append(f"- min Anthropic requests per session: {args.min_requests}")
    if args.max_requests is not None:
        bits.append(f"- max Anthropic requests per session: {args.max_requests}")
    if args.min_main_turns is not None:
        bits.append(f"- min main-agent turns per session: {args.min_main_turns}")
    if args.require_cli_min is not None:
        bits.append(f"- Claude Code CLI ≥ {args.require_cli_min} (every row)")
    if args.max_parallel_subagents is not None:
        bits.append(f"- peak concurrent sub-agent groups ≤ {args.max_parallel_subagents}")
    bits.append("- Non-image rows only (image content excluded at source)")
    bits.append("- Classifier calls excluded "
                "(`max_tokens<=64 AND no tools` → SUGGESTION MODE, title-gen, Security Monitor)")
    bits.append("- Exact-duplicate proxy rows deduped by `(timestamp, model, in, out, dur_ms, agent_id)`")
    if getattr(args, "exclude_dynamic_workflow_bug", False):
        peak = args.dwbug_min_peak or 3
        bits.append(
            f"- Dynamic-workflow-bug sessions excluded: Claude Code CLI < 2.1.174 "
            f"emitted dynamic-workflow subagents without a subagent-label header, "
            f"so they appear as many interleaved unlabeled trajectories in one "
            f"session. Dropped when peak concurrent unlabeled multi-turn "
            f"trajectories ≥ {peak} (changelog 2.1.174)."
        )
    if not cap_256k and args.base_max_isl is not None:
        bits.append(
            f"- Per-request ISL cap: input ≤ {args.base_max_isl:,} tokens "
            f"(weka `in` = hash-block count × 64). Drops KV-cache-block "
            f"overcount artifacts that read above ~1M while preserving "
            f"the surviving timeline."
        )
    if cap_256k:
        bits.append(f"- 256k per-request cap (see *256k filter rule* below)")
    return "\n".join(bits)


def _build_payload(
    args,
    per_trace_dir: Path,
    payload_dir: Path,
    *,
    repo_id: str,
    is_256k: bool,
    sampler_cmd: list,
    parent_repo_id: str | None = None,
) -> dict:
    """Assemble traces.jsonl + stats + plots + README into payload_dir."""
    payload_dir.mkdir(parents=True, exist_ok=True)
    n = _concat_traces_jsonl(per_trace_dir, payload_dir / "traces.jsonl")
    print(f"[payload] wrote traces.jsonl with {n} traces")

    stats = _compute_stats(per_trace_dir)
    (payload_dir / "stats.txt").write_text(_format_stats(stats))

    plots = payload_dir / "plots"
    plots.mkdir(exist_ok=True)
    for script in (PLOT_WEKA, PLOT_SUBAGENT):
        try:
            _run([
                sys.executable, str(script),
                "--in-dir", str(per_trace_dir),
                "--out-dir", str(plots),
            ])
        except subprocess.CalledProcessError as e:
            print(f"  WARN plot {script.name} failed: {e}", file=sys.stderr)

    readme = _build_readme(
        repo_id=repo_id,
        stats=stats,
        sampler_cmd=sampler_cmd,
        filters_block=_filters_block(args, cap_256k=is_256k),
        is_256k=is_256k,
        version_label=_version_label(args.min_trace_version, args.max_trace_version),
        parent_repo_id=parent_repo_id,
        isl_cap=(None if is_256k else args.base_max_isl),
    )
    (payload_dir / "README.md").write_text(readme)
    return stats


# ---------------------------------------------------------------------------
# Request-drop filters (256k total cap; ISL-only cap)
# ---------------------------------------------------------------------------

from typing import Callable


def _is_oversize(req: dict, cap: int = CAP_TOKENS) -> bool:
    return (req.get("in") or 0) + (req.get("out") or 0) > cap


def _is_oversize_isl(req: dict, cap: int) -> bool:
    """ISL-only predicate: drop on input length (weka `in`, i.e. the
    hash-block count × 64) regardless of output length."""
    return (req.get("in") or 0) > cap


def _filter_trace(trace: dict, is_oversize: Callable[[dict], bool]) -> dict | None:
    """Drop requests for which ``is_oversize`` is true.

      - per-request drop when ``is_oversize(req)``
      - sub-agent group dropped only if every inner is filtered
      - surviving timestamps retain their original relative offsets

    ``think_time`` is computed over the globally ordered proxy rows, so it
    cannot reconstruct overlapping top-level entries. Rebuilding timestamps
    by chaining ``prev_t + prev_api_time + think_time`` serializes concurrent
    subagents. Instead, preserve the recorded wall-clock topology. When the
    earliest request is removed, translate every surviving timestamp by the
    same offset so the trace still starts at zero.

    Returns the filtered trace (deep copy) or None if nothing survives.
    """
    out = copy.deepcopy(trace)
    requests = out.get("requests", [])
    surviving: list[dict] = []
    dropped_any = False

    for entry in requests:
        if entry.get("type") == "subagent":
            inners = [r for r in entry.get("requests", []) if not is_oversize(r)]
            if not inners:
                dropped_any = True
                continue
            if len(inners) != len(entry.get("requests", [])):
                dropped_any = True
            entry["requests"] = inners
            surviving.append(entry)
        else:
            if is_oversize(entry):
                dropped_any = True
                continue
            surviving.append(entry)

    if not surviving:
        return None

    # Most traces do not contain an oversize request. Keep those byte-for-byte
    # identical, including overlapping subagent timestamps.
    if not dropped_any:
        return out

    request_times = [
        req.get("t", 0.0)
        for entry in surviving
        for req in (
            entry.get("requests", [])
            if entry.get("type") == "subagent"
            else [entry]
        )
    ]
    origin = min(request_times)

    for entry in surviving:
        if entry.get("type") != "subagent":
            entry["t"] = entry.get("t", 0.0) - origin
            continue

        inners = entry["requests"]
        for inner in inners:
            inner["t"] = inner.get("t", 0.0) - origin

        first_t = min(inner["t"] for inner in inners)
        last_end = max(
            inner["t"] + (inner.get("api_time") or 0.0)
            for inner in inners
        )
        entry["t"] = first_t
        entry["duration_ms"] = int(round((last_end - first_t) * 1000.0))
        entry["total_tokens"] = sum(
            (r.get("in") or 0) + (r.get("out") or 0)
            for r in inners
        )

    out["requests"] = surviving
    return out


def _filter_trace_256k(trace: dict, cap: int = CAP_TOKENS) -> dict | None:
    """256k variant: drop per-request `in + out > cap`."""
    return _filter_trace(trace, lambda r: _is_oversize(r, cap))


def _filter_trace_isl(trace: dict, cap: int) -> dict | None:
    """ISL variant: drop per-request `in > cap` (hash-block overcount
    artifacts)."""
    return _filter_trace(trace, lambda r: _is_oversize_isl(r, cap))


def _stage_filter(
    per_trace_dir: Path, out_dir: Path, filter_fn, tag: str
) -> Path:
    """Apply ``filter_fn`` to every per-trace JSON; write survivors to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = 0
    dropped = 0
    for p in sorted(per_trace_dir.glob("*.json")):
        try:
            trace = json.loads(p.read_text())
        except json.JSONDecodeError:
            dropped += 1
            continue
        filtered = filter_fn(trace)
        if filtered is None:
            dropped += 1
            continue
        (out_dir / p.name).write_text(json.dumps(filtered, separators=(",", ":")))
        kept += 1
    print(f"[{tag}] kept {kept} traces, dropped {dropped} (empty after filter)")
    return out_dir


def stage_256k(per_trace_dir: Path, work_dir: Path) -> Path:
    """Apply 256k filter to every per-trace JSON; write to work_dir/per_trace_256k/."""
    return _stage_filter(
        per_trace_dir, work_dir / "per_trace_256k", _filter_trace_256k, "256k"
    )


def stage_isl_filter(per_trace_dir: Path, work_dir: Path, cap: int) -> Path:
    """Apply the ISL cap to every per-trace JSON; write to work_dir/per_trace_isl/.

    Used for the base build to drop requests whose `in` (hash-block count
    × 64) exceeds ``cap`` — the >~1M overcount artifacts."""
    return _stage_filter(
        per_trace_dir, work_dir / "per_trace_isl",
        lambda t: _filter_trace_isl(t, cap), f"isl<={cap}",
    )


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def stage_upload(args, payload_dir: Path, repo_id: str, commit_msg: str) -> None:
    if args.skip_upload:
        print(f"[upload] --skip-upload: payload ready at {payload_dir}")
        return
    from huggingface_hub import HfApi, upload_folder
    token = args.hf_token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    print(f"[upload] pushing {payload_dir} -> {repo_id}")
    upload_folder(
        folder_path=str(payload_dir),
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=commit_msg,
    )
    print(f"[upload] done: https://huggingface.co/datasets/{repo_id}")


def _reconstruct_sampler_cmd(args) -> list:
    """The exact sample_proxy_traces.py invocation, for the README."""
    cmd = ["python", "utils/agentic/datasets/sample_proxy_traces.py",
           "--out", "<workdir>/proxy", "--sampling", args.sampling]
    for flag, val in [
        ("--min-trace-version", args.min_trace_version),
        ("--max-trace-version", args.max_trace_version),
        ("--min-requests", args.min_requests),
        ("--max-requests", args.max_requests),
        ("--min-main-turns", args.min_main_turns),
        ("--require-cli-min", args.require_cli_min),
        ("--max-parallel-subagents", args.max_parallel_subagents),
        ("--limit", args.limit),
    ]:
        if val is not None:
            cmd += [flag, str(val)]
    if args.exclude_dynamic_workflow_bug:
        cmd.append("--exclude-dynamic-workflow-bug")
        if args.dwbug_min_peak is not None:
            cmd += ["--dwbug-min-peak", str(args.dwbug_min_peak)]
    return cmd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    # set DB URL into env so the subprocess inherits it cleanly even if
    # the user only passed --db-url (sampler reads $AGENTIC_PROXY_DB_URL)
    if args.db_url:
        os.environ["AGENTIC_PROXY_DB_URL"] = args.db_url

    proxy_dir = stage_sample(args, work_dir)
    per_trace_dir = stage_convert(args, proxy_dir, work_dir)
    sampler_cmd = _reconstruct_sampler_cmd(args)

    # --- base build (optionally with a per-request ISL cap to drop the
    #     hash-block overcount artifacts that read above ~1M)
    base_src = per_trace_dir
    if args.base_max_isl is not None:
        base_src = stage_isl_filter(per_trace_dir, work_dir, args.base_max_isl)
    base_payload = work_dir / "base"
    base_stats = _build_payload(
        args, base_src, base_payload,
        repo_id=args.repo_base, is_256k=False, sampler_cmd=sampler_cmd,
    )
    print(f"[base] {base_stats}")
    isl_note = f", isl<={args.base_max_isl}" if args.base_max_isl is not None else ""
    stage_upload(args, base_payload, args.repo_base,
                 commit_msg=f"build: {base_stats['traces']} traces "
                            f"(v{args.min_trace_version or '?'}{'-'+str(args.max_trace_version) if args.max_trace_version and args.max_trace_version != args.min_trace_version else ''}{isl_note})")

    # --- 256k variant
    if args.repo_256k:
        per_trace_256k = stage_256k(per_trace_dir, work_dir)
        cap_payload = work_dir / "256k"
        cap_stats = _build_payload(
            args, per_trace_256k, cap_payload,
            repo_id=args.repo_256k, is_256k=True,
            sampler_cmd=sampler_cmd, parent_repo_id=args.repo_base,
        )
        print(f"[256k] {cap_stats}")
        stage_upload(args, cap_payload, args.repo_256k,
                     commit_msg=f"build: {cap_stats['traces']} traces "
                                f"(256k cap, derived from {args.repo_base})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
