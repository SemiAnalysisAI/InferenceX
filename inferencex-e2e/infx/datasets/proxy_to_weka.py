"""Convert flat per-session JSONL dumps into weka-format trace JSON.

Reads <in-dir>/<session_id>.jsonl produced by `sample_proxy_traces.py`
and writes <in-dir>/../<out-dir>/<session_id>.json in the v1 weka trace
format consumed by the kv-cache-tester replayer (see
utils/aiperf/src/aiperf/dataset/loader/weka_trace_models.py).

Subagent grouping mirrors the conversation-view algorithm from the
SemiAnalysis claude-code-proxy:

  1. Walk session rows chronologically.
  2. A row with `subagent_label IS NULL` is a parent (main-agent) turn.
  3. A run of consecutive non-null-label rows is a "stretch". The
     stretch ends as soon as a NULL-label row appears.
  4. Inside the stretch, group by `subagent_label`. Each label group
     becomes one WekaSubagentEntry with its label rows as inner
     WekaNormalRequest entries (in chronological order).
  5. Different labels inside the same stretch produce sibling entries
     (the dashboard renders parallel groups for each).

Hash IDs (24-char hex strings in the proxy DB) are remapped to small
per-trace ints so we can emit `hash_id_scope: "local"`. The mapping is
session-scoped: first-seen hash gets 0, second 1, etc.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def _dump_trace_inline_hash_ids(trace: dict, path: Path) -> None:
    """Write indented JSON with each ``hash_ids`` array on one line."""
    placeholders: list[list[Any]] = []

    def _substitute(obj: Any) -> Any:
        if isinstance(obj, dict):
            out: dict[str, Any] = {}
            for k, v in obj.items():
                if k == "hash_ids" and isinstance(v, list):
                    idx = len(placeholders)
                    placeholders.append(v)
                    out[k] = f"@@HASHIDS_{idx}@@"
                else:
                    out[k] = _substitute(v)
            return out
        if isinstance(obj, list):
            return [_substitute(x) for x in obj]
        return obj

    text = json.dumps(_substitute(trace), indent=2)
    text = re.sub(
        r'"@@HASHIDS_(\d+)@@"',
        lambda m: json.dumps(placeholders[int(m.group(1))], separators=(", ", ": ")),
        text,
    )
    with path.open("w") as f:
        f.write(text + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--in-dir",
        "-i",
        type=Path,
        required=True,
        help="Directory containing <session_id>.jsonl files (the output of sample_proxy_traces.py).",
    )
    p.add_argument(
        "--out-dir",
        "-o",
        type=Path,
        required=True,
        help="Directory to write <session_id>.json weka traces into.",
    )
    return p.parse_args()


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(label: str) -> str:
    return _SLUG_RE.sub("_", label.lower()).strip("_") or "subagent"


def load_session_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["timestamp"])

    # Proxy duplicates inflate token counts and appear to overlap themselves.
    # Treat matching nanosecond timestamps, counts, duration, model and agent as duplicates.
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for r in rows:
        fp = (
            r.get("timestamp"),
            r.get("model"),
            r.get("input_tokens"),
            r.get("output_tokens"),
            r.get("duration_ms"),
            r.get("agent_id") or "",
        )
        if fp in seen:
            continue
        seen.add(fp)
        deduped.append(r)
    n_dropped = len(rows) - len(deduped)
    if n_dropped:
        print(
            f"  dedup: dropped {n_dropped} exact-duplicate row(s) from {path.name}",
            file=sys.stderr,
        )
    return deduped


def remap_hash(h: str, m: dict[str, int]) -> int:
    if h not in m:
        m[h] = len(m)
    return m[h]


def infer_block_size(rows: list[dict]) -> int:  # noqa: ARG001
    """Return the proxy's 64-token block size.

    Partial trailing blocks make hash_token_count / len(hash_ids) unreliable.
    """
    return 64


def effective_input_length(row: dict, block_size: int = 64) -> int:
    """Replay only hashed tokens, excluding the synthetic unhashed tail.

    Prefer hash_token_count (which accounts for partial blocks), then hash
    count times block size. Use total prompt length only without hash coverage.
    """
    hash_tok = row.get("hash_token_count") or 0
    if hash_tok > 0:
        return hash_tok
    hashes = row.get("hash_ids") or []
    if hashes:
        return len(hashes) * block_size
    return (
        (row.get("input_tokens") or 0)
        + (row.get("cache_read_input_tokens") or 0)
        + (row.get("cache_write_tokens") or 0)
    )


def build_normal_request(row: dict, hash_map: dict[str, int], think_time: float | None) -> dict:
    """Inner subagent request — Normal type, per weka v1 spec."""
    out = {
        "t": row["t_sec"],
        "type": "n",
        "model": row["model"],
        "in": effective_input_length(row),
        "out": row.get("output_tokens") or 0,
        "hash_ids": [remap_hash(h, hash_map) for h in (row.get("hash_ids") or [])],
        "api_time": (row.get("duration_ms") or 0) / 1000.0,
    }
    if think_time is not None:
        out["think_time"] = think_time
    return out


def build_top_request(row: dict, hash_map: dict[str, int], think_time: float | None) -> dict:
    """Top-level main-agent request — Normal or Streaming."""
    out = {
        "t": row["t_sec"],
        "model": row["model"],
        "in": effective_input_length(row),
        "out": row.get("output_tokens") or 0,
        "hash_ids": [remap_hash(h, hash_map) for h in (row.get("hash_ids") or [])],
        "api_time": (row.get("duration_ms") or 0) / 1000.0,
    }
    if think_time is not None:
        out["think_time"] = think_time
    if row.get("is_streaming"):
        out["type"] = "s"
        ttft_ms = row.get("ttft_ms")
        if ttft_ms is not None:
            out["ttft"] = ttft_ms / 1000.0
    else:
        out["type"] = "n"
    return out


def compute_think_times(rows: list[dict]) -> list[float | None]:
    """Wall-clock gap from the previous chronological row's end.

    First row gets None (no prior). Negative gaps clamp to 0 (the proxy
    timestamps are millisecond-precise; minor reorderings within the
    same millisecond can produce small negatives).
    """
    out: list[float | None] = []
    prev_end: float | None = None
    for r in rows:
        if prev_end is None:
            out.append(None)
        else:
            gap = r["t_sec"] - prev_end
            out.append(max(0.0, gap))
        prev_end = r["t_sec"] + (r.get("duration_ms") or 0) / 1000.0
    return out


# From this CLI version, x-claude-code-agent-id identifies subagents.
# Labelled rows without it are utility calls and become main turns.
MIN_CLI_FOR_HEADER_AS_TRUTH = (2, 1, 139)


def _parse_cli_version(s: str | None) -> tuple[int, int, int] | None:
    if not s:
        return None
    parts = s.split(".")
    if len(parts) != 3:
        return None
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return None


def _is_utility_label_only(row: dict) -> bool:
    """True if the row's `subagent_label` should be ignored on new CLI.

    A "utility" row is one labelled as a sub-agent by the proxy's
    pattern matcher but with no header-derived id. On CLI versions
    where `x-claude-code-agent-id` is authoritative, the absence of
    that header means this isn't a Task-tool-spawned sub-agent — it's
    a utility call (Title Generation / Name Generation / Statusline)
    that should appear in the trace as a regular main turn.
    """
    if not row.get("subagent_label"):
        return False
    if row.get("agent_id") or row.get("thread_id"):
        return False
    cli = _parse_cli_version(row.get("cli_version"))
    return cli is not None and cli >= MIN_CLI_FOR_HEADER_AS_TRUTH


def _id_group_key(row: dict) -> str | None:
    """Match `idGroupKey` in subagent-runs.ts.

    Returns a stable cross-session key when we have a header-derived id,
    else None (caller falls back to legacy contiguous-stretch grouping).
    """
    if not row.get("subagent_label"):
        return None
    if row.get("agent_id"):
        return f"cc-agent::{row['agent_id']}"
    if row.get("thread_id"):
        return f"{row['subagent_label']}::thread::{row['thread_id']}"
    return None


def build_subagent_entry(
    label: str,
    instance_idx: int,
    items: list[tuple[dict, float | None]],
    hash_map: dict[str, int],
) -> dict:
    inner = [build_normal_request(row, hash_map, tt) for row, tt in items]
    first_row = items[0][0]
    last_row = items[-1][0]
    end_t = last_row["t_sec"] + (last_row.get("duration_ms") or 0) / 1000.0
    duration_ms = round((end_t - first_row["t_sec"]) * 1000)
    total_tokens = sum(r["in"] + r["out"] for r in inner)
    models = sorted({row["model"] for row, _ in items})
    # agent_id suffix priority: Claude Code agent-id (canonical when
    # present) > Codex thread-id. Matches the dashboard's
    # getSubagentRunLabel which suffixes with the last 8 chars.
    cc_agent_id = first_row.get("agent_id")
    thread_id = first_row.get("thread_id")
    agent_id = f"{slugify(label)}_{instance_idx:03d}"
    suffix = cc_agent_id or thread_id
    if suffix:
        agent_id = f"{agent_id}_{suffix[-8:]}"
    return {
        "t": first_row["t_sec"],
        "type": "subagent",
        "agent_id": agent_id,
        "subagent_type": label,
        "duration_ms": duration_ms,
        "total_tokens": total_tokens,
        # tool_use_count is not tracked in the proxy DB; leave as None
        # (the model field defaults to None).
        "tool_use_count": None,
        "status": "completed",
        "requests": inner,
        "models": models,
    }


def session_to_weka(session_id: str, rows: list[dict]) -> dict:
    if not rows:
        return {
            "id": session_id,
            "models": [],
            "block_size": 64,
            "hash_id_scope": "local",
            "requests": [],
        }

    n_demoted = 0
    demoted_rows: list[dict] = []
    for r in rows:
        if _is_utility_label_only(r):
            r = {**r, "subagent_label": None}
            n_demoted += 1
        demoted_rows.append(r)
    if n_demoted:
        print(
            f"  demoted {n_demoted} utility-labelled row(s) to main turns "
            f"(no x-claude-code-agent-id on CLI >= "
            f"{'.'.join(str(x) for x in MIN_CLI_FOR_HEADER_AS_TRUTH)})",
            file=sys.stderr,
        )
    rows = demoted_rows

    think_times = compute_think_times(rows)
    hash_map: dict[str, int] = {}
    block_size = infer_block_size(rows)

    out_requests: list[dict] = []
    instance_count: dict[str, int] = {}
    models_seen: set[str] = set()

    # Group by header ID across main turns so background subagents stay intact.
    # Matches subagent-runs.ts:buildRequestRuns.
    id_groups: dict[str, list[tuple[dict, float | None]]] = {}
    for r, tt in zip(rows, think_times, strict=False):
        key = _id_group_key(r)
        if key is None:
            continue
        id_groups.setdefault(key, []).append((r, tt))

    emitted: set[str] = set()
    i = 0
    while i < len(rows):
        row = rows[i]
        if row.get("subagent_label") is None:
            out_requests.append(build_top_request(row, hash_map, think_times[i]))
            models_seen.add(row["model"])
            i += 1
            continue

        key = _id_group_key(row)
        if key is not None:
            if key not in emitted:
                emitted.add(key)
                items = id_groups[key]
                # Claude Code agent-id groups use the flat 'Subagent'
                # label since per-request system-prompt labels drift.
                use_label = "Subagent" if row.get("agent_id") else row["subagent_label"]
                instance_count[use_label] = instance_count.get(use_label, 0) + 1
                entry = build_subagent_entry(use_label, instance_count[use_label], items, hash_map)
                out_requests.append(entry)
                models_seen.update(entry["models"])
            i += 1
            continue

        # Without header IDs, group by label within stretches bounded by main turns.
        stretch_rows: list[tuple[dict, float | None]] = []
        while (
            i < len(rows)
            and rows[i].get("subagent_label") is not None
            and _id_group_key(rows[i]) is None
        ):
            stretch_rows.append((rows[i], think_times[i]))
            i += 1
        groups: dict[str, list[tuple[dict, float | None]]] = {}
        for r, tt in stretch_rows:
            groups.setdefault(r["subagent_label"], []).append((r, tt))
        for label, items in groups.items():
            instance_count[label] = instance_count.get(label, 0) + 1
            entry = build_subagent_entry(label, instance_count[label], items, hash_map)
            out_requests.append(entry)
            models_seen.update(entry["models"])

    return {
        "id": session_id,
        "models": sorted(models_seen),
        "block_size": block_size,
        "hash_id_scope": "local",
        "requests": out_requests,
    }


def main() -> int:
    args = parse_args()

    in_files = sorted(p for p in args.in_dir.glob("*.jsonl"))
    if not in_files:
        sys.exit(f"ERROR: no .jsonl files in {args.in_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_traces = 0
    n_top = 0
    n_subagent_entries = 0
    n_inner = 0
    for src in in_files:
        session_id = src.stem
        rows = load_session_rows(src)
        trace = session_to_weka(session_id, rows)

        out_path = args.out_dir / f"{session_id}.json"
        _dump_trace_inline_hash_ids(trace, out_path)

        n_traces += 1
        for entry in trace["requests"]:
            if entry.get("type") == "subagent":
                n_subagent_entries += 1
                n_inner += len(entry["requests"])
            else:
                n_top += 1

        print(
            f"{session_id}: {len(rows)} row(s) -> "
            f"{len(trace['requests'])} entries "
            f"({sum(1 for e in trace['requests'] if e.get('type') == 'subagent')} subagent groups)"
            f" -> {out_path}",
            file=sys.stderr,
        )

    print(
        f"\nWrote {n_traces} trace(s): "
        f"{n_top} main turns, "
        f"{n_subagent_entries} subagent groups ({n_inner} inner requests)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
