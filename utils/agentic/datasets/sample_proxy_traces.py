#!/usr/bin/env python3
"""Sample Claude Code traces from the agentic-proxy Neon DB.

Selects sessions in `public.requests` / `public.sessions` matching the
filter flags, then dumps one JSONL file per session under --out, plus a
manifest.json with summary stats. Filenames mirror the kv-cache-tester /
weka layout so a downstream weka converter can iterate <out>/*.jsonl one
trace at a time.

Connection string is read from AGENTIC_PROXY_DB_URL (overridable via
--db-url). All filters are explicit — no defaults. Migration
`013_add_subagent_label` ran 2026-04-16T16:00:00+00:00; rows before
that have NULL subagent_label and are not useful for trace replay.

Anthropic-only: model LIKE 'claude-%'. Non-Claude rows (gpt-5.x via
/proxy/responses) are intentionally excluded.

Dependency: pip install 'psycopg[binary]'

Example
-------
    AGENTIC_PROXY_DB_URL=postgresql://... \\
        python utils/agentic/datasets/sample_proxy_traces.py \\
            --out ./proxy_traces \\
            --min-requests 100 --max-requests 800 \\
            --max-span-hours 4 \\
            --sampling random --seed 42 \\
            --limit 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger("sample_proxy_traces")


def _redact_url(url: str) -> str:
    """Show host + db, drop credentials."""
    try:
        p = urlsplit(url)
        return f"{p.hostname}{p.path}"
    except Exception:
        return "<connection string>"

DEFAULT_ENV = "AGENTIC_PROXY_DB_URL"
# Migration `013_add_subagent_label` ran at this UTC timestamp; rows
# before this have NULL subagent_label and are unusable for trace replay.
SUBAGENT_MIGRATION_TS = "2026-04-16T16:00:00+00:00"
# gpt-5.x rows go through /proxy/responses (different shape) and are
# intentionally excluded.
ANTHROPIC_MODEL_LIKE = "claude-%"

# Subagent labeling and thread-id extraction mirror
# semianalysis-claude-code-proxy:
#   packages/app/src/lib/subagent-runs.ts  (getRequestThreadId,
#                                            getRequestSubagentLabel,
#                                            buildRequestRuns)
# Two SQL projections do the per-row work the dashboard does in JS so
# the JSONL dump and the downstream weka converter both consume the
# already-resolved label and thread id.
#
# (1) Effective subagent_label = server-set label, with a "Subagent
#     (Haiku)" fallback for unlabelled haiku calls whose max_tokens is
#     anything other than 1 (the dashboard's `req.model.includes('haiku')
#     && req.requestBody?.max_tokens !== 1` rule). max_tokens missing/non-
#     numeric is treated as "not 1", matching JS truthiness semantics.
EFFECTIVE_SUBAGENT_LABEL_EXPR = """
    CASE
        WHEN subagent_label IS NOT NULL THEN subagent_label
        WHEN model ILIKE '%%haiku%%'
             AND NOT (jsonb_typeof(request_body->'max_tokens') = 'number'
                      AND (request_body->>'max_tokens')::int = 1)
        THEN 'Subagent (Haiku)'
        ELSE NULL
    END
"""

# (2) Thread id COALESCE over raw HTTP header keys, in the same priority
#     as getRequestThreadId. Headers are stored lowercased / hyphenated
#     (raw HTTP form), so camelCase variants don't appear in the DB and
#     can be omitted. x-codex-window-id is "<id>:<msg>"; we split on `:`
#     and take the prefix. NULLIF on each branch treats empty strings as
#     absent to match the JS truthiness check.
THREAD_ID_EXPR = """
    COALESCE(
        NULLIF(request_headers->>'thread_id', ''),
        NULLIF(request_headers->>'x-codex-thread-id', ''),
        NULLIF(split_part(NULLIF(request_headers->>'x-codex-window-id', ''), ':', 1), ''),
        NULLIF(request_headers->>'session_id', '')
    )
"""

# (2b) Claude Code agent id — added in claude-cli ≥ 2.1.139. When
#      present this is the canonical sub-agent identifier per
#      `getRequestClaudeCodeAgentId` in subagent-runs.ts: one id
#      = one logical sub-agent invocation, regardless of how the
#      per-request system-prompt label drifts across its tool calls.
#      Absent on main-agent requests and on older Claude clients —
#      the converter falls back to (label, thread_id) grouping in
#      that case.
CLAUDE_CODE_AGENT_ID_EXPR = """
    NULLIF(request_headers->>'x-claude-code-agent-id', '')
"""

# (2c) Claude CLI version — extracted from the user-agent header
#      ("claude-cli/X.Y.Z (external, cli)"). Used by the converter to
#      decide whether `x-claude-code-agent-id` is reliable enough to
#      demote utility-labelled rows (Title Generation etc.) without an
#      agent-id back to main turns. Pre-2.1.139 rows fall back to the
#      legacy label-only grouping.
CLI_VERSION_EXPR = """
    substring(request_headers->>'user-agent'
              from 'claude-cli/([0-9]+\\.[0-9]+\\.[0-9]+)')
"""

# (3) Image-content exclusion (HARDCODED): v1/v2 trace versions did not
#     consistently capture or anonymize image content blocks, so an
#     aiperf replay run against them would be unreliable. Exclude any
#     session containing a row that is both `trace_version <= 2` AND has
#     a `messages[*].content[*].type == "image"` block in its
#     `request_body`. v3+ rows with image content are allowed through.
#
#     Two-phase implementation: this expensive JSONB scan runs only in
#     IMAGE_CHECK_SQL below, bounded to candidate session_ids that
#     already passed the cheap aggregates. Avoids paying the @?
#     jsonpath cost on the full ~117K-row table.

# (4) Non-conversational classifier exclusion (HARDCODED): Claude Code
#     fires several auxiliary classifier calls that aren't real agent
#     turns — "SUGGESTION MODE" next-input prediction, conversation-
#     title generation, haiku-style intent detection, the "Security
#     Monitor" subagent, etc. They share a clean shape:
#         max_tokens <= 64   (output is intentionally capped tiny)
#         tools = []         (classifiers don't get the agent toolbox)
#     and they show up as a heavy spike at OSL 5-10 in the published
#     distribution. The label-only Security Monitor filter we shipped
#     first did NOT catch them — most have subagent_label IS NULL
#     because the proxy doesn't recognize them as subagent calls.
#
#     Drop them at source by shape, not label. This is universal across
#     privacy modes (max_tokens + tools array shape survive anon
#     redaction; only the text inside system/messages is nulled).
#
#     We keep the Security Monitor label/body fallback in addition,
#     since the Security Monitor subagent itself sometimes has
#     max_tokens above 64 and would otherwise leak through.
CLASSIFIER_SHAPE_PREDICATE = """
    (request_body->>'max_tokens') IS NOT NULL
    AND (request_body->>'max_tokens')::int <= 64
    AND (
        jsonb_typeof(request_body->'tools') IS NULL
        OR jsonb_typeof(request_body->'tools') <> 'array'
        OR jsonb_array_length(request_body->'tools') = 0
    )
"""

# Kept name for backwards compatibility with the rest of the file —
# this is now a combined classifier+SecMon predicate, NOT only SecMon.
SECURITY_MONITOR_FILTER_SQL = f"""
    NOT ({CLASSIFIER_SHAPE_PREDICATE})
    AND subagent_label IS DISTINCT FROM 'Security Monitor'
    AND NOT (
        subagent_label = 'Subagent'
        AND request_body::text ILIKE '%%security monitor for autonomous%%'
    )
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--out", "-o", type=Path, required=True,
        help="Output directory. One <session_id>.jsonl per trace + manifest.json.",
    )
    p.add_argument(
        "--min-requests", type=int, default=None,
        help="Drop sessions with fewer than this many Anthropic requests.",
    )
    p.add_argument(
        "--max-requests", type=int, default=None,
        help="Drop sessions larger than this.",
    )
    p.add_argument(
        "--max-span-hours", type=float, default=None,
        help="Drop sessions whose first→last request spans more than this many hours.",
    )
    p.add_argument(
        "--min-main-turns", type=int, default=None,
        help=(
            "Drop sessions with fewer than this many MAIN-AGENT turns (rows "
            "with effective subagent_label IS NULL). This is the count that "
            "survives the subagent-stripping step downstream — use it when "
            "building the no-subagents variant to avoid pulling sessions that "
            "would end up with a tiny main-agent stream."
        ),
    )
    p.add_argument(
        "--min-trace-version", type=int, default=None,
        help=(
            "Require min(trace_version) over the session's SUCCESSFUL anon "
            "Claude rows to be >= N. Equivalent to 'every replayable request "
            "in this session is trace_version >= N'. Sessions with any mixed "
            "older-version request are excluded entirely."
        ),
    )
    p.add_argument(
        "--max-trace-version", type=int, default=None,
        help=(
            "Require max(trace_version) over the session's SUCCESSFUL anon "
            "Claude rows to be <= N. Combine with --min-trace-version=N to "
            "pin the session to exactly version N (e.g. v6-only). Sessions "
            "containing any newer-format row are excluded entirely."
        ),
    )
    p.add_argument(
        "--max-parallel-subagents", type=int, default=None,
        metavar="N",
        help=(
            "Drop sessions whose peak concurrent subagent GROUP count exceeds "
            "N. A subagent group is one x-claude-code-agent-id (one Task-tool "
            "invocation, regardless of how many inner tool-use turns it does). "
            "A group is 'in flight' from its first inner request's start "
            "timestamp through its last inner request's start + duration_ms. "
            "Peak is the max over a sweep-line of (start, +1)/(end, -1) "
            "events per session; ends are applied before starts at equal "
            "timestamps so a group starting exactly when another ends does "
            "not count as overlapping. Requires --require-cli-min 2.1.139 or "
            "newer (header-based agent_id grouping); pre-2.1.139 sessions are "
            "rejected since the legacy label-based fallback can't reliably "
            "distinguish concurrent same-label subagents."
        ),
    )
    p.add_argument(
        "--require-cli-min", type=str, default=None,
        metavar="X.Y.Z",
        help=(
            "Require every successful Claude row in the session to be on "
            "Claude Code CLI >= X.Y.Z. Rejects sessions with any row whose "
            "user-agent doesn't parse as `claude-cli/X.Y.Z`. Use 2.1.139 "
            "(when `x-claude-code-agent-id` shipped) for any pipeline that "
            "needs the canonical header-based subagent grouping — the "
            "legacy stretch-based fallback in proxy_to_weka.py is "
            "unreliable for sessions with concurrent same-label subagents."
        ),
    )
    p.add_argument(
        "--until", type=str, default=None,
        help="Latest timestamp to consider (ISO).",
    )
    p.add_argument(
        "--exclude-dynamic-workflow-bug", action="store_true",
        help=(
            "Exclude sessions hit by the Claude Code CLI<2.1.174 dynamic-"
            "workflow bug, where dynamic-workflow subagents were emitted "
            "WITHOUT a subagent-label/agent-id header and so appear as many "
            "independent, time-interleaved unlabeled trajectories in one "
            "session. A session is dropped when its max CLI < 2.1.174 AND its "
            "unlabeled main-stream requests form >= --dwbug-min-peak "
            "concurrently-active multi-turn trajectories."
        ),
    )
    p.add_argument(
        "--dwbug-min-peak", type=int, default=3, metavar="N",
        help=(
            "Peak concurrent unlabeled multi-turn trajectories at/above which "
            "a pre-2.1.174 session is treated as dynamic-workflow-buggy "
            "(default 3; only used with --exclude-dynamic-workflow-bug)."
        ),
    )
    p.add_argument(
        "--privacy-mode", choices=("anon", "full"), default="anon",
        help="Privacy filter (default: anon — request bodies are redacted; metric columns are intact). Pass `full` to include un-redacted rows. There is no opt-in for mixing both, by design.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Max sessions to dump.",
    )
    p.add_argument(
        "--sampling", choices=("top", "recent", "random"), default=None,
        help="`top` = largest req_count, `recent` = newest last activity, `random` = deterministic md5(session_id||seed). Required if --limit is set.",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Salt for --sampling random (server-side md5). Required when --sampling random.",
    )
    p.add_argument(
        "--db-url", type=str, default=None,
        help=f"Postgres connection string. Falls back to ${DEFAULT_ENV}.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print matching-session summary; do not fetch row-level data or write files.",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    p.add_argument(
        "--session-id", type=str, default=None,
        help=(
            "Dump exactly one session by its `sessions.id`. Skips phase 1 "
            "(candidate aggregates), phase 2 (v1/v2 image exclusion), and "
            "all sampling/limit/seed logic. The privacy_mode safety filter "
            "and Anthropic-only model filter still apply on row dump."
        ),
    )
    p.add_argument(
        "--allow-non-anthropic", action="store_true",
        help=(
            "Escape hatch: drop the `model LIKE 'claude-%%'` filter so "
            "Codex/GPT sessions get dumped too. The downstream weka "
            "converter is built for Claude-shaped traces and will not "
            "produce meaningful output for these — use for ad-hoc "
            "inspection only."
        ),
    )
    args = p.parse_args()

    if args.session_id is None:
        if args.sampling == "random" and args.seed is None:
            p.error("--sampling random requires --seed")
        if args.limit is not None and args.sampling is None:
            p.error("--limit requires --sampling so the LIMIT is deterministic")
        if args.max_parallel_subagents is not None:
            if args.max_parallel_subagents < 0:
                p.error("--max-parallel-subagents must be >= 0")
            # Header-based agent_id grouping is only reliable on
            # claude-cli >= 2.1.139. Pre-2.1.139 rows have no
            # x-claude-code-agent-id, so every subagent collapses to a
            # single NULL group and the peak count is meaningless.
            if not args.require_cli_min:
                p.error(
                    "--max-parallel-subagents requires --require-cli-min "
                    "2.1.139 or newer (header-based agent_id grouping)"
                )
            req_parts = args.require_cli_min.split(".")
            if len(req_parts) != 3 or not all(p.isdigit() for p in req_parts):
                # leave the X.Y.Z format error to the existing encoder
                pass
            else:
                req_int = (int(req_parts[0]) * 1_000_000
                           + int(req_parts[1]) * 1_000
                           + int(req_parts[2]))
                if req_int < 2_001_139:
                    p.error(
                        "--max-parallel-subagents requires --require-cli-min "
                        ">= 2.1.139 (got %s)" % args.require_cli_min
                    )
    else:
        # --session-id short-circuits all session-discovery logic.
        ignored = [
            name for name, val in (
                ("--sampling", args.sampling),
                ("--limit", args.limit),
                ("--seed", args.seed),
                ("--min-requests", args.min_requests),
                ("--max-requests", args.max_requests),
                ("--max-span-hours", args.max_span_hours),
                ("--until", args.until),
            ) if val is not None
        ]
        if ignored:
            sys.stderr.write(
                f"warning: --session-id is set; ignoring {' '.join(ignored)}\n"
            )
    return args


def get_db_url(args: argparse.Namespace) -> str:
    url = args.db_url or os.environ.get(DEFAULT_ENV)
    if not url:
        sys.exit(
            f"ERROR: connection string missing. Set ${DEFAULT_ENV} or pass --db-url."
        )
    return url


# Phase 1: cheap aggregates only. No image-content JSONB scan here —
# Postgres can use idx_requests_privacy_mode_timestamp for the
# (privacy_mode, timestamp) prefix and stream the GROUP BY. Per-row cost
# is dominated by the count/sum/min/max aggregates plus the
# EFFECTIVE_SUBAGENT_LABEL_EXPR (which is cheap: ILIKE + top-level JSONB
# key access).
# Compact CLI-version encoding: X * 1_000_000 + Y * 1_000 + Z. Within
# 0-999 per segment (true for every CC release to date), preserves
# lexicographic ordering as numeric ordering. NULL when the user-agent
# header is missing or doesn't match the claude-cli/X.Y.Z shape, which
# downstream interprets as "unknown CLI" — sessions with any such row
# must be excluded when --require-cli-min is set, because we cannot
# verify they're on a build where the x-claude-code-agent-id header is
# present and the subagent grouping algorithm is reliable.
CLI_VERSION_INT_EXPR = r"""
    CASE
        WHEN request_headers->>'user-agent' ~ 'claude-cli/[0-9]+\.[0-9]+\.[0-9]+'
        THEN
            substring(request_headers->>'user-agent' from 'claude-cli/([0-9]+)\.')::int * 1000000
          + substring(request_headers->>'user-agent' from 'claude-cli/[0-9]+\.([0-9]+)\.')::int * 1000
          + substring(request_headers->>'user-agent' from 'claude-cli/[0-9]+\.[0-9]+\.([0-9]+)')::int
        ELSE NULL
    END
"""

CANDIDATES_PHASE1_SQL = f"""
WITH candidates AS (
    SELECT session_id,
           count(*)                                                       AS req_count,
           count(DISTINCT model)                                          AS distinct_models,
           count(*) FILTER (WHERE ({EFFECTIVE_SUBAGENT_LABEL_EXPR}) IS NOT NULL)
                                                                          AS subagent_reqs,
           count(*) FILTER (WHERE ({EFFECTIVE_SUBAGENT_LABEL_EXPR}) IS NULL)
                                                                          AS main_turns,
           sum(input_tokens)                                              AS total_in,
           sum(output_tokens)                                             AS total_out,
           min(trace_version)                                             AS min_trace_version,
           max(trace_version)                                             AS max_trace_version,
           min(({CLI_VERSION_INT_EXPR}))                                  AS min_cli_int,
           count(*) FILTER (WHERE ({CLI_VERSION_INT_EXPR}) IS NULL)       AS rows_unknown_cli,
           min(timestamp)                                                 AS first_ts,
           max(timestamp)                                                 AS last_ts,
           extract(epoch FROM (max(timestamp) - min(timestamp)))::float   AS span_sec
      FROM requests
     WHERE model LIKE %(model_like)s
       AND response_status_code = 200
       AND error IS NULL
       AND privacy_mode = %(privacy)s
       AND timestamp >= %(migration_floor)s::timestamptz
       AND (%(until)s::timestamptz IS NULL OR timestamp < %(until)s::timestamptz)
       AND ({SECURITY_MONITOR_FILTER_SQL})
     GROUP BY session_id
)
SELECT *
  FROM candidates
 WHERE (%(min_requests)s::int       IS NULL OR req_count        >= %(min_requests)s::int)
   AND (%(max_requests)s::int       IS NULL OR req_count        <= %(max_requests)s::int)
   AND (%(max_span_sec)s::float     IS NULL OR span_sec         <= %(max_span_sec)s::float)
   AND (%(min_main_turns)s::int     IS NULL OR main_turns       >= %(min_main_turns)s::int)
   AND (%(min_trace_version)s::int  IS NULL OR min_trace_version >= %(min_trace_version)s::int)
   AND (%(max_trace_version)s::int  IS NULL OR max_trace_version <= %(max_trace_version)s::int)
   AND (
        %(min_cli_int)s::int IS NULL
        OR (min_cli_int >= %(min_cli_int)s::int AND rows_unknown_cli = 0)
   )
"""

# Phase 2: image-content check, bounded to candidate session_ids that
# passed phase 1. session_id = ANY(...) uses idx_requests_session_id so
# heap reads are bounded. The cheap text-LIKE pre-filter short-circuits
# the expensive @? jsonpath to rows whose body actually contains the
# substring "image"; postgres evaluates AND clauses left-to-right.
IMAGE_CHECK_SQL = """
SELECT DISTINCT session_id
  FROM requests
 WHERE session_id = ANY(%(session_ids)s::text[])
   AND trace_version <= 2
   AND request_body::text LIKE '%%"image"%%'
   AND COALESCE(
       request_body @? '$.messages[*].content[*] ? (@.type == "image")',
       false
   )
"""

# Phase 2.5: peak concurrent subagent group count per session via sweep-
# line over (start, +1)/(end, -1) events. A "group" is one
# x-claude-code-agent-id (one Task-tool invocation, regardless of inner
# turn count). span_start = MIN(timestamp); span_end = MAX(timestamp +
# duration_ms) over all inner rows of the same agent_id.
#
# Tie-break: events at the same instant are ordered ends-before-starts
# (delta ASC: -1 before +1) so a new group starting at the exact moment
# another ends does NOT count as overlapping. Without the explicit
# secondary sort, postgres is free to reorder ties and can overcount.
#
# NULL duration_ms is treated as a zero-length span (collapses to a
# single timestamp). Anthropic 200s should always have duration_ms; this
# is defensive.
SUBAGENT_CONCURRENCY_SQL = f"""
WITH subagent_spans AS (
    SELECT session_id,
           ({CLAUDE_CODE_AGENT_ID_EXPR}) AS agent_id,
           MIN(timestamp) AS span_start,
           MAX(timestamp
               + COALESCE(duration_ms, 0) * interval '1 millisecond') AS span_end
      FROM requests
     WHERE session_id = ANY(%(session_ids)s::text[])
       AND model LIKE %(model_like)s
       AND response_status_code = 200
       AND error IS NULL
       AND privacy_mode = %(privacy)s
       AND timestamp >= %(migration_floor)s::timestamptz
       AND (%(until)s::timestamptz IS NULL OR timestamp < %(until)s::timestamptz)
       AND ({EFFECTIVE_SUBAGENT_LABEL_EXPR}) IS NOT NULL
       AND ({CLAUDE_CODE_AGENT_ID_EXPR}) IS NOT NULL
       AND ({SECURITY_MONITOR_FILTER_SQL})
     GROUP BY session_id, ({CLAUDE_CODE_AGENT_ID_EXPR})
),
events AS (
    SELECT session_id, span_start AS t, 1 AS delta FROM subagent_spans
    UNION ALL
    SELECT session_id, span_end   AS t, -1 AS delta FROM subagent_spans
),
running AS (
    SELECT session_id,
           SUM(delta) OVER (
               PARTITION BY session_id
               ORDER BY t ASC, delta ASC
               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
           ) AS in_flight
      FROM events
)
SELECT session_id, MAX(in_flight)::int AS max_parallel_subagents
  FROM running
 GROUP BY session_id
"""


def _sort_and_limit(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    """Apply --sampling and --limit in Python after phase-2 filtering.

    md5(session_id || seed) is the same expression Postgres uses for the
    'random' mode, so deterministic-random output is identical to the
    one-pass version.
    """
    if args.sampling == "top":
        rows = sorted(rows, key=lambda r: r["req_count"], reverse=True)
    elif args.sampling == "recent":
        rows = sorted(rows, key=lambda r: r["last_ts"], reverse=True)
    elif args.sampling == "random":
        seed = str(args.seed)
        rows = sorted(
            rows,
            key=lambda r: hashlib.md5((r["session_id"] + seed).encode("utf-8")).hexdigest(),
        )
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows


def find_sessions(
    conn, args: argparse.Namespace, model_like: str
) -> list[dict]:
    def _encode_cli_int(s: str | None) -> int | None:
        if not s:
            return None
        parts = s.split(".")
        if len(parts) != 3:
            sys.exit(f"ERROR: --require-cli-min must be X.Y.Z, got {s!r}")
        try:
            x, y, z = (int(p) for p in parts)
        except ValueError:
            sys.exit(f"ERROR: --require-cli-min must be X.Y.Z numeric, got {s!r}")
        return x * 1_000_000 + y * 1_000 + z

    min_cli_int = _encode_cli_int(args.require_cli_min)

    phase1_params = {
        "model_like": model_like,
        "migration_floor": SUBAGENT_MIGRATION_TS,
        "until": args.until,
        "privacy": args.privacy_mode,
        "min_requests": args.min_requests,
        "max_requests": args.max_requests,
        "max_span_sec": (
            args.max_span_hours * 3600.0
            if args.max_span_hours is not None else None
        ),
        "min_main_turns": args.min_main_turns,
        "min_trace_version": args.min_trace_version,
        "max_trace_version": args.max_trace_version,
        "min_cli_int": min_cli_int,
    }
    logger.info(
        "phase 1 (cheap aggregates): min_requests=%s max_requests=%s "
        "max_span_hours=%s min_main_turns=%s min_trace_version=%s "
        "max_trace_version=%s require_cli_min=%s until=%s privacy=%s",
        args.min_requests, args.max_requests, args.max_span_hours,
        args.min_main_turns, args.min_trace_version, args.max_trace_version,
        args.require_cli_min, args.until, args.privacy_mode,
    )
    t1 = time.time()
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(CANDIDATES_PHASE1_SQL, phase1_params)
        candidates = cur.fetchall()
    logger.info("phase 1: %d candidate session(s) in %.1fs",
                len(candidates), time.time() - t1)

    if not candidates:
        return []

    candidate_ids = [c["session_id"] for c in candidates]
    logger.info(
        "phase 2 (v1/v2 image-content exclusion): scanning %d candidate session(s)",
        len(candidate_ids),
    )
    t2 = time.time()
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(IMAGE_CHECK_SQL, {"session_ids": candidate_ids})
        excluded_ids = {row["session_id"] for row in cur.fetchall()}
    logger.info(
        "phase 2: %d session(s) excluded for v1/v2 image content in %.1fs",
        len(excluded_ids), time.time() - t2,
    )

    surviving = [c for c in candidates if c["session_id"] not in excluded_ids]

    # Phase 2.5: optional max-parallel-subagents filter. Computes peak
    # concurrent subagent-group count per session via a sweep-line over
    # agent_id spans; drops sessions where peak > cap.
    if args.max_parallel_subagents is not None:
        surviving_ids = [c["session_id"] for c in surviving]
        logger.info(
            "phase 2.5 (max-parallel-subagents <= %d): scanning %d session(s)",
            args.max_parallel_subagents, len(surviving_ids),
        )
        t25 = time.time()
        peak_by_id: dict[str, int] = {}
        if surviving_ids:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    SUBAGENT_CONCURRENCY_SQL,
                    {
                        "session_ids": surviving_ids,
                        "model_like": model_like,
                        "migration_floor": SUBAGENT_MIGRATION_TS,
                        "until": args.until,
                        "privacy": args.privacy_mode,
                    },
                )
                for row in cur.fetchall():
                    peak_by_id[row["session_id"]] = row["max_parallel_subagents"]
        # Sessions absent from peak_by_id have zero subagents → peak 0.
        # Annotate every surviving row with its computed peak so downstream
        # logging / manifest can see it.
        kept = []
        for c in surviving:
            peak = peak_by_id.get(c["session_id"], 0)
            c["max_parallel_subagents"] = peak
            if peak <= args.max_parallel_subagents:
                kept.append(c)
        logger.info(
            "phase 2.5: %d session(s) survive peak <= %d in %.1fs (dropped %d)",
            len(kept), args.max_parallel_subagents,
            time.time() - t25, len(surviving) - len(kept),
        )
        surviving = kept

    logger.info(
        "phase 3 (sampling=%s, limit=%s): %d → %d session(s)",
        args.sampling, args.limit,
        len(surviving), min(args.limit or len(surviving), len(surviving)),
    )
    return _sort_and_limit(surviving, args)


# Session-id passthrough: compute the same summary stats as phase 1 for
# a single explicitly named session. No filtering on min_requests /
# image content / migration_floor — the caller asked for this session
# by id, we trust them. Privacy mode and Anthropic-only filters still
# apply for safety.
ONE_SESSION_SUMMARY_SQL = f"""
SELECT session_id,
       count(*)                                                       AS req_count,
       count(DISTINCT model)                                          AS distinct_models,
       count(*) FILTER (WHERE ({EFFECTIVE_SUBAGENT_LABEL_EXPR}) IS NOT NULL)
                                                                      AS subagent_reqs,
       sum(input_tokens)                                              AS total_in,
       sum(output_tokens)                                             AS total_out,
       min(timestamp)                                                 AS first_ts,
       max(timestamp)                                                 AS last_ts,
       extract(epoch FROM (max(timestamp) - min(timestamp)))::float   AS span_sec
  FROM requests
 WHERE session_id = %(sid)s
   AND model LIKE %(model_like)s
   AND response_status_code = 200
   AND error IS NULL
   AND privacy_mode = %(privacy)s
   AND ({SECURITY_MONITOR_FILTER_SQL})
 GROUP BY session_id
"""


def find_one_session(
    conn, session_id: str, privacy_mode: str, model_like: str
) -> list[dict]:
    logger.info("looking up session %s (--session-id mode)", session_id)
    t = time.time()
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            ONE_SESSION_SUMMARY_SQL,
            {
                "sid": session_id,
                "model_like": model_like,
                "privacy": privacy_mode,
            },
        )
        rows = cur.fetchall()
    if not rows:
        logger.warning(
            "no matching rows for session %s "
            "(filters: model LIKE %s, status=200, error IS NULL, privacy_mode=%s)",
            session_id, model_like, privacy_mode,
        )
        return []
    logger.info(
        "found session %s: %d rows, %d subagent, in %.1fs",
        session_id, rows[0]["req_count"], rows[0]["subagent_reqs"],
        time.time() - t,
    )
    return rows


REQUESTS_SQL = f"""
SELECT
    timestamp,
    extract(epoch FROM (timestamp - min(timestamp) OVER ()))::float AS t_sec,
    model,
    is_streaming,
    input_tokens,
    output_tokens,
    cache_read_input_tokens,
    cache_write_tokens,
    hash_token_count,
    hash_ids,
    ttft_ms,
    tpot_ms,
    duration_ms,
    ({EFFECTIVE_SUBAGENT_LABEL_EXPR})                               AS subagent_label,
    ({THREAD_ID_EXPR})                                              AS thread_id,
    ({CLAUDE_CODE_AGENT_ID_EXPR})                                   AS agent_id,
    ({CLI_VERSION_EXPR})                                            AS cli_version,
    response_status_code,
    privacy_mode,
    metadata
  FROM requests
 WHERE session_id = %(sid)s
   AND model LIKE %(model_like)s
   AND response_status_code = 200
   AND error IS NULL
   AND privacy_mode = %(privacy)s
   AND ({SECURITY_MONITOR_FILTER_SQL})
 ORDER BY timestamp
"""


# ---------------------------------------------------------------------------
# Dynamic-workflow subagent-label bug (Claude Code CLI < 2.1.174)
# ---------------------------------------------------------------------------
# Claude Code CLI versions BEFORE 2.1.174 failed to attach the subagent-label /
# x-claude-code-agent-id header to subagents launched via *dynamic workflows*
# (changelog 2.1.173 -> 2.1.174). Those subagents land in the proxy as ordinary
# UNLABELED main-stream requests, so a single session carries many independent,
# time-INTERLEAVED multi-turn trajectories with few/no subagent-label headers.
# Such sessions are not faithful single-trajectory traces (they inflate the
# apparent prefix-cache "rollback" rate) and are excluded when
# --exclude-dynamic-workflow-bug is set.
#
# Signature (validated on the 061526 corpus): pre-fix CLI AND the unlabeled
# main-stream requests cluster into >= N CONCURRENTLY-ACTIVE multi-turn
# trajectories. Concurrency (sweep-line peak over trajectory time spans), not
# mere trajectory count, is what separates interleaved unlabeled subagents from
# sequential context compaction (which restarts a trajectory but never overlaps
# it). At N=3 this cleanly separates buggy from fixed sessions.
DYNAMIC_WORKFLOW_FIX_CLI = (2, 1, 174)
# Tolerate up to this many trailing hash blocks differing when deciding whether
# a request extends a trajectory (the per-turn ephemeral footer; see the weka
# converter's small-tail handling).
_DWBUG_TAIL_TOLERANCE = 6


def _parse_cli_tuple(s: str | None) -> tuple[int, int, int] | None:
    if not s:
        return None
    parts = s.split(".")
    if len(parts) != 3:
        return None
    try:
        return tuple(int(p) for p in parts)  # type: ignore[return-value]
    except ValueError:
        return None


def _hash_lcp(a: list, b: list) -> int:
    n = 0
    for x, y in zip(a, b):
        if x == y:
            n += 1
        else:
            break
    return n


def is_dynamic_workflow_bug(rows: list[dict], min_peak: int) -> bool:
    """Return True if a session exhibits the CLI<2.1.174 dynamic-workflow
    unlabeled-subagent signature (see module note above)."""
    clis = [c for c in (_parse_cli_tuple(r.get("cli_version")) for r in rows) if c]
    if not clis or max(clis) >= DYNAMIC_WORKFLOW_FIX_CLI:
        return False
    # Main-stream = rows with no effective subagent_label. The buggy
    # dynamic-workflow subagents are exactly the unlabeled requests here.
    main = [r for r in rows if not r.get("subagent_label")]
    trajs: list[list] = []  # [head_hash_ids, [(start_s, dur_s), ...]]
    for r in main:
        H = r.get("hash_ids") or []
        if not H:
            continue
        t = r.get("t_sec") or 0.0
        dur = (r.get("duration_ms") or 0) / 1000.0
        best, bi = -1, -1
        for i, (head, _) in enumerate(trajs):
            lcp = _hash_lcp(head, H)
            # H extends this trajectory if the trajectory's head is a (near-)
            # prefix of H: at least one block matches AND at most the trailing
            # _DWBUG_TAIL_TOLERANCE blocks (the per-turn ephemeral footer) of
            # the head diverge. max(1, ...) keeps the tail tolerance from
            # collapsing the threshold below 1 for short heads (which would
            # wrongly merge unrelated short trajectories).
            need = max(1, len(head) - _DWBUG_TAIL_TOLERANCE)
            if lcp >= need and lcp > best:
                best, bi = lcp, i
        if bi >= 0:
            trajs[bi][0] = H
            trajs[bi][1].append((t, dur))
        else:
            trajs.append([H, [(t, dur)]])
    # Only multi-turn trajectories (>=2 turns) count; single-shot utility
    # calls are not interleaved subagents.
    spans = [
        (min(s for s, _ in rr), max(s + d for s, d in rr))
        for _, rr in trajs
        if len(rr) >= 2
    ]
    if not spans:
        return False
    events: list[tuple[float, int]] = []
    for start, end in spans:
        events.append((start, 1))
        events.append((end, -1))
    # Ends before starts at equal timestamps (a trajectory starting exactly when
    # another ends does not count as overlapping).
    events.sort(key=lambda x: (x[0], x[1]))
    cur = peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak >= min_peak


def fetch_session_rows(
    conn, session_id: str, privacy_mode: str, model_like: str
) -> list[dict]:
    """Fetch all filtered rows for a session (ordered by timestamp)."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            REQUESTS_SQL,
            {"sid": session_id, "model_like": model_like, "privacy": privacy_mode},
        )
        rows = cur.fetchall()
    for row in rows:
        # Belt-and-braces: SQL already filters, but a per-row check guarantees
        # no mismatched-privacy row hits disk even if the WHERE clause is ever
        # broken upstream.
        if row["privacy_mode"] != privacy_mode:
            raise RuntimeError(
                f"privacy_mode safety check failed for session "
                f"{session_id}: row has {row['privacy_mode']!r}, "
                f"expected {privacy_mode!r}"
            )
        row["timestamp"] = row["timestamp"].isoformat()
    return rows


def write_session_rows(rows: list[dict], out_path: Path) -> int:
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
    return len(rows)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )
    url = get_db_url(args)
    t0 = time.time()

    # `--allow-non-anthropic` swaps the model filter to '%' (match all).
    # Logged loud so it's never accidentally hidden in a long run.
    model_like = "%" if args.allow_non_anthropic else ANTHROPIC_MODEL_LIKE
    if args.allow_non_anthropic:
        logger.warning(
            "--allow-non-anthropic is set: Codex/GPT rows will be included. "
            "Downstream weka conversion is undefined for non-Claude data."
        )

    logger.info("connecting to %s", _redact_url(url))
    with psycopg.connect(url) as conn:
        logger.info("connected")
        if args.session_id:
            sessions = find_one_session(
                conn, args.session_id, args.privacy_mode, model_like
            )
        else:
            sessions = find_sessions(conn, args, model_like)

        logger.info("preview of matched sessions (showing up to 5):")
        for s in sessions[:5]:
            peak = s.get("max_parallel_subagents")
            peak_str = f"  peak_par={peak}" if peak is not None else ""
            logger.info(
                "    %s  reqs=%5d  sub=%5d  span=%5.1fh  in=%10s  out=%10s%s",
                s["session_id"], s["req_count"], s["subagent_reqs"],
                s["span_sec"] / 3600, s["total_in"], s["total_out"], peak_str,
            )
        if len(sessions) > 5:
            logger.info("    ... (%d more)", len(sessions) - 5)

        if not sessions:
            logger.warning("no sessions matched — nothing to dump")
            return 0

        if args.dry_run:
            logger.info("dry-run: skipping per-session dump and manifest write")
            return 0

        args.out.mkdir(parents=True, exist_ok=True)
        logger.info("dumping %d session(s) to %s/", len(sessions), args.out)
        manifest: dict = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "filters": {
                "migration_floor": SUBAGENT_MIGRATION_TS,
                "until": args.until,
                "min_requests": args.min_requests,
                "max_requests": args.max_requests,
                "max_span_hours": args.max_span_hours,
                "min_main_turns": args.min_main_turns,
                "min_trace_version": args.min_trace_version,
                "max_trace_version": args.max_trace_version,
                "require_cli_min": args.require_cli_min,
                "privacy_mode": args.privacy_mode,
                "sampling": args.sampling,
                "seed": args.seed,
                "model_like": model_like,
                "allow_non_anthropic": args.allow_non_anthropic,
                "session_id": args.session_id,
                "security_monitor_excluded": True,
                "classifier_calls_excluded": True,
                "dynamic_workflow_bug_excluded": args.exclude_dynamic_workflow_bug,
                "dwbug_min_peak": (
                    args.dwbug_min_peak if args.exclude_dynamic_workflow_bug else None
                ),
            },
            "sessions": [],
        }

        total_rows = 0
        n_excluded_dwbug = 0
        for i, s in enumerate(sessions, 1):
            sid = s["session_id"]
            fname = f"{sid}.jsonl"
            out_path = args.out / fname
            t_dump = time.time()
            rows = fetch_session_rows(conn, sid, args.privacy_mode, model_like)
            if args.exclude_dynamic_workflow_bug and is_dynamic_workflow_bug(
                rows, args.dwbug_min_peak
            ):
                n_excluded_dwbug += 1
                logger.info(
                    "[%4d/%d] %s  EXCLUDED (dynamic-workflow bug: CLI<2.1.174 + "
                    ">=%d concurrent unlabeled trajectories)",
                    i, len(sessions), sid, args.dwbug_min_peak,
                )
                continue
            n_rows = write_session_rows(rows, out_path)
            total_rows += n_rows
            manifest["sessions"].append({
                "session_id": sid,
                "file": fname,
                "request_count_filtered": n_rows,
                "request_count_raw": s["req_count"],
                "subagent_reqs": s["subagent_reqs"],
                "distinct_models": s["distinct_models"],
                "total_input_tokens": s["total_in"],
                "total_output_tokens": s["total_out"],
                "first_ts": s["first_ts"].isoformat(),
                "last_ts": s["last_ts"].isoformat(),
                "span_sec": s["span_sec"],
            })
            logger.info(
                "[%4d/%d] %s  rows=%5d  (%4.1fs)  -> %s",
                i, len(sessions), sid, n_rows, time.time() - t_dump, out_path,
            )

        manifest_path = args.out / "manifest.json"
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2, default=str)
        logger.info("wrote manifest: %s", manifest_path)
        if args.exclude_dynamic_workflow_bug:
            logger.info(
                "dynamic-workflow-bug filter: excluded %d session(s) "
                "(CLI<2.1.174 + >=%d concurrent unlabeled trajectories)",
                n_excluded_dwbug, args.dwbug_min_peak,
            )
        logger.info(
            "done. %d session(s) written, %d total rows, in %.1fs",
            len(sessions) - n_excluded_dwbug, total_rows, time.time() - t0,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
