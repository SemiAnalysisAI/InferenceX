#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert ISB1 replay bundles into `kv-cache-tester` trace files.

Produces per-conversation JSON files compatible with
`callanjfox/kv-cache-tester`'s `trace_replay_tester.py --trace-directory`
input format (see upstream ``TraceManager.load_traces`` / ``normalize_trace``
in that repo). Works on both ISB1 bundle adapters shipped by this PR:

- ``inferencex_multiturn``     (direct ``session.turns[].messages`` shape)
- ``inferencex_trace_replay``  (``events[].input_messages`` shape, with
  optional schema ``0.2.0`` ``prefix_ref`` sidecar hydration)

This shim is the ONLY glue between SemiAnalysis's trace-replay tester and our
ISB1 bundles. It emits one ``trace_<NNNN>.json`` per conversation/cell and
does not import or execute any benchmark harness. It has no third-party
dependencies: standard library only.

Schema compatibility
--------------------

The ``kv-cache-tester`` input schema (verified against
``trace_replay_tester.py@main``):

    {
      "id":              "<trace identifier>",     # str, required
      "models":          ["<canonical model id>"],  # list[str], optional
      "block_size":      64,                        # int, optional (default 64)
      "hash_id_scope":   "local",                   # str, optional
      "tool_tokens":     0,                         # int, optional
      "system_tokens":   0,                         # int, optional
      "requests": [
        {
          "t":           0.0,                       # float, arrival offset (s)
          "type":        "n",                       # "n" non-streaming | "s" streaming
          "model":       "<canonical model id>",
          "in":          <int>,                     # input tokens
          "out":         <int>,                     # output tokens
          "hash_ids":    [1, 2, ..., N],            # prefix-reuse block sequence
          "input_types":  ["text"],
          "output_types": ["text"],
          "stop":        "end_turn",
          "think_time":  <float>                    # seconds before request
        }, ...
      ]
    }

The ``hash_ids`` field is the critical one. ``kv-cache-tester`` computes the
cache hit rate by walking each turn's ``hash_ids`` and counting hits against
the previous turn's set, stopping at the first miss. For multi-turn
conversations where every turn builds on the previous history, the correct
mapping is the monotonically extending prefix ``[1, 2, ..., ceil(in/block)]``
— earlier blocks are reused across turns and the new user message + assistant
turn extend the prefix.

Usage
-----

Single bundle:

    python tools/isb1_to_kvcache_tester.py \
        --export-file datasets/isb1/exports/core/chat_8k1k_qwen3.5.json \
        --output-dir  traces_isb1/

Whole directory tree (reproduces the ``traces_isb1/<band>/<bundle>/`` layout):

    python tools/isb1_to_kvcache_tester.py \
        --export-root datasets/isb1/exports/ \
        --output-dir  traces_isb1/

Filters (subset by cell, matching ``benchmark_export_replay.py`` semantics):

    python tools/isb1_to_kvcache_tester.py \
        --export-file datasets/isb1/exports/core/chat_8k1k.json \
        --output-dir  traces_isb1/core_chat_qwen_h200/ \
        --runtime-stack-id standalone:vllm \
        --hardware-profile-id nvidia:h200_sxm_141gb \
        --canonical-model-id qwen3_5_397b_a17b

Smoke run against Cam's tester (after a vLLM OpenAI server is up on :8888):

    python /path/to/kv-cache-tester/trace_replay_tester.py \
        --api-endpoint http://127.0.0.1:8888 \
        --trace-directory traces_isb1/core_chat_qwen_h200/ \
        --output-dir       /tmp/isb1_result/ \
        --start-users 2 --max-users 2 --test-duration 60

The shim is deterministic: same input bundle, same filters, same block_size
produces byte-identical output trace files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

SUPPORTED_ADAPTERS = ("inferencex_multiturn", "inferencex_trace_replay")
DEFAULT_BLOCK_SIZE = 64       # matches kv-cache-tester default
DEFAULT_IMAGE_TOKENS = 512    # matches DEFAULT_IMAGE_TOKEN_ESTIMATE in our exporter
FALLBACK_OUTPUT_TOKENS = 256  # used only if the bundle has no expected_output_tokens


# -----------------------------------------------------------------------------
# Token counting (stdlib-only, matches benchmark_export_replay._fallback_text_token_count)
# -----------------------------------------------------------------------------

def _fallback_text_token_count(text: str) -> int:
    """Approximate token count (≈ 4 chars / token).

    This matches the fallback path in ``utils/bench_serving/benchmark_export_replay.py``
    and mirrors Cam's tester, which does its own tokenization at replay time
    from synthetic content. The only thing this shim must emit accurately is
    the *block count* (``ceil(in_tokens / block_size)``); a ~10-20% error in
    the absolute token count still produces the correct prefix-reuse pattern
    because block IDs are assigned in order.
    """
    stripped = (text or "").strip()
    if not stripped:
        return 0
    return max(1, math.ceil(len(stripped) / 4))


# -----------------------------------------------------------------------------
# Message flattening (mirrors benchmark_export_replay._extract_message_text)
# -----------------------------------------------------------------------------

def _render_block_as_text(block: dict[str, Any]) -> str:
    block_type = str(block.get("type", "text"))
    text = (block.get("text") or "").strip()
    if block_type == "text":
        return text
    if block_type == "code":
        return f"[CODE]\n{text}" if text else "[CODE]"
    if block_type == "log":
        return f"[LOG]\n{text}" if text else "[LOG]"
    if block_type == "document":
        label = block.get("asset_path") or block.get("uri") or ""
        if text and label:
            return f"[DOCUMENT: {label}]\n{text}"
        if text:
            return f"[DOCUMENT]\n{text}"
        return f"[DOCUMENT: {label}]" if label else "[DOCUMENT]"
    if block_type == "table":
        return f"[TABLE]\n{text}" if text else "[TABLE]"
    if block_type == "image":
        # images are approximated as fixed-cost tokens; no text to render
        return ""
    return text or ""


def _extract_message_text(message: dict[str, Any]) -> str:
    if isinstance(message.get("content"), str):
        return message["content"]
    blocks = message.get("content_blocks") or []
    parts = [_render_block_as_text(b) for b in blocks if isinstance(b, dict)]
    return "\n\n".join(p for p in parts if p)


def _count_image_tokens(message: dict[str, Any], image_token_estimate: int) -> int:
    blocks = message.get("content_blocks") or []
    return sum(
        image_token_estimate
        for b in blocks
        if isinstance(b, dict) and str(b.get("type")) == "image"
    )


def _count_message_tokens(message: dict[str, Any], image_token_estimate: int) -> int:
    text_tokens = _fallback_text_token_count(_extract_message_text(message))
    return text_tokens + _count_image_tokens(message, image_token_estimate)


def _count_turn_input_tokens(
    messages: list[dict[str, Any]],
    image_token_estimate: int,
) -> int:
    return sum(_count_message_tokens(m, image_token_estimate) for m in messages)


# -----------------------------------------------------------------------------
# Prefix sidecar hydration (schema 0.2.0 inferencex_trace_replay bundles)
# -----------------------------------------------------------------------------

def _schema_version_at_least(observed: Any, required: str) -> bool:
    if not isinstance(observed, str):
        return False
    try:
        obs = tuple(int(x) for x in observed.split("."))
        req = tuple(int(x) for x in required.split("."))
    except ValueError:
        return False
    return obs >= req


def _load_prefix_artifact(
    bundle_path: Path,
    prefix_ref: str,
    prefix_entry: dict[str, Any],
) -> dict[str, Any]:
    rel_path = prefix_entry.get("path")
    if not isinstance(rel_path, str) or not rel_path:
        raise ValueError(
            f"prefix_index[{prefix_ref!r}] missing 'path' in {bundle_path}"
        )
    artifact_path = (bundle_path.parent / rel_path).resolve()
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"prefix artifact not found: {artifact_path} (ref={prefix_ref!r})"
        )
    raw = artifact_path.read_bytes()

    expected_sha = prefix_entry.get("sha256")
    if isinstance(expected_sha, str) and expected_sha:
        actual_sha = hashlib.sha256(raw).hexdigest()
        if actual_sha.lower() != expected_sha.lower():
            raise ValueError(
                "prefix artifact sha256 mismatch for "
                f"{prefix_ref!r}: expected {expected_sha}, got {actual_sha}"
            )

    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(
            f"prefix artifact {artifact_path} is not valid JSON: {exc}"
        ) from exc


def _merge_prefix_into_cell(cell: dict[str, Any], prefix_payload: dict[str, Any]) -> None:
    # Schema 0.2.0 bundles store events in the prefix artifact; merge back in-place.
    # (Mirrors benchmark_export_replay._merge_prefix_into_trace_replay_cell.)
    prefix_events = prefix_payload.get("events") or []
    cell.setdefault("events", []).extend(prefix_events)
    # Also merge trace_metadata if the prefix carries extras; preserve cell priority.
    p_meta = prefix_payload.get("trace_metadata") or {}
    if isinstance(p_meta, dict) and p_meta:
        merged = dict(p_meta)
        merged.update(cell.get("trace_metadata") or {})
        cell["trace_metadata"] = merged


def _hydrate_trace_replay_payload(payload: dict[str, Any], bundle_path: Path) -> None:
    if not _schema_version_at_least(payload.get("schema_version"), "0.2.0"):
        return
    export_cells = list(payload.get("exports", []))
    if not export_cells:
        return

    has_prefix_ref = any(cell.get("prefix_ref") for cell in export_cells)
    has_embedded_events = any("events" in cell for cell in export_cells)
    if has_prefix_ref and has_embedded_events:
        raise ValueError(
            "mixed legacy/prefix-aware trace replay bundle unsupported in "
            f"{bundle_path}; rows cannot mix embedded events with prefix_ref"
        )
    if not has_prefix_ref:
        return

    missing_prefix_ref = [c for c in export_cells if not c.get("prefix_ref")]
    if missing_prefix_ref:
        raise ValueError(
            f"prefix-aware trace replay bundle missing prefix_ref in {bundle_path}"
        )

    raw_prefix_index = payload.get("prefix_index")
    prefix_index = raw_prefix_index if isinstance(raw_prefix_index, dict) else {}
    prefix_payloads: dict[str, dict[str, Any]] = {}
    for prefix_ref in {str(cell["prefix_ref"]) for cell in export_cells}:
        entry = prefix_index.get(prefix_ref)
        if not isinstance(entry, dict):
            raise ValueError(f"unknown prefix_ref {prefix_ref!r} in {bundle_path}")
        prefix_payloads[prefix_ref] = _load_prefix_artifact(
            bundle_path, prefix_ref, entry
        )

    for cell in export_cells:
        _merge_prefix_into_cell(cell, prefix_payloads[str(cell["prefix_ref"])])


# -----------------------------------------------------------------------------
# ISB1 → kv-cache-tester per-turn mapping
# -----------------------------------------------------------------------------

def _hash_id_sequence(input_tokens: int, block_size: int) -> list[int]:
    """Emit the canonical prefix-extending hash_id sequence.

    Each turn's ``hash_ids`` is ``[1, 2, ..., ceil(in/block_size)]``. Since
    multi-turn conversations are strictly-extending prefixes (turn N+1 input
    is turn N input + assistant response + new user message), the earlier
    block IDs appear in every subsequent turn; ``kv-cache-tester``'s
    hit-rate walker sees them as cache hits.
    """
    if input_tokens <= 0:
        return []
    n_blocks = max(1, math.ceil(input_tokens / block_size))
    return list(range(1, n_blocks + 1))


def _build_request_from_turn(
    *,
    turn_idx: int,
    messages: list[dict[str, Any]],
    expected_output_tokens: Optional[int],
    wait_before_ms: int,
    prior_offset_ms: int,
    canonical_model_id: str,
    image_token_estimate: int,
    block_size: int,
    fallback_output_tokens: int,
) -> tuple[dict[str, Any], int]:
    input_tokens = _count_turn_input_tokens(messages, image_token_estimate)
    out_tokens = int(expected_output_tokens) if expected_output_tokens else fallback_output_tokens
    out_tokens = max(1, out_tokens)

    arrival_ms = prior_offset_ms + max(0, int(wait_before_ms))
    think_time_s = max(0.0, arrival_ms / 1000.0 if turn_idx == 0 else (arrival_ms - prior_offset_ms) / 1000.0)

    request = {
        "t": round(arrival_ms / 1000.0, 3),
        "type": "n",
        "model": canonical_model_id,
        "in": input_tokens,
        "out": out_tokens,
        "hash_ids": _hash_id_sequence(input_tokens, block_size),
        "input_types": ["text"],
        "output_types": ["text"],
        "stop": "end_turn",
        "think_time": round(think_time_s, 3),
    }
    return request, arrival_ms


def _iter_cells_from_multiturn(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for cell in payload.get("exports") or []:
        if not isinstance(cell, dict):
            continue
        yield cell


def _iter_cells_from_trace_replay(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for cell in payload.get("exports") or []:
        if not isinstance(cell, dict):
            continue
        yield cell


def _conversation_id(cell: dict[str, Any], adapter_id: str, fallback_idx: int) -> str:
    if adapter_id == "inferencex_multiturn":
        session = cell.get("session") or {}
        cid = session.get("session_id") or cell.get("trace_id")
    else:
        meta = cell.get("trace_metadata") or {}
        cid = meta.get("session_id") or cell.get("trace_id")
    if not cid:
        cid = f"cell_{fallback_idx:04d}"
    return str(cid)


def _cell_turns_multiturn(cell: dict[str, Any]) -> list[dict[str, Any]]:
    session = cell.get("session") or {}
    out = []
    for turn in session.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        out.append({
            "messages": list(turn.get("messages") or []),
            "expected_output_tokens": turn.get("expected_output_tokens"),
            "wait_before_ms": int(turn.get("wait_before_ms") or 0),
        })
    return out


def _cell_turns_trace_replay(cell: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    prior_ms = 0
    for event in cell.get("events") or []:
        if not isinstance(event, dict):
            continue
        offset_ms = int(event.get("arrival_time_offset_ms") or 0)
        wait_ms = 0 if not out else max(0, offset_ms - prior_ms)
        prior_ms = offset_ms
        out.append({
            "messages": list(event.get("input_messages") or []),
            "expected_output_tokens": event.get("target_output_tokens"),
            "wait_before_ms": wait_ms,
        })
    return out


def _build_trace(
    *,
    cell: dict[str, Any],
    adapter_id: str,
    fallback_idx: int,
    block_size: int,
    fallback_output_tokens: int,
    image_token_estimate: int,
    max_turns_per_conversation: Optional[int],
) -> Optional[dict[str, Any]]:
    conversation_id = _conversation_id(cell, adapter_id, fallback_idx)
    canonical_model_id = str(cell.get("canonical_model_id") or "unknown")

    if adapter_id == "inferencex_multiturn":
        turns = _cell_turns_multiturn(cell)
    else:
        turns = _cell_turns_trace_replay(cell)

    if not turns:
        return None

    if max_turns_per_conversation is not None:
        turns = turns[: max(1, int(max_turns_per_conversation))]

    requests = []
    cumulative_ms = 0
    for turn_idx, turn in enumerate(turns):
        request, cumulative_ms = _build_request_from_turn(
            turn_idx=turn_idx,
            messages=turn["messages"],
            expected_output_tokens=turn.get("expected_output_tokens"),
            wait_before_ms=turn.get("wait_before_ms", 0),
            prior_offset_ms=cumulative_ms,
            canonical_model_id=canonical_model_id,
            image_token_estimate=image_token_estimate,
            block_size=block_size,
            fallback_output_tokens=fallback_output_tokens,
        )
        requests.append(request)

    total_in = sum(r["in"] for r in requests)
    total_out = sum(r["out"] for r in requests)

    return {
        "id": conversation_id,
        "models": [canonical_model_id],
        "block_size": int(block_size),
        "hash_id_scope": "local",
        "tool_tokens": 0,
        "system_tokens": 0,
        "requests": requests,
        "totals": {
            "parent_tokens": {"input": total_in, "output": total_out},
            "subagent_tokens": {"input": 0, "output": 0},
            "combined_tokens": {"input": total_in, "output": total_out},
            "subagent_count": 0,
        },
        # ISB1 passthrough tags — kv-cache-tester ignores unknown keys.
        "isb1": {
            "trace_id": cell.get("trace_id"),
            "runtime_stack_id": cell.get("runtime_stack_id"),
            "hardware_profile_id": cell.get("hardware_profile_id"),
            "canonical_model_id": canonical_model_id,
            "support_status": cell.get("support_status"),
            "benchmark_certification_status": cell.get(
                "benchmark_certification_status"
            ),
            "context_band": cell.get("context_band"),
            "adapter_id": adapter_id,
        },
    }


# -----------------------------------------------------------------------------
# Filter logic (subset of benchmark_export_replay.load_replay_sessions)
# -----------------------------------------------------------------------------

def _matches(value: Any, allowed: set[str] | None) -> bool:
    if allowed is None:
        return True
    return str(value or "") in allowed


def _csv_set(raw: Optional[str]) -> set[str] | None:
    if raw is None:
        return None
    return {v.strip() for v in raw.split(",") if v.strip()} or None


# -----------------------------------------------------------------------------
# Bundle processing
# -----------------------------------------------------------------------------

def _load_bundle(export_file: Path) -> dict[str, Any]:
    try:
        payload = json.loads(export_file.read_text())
    except Exception as exc:
        raise ValueError(f"failed to read ISB1 bundle {export_file}: {exc}") from exc
    adapter_id = str(payload.get("adapter_id") or "unknown")
    if adapter_id not in SUPPORTED_ADAPTERS:
        raise ValueError(
            f"unsupported ISB1 adapter {adapter_id!r} in {export_file}. "
            f"Expected one of {SUPPORTED_ADAPTERS}. "
            "This shim converts ISB1 replay bundles only; "
            "raw model traces from other pipelines are out of scope."
        )
    if adapter_id == "inferencex_trace_replay":
        _hydrate_trace_replay_payload(payload, export_file)
    return payload


def _convert_bundle(
    *,
    export_file: Path,
    output_dir: Path,
    runtime_stack_ids: set[str] | None,
    hardware_profile_ids: set[str] | None,
    canonical_model_ids: set[str] | None,
    trace_ids: set[str] | None,
    support_statuses: set[str] | None,
    block_size: int,
    fallback_output_tokens: int,
    image_token_estimate: int,
    max_conversations: Optional[int],
    max_turns_per_conversation: Optional[int],
    trace_prefix: str,
) -> tuple[int, int]:
    """Returns (written_count, skipped_count)."""
    payload = _load_bundle(export_file)
    adapter_id = str(payload["adapter_id"])

    iterator = (
        _iter_cells_from_multiturn(payload)
        if adapter_id == "inferencex_multiturn"
        else _iter_cells_from_trace_replay(payload)
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    emitted_ids: set[str] = set()
    for idx, cell in enumerate(iterator):
        if not _matches(cell.get("runtime_stack_id"), runtime_stack_ids):
            skipped += 1
            continue
        if not _matches(cell.get("hardware_profile_id"), hardware_profile_ids):
            skipped += 1
            continue
        if not _matches(cell.get("canonical_model_id"), canonical_model_ids):
            skipped += 1
            continue
        if not _matches(cell.get("trace_id"), trace_ids):
            skipped += 1
            continue
        if not _matches(cell.get("support_status"), support_statuses):
            skipped += 1
            continue

        trace = _build_trace(
            cell=cell,
            adapter_id=adapter_id,
            fallback_idx=idx,
            block_size=block_size,
            fallback_output_tokens=fallback_output_tokens,
            image_token_estimate=image_token_estimate,
            max_turns_per_conversation=max_turns_per_conversation,
        )
        if trace is None:
            skipped += 1
            continue

        # Make the filename unique per cell, stable across runs.
        base_id = str(trace["id"])
        safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in base_id)
        if safe_id in emitted_ids:
            safe_id = f"{safe_id}_{idx:04d}"
        emitted_ids.add(safe_id)

        fname = f"{trace_prefix}{safe_id}.json"
        out_path = output_dir / fname
        out_path.write_text(json.dumps(trace, indent=2, sort_keys=False))
        written += 1

        if max_conversations is not None and written >= int(max_conversations):
            break

    return written, skipped


def _iter_export_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.json")):
        # Skip manifests/READMEs (anything that isn't an adapter bundle).
        if path.name in ("manifest.json", "manifest_qwen3.5.json"):
            continue
        if "/prefixes/" in path.as_posix():
            continue
        try:
            with path.open() as fh:
                first_chunk = fh.read(4096)
        except Exception:
            continue
        if '"adapter_id"' not in first_chunk:
            continue
        yield path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="isb1_to_kvcache_tester",
        description=(
            "Convert ISB1 replay bundles into kv-cache-tester-compatible "
            "per-conversation JSON files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--export-file",
        type=Path,
        help="Single ISB1 bundle JSON (e.g. datasets/isb1/exports/core/chat_8k1k.json).",
    )
    src.add_argument(
        "--export-root",
        type=Path,
        help=(
            "Directory tree under which every adapter bundle is converted "
            "(e.g. datasets/isb1/exports/). Mirrors subpaths under --output-dir."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for trace_*.json files.",
    )
    parser.add_argument(
        "--trace-prefix",
        type=str,
        default="isb1_",
        help="Filename prefix for emitted trace files (default: 'isb1_').",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help=(
            f"Hash block size in tokens (default: {DEFAULT_BLOCK_SIZE}, "
            "matches kv-cache-tester default)."
        ),
    )
    parser.add_argument(
        "--image-token-estimate",
        type=int,
        default=DEFAULT_IMAGE_TOKENS,
        help=f"Tokens per image block (default: {DEFAULT_IMAGE_TOKENS}).",
    )
    parser.add_argument(
        "--fallback-output-tokens",
        type=int,
        default=FALLBACK_OUTPUT_TOKENS,
        help=(
            f"Output tokens to emit when a turn has no expected_output_tokens "
            f"(default: {FALLBACK_OUTPUT_TOKENS})."
        ),
    )

    parser.add_argument("--runtime-stack-id",
                        help="CSV of runtime_stack_ids to include (e.g. standalone:vllm,standalone:sglang).")
    parser.add_argument("--hardware-profile-id",
                        help="CSV of hardware_profile_ids to include.")
    parser.add_argument("--canonical-model-id",
                        help="CSV of canonical_model_ids to include.")
    parser.add_argument("--trace-id",
                        help="CSV of trace_ids to include.")
    parser.add_argument("--support-status",
                        help="CSV of support_status values to include (e.g. supported,reviewed_preview).")

    parser.add_argument("--max-conversations", type=int, default=None,
                        help="Stop after writing N conversations per bundle.")
    parser.add_argument("--max-turns-per-conversation", type=int, default=None,
                        help="Truncate each conversation after N turns.")

    parser.add_argument("--quiet", action="store_true", help="Suppress per-bundle progress.")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    if args.block_size <= 0:
        print(f"ERROR: --block-size must be positive (got {args.block_size})", file=sys.stderr)
        return 2

    runtime_stack_ids = _csv_set(args.runtime_stack_id)
    hardware_profile_ids = _csv_set(args.hardware_profile_id)
    canonical_model_ids = _csv_set(args.canonical_model_id)
    trace_ids = _csv_set(args.trace_id)
    support_statuses = _csv_set(args.support_status)

    bundles: list[tuple[Path, Path]] = []
    if args.export_file:
        bundles.append((args.export_file.resolve(), args.output_dir.resolve()))
    else:
        root = args.export_root.resolve()
        if not root.is_dir():
            print(f"ERROR: --export-root not a directory: {root}", file=sys.stderr)
            return 2
        for f in _iter_export_files(root):
            rel = f.relative_to(root).parent
            out = args.output_dir.resolve() / rel / f.stem
            bundles.append((f, out))

    if not bundles:
        print("ERROR: no ISB1 bundles found", file=sys.stderr)
        return 2

    total_written = 0
    total_skipped = 0
    errors = 0
    for export_file, output_dir in bundles:
        try:
            written, skipped = _convert_bundle(
                export_file=export_file,
                output_dir=output_dir,
                runtime_stack_ids=runtime_stack_ids,
                hardware_profile_ids=hardware_profile_ids,
                canonical_model_ids=canonical_model_ids,
                trace_ids=trace_ids,
                support_statuses=support_statuses,
                block_size=args.block_size,
                fallback_output_tokens=args.fallback_output_tokens,
                image_token_estimate=args.image_token_estimate,
                max_conversations=args.max_conversations,
                max_turns_per_conversation=args.max_turns_per_conversation,
                trace_prefix=args.trace_prefix,
            )
        except (ValueError, FileNotFoundError) as exc:
            print(f"ERROR: {export_file}: {exc}", file=sys.stderr)
            errors += 1
            continue

        total_written += written
        total_skipped += skipped
        if not args.quiet:
            print(
                f"ok  {export_file}: wrote {written} trace(s) to {output_dir} "
                f"(skipped {skipped} cell(s) by filter)"
            )

    if not args.quiet:
        print(
            f"\ndone: {total_written} trace file(s) written across "
            f"{len(bundles)} bundle(s); {total_skipped} cell(s) skipped; "
            f"{errors} bundle(s) errored"
        )

    if total_written == 0:
        print("ERROR: no traces written — check --runtime-stack-id / --hardware-profile-id filters",
              file=sys.stderr)
        return 1

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
