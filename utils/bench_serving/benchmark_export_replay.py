# SPDX-License-Identifier: Apache-2.0
r"""Replay ISB1 export sessions against OpenAI-compatible inference servers.

Supported export formats:
    - ``inferencex_multiturn`` (direct-ingest session turns)
    - ``inferencex_trace_replay`` (event-based trace replay)

Supported request modes:
    - ``chat``: send full message history to ``/v1/chat/completions``
    - ``completions``: project the message history into a single tagged prompt
      and send it to ``/v1/completions``
    - ``auto``: prefer chat for standalone vLLM/SGLang cells and completions
      for TRT / Dynamo projection cells
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60, sock_read=5 * 60)
DEFAULT_IMAGE_TOKEN_ESTIMATE = 2048
DEFAULT_FALLBACK_OUTPUT_LEN = 256
CHAT_NATIVE_RUNTIMES = {"standalone:vllm", "standalone:sglang"}
COMPLETIONS_PREFERRED_RUNTIMES = {
    "standalone:trt_llm",
    "dynamo:vllm",
    "dynamo:sglang",
    "dynamo:trt_llm",
}
ROLE_LABELS = {
    "system": "SYSTEM",
    "user": "USER",
    "assistant": "ASSISTANT",
    "tool": "TOOL",
    "retrieval": "RETRIEVAL",
    "execution": "EXECUTION",
}
MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))


@dataclass
class TurnResult:
    turn_idx: int
    context_len: int
    output_len: int
    ttft: float = 0.0
    tpot: float = 0.0
    e2el: float = 0.0
    itl: list[float] = field(default_factory=list)
    success: bool = True
    error: str = ""
    request_mode: str = "chat"
    actual_context_len: int = 0


@dataclass
class SessionResult:
    session_id: str
    turns: list[TurnResult] = field(default_factory=list)
    total_input_tokens: int = 0
    total_actual_input_tokens: int = 0
    total_output_tokens: int = 0
    total_duration: float = 0.0


@dataclass
class ReplayTurn:
    turn_idx: int
    turn_id: Any
    output_len: int
    wait_before_s: float
    context_len: int
    actual_context_len: int
    chat_messages: list[dict[str, Any]]
    completion_prompt: str


@dataclass
class ReplaySession:
    session_id: str
    trace_id: str
    runtime_stack_id: str
    hardware_profile_id: str
    canonical_model_id: str
    support_status: str
    benchmark_certification_status: str
    request_mode: str
    adapter_id: str
    turns: list[ReplayTurn]


def _csv_values(raw: Optional[str]) -> set[str] | None:
    if raw is None:
        return None
    values = {item.strip() for item in raw.split(",") if item.strip()}
    return values or None


def _matches_filter(value: str, allowed: set[str] | None) -> bool:
    return allowed is None or value in allowed


def _fallback_text_token_count(text: str) -> int:
    stripped = (text or "").strip()
    if not stripped:
        return 0
    return max(1, math.ceil(len(stripped) / 4))


def build_text_token_counter(
    tokenizer_id: Optional[str],
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
) -> Callable[[str], int]:
    if not tokenizer_id:
        return _fallback_text_token_count

    try:
        from backend_request_func import get_tokenizer

        tokenizer = get_tokenizer(
            tokenizer_id,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        warnings.warn(
            "Falling back to approximate token counting because tokenizer load "
            f"failed for {tokenizer_id!r}: {exc}",
            stacklevel=2,
        )
        return _fallback_text_token_count

    def _count(text: str) -> int:
        return len(tokenizer.encode(text or "", add_special_tokens=False))

    return _count


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
        label = block.get("uri") or block.get("asset_path") or text or "image"
        return f"[IMAGE: {label}]"
    return text or f"[{block_type.upper()}]"


def _extract_message_text(message: dict[str, Any]) -> str:
    if isinstance(message.get("content"), str):
        body = message["content"]
    elif isinstance(message.get("content"), list):
        parts: list[str] = []
        for part in message["content"]:
            part_type = str(part.get("type", "text"))
            if part_type == "text":
                parts.append((part.get("text") or "").strip())
            elif part_type == "image_url":
                url = ""
                if isinstance(part.get("image_url"), dict):
                    url = part["image_url"].get("url") or ""
                parts.append(f"[IMAGE: {url or 'image'}]")
        body = "\n\n".join(item for item in parts if item)
    else:
        content_blocks = message.get("content_blocks") or []
        body = "\n\n".join(
            filter(None, (_render_block_as_text(block) for block in content_blocks))
        )

    role = str(message.get("role", "user"))
    if role in {"tool", "retrieval", "execution"}:
        prefix = f"[{ROLE_LABELS.get(role, role.upper())} RESULT]"
        return f"{prefix}\n{body}" if body else prefix
    return body


def _message_to_chat_payload(message: dict[str, Any]) -> dict[str, Any]:
    role = str(message.get("role", "user"))
    projected_role = role if role in {"system", "user", "assistant"} else "user"
    content_blocks = message.get("content_blocks") or []

    if not content_blocks:
        return {"role": projected_role, "content": _extract_message_text(message)}

    parts: list[dict[str, Any]] = []
    if role not in {"system", "user", "assistant"}:
        parts.append(
            {
                "type": "text",
                "text": f"[{ROLE_LABELS.get(role, role.upper())} RESULT]",
            }
        )

    for block in content_blocks:
        block_type = str(block.get("type", "text"))
        if block_type == "image" and block.get("uri"):
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": block["uri"]},
                }
            )
            continue

        text = _render_block_as_text(block)
        if text:
            parts.append({"type": "text", "text": text})

    if not parts:
        return {"role": projected_role, "content": ""}
    if len(parts) == 1 and parts[0]["type"] == "text":
        return {"role": projected_role, "content": parts[0]["text"]}
    return {"role": projected_role, "content": parts}


def _message_token_estimate(
    message: dict[str, Any],
    count_text_tokens: Callable[[str], int],
    image_token_estimate: int,
) -> int:
    content_blocks = message.get("content_blocks") or []
    if not content_blocks:
        return count_text_tokens(_extract_message_text(message))

    total = 0
    role = str(message.get("role", "user"))
    if role in {"tool", "retrieval", "execution"}:
        total += count_text_tokens(f"[{ROLE_LABELS.get(role, role.upper())} RESULT]")

    for block in content_blocks:
        block_type = str(block.get("type", "text"))
        if block_type == "image":
            total += int(
                block.get("asset_token_count")
                or block.get("metadata", {}).get("token_count")
                or image_token_estimate
            )
            continue
        if block.get("asset_token_count") and block.get("asset_path"):
            total += int(block["asset_token_count"])
            continue
        total += count_text_tokens(_render_block_as_text(block))
    return total


def _chat_payload_token_count(
    chat_messages: list[dict[str, Any]],
    count_text_tokens: Callable[[str], int],
) -> int:
    """Count tokens in the rendered chat payload that will actually be sent over HTTP."""
    total = 0
    for msg in chat_messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_text_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    total += count_text_tokens(part.get("text", ""))
                elif part.get("type") == "image_url":
                    total += DEFAULT_IMAGE_TOKEN_ESTIMATE
    return total


def _messages_to_completion_prompt(messages: list[dict[str, Any]]) -> str:
    prompt_parts: list[str] = []
    for message in messages:
        role = ROLE_LABELS.get(str(message.get("role", "user")), "USER")
        body = _extract_message_text(message).strip()
        prompt_parts.append(f"{role}:\n{body}" if body else f"{role}:")
    prompt_parts.append("ASSISTANT:\n")
    return "\n\n".join(prompt_parts)


def resolve_request_mode(runtime_stack_id: str, requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode
    if runtime_stack_id in CHAT_NATIVE_RUNTIMES:
        return "chat"
    if runtime_stack_id in COMPLETIONS_PREFERRED_RUNTIMES:
        return "completions"
    return "chat"


def _parse_prometheus_sample(line: str) -> tuple[str, float] | None:
    """Parse a Prometheus sample line into ``(metric_name, value)``."""
    raw_line = line.strip()
    if not raw_line or raw_line.startswith("#"):
        return None

    try:
        metric_with_labels, raw_value = raw_line.rsplit(maxsplit=1)
        metric_name = metric_with_labels.split("{", 1)[0]
        return metric_name, float(raw_value)
    except (TypeError, ValueError):
        return None


def _resolve_output_len(
    raw_output_len: Any,
    fallback_output_len: int,
    output_len_cap: Optional[int],
) -> int:
    try:
        output_len = int(raw_output_len)
    except (TypeError, ValueError):
        output_len = fallback_output_len
    if output_len <= 0:
        output_len = fallback_output_len
    if output_len_cap is not None:
        output_len = min(output_len, output_len_cap)
    return output_len


def _build_turn_from_messages(
    turn_idx: int,
    turn_id: Any,
    messages: list[dict[str, Any]],
    output_len: int,
    wait_before_s: float,
    request_mode: str,
    count_text_tokens: Callable[[str], int],
    image_token_estimate: int,
) -> ReplayTurn:
    chat_messages = [_message_to_chat_payload(message) for message in messages]
    completion_prompt = _messages_to_completion_prompt(messages)
    if request_mode == "chat":
        context_len = sum(
            _message_token_estimate(message, count_text_tokens, image_token_estimate)
            for message in messages
        )
        actual_context_len = _chat_payload_token_count(chat_messages, count_text_tokens)
    else:
        context_len = count_text_tokens(completion_prompt)
        actual_context_len = context_len  # completions mode already uses rendered text
    return ReplayTurn(
        turn_idx=turn_idx,
        turn_id=turn_id,
        output_len=output_len,
        wait_before_s=wait_before_s,
        context_len=context_len,
        actual_context_len=actual_context_len,
        chat_messages=chat_messages,
        completion_prompt=completion_prompt,
    )


def _build_session_from_multiturn_cell(
    cell: dict[str, Any],
    request_mode: str,
    count_text_tokens: Callable[[str], int],
    image_token_estimate: int,
    ignore_waits: bool,
    fallback_output_len: int,
    output_len_cap: Optional[int],
    max_turns_per_session: Optional[int],
) -> ReplaySession:
    session = cell["session"]
    turns: list[ReplayTurn] = []
    for raw_turn in session.get("turns", []):
        turns.append(
            _build_turn_from_messages(
                turn_idx=int(raw_turn.get("turn_idx", len(turns))),
                turn_id=raw_turn.get("turn_id"),
                messages=list(raw_turn.get("messages", [])),
                output_len=_resolve_output_len(
                    raw_turn.get("expected_output_tokens"),
                    fallback_output_len,
                    output_len_cap,
                ),
                wait_before_s=0.0
                if ignore_waits
                else float(raw_turn.get("wait_before_ms", 0)) / 1000.0,
                request_mode=request_mode,
                count_text_tokens=count_text_tokens,
                image_token_estimate=image_token_estimate,
            )
        )
        if max_turns_per_session is not None and len(turns) >= max_turns_per_session:
            break

    return ReplaySession(
        session_id=str(session.get("session_id", cell["trace_id"])),
        trace_id=str(cell["trace_id"]),
        runtime_stack_id=str(cell["runtime_stack_id"]),
        hardware_profile_id=str(cell["hardware_profile_id"]),
        canonical_model_id=str(cell["canonical_model_id"]),
        support_status=str(cell.get("support_status", "unknown")),
        benchmark_certification_status=str(
            cell.get("benchmark_certification_status", "unknown")
        ),
        request_mode=request_mode,
        adapter_id="inferencex_multiturn",
        turns=turns,
    )


def _build_session_from_trace_replay_cell(
    cell: dict[str, Any],
    request_mode: str,
    count_text_tokens: Callable[[str], int],
    image_token_estimate: int,
    ignore_waits: bool,
    fallback_output_len: int,
    output_len_cap: Optional[int],
    max_turns_per_session: Optional[int],
) -> ReplaySession:
    turns: list[ReplayTurn] = []
    prior_offset_ms = 0
    for index, event in enumerate(cell.get("events", [])):
        offset_ms = int(event.get("arrival_time_offset_ms", 0) or 0)
        wait_before_ms = 0 if index == 0 else max(0, offset_ms - prior_offset_ms)
        prior_offset_ms = offset_ms
        turns.append(
            _build_turn_from_messages(
                turn_idx=index,
                turn_id=event.get("turn_id"),
                messages=list(event.get("input_messages", [])),
                output_len=_resolve_output_len(
                    event.get("target_output_tokens"),
                    fallback_output_len,
                    output_len_cap,
                ),
                wait_before_s=0.0 if ignore_waits else wait_before_ms / 1000.0,
                request_mode=request_mode,
                count_text_tokens=count_text_tokens,
                image_token_estimate=image_token_estimate,
            )
        )
        if max_turns_per_session is not None and len(turns) >= max_turns_per_session:
            break

    return ReplaySession(
        session_id=str(cell.get("trace_metadata", {}).get("session_id", cell["trace_id"])),
        trace_id=str(cell["trace_id"]),
        runtime_stack_id=str(cell["runtime_stack_id"]),
        hardware_profile_id=str(cell["hardware_profile_id"]),
        canonical_model_id=str(cell["canonical_model_id"]),
        support_status=str(cell.get("support_status", "unknown")),
        benchmark_certification_status=str(
            cell.get("benchmark_certification_status", "unknown")
        ),
        request_mode=request_mode,
        adapter_id="inferencex_trace_replay",
        turns=turns,
    )


def load_replay_sessions(
    export_file: str,
    count_text_tokens: Callable[[str], int],
    runtime_stack_ids: set[str] | None = None,
    hardware_profile_ids: set[str] | None = None,
    canonical_model_ids: set[str] | None = None,
    trace_ids: set[str] | None = None,
    support_statuses: set[str] | None = None,
    request_mode: str = "auto",
    image_token_estimate: int = DEFAULT_IMAGE_TOKEN_ESTIMATE,
    ignore_waits: bool = False,
    fallback_output_len: int = DEFAULT_FALLBACK_OUTPUT_LEN,
    output_len_cap: Optional[int] = None,
    session_offset: int = 0,
    max_sessions: Optional[int] = None,
    max_turns_per_session: Optional[int] = None,
    shuffle_sessions: bool = False,
    seed: int = 0,
    allow_mixed_selection: bool = False,
) -> tuple[list[ReplaySession], dict[str, Any]]:
    payload = json.loads(Path(export_file).read_text())
    adapter_id = str(payload.get("adapter_id", "unknown"))
    export_cells = list(payload.get("exports", []))
    if adapter_id not in {"inferencex_multiturn", "inferencex_trace_replay"}:
        raise ValueError(
            f"Unsupported export adapter {adapter_id!r}. Expected "
            "'inferencex_multiturn' or 'inferencex_trace_replay'."
        )

    selected_cells = [
        cell
        for cell in export_cells
        if _matches_filter(str(cell.get("runtime_stack_id", "")), runtime_stack_ids)
        and _matches_filter(str(cell.get("hardware_profile_id", "")), hardware_profile_ids)
        and _matches_filter(str(cell.get("canonical_model_id", "")), canonical_model_ids)
        and _matches_filter(str(cell.get("trace_id", "")), trace_ids)
        and _matches_filter(str(cell.get("support_status", "")), support_statuses)
    ]
    if not selected_cells:
        raise ValueError(
            "No export cells matched the requested filters. "
            "Check runtime_stack_id / hardware_profile_id / canonical_model_id / "
            "trace_id / support_status."
        )

    if shuffle_sessions:
        random.Random(seed).shuffle(selected_cells)

    if session_offset:
        selected_cells = selected_cells[session_offset:]
    if max_sessions is not None:
        selected_cells = selected_cells[:max_sessions]
    if not selected_cells:
        raise ValueError("Selection became empty after applying session_offset/max_sessions.")

    uniqueness = {
        "runtime_stack_id": sorted({str(cell["runtime_stack_id"]) for cell in selected_cells}),
        "hardware_profile_id": sorted({str(cell["hardware_profile_id"]) for cell in selected_cells}),
        "canonical_model_id": sorted({str(cell["canonical_model_id"]) for cell in selected_cells}),
    }
    if not allow_mixed_selection:
        mixed_fields = [field for field, values in uniqueness.items() if len(values) > 1]
        if mixed_fields:
            details = ", ".join(f"{field}={uniqueness[field]}" for field in mixed_fields)
            raise ValueError(
                "Selected export cells span multiple target server identities; "
                f"filter more narrowly or pass --allow-mixed-selection. Mixed fields: {details}"
            )

    sessions: list[ReplaySession] = []
    for cell in selected_cells:
        resolved_mode = resolve_request_mode(str(cell["runtime_stack_id"]), request_mode)
        if adapter_id == "inferencex_multiturn":
            sessions.append(
                _build_session_from_multiturn_cell(
                    cell=cell,
                    request_mode=resolved_mode,
                    count_text_tokens=count_text_tokens,
                    image_token_estimate=image_token_estimate,
                    ignore_waits=ignore_waits,
                    fallback_output_len=fallback_output_len,
                    output_len_cap=output_len_cap,
                    max_turns_per_session=max_turns_per_session,
                )
            )
        else:
            sessions.append(
                _build_session_from_trace_replay_cell(
                    cell=cell,
                    request_mode=resolved_mode,
                    count_text_tokens=count_text_tokens,
                    image_token_estimate=image_token_estimate,
                    ignore_waits=ignore_waits,
                    fallback_output_len=fallback_output_len,
                    output_len_cap=output_len_cap,
                    max_turns_per_session=max_turns_per_session,
                )
            )

    selection_metadata = {
        "adapter_id": adapter_id,
        "export_file": str(export_file),
        "selected_sessions": len(sessions),
        "trace_ids": [session.trace_id for session in sessions],
        "runtime_stack_ids": sorted({session.runtime_stack_id for session in sessions}),
        "hardware_profile_ids": sorted({session.hardware_profile_id for session in sessions}),
        "canonical_model_ids": sorted({session.canonical_model_id for session in sessions}),
        "support_statuses": sorted({session.support_status for session in sessions}),
        "support_status_counts": {
            status: sum(1 for session in sessions if session.support_status == status)
            for status in sorted({session.support_status for session in sessions})
        },
        "benchmark_certification_statuses": sorted(
            {session.benchmark_certification_status for session in sessions}
        ),
        "benchmark_certification_status_counts": {
            status: sum(
                1
                for session in sessions
                if session.benchmark_certification_status == status
            )
            for status in sorted(
                {session.benchmark_certification_status for session in sessions}
            )
        },
        "request_mode_mix": {
            mode: sum(1 for session in sessions if session.request_mode == mode)
            for mode in sorted({session.request_mode for session in sessions})
        },
    }
    return sessions, selection_metadata


async def _iter_sse_lines(
    response: aiohttp.ClientResponse,
):
    """Yield individual SSE data payloads from a streaming response.

    Buffers partial lines across TCP chunks and splits multi-line chunks.
    Handles the common case where multiple ``data: {...}`` frames arrive
    in a single TCP read, or a single frame is split across reads.
    """
    buffer = b""
    async for chunk in response.content:
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            decoded = line.decode("utf-8")
            if decoded.startswith(":"):
                continue  # SSE comment / keep-alive
            if decoded.startswith("data: "):
                payload_str = decoded[6:].strip()
            elif decoded.startswith("data:"):
                payload_str = decoded[5:].strip()
            else:
                continue
            if payload_str == "[DONE]":
                return
            yield payload_str
    # Flush remaining buffer
    remaining = buffer.strip()
    if remaining:
        decoded = remaining.decode("utf-8")
        for prefix in ("data: ", "data:"):
            if decoded.startswith(prefix):
                payload_str = decoded[len(prefix):].strip()
                if payload_str and payload_str != "[DONE]":
                    yield payload_str
                break


async def _stream_chat_request(
    api_url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    context_len: int,
    count_text_tokens: Callable[[str], int],
    request_mode: str,
) -> tuple[TurnResult, int]:
    turn = TurnResult(
        turn_idx=-1,
        context_len=context_len,
        output_len=0,
        success=False,
        request_mode=request_mode,
    )
    generated_text = ""
    ttft = 0.0
    st = time.perf_counter()
    most_recent_timestamp = st

    async with aiohttp.ClientSession(trust_env=True, timeout=AIOHTTP_TIMEOUT) as session:
        async with session.post(url=api_url, json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = (await response.text()).strip()
                turn.error = f"HTTP {response.status}: {error_text or response.reason}"
                return turn, response.status

            async for sse_payload in _iter_sse_lines(response):
                data = json.loads(sse_payload)
                if choices := data.get("choices"):
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if isinstance(content, list):
                        content = "".join(
                            part.get("text", "")
                            for part in content
                            if isinstance(part, dict) and part.get("type") == "text"
                        )
                    if content:
                        timestamp = time.perf_counter()
                        if ttft == 0.0:
                            ttft = timestamp - st
                            turn.ttft = ttft
                        else:
                            turn.itl.append(timestamp - most_recent_timestamp)
                        most_recent_timestamp = timestamp
                        generated_text += content
                elif usage := data.get("usage"):
                    turn.output_len = int(usage.get("completion_tokens") or 0)

    turn.e2el = max(0.0, most_recent_timestamp - st)
    turn.success = True
    if turn.output_len == 0 and generated_text:
        turn.output_len = count_text_tokens(generated_text)
    if turn.output_len > 1:
        turn.tpot = (turn.e2el - turn.ttft) / (turn.output_len - 1)
    return turn, 200


async def _send_chat_turn(
    chat_messages: list[dict[str, Any]],
    model_id: str,
    model_name: Optional[str],
    api_url: str,
    output_len: int,
    context_len: int,
    count_text_tokens: Callable[[str], int],
    ignore_eos: bool = False,
) -> TurnResult:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}",
    }
    payload_base = {
        "model": model_name or model_id,
        "messages": chat_messages,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        payload_base["ignore_eos"] = True

    errors: list[str] = []
    for max_tokens_key in ("max_completion_tokens", "max_tokens"):
        payload = {**payload_base, max_tokens_key: output_len}
        turn, status = await _stream_chat_request(
            api_url=api_url,
            payload=payload,
            headers=headers,
            context_len=context_len,
            count_text_tokens=count_text_tokens,
            request_mode="chat",
        )
        if turn.success:
            return turn
        errors.append(turn.error)
        if status not in {400, 404, 422}:
            break

    return TurnResult(
        turn_idx=-1,
        context_len=context_len,
        output_len=0,
        success=False,
        error=" | ".join(error for error in errors if error),
        request_mode="chat",
    )


async def _send_completion_turn(
    prompt: str,
    model_id: str,
    model_name: Optional[str],
    api_url: str,
    output_len: int,
    context_len: int,
    count_text_tokens: Callable[[str], int],
    ignore_eos: bool = False,
) -> TurnResult:
    payload = {
        "model": model_name or model_id,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": output_len,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}",
    }

    turn = TurnResult(
        turn_idx=-1,
        context_len=context_len,
        output_len=0,
        success=False,
        request_mode="completions",
    )
    generated_text = ""
    ttft = 0.0
    st = time.perf_counter()
    most_recent_timestamp = st

    try:
        async with aiohttp.ClientSession(trust_env=True, timeout=AIOHTTP_TIMEOUT) as session:
            async with session.post(url=api_url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = (await response.text()).strip()
                    turn.error = f"HTTP {response.status}: {error_text or response.reason}"
                    return turn

                async for sse_payload in _iter_sse_lines(response):
                    data = json.loads(sse_payload)
                    if choices := data.get("choices"):
                        choice = choices[0]
                        content = choice.get("text")
                        if content is None:
                            delta = choice.get("delta", {})
                            content = delta.get("content")
                        if isinstance(content, list):
                            content = "".join(
                                part.get("text", "")
                                for part in content
                                if isinstance(part, dict) and part.get("type") == "text"
                            )
                        if content:
                            timestamp = time.perf_counter()
                            if ttft == 0.0:
                                ttft = timestamp - st
                                turn.ttft = ttft
                            else:
                                turn.itl.append(timestamp - most_recent_timestamp)
                            most_recent_timestamp = timestamp
                            generated_text += content
                    elif usage := data.get("usage"):
                        turn.output_len = int(usage.get("completion_tokens") or 0)
    except Exception as exc:
        turn.error = str(exc)
        return turn

    turn.e2el = max(0.0, most_recent_timestamp - st)
    turn.success = True
    if turn.output_len == 0 and generated_text:
        turn.output_len = count_text_tokens(generated_text)
    if turn.output_len > 1:
        turn.tpot = (turn.e2el - turn.ttft) / (turn.output_len - 1)
    return turn


async def poll_server_metrics(api_url: str, interval: float = 2.0) -> list[dict[str, float]]:
    """Poll ``/metrics`` periodically to capture KV / cache status."""
    import urllib.parse

    parsed = urllib.parse.urlparse(api_url)
    metrics_url = f"{parsed.scheme}://{parsed.netloc}/metrics"
    metrics_history: list[dict[str, float]] = []

    try:
        async with aiohttp.ClientSession(trust_env=True) as session:
            while True:
                try:
                    async with session.get(metrics_url, timeout=aiohttp.ClientTimeout(total=5.0)) as response:
                        if response.status == 200:
                            text = await response.text()
                            snapshot: dict[str, float] = {}
                            for line in text.split("\n"):
                                parsed_line = _parse_prometheus_sample(line)
                                if parsed_line is None:
                                    continue
                                metric_name, metric_value = parsed_line
                                if metric_name == "vllm:gpu_cache_usage_perc":
                                    snapshot["vllm_gpu_cache_usage"] = metric_value
                                elif metric_name == "vllm:cpu_cache_usage_perc":
                                    snapshot["vllm_cpu_cache_usage"] = metric_value
                                elif metric_name == "sglang:cache_hit_rate":
                                    snapshot["sglang_cache_hit_rate"] = metric_value
                                elif metric_name == "sglang:kv_cache_usage":
                                    snapshot["sglang_kv_cache_usage"] = metric_value
                                elif metric_name == "sglang:token_usage":
                                    snapshot["sglang_token_usage"] = metric_value
                                elif metric_name == "vllm:num_preemptions_total":
                                    snapshot["vllm_preemptions_total"] = metric_value
                                elif metric_name == "vllm:num_requests_running":
                                    snapshot["vllm_requests_running"] = metric_value
                                elif metric_name == "vllm:num_requests_waiting":
                                    snapshot["vllm_requests_waiting"] = metric_value
                            if snapshot:
                                metrics_history.append(snapshot)
                except Exception:
                    pass
                await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass

    return metrics_history


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, percentile))


def calculate_multiturn_metrics(
    session_results: list[SessionResult],
    max_turns: int,
    selected_percentiles: list[float],
) -> dict[str, Any]:
    ms = 1000.0
    per_turn: dict[str, dict[str, Any]] = {}

    for turn_index in range(max_turns):
        ttfts: list[float] = []
        tpots: list[float] = []
        e2els: list[float] = []
        context_lens: list[int] = []
        actual_context_lens: list[int] = []
        output_lens: list[int] = []
        successes = 0
        for session in session_results:
            if turn_index < len(session.turns):
                turn = session.turns[turn_index]
                if turn.success:
                    ttfts.append(turn.ttft)
                    tpots.append(turn.tpot)
                    e2els.append(turn.e2el)
                    context_lens.append(turn.context_len)
                    actual_context_lens.append(turn.actual_context_len)
                    output_lens.append(turn.output_len)
                    successes += 1

        key = f"turn_{turn_index + 1}"
        metrics: dict[str, Any] = {
            "completed": successes,
            "mean_context_len": float(np.mean(context_lens)) if context_lens else 0.0,
            "mean_actual_context_len": float(np.mean(actual_context_lens)) if actual_context_lens else 0.0,
            "mean_output_len": float(np.mean(output_lens)) if output_lens else 0.0,
        }
        for label, values in (("ttft", ttfts), ("tpot", tpots), ("e2el", e2els)):
            metrics[f"mean_{label}_ms"] = float(np.mean(values)) * ms if values else 0.0
            metrics[f"median_{label}_ms"] = float(np.median(values)) * ms if values else 0.0
            metrics[f"std_{label}_ms"] = float(np.std(values)) * ms if values else 0.0
            for percentile in selected_percentiles:
                percentile_label = str(int(percentile)) if int(percentile) == percentile else str(percentile)
                metrics[f"p{percentile_label}_{label}_ms"] = _percentile(values, percentile) * ms
        per_turn[key] = metrics

    all_ttfts: list[float] = []
    all_tpots: list[float] = []
    all_e2els: list[float] = []
    total_input = 0
    total_actual_input = 0
    total_output = 0
    completed_sessions = 0
    total_wall = 0.0
    max_actual_context_per_turn = 0

    for session in session_results:
        if session.turns and all(turn.success for turn in session.turns):
            completed_sessions += 1
        total_input += session.total_input_tokens
        total_actual_input += session.total_actual_input_tokens
        total_output += session.total_output_tokens
        total_wall = max(total_wall, session.total_duration)
        for turn in session.turns:
            if turn.success:
                all_ttfts.append(turn.ttft)
                all_tpots.append(turn.tpot)
                all_e2els.append(turn.e2el)
                if turn.actual_context_len > max_actual_context_per_turn:
                    max_actual_context_per_turn = turn.actual_context_len

    aggregate: dict[str, Any] = {
        "completed_sessions": completed_sessions,
        "total_sessions": len(session_results),
        "total_input_tokens": total_input,
        "total_actual_input_tokens": total_actual_input,
        "max_actual_context_len_per_turn": max_actual_context_per_turn,
        "total_output_tokens": total_output,
        "total_wall_time_s": total_wall,
        "session_throughput_sps": completed_sessions / total_wall if total_wall > 0 else 0.0,
        "output_throughput_tps": total_output / total_wall if total_wall > 0 else 0.0,
        "total_token_throughput_tps": (total_input + total_output) / total_wall if total_wall > 0 else 0.0,
    }
    for label, values in (("ttft", all_ttfts), ("tpot", all_tpots), ("e2el", all_e2els)):
        aggregate[f"mean_{label}_ms"] = float(np.mean(values)) * ms if values else 0.0
        aggregate[f"median_{label}_ms"] = float(np.median(values)) * ms if values else 0.0
        aggregate[f"std_{label}_ms"] = float(np.std(values)) * ms if values else 0.0
        for percentile in selected_percentiles:
            percentile_label = str(int(percentile)) if int(percentile) == percentile else str(percentile)
            aggregate[f"p{percentile_label}_{label}_ms"] = _percentile(values, percentile) * ms

    return {"per_turn_metrics": per_turn, "aggregate_metrics": aggregate}


async def _run_replay_session(
    session: ReplaySession,
    model_id: str,
    model_name: Optional[str],
    chat_api_url: str,
    completion_api_url: str,
    count_text_tokens: Callable[[str], int],
    pbar: Optional[tqdm],
    ignore_eos: bool,
) -> SessionResult:
    result = SessionResult(session_id=session.session_id)
    start = time.perf_counter()

    for replay_turn in session.turns:
        if replay_turn.wait_before_s > 0:
            await asyncio.sleep(replay_turn.wait_before_s)

        if session.request_mode == "chat":
            turn_result = await _send_chat_turn(
                chat_messages=replay_turn.chat_messages,
                model_id=model_id,
                model_name=model_name,
                api_url=chat_api_url,
                output_len=replay_turn.output_len,
                context_len=replay_turn.context_len,
                count_text_tokens=count_text_tokens,
                ignore_eos=ignore_eos,
            )
        else:
            turn_result = await _send_completion_turn(
                prompt=replay_turn.completion_prompt,
                model_id=model_id,
                model_name=model_name,
                api_url=completion_api_url,
                output_len=replay_turn.output_len,
                context_len=replay_turn.context_len,
                count_text_tokens=count_text_tokens,
                ignore_eos=ignore_eos,
            )

        turn_result.turn_idx = replay_turn.turn_idx
        turn_result.actual_context_len = replay_turn.actual_context_len
        result.turns.append(turn_result)
        if turn_result.success:
            result.total_input_tokens += turn_result.context_len
            result.total_actual_input_tokens += turn_result.actual_context_len
            result.total_output_tokens += turn_result.output_len
        if pbar is not None:
            pbar.update(1)

    result.total_duration = time.perf_counter() - start
    return result


async def _run_warmup_sessions(
    sessions: list[ReplaySession],
    model_id: str,
    model_name: Optional[str],
    chat_api_url: str,
    completion_api_url: str,
    count_text_tokens: Callable[[str], int],
    num_warmup_sessions: int,
    ignore_eos: bool,
) -> None:
    if num_warmup_sessions <= 0 or not sessions:
        return

    print(f"Running {num_warmup_sessions} warmup session(s) (results discarded) ...")
    warmup_jobs: list[asyncio.Task[SessionResult]] = []
    for index in range(num_warmup_sessions):
        source = sessions[index % len(sessions)]
        warmup_turns = [
            ReplayTurn(
                turn_idx=turn.turn_idx,
                turn_id=turn.turn_id,
                output_len=turn.output_len,
                wait_before_s=0.0,
                context_len=turn.context_len,
                actual_context_len=turn.actual_context_len,
                chat_messages=turn.chat_messages,
                completion_prompt=turn.completion_prompt,
            )
            for turn in source.turns[: min(2, len(source.turns))]
        ]
        warmup_jobs.append(
            asyncio.create_task(
                _run_replay_session(
                    session=ReplaySession(
                        session_id=f"warmup-{index}",
                        trace_id=source.trace_id,
                        runtime_stack_id=source.runtime_stack_id,
                        hardware_profile_id=source.hardware_profile_id,
                        canonical_model_id=source.canonical_model_id,
                        support_status=source.support_status,
                        benchmark_certification_status=source.benchmark_certification_status,
                        request_mode=source.request_mode,
                        adapter_id=source.adapter_id,
                        turns=warmup_turns,
                    ),
                    model_id=model_id,
                    model_name=model_name,
                    chat_api_url=chat_api_url,
                    completion_api_url=completion_api_url,
                    count_text_tokens=count_text_tokens,
                    pbar=None,
                    ignore_eos=ignore_eos,
                )
            )
        )

    results = await asyncio.gather(*warmup_jobs, return_exceptions=True)
    succeeded = sum(
        1
        for result in results
        if isinstance(result, SessionResult) and any(turn.success for turn in result.turns)
    )
    failed = num_warmup_sessions - succeeded
    if failed:
        print(
            f"  ⚠️  {failed}/{num_warmup_sessions} warmup session(s) failed. "
            "Check the server endpoint and selected export cell."
        )
    else:
        print(f"  ✅ {succeeded} warmup session(s) completed successfully.")
    print()


async def run_export_replay_benchmark(
    sessions: list[ReplaySession],
    selection_metadata: dict[str, Any],
    model_id: str,
    model_name: Optional[str],
    chat_api_url: str,
    completion_api_url: str,
    count_text_tokens: Callable[[str], int],
    max_concurrency: int,
    selected_percentiles: list[float],
    disable_tqdm: bool,
    num_warmup_sessions: int = 1,
    ignore_eos: bool = False,
) -> dict[str, Any]:
    if not sessions:
        raise ValueError("No replay sessions were selected.")

    max_turns = max(len(session.turns) for session in sessions)
    total_turns = sum(len(session.turns) for session in sessions)

    print("============================================================")
    print(" Export Replay Selection")
    print("============================================================")
    print(f"  Adapter:               {selection_metadata['adapter_id']}")
    print(f"  Sessions selected:     {selection_metadata['selected_sessions']}")
    print(f"  Runtime stack(s):      {', '.join(selection_metadata['runtime_stack_ids'])}")
    print(f"  Hardware profile(s):   {', '.join(selection_metadata['hardware_profile_ids'])}")
    print(f"  Canonical model(s):    {', '.join(selection_metadata['canonical_model_ids'])}")
    print(
        "  Support status(es):    "
        f"{', '.join(selection_metadata['support_statuses'])}"
    )
    print(
        "  Certification status:  "
        f"{', '.join(selection_metadata['benchmark_certification_statuses'])}"
    )
    print(f"  Request mode mix:      {selection_metadata['request_mode_mix']}")
    print(f"  Total turns:           {total_turns}")
    print("============================================================")
    print()

    await _run_warmup_sessions(
        sessions=sessions,
        model_id=model_id,
        model_name=model_name,
        chat_api_url=chat_api_url,
        completion_api_url=completion_api_url,
        count_text_tokens=count_text_tokens,
        num_warmup_sessions=num_warmup_sessions,
        ignore_eos=ignore_eos,
    )

    pbar = None if disable_tqdm else tqdm(total=total_turns, desc="turns")
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _limited_run(session: ReplaySession) -> SessionResult:
        async with semaphore:
            return await _run_replay_session(
                session=session,
                model_id=model_id,
                model_name=model_name,
                chat_api_url=chat_api_url,
                completion_api_url=completion_api_url,
                count_text_tokens=count_text_tokens,
                pbar=pbar,
                ignore_eos=ignore_eos,
            )

    print(
        f"Starting export replay benchmark: {len(sessions)} sessions, "
        f"max_turns={max_turns}, max_concurrency={max_concurrency}"
    )
    benchmark_start = time.perf_counter()
    metrics_task = asyncio.create_task(poll_server_metrics(chat_api_url, interval=2.0))
    jobs = [asyncio.create_task(_limited_run(session)) for session in sessions]
    session_results = await asyncio.gather(*jobs)
    benchmark_duration = time.perf_counter() - benchmark_start

    metrics_task.cancel()
    try:
        server_metrics = await metrics_task
    except asyncio.CancelledError:
        server_metrics = []

    if pbar is not None:
        pbar.close()

    metrics = calculate_multiturn_metrics(
        session_results=session_results,
        max_turns=max_turns,
        selected_percentiles=selected_percentiles,
    )
    aggregate = metrics["aggregate_metrics"]
    per_turn = metrics["per_turn_metrics"]

    cache_usage_avg = 0.0
    cache_hit_rate_avg = 0.0
    gpu_cache_usage_avg = 0.0
    gpu_cache_usage_peak = 0.0
    cpu_cache_usage_avg = 0.0
    cpu_cache_usage_peak = 0.0
    gpu_cache_metric_name: str | None = None
    cpu_cache_metric_name: str | None = None
    observability_status = "no_cache_metrics"
    cpu_samples: list[float] = []
    kv_offload_observed = False
    if server_metrics:
        vllm_gpu_samples = [
            item["vllm_gpu_cache_usage"]
            for item in server_metrics
            if "vllm_gpu_cache_usage" in item
        ]
        sglang_gpu_samples: list[float] = []
        saw_sglang_kv_metric = False
        saw_sglang_token_metric = False
        for item in server_metrics:
            if "sglang_kv_cache_usage" in item:
                sglang_gpu_samples.append(item["sglang_kv_cache_usage"])
                saw_sglang_kv_metric = True
            elif "sglang_token_usage" in item:
                sglang_gpu_samples.append(item["sglang_token_usage"])
                saw_sglang_token_metric = True

        if saw_sglang_kv_metric:
            gpu_cache_metric_name = "sglang:kv_cache_usage"
        elif saw_sglang_token_metric:
            gpu_cache_metric_name = "sglang:token_usage"

        if vllm_gpu_samples:
            gpu_samples = vllm_gpu_samples
            gpu_cache_metric_name = "vllm:gpu_cache_usage_perc"
        else:
            gpu_samples = sglang_gpu_samples

        cpu_samples = [
            item["vllm_cpu_cache_usage"]
            for item in server_metrics
            if "vllm_cpu_cache_usage" in item
        ]
        if cpu_samples:
            cpu_cache_metric_name = "vllm:cpu_cache_usage_perc"
        cache_hit_samples = [
            item["sglang_cache_hit_rate"]
            for item in server_metrics
            if "sglang_cache_hit_rate" in item
        ]

        if gpu_samples:
            gpu_cache_usage_avg = float(np.mean(gpu_samples))
            gpu_cache_usage_peak = float(np.max(gpu_samples))
            cache_usage_avg = gpu_cache_usage_avg
        if cpu_samples:
            cpu_cache_usage_avg = float(np.mean(cpu_samples))
            cpu_cache_usage_peak = float(np.max(cpu_samples))
            kv_offload_observed = any(sample > 0.0 for sample in cpu_samples)
        if cache_hit_samples:
            cache_hit_rate_avg = float(np.mean(cache_hit_samples))
        if cpu_samples:
            observability_status = "direct_cpu_cache_metric"
        elif gpu_samples or cache_hit_samples:
            observability_status = "indirect_without_cpu_cache_metric"

    print()
    print("{s:{c}^{n}}".format(s=" Export Replay Benchmark Result ", n=60, c="="))
    print(f"  {'Completed sessions:':<35} {aggregate['completed_sessions']}/{aggregate['total_sessions']}")
    print(f"  {'Benchmark duration (s):':<35} {benchmark_duration:.2f}")
    print(f"  {'Total input tokens (estimated):':<35} {aggregate['total_input_tokens']}")
    print(f"  {'Total input tokens (actual sent):':<35} {aggregate['total_actual_input_tokens']}")
    print(f"  {'Max actual context/turn:':<35} {aggregate['max_actual_context_len_per_turn']}")
    print(f"  {'Total output tokens:':<35} {aggregate['total_output_tokens']}")
    print(f"  {'Session throughput (sessions/s):':<35} {aggregate['session_throughput_sps']:.2f}")
    print(f"  {'Output throughput (tok/s):':<35} {aggregate['output_throughput_tps']:.2f}")
    print(f"  {'Total throughput (tok/s):':<35} {aggregate['total_token_throughput_tps']:.2f}")
    if server_metrics:
        print()
        print(f"  {'Server KV Cache Usage (avg):':<35} {cache_usage_avg:.1%}")
        if cpu_cache_metric_name:
            print(f"  {'Server CPU Cache Usage (avg):':<35} {cpu_cache_usage_avg:.1%}")
        if cache_hit_rate_avg > 0:
            print(f"  {'Prefix Cache Hit Rate (avg):':<35} {cache_hit_rate_avg:.1%}")
        if observability_status == "indirect_without_cpu_cache_metric":
            print(
                f"  {'Offload observability:':<35} "
                "indirect only (no direct CPU cache metric)"
            )
    print()
    print("{s:{c}^{n}}".format(s=" Per-Turn TTFT Progression ", n=60, c="-"))
    print(f"  {'Turn':<8} {'Est Ctx':<10} {'Act Ctx':<10} {'Mean TTFT':<14} {'P99 TTFT':<14} {'Mean E2EL':<14}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*14} {'─'*14} {'─'*14}")
    for turn_index in range(max_turns):
        key = f"turn_{turn_index + 1}"
        if key not in per_turn:
            continue
        turn_metrics = per_turn[key]
        print(
            f"  {turn_index + 1:<8} "
            f"{turn_metrics['mean_context_len']:<10.0f} "
            f"{turn_metrics.get('mean_actual_context_len', 0.0):<10.0f} "
            f"{turn_metrics['mean_ttft_ms']:<14.1f} "
            f"{turn_metrics.get('p99_ttft_ms', 0.0):<14.1f} "
            f"{turn_metrics['mean_e2el_ms']:<14.1f}"
        )
    print("=" * 60)

    return {
        "mode": "export_replay",
        "adapter_id": selection_metadata["adapter_id"],
        "selection": selection_metadata,
        "duration": benchmark_duration,
        "num_sessions": len(sessions),
        "max_turns": max_turns,
        "max_concurrency": max_concurrency,
        "num_warmup_sessions": num_warmup_sessions,
        "server_metrics_summary": {
            "cache_usage_avg": cache_usage_avg,
            "cache_hit_rate_avg": cache_hit_rate_avg,
            "gpu_cache_usage_avg": gpu_cache_usage_avg,
            "gpu_cache_usage_peak": gpu_cache_usage_peak,
            "gpu_cache_metric_name": gpu_cache_metric_name,
            "cpu_cache_usage_avg": cpu_cache_usage_avg,
            "cpu_cache_usage_peak": cpu_cache_usage_peak,
            "cpu_cache_metric_name": cpu_cache_metric_name,
            "cpu_cache_metric_available": bool(cpu_samples),
            "observability_status": observability_status,
            # Observability-only signal; not a certification or quality claim.
            "kv_offload_observed": kv_offload_observed,
            "samples": len(server_metrics),
            "preemption_count": int(
                max(
                    (item.get("vllm_preemptions_total", 0.0) for item in server_metrics),
                    default=0.0,
                )
            ) if server_metrics else 0,
            "peak_requests_running": float(
                max(
                    (item.get("vllm_requests_running", 0.0) for item in server_metrics),
                    default=0.0,
                )
            ) if server_metrics else 0.0,
            "peak_requests_waiting": float(
                max(
                    (item.get("vllm_requests_waiting", 0.0) for item in server_metrics),
                    default=0.0,
                )
            ) if server_metrics else 0.0,
        },
        "depth_telemetry": {
            "total_estimated_input_tokens": aggregate["total_input_tokens"],
            "total_actual_input_tokens": aggregate["total_actual_input_tokens"],
            "max_actual_context_len_per_turn": aggregate["max_actual_context_len_per_turn"],
        },
        **metrics,
    }


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)

    base_url = args.base_url or f"http://{args.host}:{args.port}"
    base_url = base_url.rstrip("/")
    chat_api_url = args.chat_api_url or f"{base_url}{args.chat_endpoint}"
    completion_api_url = args.completion_api_url or f"{base_url}{args.completion_endpoint}"

    tokenizer_id = None if args.skip_tokenizer_load else (args.tokenizer or args.model)
    count_text_tokens = build_text_token_counter(
        tokenizer_id=tokenizer_id,
        tokenizer_mode=args.tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
    )
    sessions, selection_metadata = load_replay_sessions(
        export_file=args.export_file,
        count_text_tokens=count_text_tokens,
        runtime_stack_ids=_csv_values(args.runtime_stack_id),
        hardware_profile_ids=_csv_values(args.hardware_profile_id),
        canonical_model_ids=_csv_values(args.canonical_model_id),
        trace_ids=_csv_values(args.trace_id),
        support_statuses=_csv_values(args.support_status),
        request_mode=args.request_mode,
        image_token_estimate=args.image_token_estimate,
        ignore_waits=args.ignore_waits,
        fallback_output_len=args.fallback_output_len,
        output_len_cap=args.max_output_len,
        session_offset=args.session_offset,
        max_sessions=args.max_sessions,
        max_turns_per_session=args.max_turns_per_session,
        shuffle_sessions=args.shuffle_sessions,
        seed=args.seed,
        allow_mixed_selection=args.allow_mixed_selection,
    )

    result = asyncio.run(
        run_export_replay_benchmark(
            sessions=sessions,
            selection_metadata=selection_metadata,
            model_id=args.model,
            model_name=args.served_model_name,
            chat_api_url=chat_api_url,
            completion_api_url=completion_api_url,
            count_text_tokens=count_text_tokens,
            max_concurrency=args.max_concurrency,
            selected_percentiles=[float(item) for item in args.metric_percentiles.split(",")],
            disable_tqdm=args.disable_tqdm,
            num_warmup_sessions=args.num_warmup_sessions,
            ignore_eos=args.ignore_eos,
        )
    )

    if args.save_result:
        result_json: dict[str, Any] = {
            "date": datetime.now().strftime("%Y%m%d-%H%M%S"),
            "model_id": args.model,
        }
        if tokenizer_id is not None:
            result_json["tokenizer_id"] = tokenizer_id
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    key, value = item.split("=", 1)
                    result_json[key.strip()] = value.strip()
        result_json = {**result_json, **result}

        file_name = args.result_filename or f"export-replay-{Path(args.export_file).stem}.json"
        if args.result_dir:
            os.makedirs(args.result_dir, exist_ok=True)
            file_name = os.path.join(args.result_dir, file_name)

        with open(file_name, "w", encoding="utf-8") as handle:
            json.dump(result_json, handle, indent=2)
        print(f"\nResults saved to {file_name}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description=(
            "Replay ISB1 export sessions against an OpenAI-compatible server. "
            "Supports chat-completions replay for standalone vLLM/SGLang and "
            "prompt-projected completions replay for TRT / Dynamo-style cells."
        )
    )

    parser.add_argument("--export-file", type=str, required=True,
                        help="Path to an inferencex_multiturn or inferencex_trace_replay export JSON")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Server base URL, e.g. http://0.0.0.0:8000")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--chat-endpoint", type=str, default="/v1/chat/completions")
    parser.add_argument("--completion-endpoint", type=str, default="/v1/completions")
    parser.add_argument("--chat-api-url", type=str, default=None,
                        help="Override the full chat endpoint URL")
    parser.add_argument("--completion-api-url", type=str, default=None,
                        help="Override the full completions endpoint URL")

    parser.add_argument("--model", type=str, required=True,
                        help="Model identifier sent to the target server")
    parser.add_argument("--served-model-name", type=str, default=None,
                        help="Served model name if different from --model")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer name/path if different from --model")
    parser.add_argument("--tokenizer-mode", type=str, default="auto",
                        choices=["auto", "slow", "mistral", "custom"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-tokenizer-load", action="store_true",
                        help="Use approximate token counting instead of loading a tokenizer")

    parser.add_argument("--runtime-stack-id", type=str, default=None,
                        help="Comma-separated runtime_stack_id filter(s)")
    parser.add_argument("--hardware-profile-id", type=str, default=None,
                        help="Comma-separated hardware_profile_id filter(s)")
    parser.add_argument("--canonical-model-id", type=str, default=None,
                        help="Comma-separated canonical_model_id filter(s)")
    parser.add_argument("--trace-id", type=str, default=None,
                        help="Comma-separated trace_id filter(s)")
    parser.add_argument("--support-status", type=str, default=None,
                        help="Comma-separated support_status filter(s)")
    parser.add_argument("--request-mode", type=str, default="auto",
                        choices=["auto", "chat", "completions"])
    parser.add_argument("--allow-mixed-selection", action="store_true",
                        help="Allow multiple runtime/model/hardware identities in one run")
    parser.add_argument("--shuffle-sessions", action="store_true")
    parser.add_argument("--session-offset", type=int, default=0)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--max-turns-per-session", type=int, default=None)
    parser.add_argument("--ignore-waits", action="store_true",
                        help="Ignore export wait_before/arrival-time gaps")
    parser.add_argument("--fallback-output-len", type=int, default=DEFAULT_FALLBACK_OUTPUT_LEN,
                        help="Fallback output length when export metadata is missing")
    parser.add_argument("--max-output-len", type=int, default=None,
                        help="Optional cap applied to each exported target output length")
    parser.add_argument("--image-token-estimate", type=int, default=DEFAULT_IMAGE_TOKEN_ESTIMATE,
                        help="Approximate token cost for image blocks when no explicit token count exists")

    parser.add_argument("--max-concurrency", type=int, default=8,
                        help="Maximum concurrently active replay sessions")
    parser.add_argument("--num-warmup-sessions", type=int, default=1,
                        help="Warmup sessions to prime KV/prefix cache before measurement")
    parser.add_argument("--ignore-eos", action="store_true")

    parser.add_argument("--save-result", action="store_true")
    parser.add_argument("--result-dir", type=str, default=None)
    parser.add_argument("--result-filename", type=str, default=None)
    parser.add_argument("--metadata", metavar="KEY=VALUE", nargs="*")
    parser.add_argument("--metric-percentiles", type=str, default="90,99,99.9")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-tqdm", action="store_true")

    main(parser.parse_args())
