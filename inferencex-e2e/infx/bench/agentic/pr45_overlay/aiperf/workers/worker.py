# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import time
import uuid
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.constants import BYTES_PER_MIB, NANOS_PER_SECOND
from aiperf.common.enums import (
    CacheBustTarget,
    CommAddress,
    CommandType,
    ConversationBranchMode,
    CreditPhase,
    MemoryMapFormat,
    MessageType,
)
from aiperf.common.environment import Environment
from aiperf.common.event_loop_monitor import EventLoopMonitor
from aiperf.common.exceptions import NotInitializedError
from aiperf.common.hooks import (
    background_task,
    on_command,
    on_message,
    on_start,
    on_stop,
)
from aiperf.common.messages import (
    CommandMessage,
    DatasetConfiguredNotification,
    ErrorMessage,
    InferenceResultsMessage,
    WorkerHealthMessage,
)
from aiperf.common.messages.dataset_messages import (
    ConversationRequestMessage,
    ConversationResponseMessage,
)
from aiperf.common.mixins import ProcessHealthMixin
from aiperf.common.models import (
    Conversation,
    ErrorDetails,
    MemoryMapClientMetadata,
    ModelEndpointInfo,
    ParsedResponse,
    ProcessHealth,
    RecordContext,
    RequestInfo,
    RequestRecord,
    SSEMessage,
    Text,
    Turn,
    WorkerTaskStats,
)
from aiperf.common.models.record_models import find_last_non_empty_usage
from aiperf.common.protocols import (
    PushClientProtocol,
    RequestClientProtocol,
    StreamingDealerClientProtocol,
    StreamingPushClientProtocol,
)
from aiperf.common.scenario.context_overflow import is_context_overflow_response
from aiperf.common.scenario.registry import get_scenario
from aiperf.config.adaptive_scale_phase import (
    sla_filters_require_first_token_observation,
)
from aiperf.credit.messages import (
    CancelCredits,
    CreditReturn,
    FirstToken,
    RouterToWorkerMessage,
    StreamingContent,
    WorkerReady,
    WorkerShutdown,
)
from aiperf.credit.structs import Credit, CreditContext
from aiperf.dataset.memory_map_utils import (
    apply_max_tokens_to_wire_payload,
    turn_from_payload_turn,
)
from aiperf.dataset.protocols import DatasetClientStoreProtocol
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from aiperf.records.payload_retention import resolve_strip_record_payload_bytes
from aiperf.workers.inference_client import InferenceClient
from aiperf.workers.session_manager import UserSession, UserSessionManager

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun
    from aiperf.transports.base_transports import FirstTokenCallback


_CONVERSATION_CACHE_MIN_ENTRIES = Environment.WORKER.CONVERSATION_CACHE_MIN_ENTRIES
"""Floor on decoded conversations retained per worker (LRU), env-tunable.

The live bound is ``max(floor, live sessions on this worker)``. A cached entry
whose trace some live session is replaying costs almost nothing, because that
session's clone already shares its payload; only idle entries cost memory. So
tying the bound to the live-session count keeps idle memory at most one
conversation per live session, while a worker running more lanes than the
corpus has traces caches the whole corpus and decodes each trace once. A fixed
small bound instead misses on almost every session at 50k lanes (~500 lanes per
worker over 393 traces), and each miss pays decode plus dedup."""


def _phase_needs_first_token_callback(phase) -> bool:
    if phase.prefill_concurrency is not None:
        return True
    return bool(
        getattr(phase, "adaptive_scale", False)
        and sla_filters_require_first_token_observation(phase.sla)
    )


_logger = AIPerfLogger(__name__)


def _error_message(error: str | ErrorDetails | None) -> str | None:
    """Extract the human error text for overflow substring matching.

    ``credit_context.error`` usually holds an ``ErrorDetails``; read its
    ``.message`` rather than the model's ``__str__`` repr (which embeds
    ``code=``/``type=``/``cause=`` noise and would silently break substring
    matching if the repr ever changed). Used only for classification -- the
    wire ``CreditReturn.error`` still carries the full ``str(ErrorDetails)`` so
    no diagnostic detail (code/type/cause) is dropped downstream.
    """
    if error is None:
        return None
    return error.message if isinstance(error, ErrorDetails) else str(error)


def _is_terminal_context_overflow(
    credit: Credit, credit_context: CreditContext
) -> bool:
    """Whether this credit return ends the conversation via context overflow.

    A context-overflow error on a non-final, non-cancelled turn is terminal:
    agentic_replay recycles the lane and issues no further (final/cancel) credit
    for the session, so the worker must treat it as a terminal disposition.
    Mirrors the strategy's own ``terminal_overflow`` classification, applied to
    the same error body the worker returns on the credit.
    """
    if credit.is_final_turn or credit_context.cancelled:
        return False
    body = _error_message(credit_context.error)
    return body is not None and is_context_overflow_response(body=body)


def _apply_cache_bust_to_system_message(
    system_message: str | None, marker: str, target: CacheBustTarget
) -> str | None:
    """Apply marker to the structured system_message string.

    Returns the modified string, or `None` if the input was None — the caller
    is then expected to fall back to mutating raw_messages.
    """
    if not marker or target == CacheBustTarget.NONE or system_message is None:
        return system_message
    if target in (
        CacheBustTarget.SYSTEM_PREFIX,
        CacheBustTarget.WARMUP_ISOLATION_SYSTEM,
    ):
        return marker + system_message
    if target == CacheBustTarget.SYSTEM_SUFFIX:
        return system_message + marker
    return system_message


def _content_has_marker_at_edge(
    content: object, marker: str, *, is_prefix: bool
) -> bool:
    """Whether ``content`` already carries ``marker`` at the prefix/suffix edge.

    The injection helpers run once per credit and several paths mutate a turn
    object shared across the session's turns (delta-mode ``turn_list[0]``, and
    the unconditional every-credit first-user mark used for seeded resumes), so
    re-injecting the constant per-session marker must not stack it. This check
    is exact (the per-session marker is constant; a fresh recycled play sees
    pristine content and injects its own marker). Handles plain-string and
    OpenAI multimodal list-of-parts content.
    """
    if isinstance(content, str):
        return content.startswith(marker) if is_prefix else content.endswith(marker)
    if isinstance(content, list) and content:
        marker_part = {"type": "text", "text": marker.strip()}
        return content[0 if is_prefix else -1] == marker_part
    return False


def _inject_marker_into_raw_messages(
    raw_messages: list[dict], marker: str, *, is_prefix: bool
) -> None:
    """Mutate the first system-role message's content in-place.

    No-op when raw_messages is empty or the first message is not a system role.
    For multimodal content (``content`` is a list of parts), the marker is
    inserted as a new ``{"type": "text", "text": marker}`` part at the start
    (prefix) or end (suffix) of the parts list. Idempotent via
    :func:`_content_has_marker_at_edge`.
    """
    if not raw_messages or not marker:
        return
    first = raw_messages[0]
    if not isinstance(first, dict) or first.get("role") != "system":
        return
    content = first.get("content", "")
    if _content_has_marker_at_edge(content, marker, is_prefix=is_prefix):
        return
    if isinstance(content, str):
        raw_messages[0] = {
            **first,
            "content": (marker + content) if is_prefix else (content + marker),
        }
        return
    if isinstance(content, list):
        marker_part = {"type": "text", "text": marker.strip()}
        new_content = [marker_part, *content] if is_prefix else [*content, marker_part]
        raw_messages[0] = {**first, "content": new_content}
        return
    _logger.warning(
        f"cache-bust: cannot inject marker into raw_messages[0].content of "
        f"type {type(content).__name__}; marker dropped"
    )


def _inject_marker_into_first_user_turn(
    raw_messages: list[dict], marker: str, *, is_prefix: bool
) -> None:
    """Mutate the first user-role message's content in-place.

    No-op when raw_messages is empty. For multimodal content (``content`` is
    a list of parts), the marker is inserted as a new
    ``{"type": "text", "text": marker}`` part at the start (prefix) or end
    (suffix) of the parts list. Idempotent via
    :func:`_content_has_marker_at_edge` — FIRST_TURN_* injection runs every
    credit (to mark seeded turn 0 on mid-trajectory resumes), so repeated calls
    on the same shared turn must not stack the marker.
    """
    if not raw_messages or not marker:
        return
    for idx, msg in enumerate(raw_messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if _content_has_marker_at_edge(content, marker, is_prefix=is_prefix):
                return
            if isinstance(content, str):
                raw_messages[idx] = {
                    **msg,
                    "content": (marker + content) if is_prefix else (content + marker),
                }
                return
            if isinstance(content, list):
                marker_part = {"type": "text", "text": marker.strip()}
                new_content = (
                    [marker_part, *content] if is_prefix else [*content, marker_part]
                )
                raw_messages[idx] = {**msg, "content": new_content}
                return
            _logger.warning(
                f"cache-bust: cannot inject marker into first user-turn content "
                f"of type {type(content).__name__}; marker dropped"
            )
            return


def _find_effective_raw_system(turn_list: list[Turn]) -> list[dict] | None:
    """Return the ``raw_system`` block list the endpoint will actually send.

    Mirrors ``base_endpoint._latest_turn_attr``: walks ``turn_list`` from the
    end and returns the first non-None ``raw_system``. Only ``MessagesEndpoint``
    reads this field, and there it outranks any system-role ``raw_messages``
    entry — so a ``SYSTEM_*`` marker written to ``raw_messages`` would be
    silently dropped from the wire whenever a turn carries ``raw_system``.

    It does NOT outrank a conversation-level ``system_message``: that is
    prepended as ``system`` block 0, ahead of these blocks. Callers marking
    the front of the system section must therefore prefer ``system_message``
    when both are present; callers marking the tail must prefer these blocks.
    """
    for turn in reversed(turn_list):
        if turn.raw_system is not None:
            return turn.raw_system
    return None


def _inject_marker_into_raw_system(
    raw_system: list[dict], marker: str, *, is_prefix: bool
) -> None:
    """Insert the marker as a text block at the edge of a ``raw_system`` list.

    ``raw_system`` is a list of vendor content blocks (not role/content
    messages), so the marker becomes its own ``{"type": "text", ...}`` block
    rather than being spliced into an existing block's text — that leaves any
    neighbouring block's ``cache_control`` breakpoint attached to the same
    bytes it was authored against. Idempotent via
    :func:`_content_has_marker_at_edge`.
    """
    if not marker:
        return
    if _content_has_marker_at_edge(raw_system, marker, is_prefix=is_prefix):
        return
    marker_block = {"type": "text", "text": marker.strip()}
    if is_prefix:
        raw_system.insert(0, marker_block)
    else:
        raw_system.append(marker_block)


def _find_first_system_message(turn_list: list[Turn]) -> list[dict] | None:
    """Return the raw_messages list whose first dict has ``role == "system"``, or None.

    Walks ``turn_list`` forward and returns the first ``raw_messages`` whose
    leading dict is a system-role message. Used by cache-bust system-target
    injection so it works for both single-turn message-array mode (system
    lives in ``turn_list[-1]``, which is also ``turn_list[0]``) and
    accumulating delta mode (system in ``turn_list[0]``, deltas in
    ``turn_list[1..]``).
    """
    for turn in turn_list:
        raw = turn.raw_messages
        if raw and isinstance(raw[0], dict) and raw[0].get("role") == "system":
            return raw
    return None


def _find_first_user_turn(turn_list: list[Turn]) -> Turn | None:
    """Return the first turn whose payload carries the conversation's initial
    user message, or None.

    Walks ``turn_list`` forward. A turn qualifies when it has any
    ``raw_messages`` entry with ``role == "user"``, or when ``texts`` is
    non-empty (synthetic-Turn path). If no turn matches but at least one turn
    has neither ``raw_messages`` nor ``texts`` (truly empty synthetic Turn,
    e.g. before any prompt has been generated), returns that first empty
    turn so a marker-only-text seed path still resolves.
    """
    empty_synthetic: Turn | None = None
    for turn in turn_list:
        if turn.raw_messages:
            for msg in turn.raw_messages:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return turn
        elif turn.texts:
            return turn
        elif empty_synthetic is None:
            empty_synthetic = turn
    return empty_synthetic


def _inject_marker_into_first_user_text(
    turn: Turn, marker: str, *, is_prefix: bool
) -> None:
    """Mutate the first ``Text.contents[0]`` on a structured Turn (synthetic-Turn path).

    Used as a fallback when ``Turn.raw_messages`` is None and the endpoint
    formatter would synthesise the user message from ``Turn.texts``. If the
    Turn has no ``texts`` entries, prepends one whose content is the marker
    alone (becomes the entire turn body — fine because there was nothing else
    to merge with).
    """
    if not marker:
        return
    # Seed with the full marker (including ``\n\n``) so
    # ``_content_has_marker_at_edge`` idempotency matches the string path.
    # Multimodal list-of-parts injection still uses ``marker.strip()`` when
    # comparing/inserting text parts.
    if not turn.texts:
        turn.texts = [Text(contents=[marker])]
        return
    first = turn.texts[0]
    if not first.contents:
        first.contents = [marker]
        return
    existing = first.contents[0]
    if _content_has_marker_at_edge(existing, marker, is_prefix=is_prefix):
        return
    first.contents[0] = (marker + existing) if is_prefix else (existing + marker)


def _clone_conversation_for_session(conversation: Conversation) -> Conversation:
    """Give one session private containers over shared, immutable content.

    A session decodes its whole conversation and holds it until the session
    ends. At 50,000 lanes over 393 traces every trace is decoded and held about
    127 times, which costs hundreds of GiB; decoding once per worker and
    cloning per session keeps the message payloads shared.

    Cache-bust and media-uuid suffixing mutate only *containers*: they assign to
    list slots (``raw_messages[:] = ...``, ``images[:] = ...``,
    ``contents[0] = ...``, ``turn.texts = ...``) and replace dicts rather than
    editing them. The message dicts, parts and strings are therefore safe to
    share, while every list container and ``Turn`` object the mutators can
    write to must be private to the session.
    """

    def clone_turn(turn: Turn) -> Turn:
        return turn.model_copy(
            update={
                "raw_messages": None
                if turn.raw_messages is None
                else list(turn.raw_messages),
                "raw_tools": None if turn.raw_tools is None else list(turn.raw_tools),
                "raw_system": None
                if turn.raw_system is None
                else list(turn.raw_system),
                "texts": [
                    text.model_copy(update={"contents": list(text.contents)})
                    for text in turn.texts
                ],
                "images": list(turn.images),
                "audios": list(turn.audios),
                "videos": list(turn.videos),
                "raw_payload": None
                if turn.raw_payload is None
                else dict(turn.raw_payload),
            }
        )

    return conversation.model_copy(
        update={"turns": [clone_turn(turn) for turn in conversation.turns]}
    )


def _inject_marker_at_first_user(
    turn_list: list[Turn], marker: str, *, is_prefix: bool
) -> None:
    """Inject ``marker`` at the first user turn (raw_messages or texts).

    Wraps the lookup + dispatch shared by SYSTEM_* fallback (sub-path 3
    in :func:`_apply_cache_bust`) and the FIRST_TURN_* path. No-op when
    there is no user-bearing turn at all.
    """
    user_turn = _find_first_user_turn(turn_list)
    if user_turn is None:
        return
    if user_turn.raw_messages:
        _inject_marker_into_first_user_turn(
            user_turn.raw_messages, marker, is_prefix=is_prefix
        )
    else:
        _inject_marker_into_first_user_text(user_turn, marker, is_prefix=is_prefix)


def _apply_cache_bust(
    session: UserSession,
    credit: Credit,
    system_message: str | None,
) -> str | None:
    """Dispatch cache-bust marker injection for a single credit.

    Mutates the appropriate turn's ``raw_messages`` (or ``texts``) in-place
    when the marker attaches to the trace's pre-rendered messages. Returns
    the (possibly modified) ``system_message`` string for the caller to
    forward into request building.

    The system / first-user lookups walk ``turn_list`` forward rather than
    indexing ``[-1]``, so this works under both ``MESSAGE_ARRAY_WITH_RESPONSES``
    (single-turn ``turn_list``) and ``DELTAS_WITH_RESPONSES`` (accumulating
    ``turn_list`` where the system role lives in ``turn_list[0]`` and later
    deltas start with the prior assistant response).

    Injection targets the *effective wire prefix* (see
    :func:`_effective_prefix_turns`): the slice of ``turn_list`` that
    ``build_messages`` actually emits. A ``reset_context`` turn makes
    ``build_messages`` discard every prior turn, so the effective prefix begins
    at the last such turn — which may sit mid-history (seeded on a resume,
    never dispatched as the current turn), not just at ``turn_list[-1]``.
    Marking the discarded turn 0 instead would leave the real prefix unmarked
    and let recycled plays warm the server's cache on identical post-reset
    bytes. ``SYSTEM_*`` targets are handled in
    :func:`_apply_system_target_cache_bust`.
    """
    marker = credit.cache_bust_marker
    target = credit.cache_bust_target

    if not marker or target == CacheBustTarget.NONE:
        return system_message

    # FORK children share the parent's KV cache by design: they seed turn_list
    # from the parent (the SAME Turn objects) and must send the parent's exact
    # prefix to hit its cache. The parent already injected its marker into those
    # shared turns, so the child inherits it for free — re-busting here would
    # diverge the child's prefix from the parent's (cache miss) AND mutate the
    # parent's shared, read-only Turn objects (stacking markers). So cache-bust
    # is a no-op for FORK children. SPAWN children start fresh (no shared turns)
    # and root sessions own their prefix, so both are busted normally.
    if (
        session.parent_correlation_id is not None
        and session.branch_mode == ConversationBranchMode.FORK
    ):
        return system_message

    is_prefix = target in (
        CacheBustTarget.SYSTEM_PREFIX,
        CacheBustTarget.FIRST_TURN_PREFIX,
        CacheBustTarget.WARMUP_ISOLATION_SYSTEM,
        CacheBustTarget.WARMUP_ISOLATION_FIRST_TURN,
    )
    prefix_turns = _effective_prefix_turns(session)

    if target in (
        CacheBustTarget.SYSTEM_PREFIX,
        CacheBustTarget.SYSTEM_SUFFIX,
        CacheBustTarget.WARMUP_ISOLATION_SYSTEM,
    ):
        return _apply_system_target_cache_bust(
            prefix_turns,
            all_turns=session.turn_list,
            system_message=system_message,
            marker=marker,
            target=target,
            is_prefix=is_prefix,
        )

    # Mark the effective prefix's opening user turn every credit (idempotent).
    # Unconditional rather than turn_index==0-gated so a seeded mid-trajectory
    # resume (turn_list back-filled with turns 0..k_i at a credit whose
    # turn_index > 0) still marks the true wire prefix.
    _inject_marker_at_first_user(prefix_turns, marker, is_prefix=is_prefix)
    return system_message


def _apply_system_target_cache_bust(
    prefix_turns: list[Turn],
    *,
    all_turns: list[Turn],
    system_message: str | None,
    marker: str,
    target: CacheBustTarget,
    is_prefix: bool,
) -> str | None:
    """Inject a ``SYSTEM_PREFIX`` / ``SYSTEM_SUFFIX`` / ``WARMUP_ISOLATION_SYSTEM`` marker for one credit.

    ``prefix_turns`` is the effective wire prefix slice (see
    :func:`_effective_prefix_turns`). Four sub-paths, ordered to match how the
    endpoint resolves the system field on the wire — marking a lower-precedence
    carrier would leave the bytes that actually ship unchanged:
      1. Conversation-level ``system_message`` present: marker applied every
         turn (string mutation re-applied per credit). Unaffected by
         ``reset_context`` — the ``system_message`` rides on ``RequestInfo`` and
         is re-emitted every turn independent of ``build_messages``' reset.
         Checked first for prefix targets because every system-aware endpoint
         places it at or ahead of the other carriers: ``MessagesEndpoint``
         makes it ``system`` block 0 in front of ``raw_system``, and
         ``ChatEndpoint`` merges it into the front of a leading system
         ``raw_messages`` entry. Marking a later carrier would leave a constant
         leading block the server can still prefix-hit on, which is exactly
         what ``SYSTEM_PREFIX`` must prevent.
         For ``SYSTEM_SUFFIX`` the same wire order means ``system_message`` is
         the tail only when no ``raw_system`` follows it: ``MessagesEndpoint``
         ships ``[system_message, *raw_system]``, so a marker appended to
         ``system_message`` would sit ahead of constant trailing blocks and
         the shared bytes the suffix contract keeps cacheable would land after
         it. Suffix therefore defers to step 2 when ``raw_system`` is present.
      2. ``raw_system`` present on any turn (and either no ``system_message``
         or a suffix target): it outranks system-role ``raw_messages`` in
         ``MessagesEndpoint``, so the marker goes there. Resolved over
         ``all_turns`` (not the prefix slice) because ``_latest_turn_attr``
         ignores ``reset_context``.
      3. ``raw_messages`` first dict has ``role=="system"``: marker injected
         into the first system message of the prefix slice.
      4. No system anywhere -> first-user-turn fallback: marker injected
         into the first user turn every credit (idempotent), matching
         ``FIRST_TURN_*`` semantics so a seeded mid-trajectory resume still
         marks the prefix.

    Returns the (possibly modified) ``system_message``.
    """
    raw_system_blocks = _find_effective_raw_system(all_turns)
    if system_message is not None and (is_prefix or raw_system_blocks is None):
        return _apply_cache_bust_to_system_message(system_message, marker, target)
    if raw_system_blocks is not None:
        _inject_marker_into_raw_system(raw_system_blocks, marker, is_prefix=is_prefix)
        return system_message
    raw_system = _find_first_system_message(prefix_turns)
    if raw_system is not None:
        _inject_marker_into_raw_messages(raw_system, marker, is_prefix=is_prefix)
    else:
        _inject_marker_at_first_user(prefix_turns, marker, is_prefix=is_prefix)
    return system_message


def _effective_prefix_turns(session: UserSession) -> list[Turn]:
    """The ``turn_list`` slice that forms the wire prefix for cache-bust.

    ``base_endpoint.build_messages`` restarts the message array at every
    ``reset_context`` turn that carries ``raw_messages`` (discarding everything
    before it), so the effective prefix begins at the *last* such turn in
    ``turn_list`` — not turn 0, and not merely ``turn_list[-1]``: a reset can
    sit mid-history (e.g. seeded into a mid-trajectory resume, where it is never
    dispatched as the current turn). Returns the slice from that turn to the
    end, or the whole ``turn_list`` when there is no reset.
    """
    turns = session.turn_list
    for i in range(len(turns) - 1, -1, -1):
        if turns[i].reset_context and turns[i].raw_messages:
            return turns[i:]
    return turns


class Worker(BaseComponentService, ProcessHealthMixin):
    """Worker processes credits from the TimingManager and makes API calls to inference servers.

    Responsibilities:
    - Receives credits via DEALER socket from StickyCreditRouter
    - Processes individual turns (1 credit = 1 turn) with session caching for sticky routing
    - Manages conversation state and assistant responses across turns
    - Sends inference results to RecordProcessor for metric calculation
    - Reports health and task statistics to WorkerManager

    Architecture:

      ┌────────────────────┐
      │ StickyCreditRouter │
      │   (ROUTER socket)  │
      └────┬──────────▲────┘
           │          │
        Credit   CreditReturn
           │          │
           ▼          │  ┌─── RequestRecord ──▶ RecordProcessor
      ┌────────────────────┐
      │  Worker (DEALER)   │
      │                    │
      │ 1. Check cache     │
      │ 2. Advance session │
      │ 3. Build request   │
      └────┬──────────▲────┘
           │          │
           ▼          │
      ┌────────────────────┐
      │  InferenceClient   │
      │  (HTTP/streaming)  │
      └────┬──────────▲────┘
           │          │
           ▼          │
      ┌────────────────────┐
      │  Inference Server  │
      │   (vLLM, TRT-LLM)  │
      └────────────────────┘

    Credit Flow (All Modes):
    ═══════════════════════════════════════════════════════════════════════════
    1. Credit arrives with x_correlation_id (shared across all turns)
    2. Check session cache:
       - Cache HIT:  Reuse session → Sticky routing working!
       - Cache MISS: Fetch conversation → Create & cache session
    3. Advance session to credit.turn_index
    4. Process single turn, return credit immediately
    5. If final_turn: Evict session from cache

    Example timeline for 3-turn conversation:
    T1: credit[turn=0, x_corr=ABC] → cache MISS → fetch & cache session → return
    T2: credit[turn=1, x_corr=ABC] → cache HIT  → reuse session → return
    T3: credit[turn=2, x_corr=ABC] → cache HIT  → reuse session → evict → return
        └─▶ Same worker processes all turns (StickyCreditRouter sticky routing)

    Session Lifecycle:
    - First turn: Create session from DatasetManager, cache by x_correlation_id
    - Subsequent turns: Retrieve from cache, advance to turn_index
    - Final turn: Process and evict from cache
    - StickyCreditRouter ensures all turns route to same worker for cache hits
    """

    def __init__(
        self,
        run: BenchmarkRun,
        service_id: str | None = None,
        **kwargs,
    ):
        super().__init__(
            run=run,
            service_id=service_id,
            **kwargs,
        )

        self.debug(lambda: f"Worker process __init__ (pid: {self._process.pid})")

        self.event_loop_monitor = EventLoopMonitor(self.service_id)

        self.task_stats: WorkerTaskStats = WorkerTaskStats()

        self.credit_tasks: dict[int, asyncio.Task] = {}

        self.inference_results_push_client: PushClientProtocol = (
            self.comms.create_push_client(
                CommAddress.RAW_INFERENCE_PROXY_FRONTEND,
            )
        )

        self.model_endpoint = ModelEndpointInfo.from_run(self.run)

        self.inference_client: InferenceClient = InferenceClient(
            model_endpoint=self.model_endpoint,
            service_id=self.service_id,
            strip_record_payload_bytes=resolve_strip_record_payload_bytes(
                self.run.cfg, self.model_endpoint
            ),
        )
        self.attach_child_lifecycle(self.inference_client)
        self.debug(
            lambda: (
                f"Created inference client for {self.model_endpoint.endpoint.type}, "
                f"class: {self.inference_client.__class__.__name__}"
            ),
        )

        # Identity must be unique - ZMQ ROUTER uses it to address messages to specific
        # DEALERs. The sticky router tracks workers by this identity.
        self.credit_dealer_client: StreamingDealerClientProtocol = (
            self.comms.create_streaming_dealer_client(
                address=CommAddress.CREDIT_ROUTER,
                identity=self.service_id,
                bind=False,
            )
        )
        self.credit_dealer_client.register_receiver(self._on_credit_message)

        # Dual-channel returns: CreditReturn/FirstToken go out a dedicated typed
        # PUSH -> router PULL fan-in instead of back on the bidirectional credit
        # DEALER, so the dispatch DEALER is receive-only (no shared-FD send/recv
        # contention). WorkerReady/WorkerShutdown still go on the DEALER so the
        # ROUTER registers/tracks identity. Returns carry worker_id in-message
        # since PUSH/PULL has no ZMQ envelope identity.
        self.credit_return_push_client: StreamingPushClientProtocol = (
            self.comms.create_streaming_push_client(
                CommAddress.CREDIT_RETURN,
                bind=False,
            )
        )

        self.memory_usage_before_profiling: float | None = None

        self.session_manager: UserSessionManager = UserSessionManager()

        # Dataset client for direct data access (eliminates DatasetManager bottleneck)
        # Initialized when DatasetConfiguredNotification is received via factory
        self._dataset_client: DatasetClientStoreProtocol | None = None
        self._dataset_configured_event = asyncio.Event()
        # True when the mmap dataset ships pre-encoded per-turn payload bytes
        # (PAYLOAD_BYTES format); enables the verbatim-replay fast path.
        self._is_payload_bytes: bool = False

        # Detecting first token requires parsing each SSE chunk, so only enable
        # FirstToken messages when a downstream consumer needs them.
        self._prefill_concurrency_enabled: bool = any(
            _phase_needs_first_token_callback(phase) for phase in self.run.cfg.phases
        )
        self._streaming_coverage_enabled = bool(
            self.run.cfg.scenario
            and get_scenario(
                self.run.cfg.scenario
            ).minimum_profile_metric_coverage_ratio
            is not None
        )

        # One-shot warning gate so cache-bust diagnostics don't spam logs at
        # high concurrency — the misconfiguration is the same for every credit.
        self._cache_bust_warning_shown: bool = False

        # Only used as a fallback when dataset client is not initialized
        # or was not available when the credit was dropped. Must be created here
        # so it can be attached to the worker lifecycle.
        self.conversation_request_client: RequestClientProtocol = (
            self.comms.create_request_client(
                address=CommAddress.DATASET_MANAGER_PROXY_FRONTEND,
                bind=False,
            )
        )

    @on_start
    async def _send_worker_ready_message(self) -> None:
        """Send WorkerReady to announce presence."""
        await self.credit_dealer_client.send(WorkerReady(worker_id=self.service_id))

    @on_message(MessageType.DATASET_CONFIGURED_NOTIFICATION)
    async def _on_dataset_configured(self, msg: DatasetConfiguredNotification) -> None:
        """Initialize dataset client when configuration is received.

        Uses factory pattern to dynamically create the appropriate client.
        The factory auto-extracts client_type from client_metadata, leveraging
        the discriminated union pattern for type-safe routing. This allows new
        storage backends (S3, Redis, etc.) to work without modifying Worker code.
        """
        ClientStoreClass = plugins.get_class(
            PluginType.DATASET_CLIENT_STORE, msg.client_metadata.client_type
        )
        self._dataset_client = ClientStoreClass(client_metadata=msg.client_metadata)
        await self._dataset_client.initialize()
        self.session_manager.set_default_context_mode(msg.metadata.default_context_mode)
        # Decoded conversations are cached per worker and cloned per session
        # (_clone_conversation_for_session). Decoding a multi-MB conversation on
        # the worker's event loop for every session is both the biggest stall
        # source and, at 50k lanes, the dominant memory consumer.
        self._conversation_cache: OrderedDict[tuple[str, int], Conversation] = (
            OrderedDict()
        )
        if isinstance(msg.client_metadata, MemoryMapClientMetadata):
            self._is_payload_bytes = (
                msg.client_metadata.format == MemoryMapFormat.PAYLOAD_BYTES
            )
            if (
                self._is_payload_bytes
                and self.run.cfg.get_cache_bust_target() != CacheBustTarget.NONE
            ):
                raise RuntimeError(
                    "cache-bust is incompatible with PAYLOAD_BYTES fast path; "
                    "loader should have skipped preformat "
                    "(see DatasetManager._preformat_payloads)"
                )
        self._dataset_configured_event.set()
        self.debug(
            lambda: (
                f"Dataset client initialized: type={msg.client_metadata.client_type}"
            )
        )

    @on_stop
    async def _send_worker_shutdown_message(self) -> None:
        """Send WorkerShutdown to announce shutdown."""
        try:
            await self.credit_dealer_client.send(
                WorkerShutdown(worker_id=self.service_id)
            )
            self.debug(
                lambda: (
                    f"Sent WorkerShutdown for graceful disconnect ({self.service_id})"
                )
            )
        except Exception as e:
            self.warning(
                f"Failed to send shutdown message (already disconnected?): {e!r}"
            )

    @background_task(
        immediate=False,
        interval=Environment.WORKER.HEALTH_CHECK_INTERVAL,
    )
    async def _health_check_task(self) -> None:
        """Task to report the health of the worker to the worker manager."""
        health = await asyncio.to_thread(self.get_process_health)
        await self.publish(self.create_health_message(health))

    def create_health_message(self, health: ProcessHealth) -> WorkerHealthMessage:
        return WorkerHealthMessage(
            service_id=self.service_id,
            health=health,
            task_stats=self.task_stats,
        )

    async def _on_credit_message(self, message: RouterToWorkerMessage) -> None:
        """Handle incoming credit message from TimingManager via StickyCreditRouter."""
        match message:
            case Credit():
                self._schedule_credit_drop_task(message)
            case CancelCredits():
                await self._on_cancel_credits_message(message)
            case _:
                self.warning(
                    f"Unknown credit message type: {message.__class__.__name__}"
                )

    def _schedule_credit_drop_task(self, credit: Credit) -> None:
        """Schedule a task to handle the credit drop message from TimingManager via StickyCreditRouter.

        This method creates the credit context outside the task so it's available to the done callback.
        This simply schedules the task to be executed asynchronously and adds a done callback to
        ensure the credit is returned. It does not wait for it to actually execute.
        """
        drop_perf_ns = time.perf_counter_ns()
        credit_context = CreditContext(
            credit=credit,
            drop_perf_ns=drop_perf_ns,
        )

        task = self.execute_async(self._on_credit_drop_message_task(credit_context))
        self.credit_tasks[credit.id] = task
        task.add_done_callback(
            lambda t, ctx=credit_context: self._on_credit_drop_message_task_done(t, ctx)
        )

    def _on_credit_drop_message_task_done(
        self, task: asyncio.Task, credit_context: CreditContext
    ) -> None:
        """Handle credit task completion - ensure credit is ALWAYS returned.

        This callback runs when a credit task finishes, whether it completed normally,
        was cancelled, or errored. For cancelled tasks that never started executing,
        the finally block never runs, so we must return the credit here.
        """
        credit_id = credit_context.credit.id

        # Always remove from tracking dict when task completes
        self.credit_tasks.pop(credit_id, None)

        # The finally block handles normal/error returns. This callback only needs
        # to return credits for tasks that were cancelled before they started executing.
        if credit_context.returned:
            # Clear references explicitly since GC is disabled during profiling
            credit_context.credit = None
            credit_context.error = None
            return

        # Credit was NOT returned - this means the task was cancelled before it started
        # or failed in some way that prevented the finally block from sending the return
        self.debug(
            lambda id=credit_id: (
                f"Credit {id} task done but NOT returned! "
                f"Task likely was cancelled before finally block could execute. Returning now."
            )
        )

        # Update credit_context with cancellation status
        credit_context.cancelled = credit_context.cancelled or task.cancelled()

        # Build and send return message (synchronous context, need to schedule send)
        credit_return = CreditReturn(
            credit=credit_context.credit,
            cancelled=credit_context.cancelled,
            first_token_sent=credit_context.first_token_sent,
            error=str(credit_context.error) if credit_context.error else None,
            request_latency_ns=credit_context.request_latency_ns,
            inter_token_latency_ns=credit_context.inter_token_latency_ns,
            output_sequence_length=credit_context.output_sequence_length,
            last_streaming_content_ns=credit_context.last_streaming_content_ns,
            worker_id=self.service_id,
        )
        self.execute_async(self.credit_return_push_client.send(credit_return))
        credit_context.returned = True

        # Explicitly clear references to help refcounting (GC is disabled on workers)
        credit_context.credit = None
        credit_context.error = None

    async def _on_cancel_credits_message(self, message: CancelCredits) -> None:
        """Handle incoming cancel credits message from TimingManager via StickyCreditRouter."""
        self.debug(
            lambda: f"Received cancel credits message: credit_ids={message.credit_ids}"
        )
        for credit_id in message.credit_ids:
            if task := self.credit_tasks.get(credit_id):
                task.cancel()
            else:
                self.debug(
                    lambda id=credit_id: (
                        f"Task for credit {id} not found (already completed?)"
                    )
                )

    async def _on_credit_drop_message_task(self, credit_context: CreditContext) -> None:
        """Handle incoming credit from TimingManager via StickyCreditRouter.

        Flow:
        1. Process single turn:
           - Check session cache by x_correlation_id
           - If cache miss: Fetch conversation and create session
           - Advance session to turn_index
           - Send request to inference server
        2. ALWAYS return credit in finally block, regardless of success/failure

        Credit return is guaranteed via finally block to ensure accurate concurrency tracking.
        For tasks cancelled before they start, the done callback handles the return.
        """
        try:
            if not self.inference_client:
                raise NotInitializedError("Inference server client not initialized.")
            await self._process_credit(credit_context)
        except Exception as e:
            self.exception(
                f"Error occurred while processing credit {credit_context.credit.id}: {e!r}"
            )
        except asyncio.CancelledError:
            self.debug(lambda: f"Credit {credit_context.credit.id} cancelled")
            credit_context.cancelled = True
        finally:
            # Lockstep: a credit returned as completed MUST be accompanied by a
            # record, or the RecordsManager completion barrier (which has no
            # timeout) hangs waiting for it. If processing failed before any
            # record was emitted, forward an error record now. Cancelled credits
            # are excluded from the barrier target, so they need no record.
            #
            # The emit MUST NOT abort the credit return below: if it raised, the
            # finally would exit early, ``returned`` would stay False, and the
            # done callback would return the credit as completed anyway (no
            # record emitted) -- the exact lockstep break this guards against.
            # Contain the failure and always proceed to return the credit.
            if not credit_context.cancelled and not credit_context.record_emitted:
                try:
                    await self._emit_credit_failure_record(credit_context)
                except Exception as e:
                    self.exception(
                        f"Failed to emit lockstep failure record for credit "
                        f"{credit_context.credit.id}: {e!r}"
                    )
            # ALWAYS return the credit here to ensure accurate tracking
            credit_return = CreditReturn(
                credit=credit_context.credit,
                cancelled=credit_context.cancelled,
                first_token_sent=credit_context.first_token_sent,
                error=str(credit_context.error) if credit_context.error else None,
                request_latency_ns=credit_context.request_latency_ns,
                inter_token_latency_ns=credit_context.inter_token_latency_ns,
                output_sequence_length=credit_context.output_sequence_length,
                last_streaming_content_ns=credit_context.last_streaming_content_ns,
                worker_id=self.service_id,
            )
            await self.credit_return_push_client.send(credit_return)
            # Mark as returned AFTER send succeeds
            # If send fails/cancelled, done callback will retry
            # Router idempotency guard handles duplicates
            credit_context.returned = True
            # Note: Don't null credit_context.credit here - done callback needs
            # credit.id for cleanup. Done callback handles all reference clearing.

    async def _emit_credit_failure_record(self, credit_context: CreditContext) -> None:
        """Forward an error record for a credit whose processing failed before
        any record was emitted.

        Mirrors the conversation-retrieval error path so the records-side count
        stays in lockstep with the credit the worker is about to return as
        completed; without it the RecordsManager completion barrier (no timeout)
        hangs waiting for a record that never arrives.
        """
        credit = credit_context.credit
        err = credit_context.error
        if isinstance(err, ErrorDetails):
            error = err
        elif isinstance(err, str) and err:
            error = ErrorDetails(message=err, type="CreditProcessingError", code=500)
        else:
            error = ErrorDetails(
                message="Credit processing failed before a record was produced",
                type="CreditProcessingError",
                code=500,
            )
        await self._send_inference_result_message(
            RequestRecord(
                request_info=RecordContext(
                    conversation_id=credit.conversation_id,
                    turn_index=credit.turn_index,
                    credit_num=credit.id,
                    credit_phase=credit.phase,
                    phase_index=credit.phase_index,
                    profiling_index=credit.profiling_index,
                    phase_name=credit.phase_name,
                    phase_kind=credit.phase_kind,
                    x_request_id=str(uuid.uuid4()),
                    x_correlation_id=credit.x_correlation_id,
                    agent_depth=credit.agent_depth,
                    parent_correlation_id=credit.parent_correlation_id,
                ),
                model_name=self.model_endpoint.primary_model_name,
                timestamp_ns=time.time_ns(),
                start_perf_ns=time.perf_counter_ns(),
                end_perf_ns=time.perf_counter_ns(),
                error=error,
            )
        )
        # Surface the error on the credit so the subsequent CreditReturn carries
        # it and increment_returned(..., errored=...) counts it. Without this the
        # forwarded error record would not be reflected in the phase-complete
        # request_errors log line.
        credit_context.error = error
        credit_context.record_emitted = True

    async def _process_credit(self, credit_context: CreditContext) -> None:
        """Process a credit (1 credit = 1 request).

        Orchestrates error handling and session eviction for both paths:
        - **Payload bytes fast path**: pre-encoded bytes from mmap, bypasses
          session/conversation deserialization entirely.
        - **Normal path**: session-based conversation handling with turn
          accumulation and response storage.

        Credit return is guaranteed by the caller (_on_credit_drop_message_task).
        """
        x_request_id = str(uuid.uuid4())
        x_correlation_id = credit_context.credit.x_correlation_id
        credit = credit_context.credit

        first_token_callback = self._make_content_callback(credit_context)

        try:
            if await self._try_payload_bytes_fast_path(
                credit_context, x_request_id, first_token_callback
            ):
                return

            # Normal path: session-based conversation handling (cache-bust
            # marker injection, DAG fork/spawn seeding, response accumulation).
            await self._process_credit_with_session(
                credit_context, x_request_id, x_correlation_id, first_token_callback
            )

        except asyncio.CancelledError:
            credit_context.cancelled = True
            raise
        except Exception as e:
            credit_context.error = ErrorDetails.from_exception(e)
            self.exception(f"Error processing credit: {e!r}")
        finally:
            self._handle_terminal_disposition(credit, credit_context, x_correlation_id)

    def _make_content_callback(
        self, credit_context: CreditContext
    ) -> FirstTokenCallback | None:
        """Observe content for prefill release and scenario coverage when needed.

        Coverage keeps the callback active after the first token so cancellation
        cannot discard the last content timestamp along with the request record.
        """
        track_coverage = (
            self._streaming_coverage_enabled
            and credit_context.credit.phase == CreditPhase.PROFILING
        )
        if not self._prefill_concurrency_enabled and not track_coverage:
            return None

        credit = credit_context.credit
        wall_clock_offset_ns = time.time_ns() - time.perf_counter_ns()
        last_report_perf_ns: int | None = None

        async def on_content(ttft_ns: int, message: SSEMessage) -> bool:
            nonlocal last_report_perf_ns
            parsed = self.inference_client.endpoint.parse_response(message)
            if parsed is None or parsed.data is None:
                return False

            if track_coverage:
                credit_context.last_streaming_content_ns = (
                    wall_clock_offset_ns + message.perf_ns
                )
                if (
                    last_report_perf_ns is None
                    or message.perf_ns - last_report_perf_ns >= NANOS_PER_SECOND
                ):
                    await self.credit_return_push_client.send(
                        StreamingContent(
                            phase=credit.phase,
                            phase_index=credit.phase_index,
                            timestamp_ns=credit_context.last_streaming_content_ns,
                        )
                    )
                    last_report_perf_ns = message.perf_ns
            if (
                self._prefill_concurrency_enabled
                and not credit_context.first_token_sent
            ):
                await self.credit_return_push_client.send(
                    FirstToken(
                        credit_id=credit.id,
                        phase=credit.phase,
                        phase_index=credit.phase_index,
                        ttft_ns=ttft_ns,
                    )
                )
                credit_context.first_token_sent = True
            return not track_coverage

        return on_content

    async def _process_credit_with_session(
        self,
        credit_context: CreditContext,
        x_request_id: str,
        x_correlation_id: str,
        first_token_callback: FirstTokenCallback | None,
    ) -> None:
        """Normal credit path: session-based conversation handling.

        Flow:
        1. Check session cache using x_correlation_id:
           - Cache hit: Reuse session (enables conversation caching on inference server)
           - Cache miss: Retrieve conversation from DatasetManager, create new session
        2. Advance session to current turn index
        3. Build RequestInfo from session state and send request
        4. Store assistant response in session for multi-turn accumulation

        Session Lifecycle:
        - First turn: Session created and cached under x_correlation_id
        - Subsequent turns: Retrieved from cache (sticky routing ensures same worker)
        - Final turn: Evicted by caller (_process_credit) in its finally block
        """
        credit = credit_context.credit
        session = self.session_manager.get(x_correlation_id)
        if session is None:
            _conversation = await self._borrow_conversation_for_session(
                credit_context=credit_context,
            )
            session = self.session_manager.create_and_store(
                x_correlation_id,
                _conversation,
                credit.num_turns,
                url_index=credit.url_index,
                parent_correlation_id=credit.parent_correlation_id,
                branch_mode=credit.branch_mode,
            )
            self._pin_parent_if_fork_child(credit, x_correlation_id)
            self._seed_from_parent_if_fork_child(credit, x_correlation_id)

        session.advance_turn(credit.turn_index)

        system_message = _apply_cache_bust(
            session,
            credit,
            session.conversation.system_message,
        )
        self._maybe_warn_cache_bust_silent_drop(session, credit)

        request_info = self._create_request_info(
            session=session,
            credit_context=credit_context,
            x_request_id=x_request_id,
            system_message=system_message,
            user_context_message=session.conversation.user_context_message,
        )
        record: RequestRecord = await self._execute_request(
            credit_context, request_info, first_token_callback
        )

        self._finalize_session_response(session, credit_context, record, request_info)

    async def _try_payload_bytes_fast_path(
        self,
        credit_context: CreditContext,
        x_request_id: str,
        first_token_callback: FirstTokenCallback | None,
    ) -> bool:
        """Serve the credit from pre-encoded mmap payload bytes when possible.

        Payload bytes fast path: stream pre-encoded bytes from the mmap
        verbatim, bypassing session/conversation deserialization and
        per-request payload formatting entirely. Skipped for DAG descendants
        (agent_depth > 0), whose turn accumulation needs session state, and for
        credits carrying a ``max_tokens_override``: the override cannot be
        reflected in the pre-encoded bytes, so serving them verbatim would send
        the baked-in cap while the enriched metric turn reports the override,
        yielding a spurious OSL mismatch. Those defer to the session path, which
        formats the payload with the override applied.

        Returns:
            True when the request was fully handled (sent + result emitted);
            False when the caller should take the normal session path.
        """
        credit = credit_context.credit
        if (
            not self._is_payload_bytes
            or self._dataset_client is None
            or credit.agent_depth != 0
            or credit.max_tokens_override is not None
        ):
            return False
        payload_turn = await self._dataset_client.get_payload_turn(
            credit.conversation_id, credit.turn_index
        )
        if payload_turn is None:
            return False

        self.task_stats.total += 1
        request_info = self._create_request_info(
            credit_context=credit_context,
            x_request_id=x_request_id,
            payload_bytes=payload_turn.payload_bytes,
            # Scalars only — payload_bytes carries the wire body; enrichment
            # reads max_tokens / timestamp off this turn for OSL-mismatch and
            # schedule-lag metrics, source_kind for per-row attribution, and
            # audio_duration_seconds for the audio-duration and RTFx metrics.
            turns=[
                Turn(
                    role="user",
                    max_tokens=payload_turn.max_tokens,
                    timestamp=payload_turn.timestamp,
                    source_kind=payload_turn.source_kind,
                    audio_duration_seconds=payload_turn.audio_duration_seconds,
                )
            ],
        )
        record = await self.inference_client.send_request(
            request_info, first_token_callback=first_token_callback
        )
        self._populate_response_metrics(credit_context, record)
        await self._send_inference_result_message(record)
        credit_context.record_emitted = True
        if record.error is not None:
            credit_context.error = record.error
        return True

    def _process_responses_for_record(
        self,
        record: RequestRecord,
        *,
        capture_assistant_turn: bool,
    ) -> tuple[list[ParsedResponse], Turn | None]:
        """Process responses while preserving protocol-only third-party endpoints.

        The fallback supports ``EndpointProtocol`` implementations that do not
        inherit ``BaseEndpoint`` and therefore lack its optional replay helpers.
        """
        endpoint = self.inference_client.endpoint
        process_responses = getattr(endpoint, "process_responses", None)
        if callable(process_responses):
            return process_responses(
                record, capture_assistant_turn=capture_assistant_turn
            )

        parsed_responses = endpoint.extract_response_data(record)
        build_assistant_turn = getattr(endpoint, "build_assistant_turn", None)
        assistant_turn = (
            build_assistant_turn(record)
            if capture_assistant_turn and callable(build_assistant_turn)
            else None
        )
        return parsed_responses, assistant_turn

    def _finalize_session_response(
        self,
        session: UserSession,
        credit_context: CreditContext,
        record: RequestRecord,
        request_info: RequestInfo | None = None,
    ) -> None:
        """Store the assistant turn (when retained) and populate metrics from a
        single response-processing pass, shared with the payload-bytes path.

        ``request_info`` is the full request that produced ``record``; the
        record itself only carries the slim ``RecordContext``.
        """
        parsed_responses, assistant_turn = self._process_responses_for_record(
            record,
            capture_assistant_turn=session.should_store_response(),
        )
        if assistant_turn is not None:
            session.store_response(assistant_turn)
        # Response-ID chaining is only coherent for the mode that captures live
        # responses and accumulates genuine deltas (DELTAS_WITHOUT_RESPONSES).
        # In the *_WITH_RESPONSES modes ``turns[-1]`` already carries the full
        # authored history, so chaining would double the context on the wire.
        # A failed turn may still emit a ``response.created`` id for a partial or
        # aborted response; chaining onto it would corrupt the server context, so
        # keep the last good id (from a prior successful turn) instead.
        if (
            not record.has_error
            and session.should_store_response()
            and (
                extract_response_id := getattr(
                    self.inference_client.endpoint, "extract_response_id", None
                )
            )
        ):
            compute_stored_flag = getattr(
                self.inference_client.endpoint,
                "stored_history_has_system_prompt",
                None,
            )
            session.store_response_id(
                extract_response_id(record),
                system_prompt_in_history=bool(
                    compute_stored_flag
                    and request_info is not None
                    and compute_stored_flag(request_info)
                ),
            )
        self._populate_response_metrics(credit_context, record, parsed_responses)

    def _populate_response_metrics(
        self,
        credit_context: CreditContext,
        record: RequestRecord,
        parsed_responses: list[ParsedResponse] | None = None,
    ) -> None:
        """Derive latency/OSL/ITL from ``record`` onto ``credit_context``.

        Shared by the normal session path and the payload-bytes fast path so
        both emit identical CreditReturn timing fields from the same record.
        """
        if parsed_responses is None:
            parsed_responses = self.inference_client.endpoint.extract_response_data(
                record
            )
        content_perf_ns = self._content_response_perf_ns_for_record(
            record, parsed_responses
        )
        credit_context.request_latency_ns = self._request_latency_ns_for_record(
            record, content_perf_ns
        )
        credit_context.output_sequence_length = (
            self._output_sequence_length_for_responses(parsed_responses)
        )
        credit_context.inter_token_latency_ns = self._inter_token_latency_ns_for_record(
            record,
            content_perf_ns,
            parsed_responses,
            credit_context.output_sequence_length,
        )

    def _content_response_perf_ns_for_record(
        self,
        record: RequestRecord,
        parsed_responses: list[ParsedResponse] | None = None,
    ) -> list[int]:
        """Return perf timestamps for parsed responses with meaningful content."""
        if parsed_responses is None:
            parsed_responses = self.inference_client.endpoint.extract_response_data(
                record
            )
        return [parsed.perf_ns for parsed in parsed_responses if parsed.data]

    def _request_latency_ns_for_record(
        self, record: RequestRecord, content_perf_ns: list[int] | None = None
    ) -> int | None:
        """Return the same latency sample used by RequestLatencyMetric."""
        if content_perf_ns is None:
            content_perf_ns = self._content_response_perf_ns_for_record(record)
        if not content_perf_ns:
            return None
        final_response_perf_ns = content_perf_ns[-1]
        if final_response_perf_ns < record.start_perf_ns:
            return None
        return final_response_perf_ns - record.start_perf_ns

    def _output_sequence_length_for_responses(
        self, parsed_responses: list[ParsedResponse]
    ) -> int | None:
        usage = find_last_non_empty_usage(parsed_responses)
        if usage is None or usage.completion_tokens is None:
            return None
        return usage.completion_tokens

    def _inter_token_latency_ns_for_record(
        self,
        record: RequestRecord,
        content_perf_ns: list[int] | None = None,
        parsed_responses: list[ParsedResponse] | None = None,
        output_sequence_length: int | None = None,
    ) -> float | None:
        """Return ITL using the records-pipeline output sequence formula."""
        if parsed_responses is None:
            parsed_responses = self.inference_client.endpoint.extract_response_data(
                record
            )
        if content_perf_ns is None:
            content_perf_ns = self._content_response_perf_ns_for_record(
                record, parsed_responses
            )
        if output_sequence_length is None:
            output_sequence_length = self._output_sequence_length_for_responses(
                parsed_responses
            )
        if len(content_perf_ns) < 2 or output_sequence_length is None:
            return None
        if output_sequence_length < 2:
            return None
        request_latency_ns = content_perf_ns[-1] - record.start_perf_ns
        ttft_ns = content_perf_ns[0] - record.start_perf_ns
        if request_latency_ns < 0 or ttft_ns < 0:
            return None
        return (request_latency_ns - ttft_ns) / (output_sequence_length - 1)

    def _pin_parent_if_fork_child(self, credit: Credit, x_correlation_id: str) -> None:
        """FORK child seed: pin the parent so its session stays resident in
        the cache until every FORK child has joined. FORK children
        sticky-route to the parent's worker, so the parent's session
        lives on this same SessionManager.
        """
        if (
            credit.parent_correlation_id is None
            or credit.branch_mode != ConversationBranchMode.FORK
        ):
            return
        try:
            self.session_manager.pin_for_fork_child(credit.parent_correlation_id)
        except KeyError:
            # Parent already evicted — child arrived too late to pin; let
            # the request proceed without seed context rather than failing.
            self.warning(
                f"FORK child {x_correlation_id!r} arrived after parent "
                f"{credit.parent_correlation_id!r} was evicted; not pinning"
            )

    def _seed_from_parent_if_fork_child(
        self, credit: Credit, x_correlation_id: str
    ) -> None:
        """Copy the parent session's accumulated ``turn_list`` into the
        freshly-created FORK child session.

        Companion to ``_pin_parent_if_fork_child``: pinning keeps the
        parent resident, this seeds the child with the parent's context
        so the request-builder prepends parent prompt + captured
        responses before the child's own messages. SPAWN-mode children
        skip this — fresh-context is the whole point of SPAWN.
        """
        if (
            credit.parent_correlation_id is None
            or credit.branch_mode != ConversationBranchMode.FORK
        ):
            return
        self.session_manager.seed_from_parent(
            x_correlation_id, credit.parent_correlation_id
        )

    def _handle_terminal_disposition(
        self, credit: Credit, credit_context: CreditContext, x_correlation_id: str
    ) -> None:
        """Evict on final turn, cancellation, or terminal context overflow.

        Context overflow on a non-final, non-cancelled turn is terminal for
        agentic_replay (lane recycled, no final/cancel credit follows), so it
        must take the same eviction path.
        """
        if (
            credit.is_final_turn
            or credit_context.cancelled
            or _is_terminal_context_overflow(credit, credit_context)
        ):
            self._release_and_evict_for_terminal(credit, x_correlation_id)

    def _release_and_evict_for_terminal(
        self, credit: Credit, x_correlation_id: str
    ) -> None:
        """Release the parent pin (if FORK child) then evict this session.

        FORK parents whose terminal turn declared forks (``has_forks``)
        defer eviction: children arrive on the orchestrator's intercept
        path AFTER this credit return runs, so ``evict_if_unpinned``
        cannot find any pin to honor here. Setting ``pending_fork_eviction``
        signals ``release_fork_child`` to auto-evict the moment the last
        child joins.

        Non-FORK and non-parent sessions evict immediately.
        """
        if (
            credit.parent_correlation_id is not None
            and credit.branch_mode == ConversationBranchMode.FORK
        ):
            self.session_manager.release_fork_child(credit.parent_correlation_id)
        cur_session = self.session_manager.get(x_correlation_id)
        if cur_session is not None and cur_session.is_fork_parent:
            if credit.has_forks:
                cur_session.pending_fork_eviction = True
            self.session_manager.evict_if_unpinned(x_correlation_id)
        else:
            self.session_manager.evict(x_correlation_id)

    def _maybe_warn_cache_bust_silent_drop(
        self,
        session: UserSession,
        credit: Credit,
    ) -> None:
        """Emit a one-shot warning if cache-bust was requested but had nowhere
        to land on this credit (an empty ``session.turn_list``).

        Rate-limited to once per worker via ``self._cache_bust_warning_shown`` —
        the misconfiguration is identical for every credit, so a single
        actionable line beats N-thousand duplicates at scale.
        """
        if self._cache_bust_warning_shown:
            return
        target = credit.cache_bust_target
        marker = credit.cache_bust_marker
        if not marker or target == CacheBustTarget.NONE:
            return
        if not session.turn_list:
            self._cache_bust_warning_shown = True
            self.warning(
                f"cache-bust target={target.value} requested but session.turn_list "
                f"is empty — marker NOT injected (further occurrences suppressed)."
            )

    async def _execute_request(
        self,
        credit_context: CreditContext,
        request_info: RequestInfo,
        first_token_callback: FirstTokenCallback | None,
    ) -> RequestRecord:
        """Send request, record result, and propagate errors to credit context.

        Metrics/response storage are handled once by the caller
        (``_finalize_session_response``) from a single response-processing pass.
        """
        self.task_stats.total += 1
        record = await self.inference_client.send_request(
            request_info, first_token_callback=first_token_callback
        )
        await self._send_inference_result_message(record)
        credit_context.record_emitted = True
        if record.error is not None:
            credit_context.error = record.error
        return record

    def _create_request_info(
        self,
        *,
        x_request_id: str,
        credit_context: CreditContext,
        session: UserSession | None = None,
        system_message: str | None = None,
        user_context_message: str | None = None,
        payload_bytes: bytes | None = None,
        turns: list[Turn] | None = None,
    ) -> RequestInfo:
        """Create RequestInfo for inference request.

        When ``session`` is provided (normal path), conversation state comes from
        the session. When omitted (raw payload fast path), fields are taken
        directly from the credit.

        When ``session`` is provided (normal path), conversation state comes
        from the session. When omitted (payload-bytes fast path), fields are
        taken directly from the credit.

        Args:
            x_request_id: Unique ID for this request (X-Request-ID header)
            credit_context: Context with credit metadata (num, phase, timestamps)
            session: Session with conversation history (None for raw payload path)
            system_message: Optional shared system message to prepend to first turn
            user_context_message: Optional per-conversation user context message
            payload_bytes: Pre-encoded payload bytes from mmap (raw payload path)
            turns: Explicit turns list (raw payload fast path with image metadata).
                   Takes precedence over session-derived turns when provided.

        Returns:
            RequestInfo with all data needed to send inference request
        """
        credit = credit_context.credit
        if turns is None:
            turns = session.turn_list if session else []
        if credit.max_tokens_override is not None and turns:
            last_turn = turns[-1]
            update: dict[str, Any] = {"max_tokens": credit.max_tokens_override}
            # A verbatim raw_payload dict is shipped as-is by inference_client,
            # so Turn.max_tokens alone is ignored on the wire; rewrite the wire
            # cap too or the recorded (baked-in) value would be sent instead.
            if isinstance(last_turn.raw_payload, dict):
                update["raw_payload"] = apply_max_tokens_to_wire_payload(
                    last_turn.raw_payload, credit.max_tokens_override
                )
            turns = [*turns[:-1], last_turn.model_copy(update=update)]
        source_turn = turns[-1] if turns else None
        return RequestInfo(
            model_endpoint=self.model_endpoint,
            credit_num=credit.id,
            credit_phase=credit.phase,
            phase_index=credit.phase_index,
            profiling_index=credit.profiling_index,
            phase_name=credit.phase_name,
            phase_kind=credit.phase_kind,
            cancel_after_ns=credit.cancel_after_ns,
            x_request_id=x_request_id,
            x_correlation_id=session.x_correlation_id
            if session
            else credit.x_correlation_id,
            conversation_id=session.conversation.session_id
            if session
            else credit.conversation_id,
            turn_index=session.turn_index if session else credit.turn_index,
            source_trace_id=source_turn.source_trace_id if source_turn else None,
            source_outer_idx=source_turn.source_outer_idx if source_turn else None,
            source_inner_idx=source_turn.source_inner_idx if source_turn else None,
            source_kind=source_turn.source_kind if source_turn else None,
            turns=turns,
            previous_response_id=session.previous_response_id if session else None,
            system_prompt_in_stored_history=session.system_prompt_in_stored_history
            if session
            else False,
            drop_perf_ns=credit_context.drop_perf_ns,
            credit_issued_ns=credit.issued_at_ns,
            system_message=system_message,
            user_context_message=user_context_message,
            is_final_turn=credit.is_final_turn,
            url_index=session.url_index if session else credit.url_index,
            payload_bytes=payload_bytes,
            agent_depth=credit.agent_depth,
            parent_correlation_id=credit.parent_correlation_id,
            root_correlation_id=credit.effective_root_correlation_id,
            cache_bust_marker=credit.cache_bust_marker,
            cache_bust_target=credit.cache_bust_target
            if credit.cache_bust_marker is not None
            else None,
        )

    async def _retrieve_conversation(
        self,
        *,
        conversation_id: str,
        credit_context: CreditContext,
    ) -> Conversation:
        """Retrieve conversation from dataset client.

        The dataset client is initialized via factory when DatasetConfiguredNotification
        is received. The client type (mmap, S3, etc.) is transparent to this method.

        Args:
            conversation_id: ID of conversation to retrieve (from dataset)
            credit_context: Credit context

        Returns:
            Conversation object with turns and metadata

        Raises:
            RuntimeError: If dataset client not initialized
            KeyError: If conversation_id not found in dataset
        """
        if self._dataset_client is not None:
            return await self._dataset_client.get_conversation(conversation_id)
        elif self.stop_requested:
            raise asyncio.CancelledError("Stop requested while retrieving conversation")

        return await self._request_conversation_from_dataset_manager(
            conversation_id, credit_context
        )

    async def _borrow_conversation_for_session(
        self,
        *,
        credit_context: CreditContext,
    ) -> Conversation:
        """Return a session-private clone of this conversation.

        The decoded conversation is cached and shared; each session gets its own
        containers over the shared payload (see
        :func:`_clone_conversation_for_session`), so a trace replayed by many
        lanes is decoded once per worker rather than once per lane.

        The key includes ``num_turns`` because the PAYLOAD_BYTES reconstruction
        loops ``range(num_turns)`` and the cheap-path scan uses the credit's
        count; keying on the conversation id alone would hand a later credit a
        reconstruction built for a different turn count.
        """
        credit = credit_context.credit
        key = (credit.conversation_id, credit.num_turns)
        cached = self._conversation_cache.get(key)
        if cached is None:
            cached = await self._retrieve_conversation_for_session(
                credit_context=credit_context,
            )
            self._conversation_cache[key] = cached
            bound = max(_CONVERSATION_CACHE_MIN_ENTRIES, len(self.session_manager))
            while len(self._conversation_cache) > bound:
                self._conversation_cache.popitem(last=False)
        else:
            self._conversation_cache.move_to_end(key)
        return _clone_conversation_for_session(cached)

    async def _retrieve_conversation_for_session(
        self,
        *,
        credit_context: CreditContext,
    ) -> Conversation:
        """Retrieve a Conversation suitable for session-based processing.

        In the PAYLOAD_BYTES memory-map format the client's ``get_conversation``
        path raises because the full authoring shape is not persisted — only
        the per-turn payload bytes. For session-mode processing we reconstruct
        a minimal ``Conversation`` from per-turn payload bytes so
        ``session_manager`` can still advance turns.
        """
        conversation_id = credit_context.credit.conversation_id
        num_turns = credit_context.credit.num_turns

        if self._is_payload_bytes and self._dataset_client is not None:
            turns: list[Turn] = []
            for turn_index in range(num_turns):
                payload_turn = await self._dataset_client.get_payload_turn(
                    conversation_id, turn_index
                )
                if payload_turn is None:
                    turns.append(Turn(role="user"))
                    continue
                turns.append(turn_from_payload_turn(payload_turn))
            return Conversation(
                session_id=conversation_id,
                turns=turns,
                context_mode=self.session_manager.default_context_mode,
            )

        return await self._retrieve_conversation(
            conversation_id=conversation_id,
            credit_context=credit_context,
        )

    async def _request_conversation_from_dataset_manager(
        self, conversation_id: str, credit_context: CreditContext
    ) -> Conversation:
        """Fallback: Request from DatasetManager via ZMQ"""
        conversation_response: (
            ConversationResponseMessage | ErrorMessage
        ) = await self.conversation_request_client.request(
            ConversationRequestMessage(
                service_id=self.service_id,
                conversation_id=conversation_id,
                credit_phase=credit_context.credit.phase,
            )
        )
        if self.is_trace_enabled:
            self.trace(f"Received response message: {conversation_response}")

        # Check for error in conversation response
        if isinstance(conversation_response, ErrorMessage):
            error = conversation_response.error
            await self._send_inference_result_message(
                RequestRecord(
                    request_info=RecordContext(
                        conversation_id=conversation_id,
                        turn_index=0,
                        credit_num=credit_context.credit.id,
                        credit_phase=credit_context.credit.phase,
                        phase_index=credit_context.credit.phase_index,
                        profiling_index=credit_context.credit.profiling_index,
                        phase_name=credit_context.credit.phase_name,
                        phase_kind=credit_context.credit.phase_kind,
                        x_request_id=str(uuid.uuid4()),
                        x_correlation_id=credit_context.credit.x_correlation_id,
                        agent_depth=credit_context.credit.agent_depth,
                        parent_correlation_id=credit_context.credit.parent_correlation_id,
                    ),
                    model_name=self.model_endpoint.primary_model_name,
                    timestamp_ns=time.time_ns(),
                    start_perf_ns=time.perf_counter_ns(),
                    end_perf_ns=time.perf_counter_ns(),
                    error=error,
                )
            )
            credit_context.record_emitted = True
            raise ValueError(f"Failed to retrieve conversation response: {error}")

        return conversation_response.conversation

    async def _send_inference_result_message(self, record: RequestRecord) -> None:
        """Send RequestRecord to RecordProcessor for metric calculation.

        All records (success and error) flow through this method to ensure consistent
        metric calculation and error tracking.

        Flow:
        1. Update task statistics (total and success/failure counts)
        2. Wrap record in InferenceResultsMessage
        3. Push to RecordProcessor via PUSH socket (fire-and-forget). This channel
           is encoded as msgpack, not JSON: see ``aiperf.zmq.inference_results_wire``.

        Note: Uses execute_async() to avoid blocking on network I/O.
        """
        # All records will flow through here to be sent to the inference results push client.
        self.task_stats.task_finished(record.valid)

        msg = InferenceResultsMessage(
            service_id=self.service_id,
            record=record,
        )
        self.execute_async(self._push_inference_result(msg))

    async def _push_inference_result(self, msg: InferenceResultsMessage) -> None:
        # Fire-and-forget: log a failed push here, since nothing awaits the task.
        try:
            await self.inference_results_push_client.push(msg)
        except Exception as e:
            self.exception(f"Dropped an inference result: {e!r}")

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _on_profile_configure_command(self, message: CommandMessage) -> None:
        """Configure the worker."""
        self.debug("Waiting for dataset to be configured before starting profiling")
        await asyncio.wait_for(
            self._dataset_configured_event.wait(),
            timeout=Environment.DATASET.CONFIGURATION_TIMEOUT,
        )
        if self.is_debug_enabled:
            health = await asyncio.to_thread(self.get_process_health)
            memory_usage = health.memory_usage / BYTES_PER_MIB
            self.memory_usage_before_profiling = memory_usage
            self.debug(f"Memory usage before profiling: {memory_usage:.2f} MiB")

        self.event_loop_monitor.start()

    @on_stop
    async def _worker_stop(self) -> None:
        # Clean up dataset client resources using protocol lifecycle
        if self._dataset_client is not None:
            dataset_client = self._dataset_client
            self._dataset_client = None
            await dataset_client.stop()
            self.debug("Dataset client stopped")

        self.event_loop_monitor.stop()


def main() -> None:
    """Main entry point for the worker."""
    from aiperf.common.bootstrap import bootstrap_and_run_service
    from aiperf.plugin.enums import ServiceType

    bootstrap_and_run_service(ServiceType.WORKER)


if __name__ == "__main__":
    main()
