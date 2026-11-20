# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native msgspec structs for credit router communication.

All over-the-wire structs use tag_field="t" for efficient polymorphic decoding via tagged unions.
Tag values are short strings for minimal wire overhead.
"""

from typing import TYPE_CHECKING, Self

from msgspec import Struct

from aiperf.common.enums import CacheBustTarget, ConversationBranchMode, CreditPhase
from aiperf.common.models.error_models import ErrorDetails
from aiperf.common.types import PhaseKind

if TYPE_CHECKING:
    from aiperf.common.models.dataset_models import TurnMetadata

# =============================================================================
# Credit Struct (sent from router to worker)
# =============================================================================


class Credit(
    Struct, omit_defaults=True, frozen=True, kw_only=True, tag_field="t", tag="c"
):
    """Credit representing the right to make a single request to an inference server.

    Sent directly from router to worker (no wrapper message).

    Attributes:
        id: Sequential number of the credit in the credit phase.
        phase: Type of credit phase (e.g., "warmup", "profile").
        conversation_id: Template ID from the dataset.
        x_correlation_id: Conversation instance ID for sticky routing (X-Correlation-ID header).
        turn_index: Index of the turn in the conversation (0-based).
        num_turns: Total number of turns in the conversation.
        issued_at_ns: Wall clock timestamp when issued (time.time_ns).
        cancel_after_ns: Delay in nanoseconds after which the request should be cancelled for simulated client disconnections (optional).
                         Note: this is NOT the same as the credit being cancelled!
        url_index: Index of the URL to use when multiple --url values are configured (optional).
                   None means use the default (first) URL.
    """

    id: int
    phase: CreditPhase
    phase_index: int | None = None
    profiling_index: int | None = None
    phase_name: str | None = None
    phase_kind: PhaseKind | None = None
    conversation_id: str
    x_correlation_id: str
    turn_index: int
    num_turns: int
    issued_at_ns: int
    cancel_after_ns: int | None = None
    url_index: int | None = None
    agent_depth: int = 0
    parent_correlation_id: str | None = None
    root_correlation_id: str | None = None
    """x_correlation_id of the depth-0 root of this credit's session TREE.

    Stable across the whole tree: the root carries its own x_correlation_id
    (left None on the wire when it equals x_correlation_id to keep the struct
    small), and every descendant (child, subchild, background subagent) inherits
    the root's id. This is the key used for per-tree session-slot accounting
    (``SessionTreeRegistry``) — the slot is held until the whole tree drains —
    and is persisted in the export so analysis groups a tree under one lane.
    Effective value is ``root_correlation_id or x_correlation_id``."""
    counts_toward_phase_target: bool = True
    """Whether this credit can satisfy the phase's planned send target.

    Reactive DAG children set this False because they are spawned after the
    root plan has been sampled. Snapshot replay can dispatch subagent states as
    planned phase work, so target membership is intentionally separate from
    ``agent_depth``.
    """
    has_forks: bool = False
    branch_mode: ConversationBranchMode = ConversationBranchMode.FORK
    """DAG branch mode for this credit. Ignored when parent_correlation_id is None
    (i.e. for root sessions). FORK = inherit parent turn_list; SPAWN =
    fresh context. Default FORK keeps wire footprint small via msgspec omit_defaults."""

    cache_bust_marker: str | None = None
    """Pre-rendered cache-bust marker text (already includes whitespace boundaries).
    None when the cache-bust feature is disabled."""

    cache_bust_target: CacheBustTarget = CacheBustTarget.NONE
    """Where (and how) to inject `cache_bust_marker` at request-build time."""

    max_tokens_override: int | None = None
    """Per-request generation limit override.

    Agentic replay uses this for warmup priming requests so they prefill the
    recorded context but decode only one token. It is intentionally attached
    to the credit rather than the dataset turn so profiling retains the
    recorded output limit.
    """

    @property
    def is_final_turn(self) -> bool:
        return self.turn_index == self.num_turns - 1

    @property
    def effective_root_correlation_id(self) -> str:
        """Tree root id, defaulting to this credit's own ``x_correlation_id``."""
        return self.root_correlation_id or self.x_correlation_id


class CreditContext(
    Struct, omit_defaults=True, kw_only=True, tag_field="t", tag="cctx"
):
    """Context for a credit. This is used by the worker to track details of a credit.

    Attributes:
        credit: The credit being processed.
        drop_perf_ns: The performance timestamp when the credit was dropped.
        cancelled: True if the credit was cancelled before completion.
        returned: True if the credit was returned after completion.
        first_token_sent: True if the first token was sent before this return.
        error: The error if the request failed (None on success). Usually an
            ``ErrorDetails``; a plain string is also accepted for back-compat.
        record_emitted: True once an inference record has been pushed for this
            credit. Used to keep the records-side count in lockstep with the
            credit-side count: a completed (non-cancelled) credit with no
            record would hang the RecordsManager completion barrier.
        request_latency_ns: Request latency in nanoseconds using records-pipeline
            semantics.
        inter_token_latency_ns: Inter-token latency in nanoseconds using
            adaptive records-pipeline semantics.
        output_sequence_length: Output sequence length in tokens from usage
            data, when available.
        last_streaming_content_ns: Wall-clock timestamp of the latest parsed
            content chunk, retained even if the request is cancelled.
    """

    credit: Credit
    drop_perf_ns: int
    cancelled: bool = False
    returned: bool = False
    first_token_sent: bool = False
    error: str | ErrorDetails | None = None
    record_emitted: bool = False
    request_latency_ns: int | None = None
    inter_token_latency_ns: float | None = None
    output_sequence_length: int | None = None
    last_streaming_content_ns: int | None = None


# =============================================================================
# Turn Structs (pre-credit issuance structs)
# =============================================================================


class TurnToSend(Struct, frozen=True):
    """A turn that needs to be sent.

    Attributes:
        conversation_id: Template ID from the dataset.
        x_correlation_id: Conversation instance ID for sticky routing (X-Correlation-ID header).
        turn_index: The index of the turn in the conversation (0-based).
        num_turns: The total number of turns in the conversation.
    """

    conversation_id: str
    x_correlation_id: str
    turn_index: int
    num_turns: int
    agent_depth: int = 0
    parent_correlation_id: str | None = None
    root_correlation_id: str | None = None
    """x_correlation_id of the depth-0 root of this turn's session TREE.

    None for a root turn (the root IS its own tree root); set on every
    descendant to the root's id. Propagated onto the issued ``Credit`` and used
    for per-tree session-slot accounting. Effective value is
    ``root_correlation_id or x_correlation_id``."""
    counts_toward_phase_target: bool = True
    """Whether this turn can satisfy the phase's planned send target."""
    is_session_start: bool = False
    """True when this credit begins a new root-session occupancy in the phase
    and must acquire a session slot + bump ``sent_sessions``, even when
    ``turn_index > 0``. Agentic replay resumes a sampled trajectory mid-trace
    (warmup at k_i, profiling at k_i+1), so its first credit is a session start
    despite a non-zero ``turn_index``. ``turn_index == 0`` always implies a
    session start regardless of this flag. A mid-trace session start can only
    legitimately occur during a phase's initial dispatch (execute_phase)."""
    has_forks: bool = False
    branch_mode: ConversationBranchMode = ConversationBranchMode.FORK

    cache_bust_marker: str | None = None
    """Pre-rendered cache-bust marker text (already includes whitespace boundaries).
    None when the cache-bust feature is disabled."""

    cache_bust_target: CacheBustTarget = CacheBustTarget.NONE
    """Where (and how) to inject `cache_bust_marker` at request-build time."""

    max_tokens_override: int | None = None
    """Per-request generation limit override; omitted for normal requests."""

    @property
    def is_final_turn(self) -> bool:
        return self.turn_index == self.num_turns - 1

    @property
    def effective_root_correlation_id(self) -> str:
        """Tree root id, defaulting to this turn's own ``x_correlation_id``."""
        return self.root_correlation_id or self.x_correlation_id

    @classmethod
    def from_previous_credit(
        cls, credit: Credit, next_meta: "TurnMetadata | None" = None
    ) -> Self:
        """Create the next turn to send from the previous turn's credit.

        Args:
            credit: The previous turn's credit.
            next_meta: Metadata for the NEW turn being built. When provided, the
                ``has_forks`` flag is derived from it so the sticky
                router can defer parent-entry eviction until DAG children drain.
        """
        return cls(
            conversation_id=credit.conversation_id,
            x_correlation_id=credit.x_correlation_id,
            turn_index=credit.turn_index + 1,
            num_turns=credit.num_turns,
            agent_depth=credit.agent_depth,
            parent_correlation_id=credit.parent_correlation_id,
            root_correlation_id=credit.root_correlation_id,
            counts_toward_phase_target=credit.counts_toward_phase_target,
            has_forks=next_meta.has_forks if next_meta is not None else False,
            branch_mode=credit.branch_mode,
            cache_bust_marker=credit.cache_bust_marker,
            cache_bust_target=credit.cache_bust_target,
        )
