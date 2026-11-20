# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Credit callback handler for credit lifecycle events.

Handles ALL credit lifecycle callbacks (returns + TTFT) directly from CreditRouter.

Key responsibilities:
- Track credit returns (increment_returned, release slots)
- Handle TTFT events (increment_prefill_released, release prefill slot)
- Dispatch next turn to timing strategy (handle_credit_return)
- Cleanup in-flight sessions on phase end
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import CreditPhase
from aiperf.common.scenario.context_overflow import is_context_overflow_response
from aiperf.timing.concurrency import PhaseRuntimeKey

if TYPE_CHECKING:
    from aiperf.credit.messages import CreditReturn, FirstToken, StreamingContent
    from aiperf.credit.structs import Credit
    from aiperf.timing.branch_orchestrator import BranchOrchestrator
    from aiperf.timing.concurrency import ConcurrencyManager
    from aiperf.timing.phase.lifecycle import PhaseLifecycle
    from aiperf.timing.phase.progress_tracker import PhaseProgressTracker
    from aiperf.timing.phase.stop_conditions import StopConditionChecker
    from aiperf.timing.session_tree import SessionTreeRegistry
    from aiperf.timing.strategies.core import TimingStrategyProtocol

_logger = AIPerfLogger(__name__)


@dataclass(slots=True)
class PhaseCallbackContext:
    """Context for handling callbacks for a specific phase.

    Registered by PhaseRunner before phase execution starts.
    Contains all components needed to handle credit returns for this phase.
    """

    progress: PhaseProgressTracker
    lifecycle: PhaseLifecycle
    stop_checker: StopConditionChecker
    strategy: TimingStrategyProtocol
    concurrency_manager: ConcurrencyManager
    handle_credit_result: Callable[[CreditReturn], Awaitable[None]] | None = None
    handle_first_token: Callable[[FirstToken], Awaitable[None]] | None = None


# =============================================================================
# CreditCallbackHandler - Handle credit lifecycle callbacks
# =============================================================================


class CreditCallbackHandler:
    """Handles credit lifecycle callbacks from CreditRouter.

    Unified callback handler for all phases.

    Callback flow:
        Worker → CreditRouter → CreditCallbackHandler → [count, release slots, dispatch]

    Processing order for credit returns:
        1. Atomic counting (increment_returned)
        2. Track prefill release if TTFT never arrived
        3. Release concurrency slots
        4. Dispatch next turn via timing strategy (if applicable)

    Processing order for TTFT:
        1. Track prefill release (increment_prefill_released)
        2. Release prefill slot

    Phase Registration:
        PhaseRunner calls register_phase() BEFORE any credits are sent.
        This ensures callbacks work from the first credit.
    """

    def __init__(
        self,
        concurrency_manager: ConcurrencyManager,
        branch_orchestrator: BranchOrchestrator | None = None,
        session_tree_registry: SessionTreeRegistry | None = None,
        on_warmup_abort: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Initialize callback handler.

        Args:
            concurrency_manager: Manages concurrency slots (shared across phases).
            branch_orchestrator: Optional DAG subagent orchestrator. When
                provided, non-overflow credit returns are offered to
                ``orchestrator.intercept`` before the strategy's
                ``handle_credit_return`` is called. If intercept returns True
                the strategy dispatch is suppressed (the orchestrator has taken
                over the next-turn path by spawning children / queuing a join
                turn). Context-overflow terminals skip intercept entirely so
                ``on_root_terminal`` / overflow recycle / warmup-failure
                handling still run.
            session_tree_registry: Optional per-tree session-slot ledger (agentic
                replay only). When engaged (PROFILING), the depth-0 root session
                slot is NOT released on the root's terminal return -- it is held
                until the whole tree drains. The release is deferred to
                ``registry.on_root_terminal`` (after intercept for authored-final
                turns, so final-turn spawns are counted first); terminal means
                the authored final turn or a context-overflow early abort. The
                per-phase teardown releases any still-open trees via the
                runner's ``release_all``.
            on_warmup_abort: Optional async callback fired ONCE on the first
                terminal WARMUP failure. Used by agentic replay to abort the run
                early (broadcast ProfileCancelCommand) instead of letting warmup
                drain to teardown -- a degraded trajectory pool means PROFILING
                must not start, so there is no value in waiting for the rest of
                the warmup credits to return. ``None`` -> legacy teardown-time
                abort via the strategy's ``report_warmup_failures`` only.
        """
        self._concurrency_manager = concurrency_manager
        self._branch_orchestrator = branch_orchestrator
        self._session_tree_registry = session_tree_registry
        self._on_warmup_abort = on_warmup_abort
        self._warmup_abort_triggered = False
        self._phase_handlers: dict[PhaseRuntimeKey, PhaseCallbackContext] = {}

    @property
    def on_warmup_abort(self) -> Callable[[], Awaitable[None]] | None:
        """The wired live warmup-abort callback, or None if not enabled.

        When non-None, the live path (first terminal warmup failure ->
        ProfileCancelCommand) is authoritative and PhaseRunner skips its
        teardown ``report_warmup_failures`` raise (which is only a backstop for
        the un-wired case).
        """
        return self._on_warmup_abort

    def _tree_registry_engaged(self, credit: Credit) -> bool:
        """True when per-tree session-slot accounting owns this credit's slot.

        Ownership is determined by whether the registry has an open tree for
        this credit, not by phase name. Accelerated agentic warmup opens trees
        during WARMUP, while ordinary one-shot warmup does not.
        """
        return (
            self._session_tree_registry is not None
            and self._session_tree_registry.has_tree(
                credit.effective_root_correlation_id
            )
        )

    def set_branch_orchestrator(self, orchestrator: BranchOrchestrator | None) -> None:
        """Inject the subagent orchestrator post-construction.

        Also registers a drain observer on the orchestrator so the deferred
        completion check fires when the orchestrator's last drain step lands
        AFTER the final ``on_credit_return`` callback (concurrency race:
        under N>1, ``has_pending_branch_work()`` can flip False between
        credit returns, with no further return arriving to re-trigger the
        check). Without this hook the phase runner relies on the pre-wait
        short-circuit + drain-timeout backstop; the drain timeout cost is
        avoided here.
        """
        if (
            self._branch_orchestrator is not None
            and self._branch_orchestrator is not orchestrator
        ):
            self._branch_orchestrator.set_drain_observer(None)
        self._branch_orchestrator = orchestrator
        if orchestrator is not None:
            orchestrator.set_drain_observer(self._on_orchestrator_drain)

    def _on_orchestrator_drain(self) -> None:
        """Re-evaluate the deferred all-credits-returned check across every
        active phase handler. Idempotent: per-handler check no-ops if the
        event is already set or the predicate disagrees.
        """
        for handler in self._phase_handlers.values():
            if handler.lifecycle.is_complete:
                continue
            if (
                self._branch_orchestrator is not None
                and not handler.progress.all_credits_returned_event.is_set()
                and handler.progress.check_all_returned_or_cancelled()
                and not self._branch_orchestrator.has_pending_branch_work()
            ):
                handler.progress.all_credits_returned_event.set()

    def _dag_work_pending(self, credit: Credit) -> bool:
        """True iff the orchestrator has work in flight or will spawn on
        this credit return (so the all-credits-returned signal must defer
        until after ``intercept`` runs).

        ``intercept`` runs at every ``agent_depth`` (nested DAGs are
        supported), so the branch-id lookup must run at every depth too.
        Restricting it to root credits previously let nested grandchildren
        be truncated when their parent's return was the final outstanding
        credit at that moment. ``has_pending_branch_work`` short-circuits
        before the (potentially expensive) ``get_branch_ids`` walk, and any
        exception from that walk degrades to False so the credit-return
        callback keeps running for every credit.
        """
        if self._branch_orchestrator is None:
            return False
        if self._branch_orchestrator.has_pending_branch_work():
            return True
        try:
            if self._branch_orchestrator.get_branch_ids(credit):
                return True
        except Exception:
            return False
        return False

    @staticmethod
    def _phase_key(
        phase: CreditPhase, phase_index: int | None = None
    ) -> PhaseRuntimeKey:
        return phase_index if phase_index is not None else phase

    def register_phase(
        self,
        *,
        phase: CreditPhase,
        phase_index: int | None = None,
        progress: PhaseProgressTracker,
        lifecycle: PhaseLifecycle,
        stop_checker: StopConditionChecker,
        strategy: TimingStrategyProtocol,
    ) -> None:
        """Register phase for callback handling.

        Called by PhaseRunner BEFORE phase execution starts.
        Must be called before any credits are sent for this phase.

        Args:
            phase: Phase enum (WARMUP or PROFILING).
            progress: Progress tracker for counting.
            lifecycle: Phase lifecycle for state checks.
            stop_checker: Evaluates stop conditions.
            strategy: Timing strategy for dispatching next turns.
        """
        handle_credit_result = getattr(strategy, "handle_credit_result", None)
        handle_first_token = getattr(strategy, "handle_first_token", None)
        key = self._phase_key(phase, phase_index)
        self._phase_handlers[key] = PhaseCallbackContext(
            progress=progress,
            lifecycle=lifecycle,
            stop_checker=stop_checker,
            strategy=strategy,
            concurrency_manager=self._concurrency_manager,
            handle_credit_result=handle_credit_result
            if inspect.iscoroutinefunction(handle_credit_result)
            else None,
            handle_first_token=handle_first_token
            if inspect.iscoroutinefunction(handle_first_token)
            else None,
        )
        _logger.debug(
            lambda: f"Registered callback handler for phase {phase} key={key}"
        )

    async def _handle_warmup_failure(
        self,
        credit: Credit,
        credit_return: CreditReturn,
        handler: PhaseCallbackContext,
        phase: CreditPhase,
    ) -> None:
        """Accumulate, and live-abort on, a terminal WARMUP root failure.

        AgenticReplayStrategy exposes ``record_warmup_failure(trace_id)``;
        PhaseRunner calls ``report_warmup_failures()`` at WARMUP teardown to abort
        PROFILING if any trajectory burned its only warmup credit on a terminal
        error or cancellation. Duck-typed: only fires when the active strategy
        implements the hook, so non-replay strategies are unaffected.

        Do NOT gate on ``credit.is_final_turn``: a WARMUP credit primes the single
        turn k_i (the last request before t*), and PROFILING resumes the same
        trajectory at k_i+1, so for a session active at t* the warmed turn is never
        the trajectory's final turn (k_i < num_turns-1) and ``is_final_turn`` is
        False. WARMUP dispatches exactly one credit per session and its return is a
        strategy-level no-op, so every WARMUP root return IS the terminal warmup
        event for that trajectory -- gating on ``is_final_turn`` made this
        accumulation dead for the entire normal warmup population, silently letting
        a degraded pool proceed to PROFILING.

        When ``on_warmup_abort`` is wired, the FIRST terminal failure also aborts
        the run live (broadcast ProfileCancelCommand) instead of waiting the full
        warmup drain for the teardown-time ``report_warmup_failures`` raise: a
        single failure already breaks the contract, and the broadcast cancels
        in-flight warmup and drives a clean records-manager + system-controller
        shutdown. Fired at most once; on broadcast failure the trigger flag resets
        so a later return retries and the teardown backstop can still surface it.
        """
        if not (
            phase == CreditPhase.WARMUP
            and credit.agent_depth == 0
            and (credit_return.error is not None or credit_return.cancelled)
        ):
            return
        record_warmup_failure = getattr(handler.strategy, "record_warmup_failure", None)
        if record_warmup_failure is None:
            return
        record_warmup_failure(credit.conversation_id)
        if self._on_warmup_abort is None or self._warmup_abort_triggered:
            return
        self._warmup_abort_triggered = True
        _logger.warning(
            lambda: f"Terminal warmup failure for trace {credit.conversation_id}; "
            f"aborting run early (broadcasting ProfileCancelCommand)."
        )
        try:
            await self._on_warmup_abort()
        except Exception as exc:
            # Mirror the records-side threshold abort: if the broadcast fails,
            # reset the flag so a later warmup return retries, and the runner's
            # teardown backstop can still surface the failure.
            self._warmup_abort_triggered = False
            _logger.warning(
                lambda exc=exc: f"Failed to broadcast warmup abort: {exc!r}"
            )

    async def on_credit_return(
        self, worker_id: str, credit_return: CreditReturn
    ) -> None:
        """Handle credit return from worker.

        Processing order:
        1. Atomic counting (increment_returned)
        2. Track prefill release if TTFT never arrived
        3. Release concurrency slots
        4. Dispatch next turn via strategy (if applicable)

        Args:
            worker_id: ID of the worker returning the credit.
            credit_return: Return details including credit and status.
        """
        credit = credit_return.credit
        phase = credit.phase
        key = self._phase_key(phase, credit.phase_index)

        # Get phase handler (returns None if phase already cleaned up)
        handler = self._phase_handlers.get(key)
        if not handler:
            _logger.debug(
                lambda: f"Credit return for unregistered phase {phase} key={key}, "
                f"credit_id={credit.id}, worker={worker_id}"
            )
            return

        # Late arrivals after phase complete are logged but don't affect counts
        if handler.lifecycle.is_complete:
            _logger.warning(
                lambda: f"Credit return after phase {phase} key={key} complete, "
                f"credit_id={credit.id}, worker={worker_id}"
            )
            return

        handler.progress.observe_streaming_content(
            credit_return.last_streaming_content_ns
        )
        # 1. ATOMIC COUNTING (no await before this!)
        # DAG children are off the phase's planning books — they inherit
        # the root's session slot and are tracked by the
        # ``BranchOrchestrator``. Their returns are signalled via the
        # ``on_child_*`` hooks below; passing ``is_child=True`` keeps
        # ``requests_completed`` / ``requests_cancelled`` root-only.
        is_final_returned = handler.progress.increment_returned(
            credit.is_final_turn,
            credit_return.cancelled,
            errored=credit_return.error is not None,
            is_child=credit.agent_depth > 0,
        )

        # 2. Track prefill release if TTFT never arrived
        if not credit_return.first_token_sent:
            handler.progress.increment_prefill_released()

        # 3. Release concurrency slots
        self._release_slots_for_return(
            key, credit, credit_return, is_final_returned, handler
        )

        # 4. Signal completion if this was the final return. Deferred for
        # DAG runs: if the orchestrator already has pending descendants in
        # flight, or if this credit's intercept will spawn fresh children,
        # more credits will be sent/returned. We set the event only after
        # the orchestrator has confirmed no more work (see the post-intercept
        # guard below).
        if is_final_returned and not self._dag_work_pending(credit):
            handler.progress.all_credits_returned_event.set()

        if handler.handle_credit_result is not None:
            await handler.handle_credit_result(credit_return)

        # 4b. DAG child completion hook.
        # When a child session's final turn returns, notify the orchestrator so
        # it can decrement join refcounts, release sticky-routing entries, and
        # dispatch the parent's join turn (if any). Runs regardless of whether
        # the phase can still send, because children may finish after the
        # parent has already sent its terminal turn.
        # NOTE: credit_return.error is a free-form string produced by the
        # worker's transport/server error path. We treat any non-None value as
        # an error signal; cancellation is tracked separately via
        # credit_return.cancelled and is NOT treated as a child error.
        if (
            credit.is_final_turn
            and credit.agent_depth > 0
            and self._branch_orchestrator is not None
        ):
            try:
                if credit_return.error is not None:
                    await self._branch_orchestrator.on_child_errored(
                        credit.x_correlation_id
                    )
                else:
                    await self._branch_orchestrator.on_child_leaf_reached(
                        credit.x_correlation_id
                    )
            except Exception as exc:
                _logger.warning(
                    lambda exc=exc: f"BranchOrchestrator child-completion "
                    f"hook failed for x_correlation_id="
                    f"{credit.x_correlation_id}: {exc}"
                )

        observe_credit_return = getattr(handler.strategy, "observe_credit_return", None)
        if observe_credit_return is not None:
            observe_credit_return(credit)

        # 5. Dispatch next turn / DAG spawn.
        #
        # Context-overflow on a non-final turn is trajectory death (agentic
        # replay treats it as terminal). Detect it BEFORE honoring intercept
        # suspension: a gated next turn must not early-return past
        # ``on_root_terminal`` / strategy overflow recycle /
        # ``_handle_warmup_failure``. Skip intercept entirely on overflow
        # terminal -- spawning or suspending a dead parent is wrong.
        overflow_terminal = (
            not credit.is_final_turn
            and credit_return.error is not None
            and is_context_overflow_response(body=credit_return.error)
        )
        root_terminal = credit.is_final_turn or overflow_terminal

        # The orchestrator intercept runs FIRST (when not overflow-terminal)
        # and unconditionally (not gated behind ``can_send_any_turn``), because
        # when a DAG root finishes its own terminal turn the phase's "sending
        # complete" lifecycle flag has already flipped — but the children still
        # need to dispatch. The orchestrator owns its own dispatch path
        # (``CreditIssuer.dispatch_first_turn``) which bypasses the
        # session-level stop checks for DAG children (they inherit the root's
        # session slot).
        #
        # Strategy dispatch (for regular multi-turn continuation) remains gated
        # behind ``can_send_any_turn`` as before. Normal gated suspend
        # (intercept True, non-overflow) still early-returns past strategy
        # dispatch, but WARMUP failure accounting must still run: accelerated
        # cache-pressure warmup enables DAG intercept, and a non-overflow
        # error on a gated root would otherwise skip ``record_warmup_failure``.
        if self._branch_orchestrator is not None and not overflow_terminal:
            intercepted = await self._branch_orchestrator.intercept(credit)
            if intercepted:
                await self._handle_warmup_failure(credit, credit_return, handler, phase)
                self._finish_return_processing(handler)
                return

        # Per-tree slot release: a root's terminal return marks its tree's root
        # token complete. Terminal means either the authored final turn OR a
        # context-overflow early abort (non-final turn whose error body matches
        # the AgentX allowlist -- agentic_replay treats that as trajectory
        # death and must not leave root_pending stuck True). Run AFTER intercept
        # so any children spawned on the final turn are already registered with
        # the registry (else the tree could drain a beat too early). The
        # registry releases the session slot and recycles the freed lane only
        # once every descendant has also drained -- which may be now (no
        # outstanding descendants) or later (when the last background subagent
        # finishes). Overflow-terminal skips intercept above so this path is
        # always reached for that case; authored-final + gated suspend still
        # early-returns (next turn gated implies not authored-final).
        if (
            root_terminal
            and credit.agent_depth == 0
            and self._tree_registry_engaged(credit)
        ):
            self._session_tree_registry.on_root_terminal(
                credit.effective_root_correlation_id
            )

        # Strategy dispatch (queue next turn of the same session). Normally
        # gated behind ``can_send_any_turn``; however, for DAG-spawned
        # descendants (``credit.agent_depth > 0``) the next turn is gated
        # behind ``can_send_child_turn`` instead — the phase-level
        # sending-complete flag is driven by root sampling exhaustion, not
        # by DAG work, but the global ``--request-count`` cap still
        # applies. In terminal phases, a blocked non-final child notifies the
        # orchestrator (``on_child_stopped``) so the parent's join can drain.
        # Strategies requesting stopped returns (such as quota warmup) instead
        # preserve resumable children across a phase handoff. Final-turn child
        # returns are always passed through (the strategy is a no-op for them,
        # but observer hooks still need to fire).
        wants_stopped_returns = (
            getattr(handler.strategy, "wants_returns_after_sending_complete", False)
            is True
        )
        is_child = credit.agent_depth > 0
        if not is_child:
            if handler.stop_checker.can_send_any_turn() or wants_stopped_returns:
                await handler.strategy.handle_credit_return(
                    credit, error=credit_return.error
                )
        elif (
            credit.is_final_turn
            or handler.stop_checker.can_send_child_turn()
            or wants_stopped_returns
        ):
            await handler.strategy.handle_credit_return(
                credit, error=credit_return.error
            )
        elif self._branch_orchestrator is not None:
            try:
                await self._branch_orchestrator.on_child_stopped(
                    credit.x_correlation_id
                )
            except Exception as exc:
                _logger.warning(
                    lambda exc=exc: f"BranchOrchestrator on_child_stopped "
                    f"hook failed for x_correlation_id="
                    f"{credit.x_correlation_id}: {exc}"
                )

        # WARMUP terminal-failure handling: accumulate, and live-abort on, a
        # terminal WARMUP root failure.
        await self._handle_warmup_failure(credit, credit_return, handler, phase)

        # Deferred all-credits-returned check. Runs on EVERY return — root
        # or child — because child returns don't bump the phase counters
        # (they're tracked by the BranchOrchestrator, not ``CreditCounter``)
        # and so can't flip ``is_final_returned`` themselves. The last
        # child's evict-and-drain cascade is what clears
        # ``has_pending_branch_work``, at which point this check on the
        # child's own return path fires the event.
        self._finish_return_processing(handler)

    def _finish_return_processing(self, handler: PhaseCallbackContext) -> None:
        """Run completion and global-idle checks after return-driven dispatch."""
        self._signal_all_credits_returned_if_ready(handler)
        enforce_system_idle_cap = getattr(
            handler.strategy, "enforce_system_idle_cap", None
        )
        if enforce_system_idle_cap is not None:
            enforce_system_idle_cap(handler.progress.in_flight)

    def _signal_all_credits_returned_if_ready(
        self, handler: PhaseCallbackContext
    ) -> None:
        """Complete a drained phase, preserving paused warmup DAG state.

        Accelerated cache-pressure warmup hands the live (paused) DAG branches
        off to profiling, so once issuance has stopped it completes on
        ``in_flight == 0`` alone -- without waiting for the orchestrator's
        pending branch work to drain (that work IS the handoff payload).
        """
        allows_pending_branch_handoff = (
            getattr(
                handler.strategy,
                "allows_pending_branch_handoff_after_sending_complete",
                False,
            )
            is True
            and handler.lifecycle.is_sending_complete
        )
        all_wire_requests_returned = (
            handler.progress.in_flight == 0
            if allows_pending_branch_handoff
            else handler.progress.check_all_returned_or_cancelled()
        )
        if (
            self._branch_orchestrator is not None
            and not handler.progress.all_credits_returned_event.is_set()
            and all_wire_requests_returned
            and (
                allows_pending_branch_handoff
                or not self._branch_orchestrator.has_pending_branch_work()
            )
        ):
            handler.progress.all_credits_returned_event.set()

    def _release_slots_for_return(
        self,
        phase: PhaseRuntimeKey,
        credit: Credit,
        credit_return: CreditReturn,
        is_final_returned: bool,
        handler: PhaseCallbackContext,
    ) -> None:
        """Release slots based on credit state.

        Slot release rules:
        - Session slot: Released when conversation ends (final turn)
        - Prefill slot: Released if TTFT never arrived (error/cancellation path)
        - On final return: Cleanup in-flight sessions

        Args:
            phase: Credit phase.
            credit: The returned credit.
            credit_return: Return details.
            is_final_returned: True if this is the last credit of the phase.
            handler: Phase callback context.
        """
        concurrency = handler.concurrency_manager
        tree_engaged = self._tree_registry_engaged(credit)

        # Release session slot when a root conversation ends (final turn,
        # whether completed or cancelled). DAG children (agent_depth > 0)
        # inherit the root's session slot via ``issue_credit``'s is_child
        # bypass and therefore never acquired one of their own; releasing
        # here would underflow the session semaphore.
        #
        # Under per-tree accounting the slot belongs to the whole TREE, not the
        # root credit: do NOT release on the root's final turn here. The release
        # is deferred to ``registry.on_root_terminal`` (called after intercept,
        # so children spawned on the final turn are counted first), which frees
        # the slot only once every descendant has also drained.
        if credit.is_final_turn and credit.agent_depth == 0:
            root_corr = credit.effective_root_correlation_id
            if not (tree_engaged and self._session_tree_registry.has_tree(root_corr)):
                concurrency.release_session_slot(phase)

        # On phase end, release slots for sessions still in flight.
        # These are sessions that started but whose final turn was never sent/returned.
        # Under per-tree accounting the registry owns every session slot, so the
        # runner releases any still-open trees at phase cleanup (release_all);
        # releasing here would race the registry and over-release.
        if is_final_returned and not tree_engaged:
            in_flight = handler.progress.in_flight_sessions
            if in_flight > 0:
                _logger.debug(
                    lambda: f"Releasing {in_flight} in-flight session slots for phase {phase}"
                )
                for _ in range(in_flight):
                    concurrency.release_session_slot(phase)

        # Prefill slot is normally released on TTFT. If the request failed or was
        # cancelled before first token, we release here to prevent slot leaks.
        if not credit_return.first_token_sent:
            concurrency.release_prefill_slot(phase)

    async def on_streaming_content(self, message: StreamingContent) -> None:
        """Retain in-flight liveness before a missing return can force completion."""
        handler = self._phase_handlers.get(
            self._phase_key(message.phase, message.phase_index)
        )
        if handler is not None and not handler.lifecycle.is_complete:
            handler.progress.observe_streaming_content(message.timestamp_ns)

    async def on_first_token(self, first_token: FirstToken) -> None:
        """Handle first token event (TTFT) from worker.

        Releases prefill concurrency slot, allowing another request
        to start prefilling.

        Args:
            first_token: TTFT event details including credit_id and phase.
        """
        phase = first_token.phase
        key = self._phase_key(first_token.phase, first_token.phase_index)
        handler = self._phase_handlers.get(key)

        if not handler:
            _logger.debug(
                lambda: f"TTFT for unregistered phase {phase}, "
                f"credit_id={first_token.credit_id}"
            )
            return

        if handler.handle_first_token is not None:
            await handler.handle_first_token(first_token)

        # Track the release
        handler.progress.increment_prefill_released()

        # Release the prefill slot
        handler.concurrency_manager.release_prefill_slot(key)
