# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase orchestrator for credit phase execution.

The orchestrator handles all orchestration concerns:
- Lifecycle management (init, start, stop)
- Phase execution loop (creates PhaseRunner per phase)
- Cancellation

The actual timing logic is delegated to a pluggable TimingMode (created per-phase).
Credit callbacks are handled by CreditCallbackHandler (registered directly with router).
Progress reporting is delegated to PhaseRunner.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from aiperf.common.enums import CacheBustTarget, CreditPhase
from aiperf.common.hooks import on_init, on_start, on_stop
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.credit.callback_handler import CreditCallbackHandler
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, TimingMode
from aiperf.timing.concurrency import ConcurrencyManager
from aiperf.timing.conversation_source import ConversationSource
from aiperf.timing.phase.runner import PhaseRunner
from aiperf.timing.request_cancellation import RequestCancellationSimulator
from aiperf.timing.session_tree import SessionTreeRegistry
from aiperf.timing.trajectory_source import TrajectorySource
from aiperf.timing.url_samplers import URLSelectionStrategyProtocol

if TYPE_CHECKING:
    from aiperf.common.models import DatasetMetadata
    from aiperf.config.resolution.plan import BenchmarkRun
    from aiperf.credit.sticky_router import CreditRouterProtocol
    from aiperf.timing.config import TimingConfig
    from aiperf.timing.phase.publisher import PhasePublisher


class PhaseOrchestrator(AIPerfLifecycleMixin):
    """Orchestrates credit phase execution (warmup → profiling).

    The orchestrator handles:
    - Component composition (ConversationSource, ConcurrencyManager, CancellationPolicy)
    - Lifecycle hooks (@on_init, @on_start)
    - Phase execution loop (creates PhaseRunner per phase)
    - Cancellation

    The orchestrator does NOT handle:
    - Credit callbacks (handled by CreditCallbackHandler, registered directly with router)
    - Per-phase lifecycle (handled by PhaseRunner)

    The TimingMode (created per-phase by PhaseRunner) handles:
    - Timing logic (execute_phase)
    - Dispatching subsequent turns on credit return (handle_credit_return)

    ```
    Architecture (Simplified)
    =========================

    TimingManager
        └── PhaseOrchestrator
                │
                │ owns (long-lived, shared across phases):
                ├── ConcurrencyManager
                ├── CancellationPolicy
                ├── ConversationSource
                └── CreditCallbackHandler ──► registered with CreditRouter
                │
                │ creates per phase:
                └── PhaseRunner ──► TimingMode
                        │
                        ├── LoopScheduler (SINGLE owner)
                        ├── PhaseLifecycle
                        ├── PhaseProgressTracker ──► CreditCounter
                        ├── StopConditionChecker
                        └── CreditIssuer

    Callback Flow (direct, no orchestrator in middle):
        Worker ──► CreditRouter ──► CreditCallbackHandler ──► [count, release slots, dispatch]
    ```
    """

    def __init__(
        self,
        *,
        config: TimingConfig,
        phase_publisher: PhasePublisher,
        credit_router: CreditRouterProtocol,
        dataset_metadata: DatasetMetadata,
        run: BenchmarkRun | None = None,
        **kwargs,
    ) -> None:
        """Initialize timing strategy and orchestration components.

        Args:
            config: Timing configuration (phases, limits, etc.)
            phase_publisher: Publishes phase events to message bus
            credit_router: Routes credits to workers
            dataset_metadata: Dataset for conversation sampling
            run: Full ``BenchmarkRun`` for strategies that need it (e.g.
                AgenticReplayStrategy reads ``run.cfg.get_cache_bust_target()``
                and ``run.benchmark_id``). Optional; strategies that don't need
                it ignore the value.
        """
        super().__init__(**kwargs)
        self._config = config
        self._phase_publisher = phase_publisher
        self._credit_router = credit_router
        self._dataset_metadata = dataset_metadata
        self._run = run

        # Create dataset sampler
        SamplerClass = plugins.get_class(
            PluginType.DATASET_SAMPLER,
            self._dataset_metadata.sampling_strategy,
        )
        # Only root conversations are sampled by the strategy. DAG
        # children belong to their root's session and are dispatched by
        # the BranchOrchestrator on credit return — sampling them as
        # roots would create duplicate root sessions. Filter on
        # ``is_root`` rather than ``agent_depth == 0`` so SPAWN-mode
        # children (which keep ``agent_depth == 0`` for fresh-context
        # semantics but carry ``is_root=False``) are also excluded.
        root_conv_ids = [
            c.conversation_id
            for c in self._dataset_metadata.conversations
            if getattr(c, "is_root", True)
        ]
        self._dataset_sampler = SamplerClass(
            conversation_ids=root_conv_ids
            or [c.conversation_id for c in self._dataset_metadata.conversations]
        )

        # Long-lived components (shared across phases).
        # AGENTIC_REPLAY needs trajectories built once at orchestrator-
        # construction time so trajectory state survives the WARMUP ->
        # PROFILING boundary; it also needs a per-session-TREE slot ledger
        # (a session = root + every subagent it spawns; the slot is held
        # until the whole tree drains, giving exactly-N concurrency). Other
        # timing modes (including dag_jsonl) keep the plain ConversationSource
        # + legacy per-root-credit slot release (registry stays None).
        is_agentic_replay = any(
            pc.timing_mode == TimingMode.AGENTIC_REPLAY for pc in config.phase_configs
        )
        benchmark_id = run.benchmark_id if run is not None else "unknown"
        cache_bust_target = (
            run.cfg.get_cache_bust_target() if run is not None else CacheBustTarget.NONE
        )
        if is_agentic_replay:
            if config.concurrency is None:
                raise ValueError(
                    "AGENTIC_REPLAY timing mode requires concurrency to be set on "
                    "TimingConfig (sourced from the profiling phase concurrency)."
                )
            profiling = next(
                (
                    pc
                    for pc in config.phase_configs
                    if pc.phase == CreditPhase.PROFILING
                ),
                None,
            )
            self._conversation_source = TrajectorySource(
                dataset_metadata=self._dataset_metadata,
                dataset_sampler=self._dataset_sampler,
                concurrency=config.concurrency,
                random_seed=config.random_seed if config.random_seed is not None else 0,
                start_min_ratio=config.trajectory_start_min_ratio,
                start_max_ratio=config.trajectory_start_max_ratio,
                allow_dataset_wrap=config.allow_dataset_wrap,
                cache_bust_enabled=config.cache_bust_enabled,
                expected_num_sessions=(
                    profiling.expected_num_sessions if profiling is not None else None
                ),
                total_expected_requests=(
                    profiling.total_expected_requests if profiling is not None else None
                ),
                expected_duration_sec=(
                    profiling.expected_duration_sec if profiling is not None else None
                ),
            )
        else:
            self._conversation_source = ConversationSource(
                self._dataset_metadata,
                self._dataset_sampler,
                benchmark_id=benchmark_id,
                cache_bust_target=cache_bust_target,
            )
        self._concurrency_manager = ConcurrencyManager()
        self._session_tree_registry = (
            SessionTreeRegistry(self._concurrency_manager)
            if is_agentic_replay
            else None
        )
        self._cancellation_policy = RequestCancellationSimulator(
            config.request_cancellation
        )

        # URL sampler for multi-URL load balancing (None if single URL)
        self._url_sampler: URLSelectionStrategyProtocol | None = None
        if len(config.urls) > 1:
            StrategyClass = plugins.get_class(
                PluginType.URL_SELECTION_STRATEGY, config.url_selection_strategy
            )
            self._url_sampler = StrategyClass(urls=config.urls)

        # Callback handler registered directly with router (no orchestrator in middle)
        self._callback_handler = CreditCallbackHandler(
            self._concurrency_manager,
            session_tree_registry=self._session_tree_registry,
            on_warmup_abort=self._phase_publisher.publish_profile_cancel,
        )
        self._credit_router.set_return_callback(self._callback_handler.on_credit_return)
        self._credit_router.set_first_token_callback(
            self._callback_handler.on_first_token
        )
        self._credit_router.set_streaming_content_callback(
            self._callback_handler.on_streaming_content
        )

        # Phase configuration
        self._ordered_phase_configs = config.phase_configs

        # Active phase runners (for cancellation) - multiple possible with seamless mode
        self._active_runners: list[PhaseRunner] = []

    @property
    def conversation_source(self) -> ConversationSource:
        """Conversation source for dataset access."""
        return self._conversation_source

    @on_init
    async def _init_orchestrator(self) -> None:
        """Log configured phases (actual initialization happens per-phase in _execute_phases)."""
        self.info(
            lambda: (
                f"Initialized {len(self._ordered_phase_configs)} phase(s): "
                f"{[p.phase.replace('_', ' ').title() for p in self._ordered_phase_configs]}"
            )
        )

    @on_start
    async def _start_orchestrator(self) -> None:
        """Execute all phases and publish completion when done."""
        self.debug(lambda: "Starting PhaseOrchestrator")

        try:
            # Execute all phases sequentially (each PhaseRunner handles its own progress reporting)
            await self._execute_phases()
        finally:
            # Cleanup
            self.notice("All credits completed")
            self._credit_router.mark_credits_complete()
            await self._phase_publisher.publish_credits_complete()

    async def _execute_phases(self) -> None:
        """Execute phases in order (typically: warmup → profiling).

        For each phase:
        1. Create PhaseRunner with conversation_source
        2. Execute phase via runner.run() (runner creates timing strategy internally)
        3. Runner handles setup, execution, and cleanup

        Seamless Mode:
            With seamless=True, a phase can start before the previous phase
            completes waiting for returns. This allows smooth phase transitions
            without gaps in request issuance. Multiple runners may be active
            simultaneously (old phase waiting for returns while new phase sends).
        """
        for i, phase_config in enumerate(self._ordered_phase_configs):
            is_final_phase = i == len(self._ordered_phase_configs) - 1
            is_seamless_non_final = phase_config.seamless and not is_final_phase

            runner = PhaseRunner(
                config=phase_config,
                conversation_source=self._conversation_source,
                phase_publisher=self._phase_publisher,
                credit_router=self._credit_router,
                concurrency_manager=self._concurrency_manager,
                cancellation_policy=RequestCancellationSimulator(
                    phase_config.request_cancellation
                ),
                callback_handler=self._callback_handler,
                url_selection_strategy=self._url_sampler,
                run=self._run,
                session_tree_registry=self._session_tree_registry,
            )

            # For seamless non-final phases, set callback to remove from active runners
            # when background return wait completes
            if is_seamless_non_final:
                runner.set_phase_complete_callback(
                    self._phase_runner_cleanup_callback(runner)
                )

            # Track active runner (multiple possible with seamless mode)
            self._active_runners.append(runner)

            try:
                # Execute phase (runner.run() returns after sending complete for seamless,
                # or after all returns complete for non-seamless/final phases)
                await runner.run(is_final_phase=is_final_phase)
            except Exception as e:
                self.error(f"Error executing phase {runner.phase}: {e!r}")
                await self.cancel()
                raise e

            # Remove from active runners when fully complete
            # For seamless phases, this happens after returns complete (background task)
            if not is_seamless_non_final:
                self._active_runners.remove(runner)

    def _phase_runner_cleanup_callback(self, runner: PhaseRunner) -> Callable[[], None]:
        """Create callback that removes runner from active list when phase completes."""

        def cleanup() -> None:
            if runner in self._active_runners:
                self._active_runners.remove(runner)
                self.debug(f"Removed completed runner for phase {runner.phase}")

        return cleanup

    async def cancel(self) -> None:
        """Cancel the orchestrator gracefully.

        Stops issuing new credits and cancels in-flight requests.
        Called when user requests cancellation (e.g., Ctrl+C).
        """
        self.warning("Cancelling phase orchestrator")

        # Cancel all in-flight credits first
        await self._credit_router.cancel_all_credits()

        self._cancel_active_runners()

    @on_stop
    async def _stop_orchestrator(self) -> None:
        """Clean up orchestrator state on normal stop.

        Cancels any still-active phase runners. Without this hook, runners
        tracked in ``_active_runners`` are leaked on the non-cancellation
        shutdown path (only ``cancel()`` cleaned them up before, and it is
        only called for Ctrl+C).

        Callback registrations on the credit router are not explicitly
        unregistered: the router is a child lifecycle of ``TimingManager``
        and is torn down alongside the orchestrator, so its callback table
        does not outlive us.
        """
        if self._active_runners:
            self.debug(
                lambda: (
                    f"Stopping orchestrator with {len(self._active_runners)} active runner(s)"
                )
            )
            self._cancel_active_runners()

    def _cancel_active_runners(self) -> None:
        """Cancel every tracked phase runner and clear the active list."""
        for runner in self._active_runners:
            runner.cancel()
            self.debug(f"Cancelled active phase runner for phase {runner.phase}")
        self._active_runners.clear()
