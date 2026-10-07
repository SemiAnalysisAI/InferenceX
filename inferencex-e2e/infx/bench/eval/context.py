"""The contract between the eval dispatcher and each eval framework."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalContext:
    """One eval invocation against a ready OpenAI-compatible server."""

    base_url: str
    """Server root without a trailing slash."""
    model: str
    """Served model name sent in requests."""
    concurrency: int
    context_length: int
    """Prompt-plus-generation token budget; 0 for vendor suites, which fix their own."""
    results_dir: Path
    """Empty directory for the framework's raw artifacts."""
    suite: str | None
    """``EVAL_SUITE``, if set."""
    env: Mapping[str, str]


@dataclass(frozen=True)
class EvalOutcome:
    """What a framework reports back to the dispatcher."""

    returncode: int
    """Shell-style exit status: 128 + N after signal N."""
    suite: str
    """Suite actually run; recorded as ``eval_suite`` in ``meta_env.json``."""
