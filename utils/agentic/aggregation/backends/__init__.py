"""Server metric backend adapters for agentic aggregation."""

from __future__ import annotations

from typing import Any

from .base import ServerMetricsBackend
from .dynamo_vllm import DynamoVllmBackend
from .sglang import SglangBackend
from .vllm import VllmBackend


BACKENDS: tuple[ServerMetricsBackend, ...] = (
    DynamoVllmBackend(),
    SglangBackend(),
    VllmBackend(),
)


def detect_backend(
    metrics: dict[str, dict[str, Any]],
    framework: str,
) -> ServerMetricsBackend | None:
    for backend in BACKENDS:
        if backend.matches(metrics, framework):
            return backend
    return None
