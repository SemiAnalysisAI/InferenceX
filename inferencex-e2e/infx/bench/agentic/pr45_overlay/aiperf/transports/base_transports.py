# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from aiperf.common.environment import Environment
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import (
    RequestInfo,
    RequestRecord,
    SSEMessage,
)
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.common.protocols import AIPerfLifecycleProtocol
from aiperf.common.types import RequestInputT
from aiperf.plugin.schema.schemas import TransportMetadata

FirstTokenCallback = Callable[[int, SSEMessage], Awaitable[bool]]
"""
Callback invoked for each SSE message until it returns True.

Args:
    ttft_ns: elapsed duration from request start to this message
    message: the current SSE message

Returns:
    True to stop observing; False to continue receiving subsequent messages.

The worker uses this to release prefill concurrency once on meaningful content.
Scenario coverage keeps observing afterward to retain the latest content timestamp.
"""


def _remove_headers_case_insensitive(
    headers: dict[str, str], names: tuple[str, ...]
) -> None:
    """Remove every key in ``headers`` that case-insensitively matches ``names``."""
    lowered = {name.lower() for name in names}
    for key in [k for k in headers if k.lower() in lowered]:
        del headers[key]


@runtime_checkable
class TransportProtocol(AIPerfLifecycleProtocol, Protocol):
    """Protocol for a transport that sends requests to an inference server."""

    def __init__(self, **kwargs) -> None: ...

    @classmethod
    def metadata(cls) -> TransportMetadata: ...

    def get_transport_headers(self, request_info: RequestInfo) -> dict[str, str]: ...

    def build_headers(self, request_info: RequestInfo) -> dict[str, str]: ...

    def build_url(self, request_info: RequestInfo) -> str: ...

    def get_url(self, request_info: RequestInfo) -> str: ...

    async def send_request(
        self, request_info: RequestInfo, payload: RequestInputT
    ) -> RequestRecord: ...


class BaseTransport(AIPerfLifecycleMixin, ABC):
    """Base class for all transport protocol implementations.

    Transports handle the protocol layer (HTTP, gRPC, etc.).
    """

    def __init__(self, model_endpoint: ModelEndpointInfo, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model_endpoint: ModelEndpointInfo = model_endpoint
        from aiperf import __version__

        self.user_agent: str = f"aiperf/{__version__}"
        self.base_headers: dict[str, str] = {
            "User-Agent": self.user_agent,
        }

    @classmethod
    @abstractmethod
    def metadata(cls) -> TransportMetadata:
        """Return transport metadata for discovery and registration.

        Returns:
            Metadata describing transport type and supported URL schemes
        """
        ...

    def get_transport_headers(self, request_info: RequestInfo) -> dict[str, str]:
        """Get protocol-specific headers (e.g., Content-Type, Accept).

        Override in subclasses to add transport-specific headers.

        Args:
            request_info: Request context

        Returns:
            Dictionary of transport-specific HTTP headers
        """
        return {}

    def build_headers(self, request_info: RequestInfo) -> dict[str, str]:
        """Compose final headers from universal, endpoint, and transport sources.

        Merges headers in priority order:
        1. Universal headers (User-Agent, correlation IDs)
        2. Endpoint-specific headers (auth, custom)
        3. Transport-specific headers (Content-Type, Accept)

        Args:
            request_info: Request context with endpoint headers

        Returns:
            Complete header dictionary for request
        """
        headers: dict[str, str] = self.base_headers.copy()

        if request_info.x_request_id:
            headers["X-Request-ID"] = request_info.x_request_id
        if request_info.x_correlation_id:
            correlation_header = (
                request_info.model_endpoint.endpoint.session_header
                or "X-Correlation-ID"
            )
            headers[correlation_header] = request_info.x_correlation_id

        headers.update(request_info.endpoint_headers)
        if request_info.turns and request_info.turns[-1].extra_headers:
            headers.update(request_info.turns[-1].extra_headers)
        headers.update(self.get_transport_headers(request_info))

        # Apply derived session-affinity headers last so they are authoritative
        # and cannot be silently overwritten by endpoint, extra, or transport
        # headers. Strip caller-supplied variants case-insensitively first, since
        # HTTP header names are case-insensitive and these merge into a plain dict.
        if request_info.x_correlation_id:
            # Additive (distinct from --session-header, which RENAMES the single
            # correlation header above): some routers require X-Session-ID
            # ALONGSIDE the correlation header for session affinity.
            if Environment.HTTP.X_SESSION_ID_FROM_CORRELATION_ID:
                _remove_headers_case_insensitive(headers, ("X-Session-ID",))
                headers["X-Session-ID"] = request_info.x_correlation_id
            if Environment.HTTP.X_SESSION_AFFINITY_FROM_CORRELATION_ID:
                _remove_headers_case_insensitive(headers, ("X-Session-Affinity",))
                headers["X-Session-Affinity"] = request_info.x_correlation_id
            # Additive SGLang Model Gateway routing key: co-locate a session's
            # requests on one worker via the stable correlation ID.
            if Environment.HTTP.X_SMG_ROUTING_KEY_FROM_CORRELATION_ID:
                _remove_headers_case_insensitive(headers, ("X-SMG-Routing-Key",))
                headers["X-SMG-Routing-Key"] = request_info.x_correlation_id

        # Apply derived Dynamo session headers last so they are authoritative
        # and cannot be overwritten by endpoint, extra, or transport headers.
        # Strip any caller-supplied variants case-insensitively first, since HTTP
        # header names are case-insensitive and these are merged into a plain dict.
        if (
            request_info.x_correlation_id
            and Environment.HTTP.X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID
        ):
            _remove_headers_case_insensitive(
                headers, ("X-Dynamo-Session-ID", "X-Dynamo-Parent-Session-ID")
            )
            headers["X-Dynamo-Session-ID"] = request_info.x_correlation_id
            if request_info.parent_correlation_id:
                headers["X-Dynamo-Parent-Session-ID"] = (
                    request_info.parent_correlation_id
                )

        return headers

    def build_url(self, request_info: RequestInfo) -> str:
        """Build complete URL with query parameters from the request context.

        Preserves existing query params from base URL and merges with
        endpoint-specific params (endpoint params take precedence).

        Args:
            request_info: Request context with endpoint params

        Returns:
            Complete URL with merged query parameters
        """
        base_url = self.get_url(request_info)
        parsed = urlparse(base_url)

        # Parse existing query params from URL
        existing_params = parse_qs(parsed.query, keep_blank_values=True)
        # Flatten from lists to single values (take first)
        params = {k: v[0] if v else "" for k, v in existing_params.items()}

        # Merge endpoint params (these override existing params)
        if request_info.endpoint_params:
            params.update(request_info.endpoint_params)

        # Rebuild URL with merged params
        if params:
            return urlunparse(parsed._replace(query=urlencode(params)))

        return base_url

    @abstractmethod
    def get_url(self, request_info: RequestInfo) -> str:
        """Build base URL without query parameters.

        Args:
            request_info: Request context with model endpoint info

        Returns:
            Base URL for the request
        """
        ...

    @abstractmethod
    async def send_request(
        self,
        request_info: RequestInfo,
        payload: RequestInputT,
        *,
        first_token_callback: FirstTokenCallback | None = None,
    ) -> RequestRecord:
        """Execute request via this transport protocol.

        Args:
            request_info: Request context and metadata
            payload: Request payload (format depends on transport)
            first_token_callback: Optional callback fired on first SSE message with ttft_ns

        Returns:
            Record containing responses, timing, and any errors
        """
        ...
