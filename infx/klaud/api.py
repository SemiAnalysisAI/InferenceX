"""Public image discovery and private capacity through one bounded HTTP reader."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Never

from pydantic import ValidationError

from .models import Feed, Policy, PublicRow, identity, stamp, utc

PUBLIC = "https://inferencex.semianalysis.com"
PRIVATE = "https://dash.inferencex.semianalysis.com"
ENDPOINTS = {
    "images": PUBLIC + "/api/v1/latest-images",
    "releases": PUBLIC + "/api/v1/framework-releases",
    "clusters": PRIVATE + "/api/status/clusters",
    "benchmarks": PUBLIC + "/api/v1/benchmarks",
    "workflow-info": PUBLIC + "/api/v1/workflow-info",
    "evaluations": PUBLIC + "/api/v1/evaluations",
}
USER_AGENT = "InferenceX-Klaud-Cold/1.0"
MAX_BYTES = 16 * 1024 * 1024
RETRY_DELAYS_SECONDS = (0.5, 1.5)


class ReadError(ValueError):
    """Sanitized fixed error code safe for reports."""


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,  # noqa: ARG002
        fp: Any,  # noqa: ARG002
        code: int,  # noqa: ARG002
        msg: str,  # noqa: ARG002
        headers: Any,  # noqa: ARG002
        newurl: str,  # noqa: ARG002
    ) -> None:
        return None


def reject_nonfinite(value: str) -> Never:  # noqa: ARG001
    raise ReadError("invalid-json-number")


def finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ReadError("invalid-json-number")
    return result


def fetch(
    resource: str,
    *,
    token: str | None = None,
    model: str | None = None,
    date: str | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    opener: Callable[..., Any] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> Feed:
    query = {}
    if resource == "benchmarks" and model:
        query["model"] = model
    if resource in ("benchmarks", "workflow-info") and date:
        from datetime import date as calendar_date

        if calendar_date.fromisoformat(date).isoformat() != date:
            raise ReadError("invalid-baseline-date")
        query["date"] = date
        if resource == "benchmarks":
            query["exact"] = "true"
    url = ENDPOINTS[resource] + ("?" + urllib.parse.urlencode(query) if query else "")
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if resource == "clusters":
        if not token or not token.strip():
            raise ReadError("private-status-key-unavailable")
        if any(ord(char) < 33 or ord(char) > 126 for char in token.strip()):
            raise ReadError("private-status-key-invalid")
        headers["Authorization"] = "Bearer " + token.strip()
    request = urllib.request.Request(url, headers=headers, method="GET")  # noqa: S310
    open_request = opener or urllib.request.build_opener(NoRedirects()).open
    for attempt in range(len(RETRY_DELAYS_SECONDS) + 1):
        try:
            with open_request(request, timeout=15) as response:
                if response.status != 200:
                    if (response.status in (408, 429) or response.status >= 500) and attempt < len(
                        RETRY_DELAYS_SECONDS
                    ):
                        sleeper(RETRY_DELAYS_SECONDS[attempt])
                        continue
                    raise ReadError("http-status-error")
                raw = response.read(MAX_BYTES + 1)
                if len(raw) > MAX_BYTES:
                    raise ReadError("response-too-large")
                if "application/json" not in response.headers.get("Content-Type", "").lower():
                    raise ReadError("response-not-json")
                encoding = response.headers.get("Content-Encoding", "").strip().lower()
                if encoding == "gzip":
                    try:
                        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as compressed:
                            decoded = compressed.read(MAX_BYTES + 1)
                    except (gzip.BadGzipFile, EOFError):
                        raise ReadError("invalid-content-encoding") from None
                    if len(decoded) > MAX_BYTES:
                        raise ReadError("response-too-large")
                elif encoding in ("", "identity"):
                    decoded = raw
                else:
                    raise ReadError("unsupported-content-encoding")
                metadata = {
                    key.lower(): response.headers[key]
                    for key in ("Age", "Cache-Control", "Date", "ETag")
                    if key in response.headers
                }
            break
        except urllib.error.HTTPError as error:
            if (error.code in (408, 429) or error.code >= 500) and attempt < len(
                RETRY_DELAYS_SECONDS
            ):
                sleeper(RETRY_DELAYS_SECONDS[attempt])
                continue
            raise ReadError(f"http-{error.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt < len(RETRY_DELAYS_SECONDS):
                sleeper(RETRY_DELAYS_SECONDS[attempt])
                continue
            raise ReadError("network-error") from None
    try:
        payload = json.loads(decoded, parse_constant=reject_nonfinite, parse_float=finite_float)
        if resource == "images" and not isinstance(payload, list):
            raise ReadError("invalid-images-payload")
        if resource == "releases" and (
            not isinstance(payload, dict)
            or any(
                value is not None and (not isinstance(value, str) or not value.strip())
                for value in payload.values()
            )
        ):
            raise ReadError("invalid-releases-payload")
    except (UnicodeError, json.JSONDecodeError):
        raise ReadError("invalid-json") from None
    except ReadError:
        raise
    except ValueError:
        raise ReadError("invalid-response") from None
    return Feed(
        url=url,
        retrieved_at=stamp(clock()),
        sha256=hashlib.sha256(decoded).hexdigest(),
        payload=payload,
        headers=metadata,
    )


def feed_issues(feed: Feed | None, now: datetime, policy: Policy) -> list[str]:
    if feed is None:
        return ["feed-unavailable"]
    issues = ["feed-refresh-failed"] if feed.error else []
    age = (utc(now) - utc(feed.retrieved_at)).total_seconds()
    if age < -policy.clock_skew_seconds or age > policy.public_max_age_seconds:
        issues.append("feed-retrieval-stale")
    # The public API/CDN owns cache freshness. Age is time in a shared cache,
    # not the age of the benchmark data; do not impose a second CDN TTL here.
    return issues


def catalog(images: Feed | None, now: datetime, policy: Policy) -> tuple[list[dict], list[str]]:
    issues = [f"images:{issue}" for issue in feed_issues(images, now, policy)]
    if images is None or not isinstance(images.payload, list):
        return [], [*issues, "images:invalid-or-missing-payload"]
    result = []
    for index, raw in enumerate(images.payload):
        item: dict[str, Any] = {
            "source-index": index,
            "source-id": identity(raw),
            "source": raw,
        }
        try:
            row = PublicRow.model_validate(raw)
        except ValidationError as error:
            item.update(
                {
                    "source-status": "invalid",
                    "invalid-fields": sorted(
                        {str(part["loc"][0]) for part in error.errors() if part["loc"]}
                    ),
                }
            )
        else:
            days = max(0, (utc(now) - utc(f"{row.date}T00:00:00Z")).days)
            item.update(
                {
                    "source-status": "baseline",
                    "benchmark-age-days": days,
                }
            )
        result.append(item)
    return result, sorted(set(issues))


def fetch_catalog(policy: Policy) -> tuple[list[dict], list[str]]:
    try:
        images = fetch("images")
    except ReadError as error:
        raise ReadError(f"images:{error}") from None
    return catalog(images, datetime.now(UTC), policy)


def fresh(value: Any, now: datetime, policy: Policy) -> bool:
    try:
        return (
            -policy.clock_skew_seconds
            <= (utc(now) - utc(value)).total_seconds()
            <= policy.response_max_age_seconds
        )
    except (ValueError, TypeError, AttributeError):
        return False


def available_clusters(feed: Feed, policy: Policy, now: datetime) -> set[str]:
    """Return clusters below 80% node utilization; never publish private status."""
    try:
        raw = feed.payload
        if (
            feed.error
            or not fresh(feed.retrieved_at, now, policy)
            or raw["kind"] != "inferencex.status.clusters"
            or not fresh(raw["generatedAt"], now, policy)
            or raw["data"]["available"] is not True
        ):
            return set()
        # Validate the consumed fields, not schemaVersion: additive API changes
        # must not turn healthy capacity into an empty candidate list.
        available: set[str] = set()
        seen: set[str] = set()
        for cluster in raw["data"]["clusters"]:
            cluster_id = cluster["clusterId"]
            if not isinstance(cluster_id, str) or not cluster_id or cluster_id in seen:
                return set()
            seen.add(cluster_id)
            # The API owns the cluster-age cutoff. Check timestamp validity/order,
            # but do not impose a second cutoff on its current snapshots.
            observed, received = utc(cluster["observedAt"]), utc(cluster["receivedAt"])
            generated = utc(raw["generatedAt"])
            if (
                cluster["stale"] is not False
                or cluster["status"] not in ("operational", "degraded")
                or (observed - received).total_seconds() > policy.clock_skew_seconds
                or (received - generated).total_seconds() > policy.clock_skew_seconds
            ):
                continue
            summary = cluster["summary"]
            total = summary["totalNodes"]
            counts = [
                summary[k]
                for k in (
                    "allocatedNodes",
                    "mixedNodes",
                    "idleNodes",
                    "downNodes",
                    "otherNodes",
                )
            ]
            if (
                type(total) is not int
                or total <= 0
                or any(type(n) is not int or n < 0 for n in counts)
                or sum(counts) != total
            ):
                continue
            # An entirely down/unavailable cluster can also report 0% utilization.
            if (
                summary["idleNodes"] > 0
                and (summary["allocatedNodes"] + summary["mixedNodes"]) * 5 < total * 4
            ):
                available.add(cluster_id)
        return available
    except (KeyError, ValueError, TypeError, AttributeError):
        return set()


def fetch_capacity(policy: Policy) -> set[str]:
    return available_clusters(
        fetch("clusters", token=os.environ.get("KLAUD_DASHBOARD_API_KEY")),
        policy,
        datetime.now(UTC),
    )


def capacity_context(policy: Policy) -> dict:
    """Private routing hints for review, without node counts or raw responses."""
    feed = fetch("clusters", token=os.environ.get("KLAUD_DASHBOARD_API_KEY"))
    available = available_clusters(feed, policy, datetime.now(UTC))
    try:
        clusters = sorted(
            {
                cluster["clusterId"]
                for cluster in feed.payload["data"]["clusters"]
                if isinstance(cluster["clusterId"], str)
            }
        )
    except (KeyError, TypeError):
        clusters = []
    return {
        "telemetry-clusters": clusters,
        "eligible-telemetry-clusters": sorted(available),
    }
