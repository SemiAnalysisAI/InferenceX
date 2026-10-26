"""Public image discovery and private capacity through one bounded HTTP reader."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
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
    "scheduler": PRIVATE + "/api/status/scheduler",
    "benchmarks": PUBLIC + "/api/v1/benchmarks",
    "workflow-info": PUBLIC + "/api/v1/workflow-info",
    "evaluations": PUBLIC + "/api/v1/evaluations",
    "benchmark-siblings": PUBLIC + "/api/v1/benchmark-siblings",
}
USER_AGENT = "InferenceX-Klaud-Cold/1.0"
MAX_BYTES = 16 * 1024 * 1024
RETRY_DELAYS_SECONDS = (0.5, 1.5)
KLAUD_JOB = re.compile(r"(?:^| / )klaud \| ", re.IGNORECASE)


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
        """Refuse every redirect by returning None."""
        return


def reject_nonfinite(value: str) -> Never:  # noqa: ARG001
    """Raise ReadError for a NaN or Infinity constant ``value`` in JSON."""
    raise ReadError("invalid-json-number")


def finite_float(value: str) -> float:
    """Return JSON number text ``value`` as a float, raising ReadError unless finite."""
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
    exact: bool = True,
    benchmark: int | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    opener: Callable[..., Any] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> Feed:
    """GET one ``resource`` from ENDPOINTS and return it as a Feed.

    ``model``, ``date``, ``benchmark`` and ``token`` fill the query or auth header where the
    resource takes them. Retries transient failures, then raises ReadError with a fixed code.
    """
    query = {}
    if resource == "benchmark-siblings":
        if benchmark is None:
            raise ReadError("benchmark-id-required")
        query["id"] = str(benchmark)
    if resource == "benchmarks" and model:
        query["model"] = model
    if resource in ("benchmarks", "workflow-info") and date:
        from datetime import date as calendar_date

        if calendar_date.fromisoformat(date).isoformat() != date:
            raise ReadError("invalid-baseline-date")
        query["date"] = date
        if resource == "benchmarks" and exact:
            query["exact"] = "true"
    url = ENDPOINTS[resource] + ("?" + urllib.parse.urlencode(query) if query else "")
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if resource == "scheduler":
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
    """Return issue codes for a missing, failed or stale ``feed`` at time ``now``."""
    if feed is None:
        return ["feed-unavailable"]
    issues = ["feed-refresh-failed"] if feed.error else []
    age = (utc(now) - utc(feed.retrieved_at)).total_seconds()
    if age < -policy.clock_skew_seconds or age > policy.public_max_age_seconds:
        issues.append("feed-retrieval-stale")
    return issues


def catalog(images: Feed | None, now: datetime, policy: Policy) -> tuple[list[dict], list[str]]:
    """Return one catalog item per public image row plus the sorted feed issue codes."""
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
    """Fetch the public image feed and return its catalog and issues, or raise ReadError."""
    try:
        images = fetch("images")
    except ReadError as error:
        raise ReadError(f"images:{error}") from None
    return catalog(images, datetime.now(UTC), policy)


def fresh(value: Any, now: datetime, policy: Policy) -> bool:
    """Return True if timestamp ``value`` is within the response age limit at ``now``."""
    try:
        return (
            -policy.clock_skew_seconds
            <= (utc(now) - utc(value)).total_seconds()
            <= policy.response_max_age_seconds
        )
    except (ValueError, TypeError, AttributeError):
        return False


def cluster_headroom(feed: Feed, policy: Policy, now: datetime) -> dict[str, int]:
    """Return the idle nodes of each cluster that waiting Klaud jobs have not already requested.

    Clusters a waiting human job could run on, and clusters without headroom, are omitted.
    A queued job without a node count needs one node, and jobs already holding a runner are
    counted in allocatedNodes. Malformed or duplicate scheduler status yields no clusters.
    """
    try:
        raw = feed.payload
        snapshot = raw["data"]["snapshot"]
        snapshot_age = (utc(now) - utc(snapshot["snapshotAt"])).total_seconds()
        if (
            feed.error
            or not fresh(feed.retrieved_at, now, policy)
            or raw["kind"] != "inferencex.status.scheduler"
            or not fresh(raw["generatedAt"], now, policy)
            or raw["data"]["available"] is not True
            or snapshot["stalled"] is not False
            or not -policy.clock_skew_seconds <= snapshot_age <= policy.scheduler_max_age_seconds
        ):
            return {}
        human: set[str] = set()
        klaud: Counter[str] = Counter()
        for job in snapshot["queue"]:
            nodes = 1 if job["nodeCount"] is None else job["nodeCount"]
            labels, runner = job["clusterIds"], job["runnerName"]
            if (
                type(nodes) is not int
                or nodes < 1
                or not isinstance(labels, list)
                or not all(isinstance(label, str) for label in labels)
                or not isinstance(runner, str | None)
            ):
                return {}
            if runner is not None:
                continue
            if KLAUD_JOB.search(job["name"]):
                for label in set(labels):
                    klaud[label] += nodes
            else:
                human.update(labels)
        headroom: dict[str, int] = {}
        seen: set[str] = set()
        for metric in snapshot["packingMetrics"]:
            label = metric["clusterLabel"]
            if not isinstance(label, str) or not label.startswith("cluster:") or label in seen:
                return {}
            seen.add(label)
            schedulable, allocated = metric["schedulableNodes"], metric["allocatedNodes"]
            if type(schedulable) is not int or type(allocated) is not int or allocated < 0:
                continue
            free = schedulable - allocated - klaud[label]
            if label not in human and free > 0:
                headroom[label.removeprefix("cluster:")] = free
        return headroom
    except (KeyError, ValueError, TypeError, AttributeError):
        return {}


def fetch_capacity(policy: Policy) -> dict[str, int]:
    """Return per-cluster idle headroom from the private scheduler status feed."""
    return cluster_headroom(
        fetch("scheduler", token=os.environ.get("KLAUD_DASHBOARD_API_KEY")),
        policy,
        datetime.now(UTC),
    )
