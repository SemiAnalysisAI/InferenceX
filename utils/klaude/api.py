"""Public image discovery and private capacity through one bounded HTTP reader."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request

from pydantic import ValidationError

from .models import Feed, Policy, PublicRow, identity, stamp, utc

PUBLIC = "https://inferencex.semianalysis.com"
PRIVATE = "https://dash.inferencex.semianalysis.com"
ENDPOINTS = {
    "images": PUBLIC + "/api/v1/latest-images",
    "releases": PUBLIC + "/api/v1/framework-releases",
    "clusters": PRIVATE + "/api/status/clusters",
}
USER_AGENT = "InferenceX-Klaud-Cold/1.0"
MAX_BYTES = 16 * 1024 * 1024


class ReadError(ValueError):
    """Sanitized fixed error code safe for reports."""


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def reject_nonfinite(value: str):
    raise ReadError("invalid-json-number")


def finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ReadError("invalid-json-number")
    return result


def fetch(resource: str, *, token: str | None = None,
          model: str | None = None,
          clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc), opener=None) -> Feed:
    url = (PUBLIC + "/api/v1/benchmarks?" + urllib.parse.urlencode({"model": model})
           if resource == "benchmarks" and model else ENDPOINTS[resource])
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if resource == "clusters":
        if not token or not token.strip():
            raise ReadError("private-status-key-unavailable")
        if any(ord(char) < 33 or ord(char) > 126 for char in token.strip()):
            raise ReadError("private-status-key-invalid")
        headers["Authorization"] = "Bearer " + token.strip()
    request = urllib.request.Request(url, headers=headers, method="GET")
    open_request = opener or urllib.request.build_opener(NoRedirects()).open
    try:
        with open_request(request, timeout=15) as response:
            if response.status != 200:
                raise ReadError("http-status-error")
            raw = response.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise ReadError("response-too-large")
            if "application/json" not in response.headers.get("Content-Type", "").lower():
                raise ReadError("response-not-json")
            metadata = {key.lower(): response.headers[key] for key in ("Age", "Cache-Control", "Date", "ETag") if key in response.headers}
        payload = json.loads(raw, parse_constant=reject_nonfinite, parse_float=finite_float)
        if resource == "images" and not isinstance(payload, list):
            raise ReadError("invalid-images-payload")
        if resource == "releases" and (not isinstance(payload, dict) or any(
            value is not None and (not isinstance(value, str) or not value.strip())
            for value in payload.values()
        )):
            raise ReadError("invalid-releases-payload")
    except urllib.error.HTTPError as error:
        raise ReadError(f"http-{error.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise ReadError("network-error") from None
    except (UnicodeError, json.JSONDecodeError):
        raise ReadError("invalid-json") from None
    except ReadError:
        raise
    except ValueError:
        raise ReadError("invalid-response") from None
    return Feed(url=url, retrieved_at=stamp(clock()), sha256=hashlib.sha256(raw).hexdigest(), payload=payload, headers=metadata)


UNSTABLE_MARKERS = ("nightly", "rocm/sgl-dev", "sglang-rocm")


def image_reasons(image: str, release: str | None) -> list[str]:
    reasons = []
    if any(marker in image.lower() for marker in UNSTABLE_MARKERS):
        reasons.append("unstable-image")
    if release is None:
        reasons.append("release-comparison-unknown")
    elif release not in image:
        reasons.append("release-string-mismatch")
    return reasons


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


def catalog(images: Feed | None, releases: Feed | None, now: datetime, policy: Policy) -> tuple[list[dict], list[str]]:
    issues = [f"images:{issue}" for issue in feed_issues(images, now, policy)]
    issues += [f"releases:{issue}" for issue in feed_issues(releases, now, policy)]
    release_map = releases.payload if releases and isinstance(releases.payload, dict) else {}
    if releases is not None and not isinstance(releases.payload, dict):
        issues.append("releases:invalid-payload")
    for value in release_map.values():
        if value is not None and (not isinstance(value, str) or not value.strip()):
            issues.append("releases:invalid-tag")
            release_map = {}
            break
    if images is None or not isinstance(images.payload, list):
        return [], issues + ["images:invalid-or-missing-payload"]
    result = []
    for index, raw in enumerate(images.payload):
        item: dict[str, Any] = {"source-index": index, "source-id": identity(raw), "source": raw}
        try:
            row = PublicRow.model_validate(raw)
        except ValidationError as error:
            item.update({"source-status": "invalid", "review-reasons": ["invalid-public-row"],
                         "invalid-fields": sorted({str(part["loc"][0]) for part in error.errors() if part["loc"]}),
                         "release": None, "needs-review": False})
        else:
            bases = [key for key in release_map if row.framework == key or row.framework.endswith('-' + key)]
            release = release_map[max(bases, key=len)] if bases else None
            reasons = image_reasons(row.image, release)
            days = max(0, (utc(now) - utc(f"{row.date}T00:00:00Z")).days)
            if row.benchmark_type == "agentic_traces" and days > 14:
                reasons.append("agentx-age")
            needs_review = any(reason != "release-comparison-unknown" for reason in reasons)
            item.update({"source-status": "review" if needs_review else "unknown" if reasons else "no-review-signal",
                         "review-reasons": reasons, "release": release, "needs-review": needs_review,
                         "benchmark-age-days": days})
        result.append(item)
    return result, sorted(set(issues))


def fetch_catalog(policy: Policy) -> tuple[list[dict], list[str]]:
    feeds = {}
    for resource in ('images', 'releases'):
        try:
            feeds[resource] = fetch(resource)
        except ReadError as error:
            raise ReadError(f'{resource}:{error}') from None
    return catalog(feeds['images'], feeds['releases'], datetime.now(timezone.utc), policy)


def fresh(value: Any, now: datetime, policy: Policy) -> bool:
    try:
        return -policy.clock_skew_seconds <= (utc(now) - utc(value)).total_seconds() <= policy.response_max_age_seconds
    except (ValueError, TypeError, AttributeError):
        return False


def available_clusters(feed: Feed, policy: Policy, now: datetime) -> set[str]:
    """Return clusters below 20% node utilization; never publish private status."""
    try:
        raw = feed.payload
        if (feed.error or not fresh(feed.retrieved_at, now, policy)
                or raw['schemaVersion'] != 6 or raw['kind'] != 'inferencex.status.clusters'
                or not fresh(raw['generatedAt'], now, policy) or raw['data']['available'] is not True):
            return set()
        available: set[str] = set()
        seen: set[str] = set()
        for cluster in raw['data']['clusters']:
            cluster_id = cluster['clusterId']
            if not isinstance(cluster_id, str) or not cluster_id or cluster_id in seen:
                return set()
            seen.add(cluster_id)
            # The API owns the cluster-age cutoff. Check timestamp validity/order,
            # but do not impose a second cutoff on its current snapshots.
            observed, received = utc(cluster['observedAt']), utc(cluster['receivedAt'])
            generated = utc(raw['generatedAt'])
            if (cluster['stale'] is not False or cluster['status'] not in ('operational', 'degraded')
                    or (observed - received).total_seconds() > policy.clock_skew_seconds
                    or (received - generated).total_seconds() > policy.clock_skew_seconds):
                continue
            summary = cluster['summary']
            total = summary['totalNodes']
            counts = [summary[k] for k in ('allocatedNodes', 'mixedNodes', 'idleNodes', 'downNodes', 'otherNodes')]
            if (type(total) is not int or total <= 0
                    or any(type(n) is not int or n < 0 for n in counts) or sum(counts) != total):
                continue
            # An entirely down/unavailable cluster can also report 0% utilization.
            if summary['idleNodes'] > 0 and (summary['allocatedNodes'] + summary['mixedNodes']) * 5 < total:
                available.add(cluster_id)
        return available
    except (KeyError, ValueError, TypeError, AttributeError):
        return set()


def fetch_capacity(policy: Policy) -> set[str]:
    return available_clusters(fetch('clusters', token=os.environ.get('KLAUDE_DASHBOARD_API_KEY')),
                              policy, datetime.now(timezone.utc))


def capacity_context(policy: Policy) -> dict:
    """Private routing hints for review, without node counts or raw responses."""
    feed = fetch('clusters', token=os.environ.get('KLAUDE_DASHBOARD_API_KEY'))
    available = available_clusters(feed, policy, datetime.now(timezone.utc))
    try:
        clusters = sorted({cluster['clusterId'] for cluster in feed.payload['data']['clusters']
                           if isinstance(cluster['clusterId'], str)})
    except (KeyError, TypeError):
        clusters = []
    return {'telemetry-clusters': clusters, 'eligible-telemetry-clusters': sorted(available)}
