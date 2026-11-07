"""Auditable serial and closed-loop serving client for an operator-managed video endpoint.

This module does not launch a server, download weights, verify server identity,
or contact MiniMax's paid API. ``preview_plan`` is network-free; ``run_plan``
submits real requests to the explicitly supplied endpoint. Local HTTP fixtures
can exercise that transport, but are not evidence of model inference.

Protocol implementations were inspected at SGLang 253020450290328e9deb307eece1e402fa17f35e
and vLLM-Omni eb11446b7f2e30ca582f8aff3afe12e9a2e66f6c. They are deliberately
separate: SGLang accepts the H3 JSON contract, while vLLM-Omni expects multipart
form fields. Those source pins do not attest to a supplied server's revision.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import math
import os
import platform
import re
import socket
import ssl
import statistics
import threading
import time
import urllib.parse
import uuid
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from specs.video_spec import canonical_json_bytes, schema_version

from .serving import settings as serving_settings, summarize as serving_summary


MODEL_ID = "MiniMaxAI/MiniMax-H3"
RUNTIMES = {"sglang", "vllm-omni"}
MAX_JSON_BYTES = 1024 * 1024
# A 1344x768 RGB PNG is at most about 3.1 MB, about 4.2 MB once base64-encoded for SGLang.
MAX_IMAGE_REQUEST_BYTES = 8 * 1024 * 1024
MAX_MEDIA_BYTES = 512 * 1024 * 1024
MAX_SLOTS = 10000
MAX_POLLS = 10000
SOCKET_TIMEOUT_SECONDS = 30.0
POLL_INTERVAL_SECONDS = 0.5
_ACTIVE_STATUSES = {"queued", "pending", "in_progress", "processing", "running"}
_FAILED_STATUSES = {"failed", "cancelled", "canceled"}
_TERMINAL_STATUSES = _FAILED_STATUSES | {"completed", "succeeded", "success"}
_CHUNK_BYTES = 64 * 1024
_ERROR_EXCERPT_CHARS = 300
_PAID_MINIMAX_DOMAINS = ("minimax.io", "minimaxi.com")
_EVENT_LOCK = threading.Lock()
_CASE_FIELDS = {"case_id", "prompt", "seed", "requires_motion", "requires_sound"}
_FIRST_FRAME = {"task": "fl2va", "role": "keyframe", "frame_index": 0, "media_type": "image/png"}
_CONDITION_FIELDS = {*_FIRST_FRAME, "sha256", "width", "height"}
_GENERATION_CONTROLS = {
    "duration_seconds",
    "aspect_ratio",
    "width",
    "height",
    "frame_count",
    "fps",
    "audio_sample_rate_hz",
    "audio_channels",
    "num_inference_steps",
    "flow_shift",
    "audio_flow_shift",
}
_H3_CELL = {
    "aspect_ratio": "16:9",
    "width": 1344,
    "height": 768,
    "fps": 24,
    "audio_sample_rate_hz": 32000,
    "audio_channels": 2,
}
# H3 snaps each requested duration to its own delivered frame count.
_H3_DURATION_FRAMES = ((4, 107), (5, 124), (8, 192))
# SGLang has no crf field; it encodes with crf = int((1 - output_compression / 100) * 51).
SGLANG_OUTPUT_COMPRESSION = {18: 63}


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _positive(value: Any, name: str, *, maximum: float, integer: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 < value <= maximum
        or not math.isfinite(value)
        or (integer and not isinstance(value, int))
    ):
        raise ValueError(
            f"{name} must be a positive {'integer' if integer else 'number'} <= {maximum}"
        )


def _text(value: Any, name: str, *, limit: int = 1000) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must be a nonempty string of at most {limit} characters")


def valid_numerical_identity(identity: Any, runtime_revision: str) -> bool:
    if not isinstance(identity, dict) or set(identity) != {
        "runtime_revision",
        "image_digest",
        "server_argv",
        "server_environment",
        "gpu_topology",
    }:
        return False
    argv, environment, topology = (
        identity["server_argv"],
        identity["server_environment"],
        identity["gpu_topology"],
    )
    return (
        identity["runtime_revision"] == runtime_revision
        and isinstance(identity["image_digest"], str)
        and re.fullmatch(r"[0-9a-f]{64}", identity["image_digest"]) is not None
        and isinstance(argv, list)
        and bool(argv)
        and all(isinstance(part, str) and bool(part) for part in argv)
        and isinstance(environment, dict)
        and bool(environment)
        and all(
            isinstance(key, str) and bool(key) and isinstance(value, str) and bool(value)
            for key, value in environment.items()
        )
        and isinstance(topology, dict)
        and set(topology) == {"gpu_uuids"}
        and isinstance(topology["gpu_uuids"], list)
        and bool(topology["gpu_uuids"])
        and all(isinstance(uuid, str) and bool(uuid) for uuid in topology["gpu_uuids"])
        and len(set(topology["gpu_uuids"])) == len(topology["gpu_uuids"])
    )


def requested_encoding(plan: dict[str, Any]) -> dict[str, Any] | None:
    return plan["fairness"]["encoding"] if schema_version(plan) >= 2 else None


def _case(case: Any, name: str, canvas: tuple[Any, Any]) -> None:
    if not isinstance(case, dict) or set(case) - {"condition"} != _CASE_FIELDS:
        raise ValueError(
            f"{name} must declare exactly {sorted(_CASE_FIELDS)} and may add condition"
        )
    condition = case.get("condition")
    if "condition" in case and (
        not isinstance(condition, dict)
        or set(condition) != _CONDITION_FIELDS
        or any(condition[key] != value for key, value in _FIRST_FRAME.items())
        or (condition["width"], condition["height"]) != canvas
        or not isinstance(condition["sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", condition["sha256"])
    ):
        raise ValueError(
            f"{name}.condition must be one canvas-sized fl2va PNG keyframe at frame 0, by sha256"
        )
    _text(case["case_id"], f"{name}.case_id", limit=200)
    _text(case["prompt"], f"{name}.prompt", limit=32000)
    seed = case["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError(f"{name}.seed must be an integer in [0, 2**63)")
    for field in ("requires_motion", "requires_sound"):
        if not isinstance(case[field], bool):
            raise ValueError(f"{name}.{field} must be an explicit boolean")


def validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Check the structure a run needs, without making a request.

    Workload policy, such as the schema version 2 prompt count and warmup rules, is validated once
    by ``specs/video_spec.py``. Unknown root metadata is preserved in the plan digest.
    """
    if not isinstance(plan, dict):
        raise ValueError("plan must be an object")
    try:
        frozen = json.loads(canonical_json_bytes(plan))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("plan must contain finite JSON values") from exc
    for field in ("plan_id", "model_id", "model_revision"):
        _text(frozen.get(field), field)
    if frozen["model_id"] != MODEL_ID:
        raise ValueError(f"this runner supports only {MODEL_ID}")
    if not re.fullmatch(r"[a-fA-F0-9]{40}", frozen["model_revision"]):
        raise ValueError("model_revision must be an immutable 40-character commit")
    if schema_version(frozen) not in (1, 2, 3):
        raise ValueError("schema_version must be 1, 2 or 3")
    if schema_version(frozen) >= 2:
        fairness = frozen.get("fairness")
        encoding = fairness.get("encoding") if isinstance(fairness, dict) else None
        if (
            not isinstance(encoding, dict)
            or not isinstance(encoding.get("preset"), str)
            or type(encoding.get("crf")) is not int
        ):
            raise ValueError(
                "schema 2 and 3 plans must request fairness.encoding preset and integer crf"
            )
    generation = frozen.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("generation must be an object with explicit controls")
    if set(generation) - _GENERATION_CONTROLS:
        raise ValueError("unsupported generation controls must not be silently ignored")
    _positive(
        generation.get("num_inference_steps"),
        "generation.num_inference_steps",
        maximum=1000,
        integer=True,
    )
    for field in ("flow_shift", "audio_flow_shift"):
        _positive(generation.get(field), f"generation.{field}", maximum=1000)
    if (
        any(generation.get(key) != value for key, value in _H3_CELL.items())
        or (
            generation.get("duration_seconds"),
            generation.get("frame_count"),
        )
        not in _H3_DURATION_FRAMES
    ):
        raise ValueError(
            "this runner supports the 16:9 1344x768 24 fps H3 cell with 32 kHz stereo audio"
            " and 4s/107, 5s/124, or 8s/192 frames"
        )
    cases = frozen.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty array")
    canvas = (generation["width"], generation["height"])
    for index, case in enumerate(cases):
        _case(case, f"cases[{index}]", canvas)
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("case_id must be unique")
    if frozen.get("warmup_case") is not None:
        _case(frozen["warmup_case"], "warmup_case", canvas)
    _positive(frozen.get("repetitions"), "repetitions", maximum=MAX_SLOTS, integer=True)
    warmups = frozen.get("warmup_runs")
    if isinstance(warmups, bool) or not isinstance(warmups, int) or warmups < 0:
        raise ValueError("warmup_runs must be an explicit nonnegative integer")
    if len(cases) * frozen["repetitions"] + warmups > MAX_SLOTS:
        raise ValueError(f"plan exceeds {MAX_SLOTS} total requests")
    return frozen


def _slots(plan: dict[str, Any]) -> list[dict[str, Any]]:
    slots = []
    for index in range(1, plan["warmup_runs"] + 1):
        slots.append(
            {
                **(plan.get("warmup_case") or plan["cases"][(index - 1) % len(plan["cases"])]),
                "slot_id": f"warmup-{index:03d}",
                "phase": "warmup",
                "repetition": 0,
            }
        )
    for repetition in range(1, plan["repetitions"] + 1):
        for index, case in enumerate(plan["cases"], 1):
            slots.append(
                {
                    **case,
                    "slot_id": f"measurement-r{repetition:03d}-c{index:03d}",
                    "phase": "measurement",
                    "repetition": repetition,
                }
            )
    return slots


def _payload(plan: dict[str, Any], slot: dict[str, Any], runtime: str) -> dict[str, Any]:
    """The request as recorded: an image condition appears as ``sha256:<hex>``, not its bytes."""
    generation = plan["generation"]
    condition = slot.get("condition")
    task = condition["task"] if condition else "t2va"
    encoding = requested_encoding(plan)
    target = {
        "duration_seconds": generation["duration_seconds"],
        "aspect_ratio": generation["aspect_ratio"],
        "short_edge": min(generation["width"], generation["height"]),
    }
    common = {
        "model": plan["model_id"],
        "prompt": slot["prompt"],
        "seed": slot["seed"],
        "num_inference_steps": generation["num_inference_steps"],
        "flow_shift": generation["flow_shift"],
    }
    if runtime == "sglang":
        # H3's SGLang admission explicitly rejects caller-set fps/num_frames;
        # its target resolves both. Expected dimensions are verified on delivery.
        request = {
            **common,
            "task": task,
            "conditions": [
                {
                    "type": "image",
                    "uri": "sha256:" + condition["sha256"],
                    "role": condition["role"],
                    "frame_index": condition["frame_index"],
                }
            ]
            if condition
            else [],
            "target": target,
            "audio_flow_shift": generation["audio_flow_shift"],
        }
        if encoding:
            if encoding["crf"] not in SGLANG_OUTPUT_COMPRESSION:
                raise ValueError(f"SGLang cannot request CRF {encoding['crf']}")
            request["x264_preset"] = encoding["preset"]
            request["output_compression"] = SGLANG_OUTPUT_COMPRESSION[encoding["crf"]]
        return request
    extra = {
        "task": task,
        "target": target,
        "audio_flow_shift": generation["audio_flow_shift"],
    }
    if encoding:
        extra["video_codec_options"] = {"preset": encoding["preset"], "crf": str(encoding["crf"])}
    request = {
        **common,
        "width": generation["width"],
        "height": generation["height"],
        "fps": generation["fps"],
        "num_frames": generation["frame_count"],
        "num_outputs_per_prompt": 1,
        "extra_params": extra,
    }
    if condition:
        request["input_reference"] = "sha256:" + condition["sha256"]
    return request


def _conditioning_images(plan: dict[str, Any], image_dir: Path | None) -> dict[str, bytes]:
    """Read each conditioning image as ``<sha256>.png`` and refuse bytes the plan did not seal."""
    cases = [*plan["cases"], plan.get("warmup_case") or {}]
    digests = sorted({case["condition"]["sha256"] for case in cases if "condition" in case})
    if digests and image_dir is None:
        raise ValueError("image-conditioned plans need image_dir holding <sha256>.png files")
    images = {}
    for digest in digests:
        data = (Path(image_dir) / f"{digest}.png").read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"{digest}.png does not match the sha256 the plan names")
        images[digest] = data
    return images


def _conditioning_cache_evidence(runtime: str, identity: dict[str, Any] | None) -> str:
    """The recorded server launch setting that keeps conditioning caches off across requests."""
    if identity is not None:
        if runtime == "sglang" and "--disable-conditioning-cache" in identity["server_argv"]:
            return "server_argv --disable-conditioning-cache"
        environment = identity["server_environment"]
        if runtime == "vllm-omni" and environment.get("OMNI_DIFFUSION_PROMPT_EMBED_CACHE") == "0":
            return "server_environment OMNI_DIFFUSION_PROMPT_EMBED_CACHE=0"
    raise ValueError(
        "image-conditioned plans need a numerical_identity whose server launch turns off"
        " cross-request conditioning caches"
    )


def _body(payload: dict[str, Any], runtime: str, image: bytes | None) -> tuple[bytes, str]:
    """Encode a recorded payload for the wire, putting the image bytes where its hash stands."""
    if runtime == "vllm-omni":
        return _multipart(payload, image)
    if image is not None:
        uri = "data:image/png;base64," + base64.b64encode(image).decode("ascii")
        payload = {**payload, "conditions": [{**payload["conditions"][0], "uri": uri}]}
    return canonical_json_bytes(payload), "application/json"


def preview_plan(plan: dict[str, Any], *, runtime: str = "sglang") -> dict[str, Any]:
    """Return exact semantic payloads and counts, with no disk/network effects."""
    if runtime not in RUNTIMES:
        raise ValueError("runtime must be sglang or vllm-omni")
    frozen = validate_plan(plan)
    return {
        "evidence_kind": "request_preview_no_generation",
        "plan_id": frozen["plan_id"],
        "plan_sha256": _digest(frozen),
        "runtime": runtime,
        "measurement_count": len(frozen["cases"]) * frozen["repetitions"],
        "warmup_count": frozen["warmup_runs"],
        "total_requests": len(frozen["cases"]) * frozen["repetitions"] + frozen["warmup_runs"],
        "slots": [{**slot, "request": _payload(frozen, slot, runtime)} for slot in _slots(frozen)],
    }


class _RequestError(RuntimeError):
    """A locally composed message; HTTP error bodies appear only as redacted excerpts."""


def _endpoint(value: str) -> tuple[str, urllib.parse.SplitResult, str]:
    try:
        parts = urllib.parse.urlsplit(value)
        port = parts.port
    except (TypeError, ValueError) as exc:
        raise ValueError("endpoint must be a valid absolute HTTP(S) base URL") from exc
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or any(char.isspace() for char in value)
    ):
        raise ValueError(
            "endpoint requires HTTP(S) and may not include credentials, query, or fragment"
        )
    host = parts.hostname.lower().rstrip(".")
    if any(host == domain or host.endswith("." + domain) for domain in _PAID_MINIMAX_DOMAINS):
        raise ValueError(
            "the MVP runner supports operator-managed endpoints, not the paid hosted MiniMax API"
        )
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("endpoint port is invalid")
    base = value.rstrip("/")
    path = parts.path.rstrip("/")
    api_path = (
        path
        if path.endswith("/v1/videos")
        else path + ("/videos" if path.endswith("/v1") else "/v1/videos")
    )
    return base, parts, api_path


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("attempt deadline exceeded")
    return remaining


def _multipart(payload: dict[str, Any], image: bytes | None = None) -> tuple[bytes, str]:
    boundary = "vgbench-" + uuid.uuid4().hex
    chunks: list[bytes] = []
    for name, value in payload.items():
        if name == "input_reference":
            chunks.append(
                (
                    f'--{boundary}\r\nContent-Disposition: form-data; name="{name}";'
                    ' filename="input_reference.png"\r\nContent-Type: image/png\r\n\r\n'
                ).encode("ascii")
                + image
                + b"\r\n"
            )
            continue
        text = (
            json.dumps(value, ensure_ascii=False, allow_nan=False)
            if isinstance(value, (dict, list))
            else str(value)
        )
        chunks.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
                + text
                + "\r\n"
            ).encode("utf-8")
        )
    chunks.append(f"--{boundary}--\r\n".encode("ascii"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _open_response(
    parts: urllib.parse.SplitResult,
    method: str,
    path: str,
    *,
    deadline: float,
    credential: str | None,
    body: bytes | None = None,
    content_type: str | None = None,
    sent: Callable[[], None] | None = None,
) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse, threading.Timer]:
    """No redirect following, environment proxies, cookies, or server URLs.

    A deadline watchdog shuts down an established socket even when a peer
    trickles response headers. DNS is bounded in a daemon resolver; its late
    result is discarded before any request can be sent.
    """
    result: list[Any] = []
    resolved = threading.Event()

    def resolve() -> None:
        try:
            result.append(
                socket.getaddrinfo(
                    parts.hostname,
                    parts.port or (443 if parts.scheme == "https" else 80),
                    type=socket.SOCK_STREAM,
                )
            )
        except OSError:
            result.append(None)
        finally:
            resolved.set()

    threading.Thread(target=resolve, daemon=True).start()
    if not resolved.wait(min(SOCKET_TIMEOUT_SECONDS, _remaining(deadline))):
        raise TimeoutError("endpoint DNS resolution timed out")
    if not result[0]:
        raise _RequestError("endpoint DNS resolution failed")
    cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    connection = cls(
        parts.hostname, parts.port, timeout=min(SOCKET_TIMEOUT_SECONDS, _remaining(deadline))
    )
    connected_socket = None
    for family, socktype, proto, _, address in result[0]:
        candidate = socket.socket(family, socktype, proto)
        try:
            candidate.settimeout(min(SOCKET_TIMEOUT_SECONDS, _remaining(deadline)))
            candidate.connect(address)
            connected_socket = candidate
            break
        except OSError:
            candidate.close()
    if connected_socket is None:
        raise _RequestError("endpoint connection failed")
    connection.sock = connected_socket

    def interrupt_socket() -> None:
        try:
            connected_socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        connection.close()

    try:
        timer = threading.Timer(_remaining(deadline), interrupt_socket)
    except BaseException:
        connection.close()
        raise
    timer.daemon = True
    timer.start()
    try:
        if parts.scheme == "https":
            # Use system trust roots and hostname verification, never disable TLS.
            context = ssl.create_default_context()
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            connection.sock = context.wrap_socket(connected_socket, server_hostname=parts.hostname)
            connected_socket = connection.sock
        connection.sock.settimeout(min(SOCKET_TIMEOUT_SECONDS, _remaining(deadline)))
        headers = {"Accept": "application/json, video/mp4", "User-Agent": "vgbench-mvp/0.1.0"}
        if content_type:
            headers["Content-Type"] = content_type
        if credential:
            headers["Authorization"] = "Bearer " + credential
        connection.request(method, path, body=body, headers=headers)
        if sent:
            sent()
        response = connection.getresponse()
        if not 200 <= response.status < 300:
            body = response.read(_ERROR_EXCERPT_CHARS + len(credential or ""))
            text = body.decode("utf-8", "replace")
            if credential:
                text = text.replace(credential, "[redacted]")
            text = " ".join("".join(c if c.isprintable() else " " for c in text).split())
            raise _RequestError(
                f"{method} received HTTP {response.status}: {text[:_ERROR_EXCERPT_CHARS]}"
            )
        return connection, response, timer
    except BaseException:
        timer.cancel()
        connection.close()
        raise


def _transfer(
    parts: urllib.parse.SplitResult,
    method: str,
    path: str,
    *,
    deadline: float,
    credential: str | None,
    body: bytes | None = None,
    content_type: str | None = None,
    destination: Path | None = None,
    sent: Callable[[], None] | None = None,
) -> bytes | str:
    connection, response, timer = _open_response(
        parts,
        method,
        path,
        deadline=deadline,
        credential=credential,
        body=body,
        content_type=content_type,
        sent=sent,
    )
    limit = MAX_MEDIA_BYTES if destination else MAX_JSON_BYTES
    sink = None
    temporary = destination.with_suffix(".mp4.part") if destination else None
    try:
        length = response.getheader("Content-Length")
        if length is not None and (not length.isdigit() or int(length) > limit):
            raise _RequestError("response Content-Length is invalid or exceeds byte limit")
        if response.getheader("Content-Encoding", "identity").lower() != "identity":
            raise _RequestError("compressed HTTP responses are not accepted")
        sink = temporary.open("xb") if temporary else None
        hasher = hashlib.sha256()
        chunks: list[bytes] = []
        size = 0
        while True:
            if connection.sock:
                connection.sock.settimeout(min(SOCKET_TIMEOUT_SECONDS, _remaining(deadline)))
            _remaining(deadline)
            chunk = response.read1(min(_CHUNK_BYTES, limit - size + 1))
            _remaining(deadline)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                raise _RequestError("response exceeds byte limit")
            if sink:
                sink.write(chunk)
                hasher.update(chunk)
            else:
                chunks.append(chunk)
        if length is not None and size != int(length):
            raise _RequestError("response body is truncated")
        if not size:
            raise _RequestError("empty response body")
        if sink:
            sink.flush()
            os.fsync(sink.fileno())
            sink.close()
            sink = None
            temporary.replace(destination)
            return hasher.hexdigest()
        return b"".join(chunks)
    finally:
        if sink:
            sink.close()
        timer.cancel()
        response.close()
        connection.close()


def _json_request(
    parts: urllib.parse.SplitResult, method: str, path: str, **kwargs: Any
) -> dict[str, Any]:
    data = _transfer(parts, method, path, **kwargs)
    try:
        value = json.loads(data)
    except (ValueError, TypeError, UnicodeDecodeError) as exc:
        raise _RequestError("endpoint returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise _RequestError("endpoint returned a non-object JSON response")
    return value


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    target = path if exclusive else path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with target.open("xb") as handle:
        handle.write(canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    if not exclusive:
        target.replace(path)


def _event(path: Path, event: str, **fields: Any) -> None:
    with _EVENT_LOCK, path.open("ab") as handle:
        handle.write(canonical_json_bytes({"event": event, "at": _timestamp(), **fields}) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _safe_error(exc: BaseException) -> str:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return "attempt timed out; server completion may be unknown"
    if isinstance(exc, _RequestError):
        return str(exc)
    if isinstance(exc, KeyboardInterrupt):
        return "interrupted by operator; server completion may be unknown"
    # Arbitrary exception text can carry URLs or credentials, so traces keep only the type.
    return f"{type(exc).__name__}: local transport or media analysis failed; details omitted"


def _summary(records: list[dict[str, Any]], scheduled: int, wall: float) -> dict[str, Any]:
    measured = [record for record in records if record["phase"] == "measurement"]
    completed = [record for record in measured if record["status"] == "succeeded"]
    valid = [record for record in completed if record["media"]["valid"] is True]
    latencies = [record["latency_seconds"] for record in valid]
    stage_medians = {}
    for field in (
        "submit_to_terminal_seconds",
        "submit_to_media_seconds",
        "media_validation_seconds",
    ):
        values = [record.get(field) for record in valid]
        # Never silently calculate a successful-subset timing from missing
        # stages. The request denominator and missing measurements stay visible.
        stage_medians[field.removesuffix("_seconds") + "_median_seconds"] = (
            statistics.median(values)
            if values and all(value is not None for value in values)
            else None
        )
    return {
        "scheduled": scheduled,
        "completed": len(completed),
        "valid": len(valid),
        "failed": scheduled - len(valid),
        "failed_attempts": sum(
            record["status"] == "failed" and record.get("attempted", False) for record in measured
        ),
        "not_started": sum(not record.get("attempted", False) for record in measured),
        "invalid_completed": len(completed) - len(valid),
        "technical_success_rate": len(valid) / scheduled,
        "latency_samples": len(latencies),
        "latency_median_seconds": statistics.median(latencies) if latencies else None,
        "latency_mean_seconds": statistics.mean(latencies) if latencies else None,
        "latency_min_seconds": min(latencies) if latencies else None,
        "latency_max_seconds": max(latencies) if latencies else None,
        "latency_sample_stddev_seconds": statistics.stdev(latencies)
        if len(latencies) > 1
        else None,
        **stage_medians,
        "valid_clips_per_second": len(valid) / wall if wall > 0 else None,
    }


def run_plan(
    plan: dict[str, Any],
    output_dir: Path,
    *,
    endpoint: str,
    runtime: str = "sglang",
    runtime_revision: str,
    hardware_label: str,
    timeout_seconds: float = 3600,
    api_key_env: str | None = None,
    serving_concurrency: int | None = None,
    delivery_deadline_seconds: float | None = None,
    numerical_identity: dict[str, Any] | None = None,
    image_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute a serial or explicit closed-loop run, retaining every outcome.

    ``completed`` counts completed downloads/analyses; ``valid`` additionally
    requires media-contract success. ``failed = scheduled - valid`` includes
    invalid media and slots not started after uncertain remote completion.
    Latency is conditional on valid clips and is never reported without the
    all-scheduled technical-success denominator. No submission is retried.

    A client timeout does not cancel server work. To prevent such an unknown
    job from violating the declared concurrency, remaining slots are not submitted.
    Native media decoding observes a cooperative, not preemptive, deadline.

    An image-conditioned plan reads each image from ``image_dir/<sha256>.png`` and refuses to
    start unless every file matches its hash and ``numerical_identity`` shows the server's
    cross-request conditioning cache is off.
    """
    frozen = validate_plan(plan)
    serving = serving_settings(serving_concurrency, delivery_deadline_seconds)
    if runtime not in RUNTIMES:
        raise ValueError("runtime must be sglang or vllm-omni")
    for field, value in (
        ("runtime_revision", runtime_revision),
        ("hardware_label", hardware_label),
    ):
        _text(value, field)
    if not re.fullmatch(r"[a-fA-F0-9]{40}", runtime_revision):
        raise ValueError(
            "runtime_revision must be an operator-declared immutable 40-character commit"
        )
    if numerical_identity is not None:
        if not valid_numerical_identity(numerical_identity, runtime_revision):
            raise ValueError(
                "numerical_identity must contain a complete matching server configuration"
            )
        canonical_json_bytes(numerical_identity)
    _positive(timeout_seconds, "timeout_seconds", maximum=86400)
    images = _conditioning_images(frozen, image_dir)
    cache_evidence = _conditioning_cache_evidence(runtime, numerical_identity) if images else None
    safe_endpoint, parts, api_path = _endpoint(endpoint)
    credential = None
    if api_key_env:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
            raise ValueError("api_key_env must be an environment-variable name")
        credential = os.environ.get(api_key_env)
        if not credential or any(char in credential for char in "\r\n"):
            raise ValueError("configured API-key environment variable is missing or invalid")
        if parts.scheme != "https" and parts.hostname not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("authenticated non-loopback endpoints require HTTPS")
    # Resolve dependency availability before generating anything or creating a
    # run directory. The analyzer is imported lazily so preview is lightweight.
    import av
    import numpy
    from .media import IMPLEMENTATION_VERSION, __file__ as media_source_file, analyze_media

    media_evaluator = {
        "implementation_version": IMPLEMENTATION_VERSION,
        "source_sha256": hashlib.sha256(Path(media_source_file).read_bytes()).hexdigest(),
        "pyav_version": av.__version__,
        "numpy_version": numpy.__version__,
        "ffmpeg_libraries": {
            name: ".".join(map(str, version)) for name, version in av.library_versions.items()
        },
    }

    slots = _slots(frozen)
    payloads = {slot["slot_id"]: _payload(frozen, slot, runtime) for slot in slots}
    encoding = requested_encoding(frozen)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "artifacts").mkdir()
    (directory / "requests").mkdir()
    plan_sha256 = _digest(frozen)
    _write_json(directory / "plan.json", frozen, exclusive=True)
    configuration: dict[str, Any] = {
        "runtime": runtime,
        "runtime_revision": runtime_revision,
        "hardware_label": hardware_label,
        "model_id": frozen["model_id"],
        "model_revision": frozen["model_revision"],
        "endpoint": safe_endpoint,
        "identity_verification": "operator_declared",
        "plan_sha256": plan_sha256,
        "client_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "media_evaluator": media_evaluator,
        "client_environment": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "av": av.__version__,
            "numpy": numpy.__version__,
        },
        "limits": {
            "attempt_timeout_seconds": timeout_seconds,
            "socket_timeout_seconds": SOCKET_TIMEOUT_SECONDS,
            "max_json_bytes": MAX_JSON_BYTES,
            "max_image_request_bytes": MAX_IMAGE_REQUEST_BYTES,
            "max_media_bytes": MAX_MEDIA_BYTES,
            "max_polls": MAX_POLLS,
            "poll_interval_seconds": POLL_INTERVAL_SECONDS,
        },
        "hash_convention": "SHA256 of sorted-key compact UTF-8 JSON; configuration hash excludes configuration_sha256",
    }
    if numerical_identity is not None:
        configuration["numerical_identity"] = numerical_identity
    if cache_evidence:
        configuration["conditioning_cache"] = {
            "state": "disabled_across_requests",
            "evidence": cache_evidence,
        }
    if serving:
        from . import serving as serving_module

        configuration["serving"] = serving
        configuration["serving_source_sha256"] = hashlib.sha256(
            Path(serving_module.__file__).read_bytes()
        ).hexdigest()
    configuration["configuration_sha256"] = _digest(configuration)
    _write_json(directory / "configuration.json", configuration, exclusive=True)
    scheduled = len(frozen["cases"]) * frozen["repetitions"]
    run: dict[str, Any] = {
        "bundle_version": "0.1.0",
        "bundle_type": "mvp_run",
        "run_id": "video-" + uuid.uuid4().hex,
        "plan_id": frozen["plan_id"],
        "plan_sha256": plan_sha256,
        "plan": frozen,
        "configuration": configuration,
        "evidence_kind": "operator_endpoint",
        "started_at": _timestamp(),
        "finished_at": None,
        "status": "partial",
        "measurement": {
            "boundary": "submit_to_validated_media",
            "concurrency": 1,
            "warmup_runs": frozen["warmup_runs"],
            "wall_seconds": 0.0,
        },
        "records": [],
        "summary": _summary([], scheduled, 0.0),
    }
    if serving:
        run["measurement"].update(
            boundary="submit_to_downloaded_media", concurrency=serving["concurrency"]
        )
    journal = directory / "events.jsonl"
    _write_json(directory / "run.json", run)
    measured_start = None
    abort_reason = None
    abort_code = "not_started_after_uncertain_remote_completion"
    interrupted = None
    stop = threading.Event()

    def validate_media(record: dict, deadline: float) -> None:
        validation_started = time.monotonic()
        try:
            media = analyze_media(
                directory / record["artifact_path"],
                {**record["expected_media"], "timeout_seconds": _remaining(deadline)},
                first_frame=images[record["condition"]["sha256"]] if record["condition"] else None,
            )
        finally:
            record["media_validation_seconds"] = time.monotonic() - validation_started
        if not isinstance(media, dict) or not isinstance(media.get("valid"), bool):
            raise _RequestError("media analyzer returned an invalid contract result")
        canonical_json_bytes(media)
        _remaining(deadline)
        record.update(
            status="succeeded",
            media=media,
            outcome="completed" if media["valid"] else "invalid_media",
        )

    def attempt(slot: dict, *, defer_validation: bool = False) -> dict:
        nonlocal abort_reason, interrupted
        expected = {
            **frozen["generation"],
            "requires_motion": slot["requires_motion"],
            "requires_sound": slot["requires_sound"],
            "audio_required": True,
        }
        expected["duration_seconds"] = expected["frame_count"] / expected["fps"]
        if encoding:
            expected["encoding"] = encoding
        condition = slot.get("condition")
        image = images[condition["sha256"]] if condition else None
        record: dict[str, Any] = {
            **{
                key: slot[key]
                for key in ("slot_id", "case_id", "prompt", "seed", "repetition", "phase")
            },
            "status": "failed",
            "artifact_path": None,
            "sha256": None,
            "latency_seconds": 0.0,
            "media": None,
            "error": None,
            "submit_to_terminal_seconds": None,
            "submit_to_media_seconds": None,
            "media_validation_seconds": None,
            "attempted": False,
            "expected_media": expected,
            "outcome": "not_started",
            "job_id": None,
            "provider_status": None,
            "condition": {**condition, "byte_size": len(image)} if condition else None,
            "request_bytes": None,
            "submit_to_sent_seconds": None,
            "submit_to_accepted_seconds": None,
        }
        if abort_reason:
            record["error"] = abort_code
            _event(journal, "slot_not_started", slot_id=slot["slot_id"], reason=record["error"])
        else:
            payload = payloads[slot["slot_id"]]
            _write_json(
                directory / "requests" / (slot["slot_id"] + ".json"), payload, exclusive=True
            )
            body, content_type = _body(payload, runtime, image)
            if len(body) > (MAX_IMAGE_REQUEST_BYTES if image else MAX_JSON_BYTES):
                raise ValueError("request exceeds byte limit")
            record["request_bytes"] = len(body)
            # fsync the intent BEFORE the first byte can leave this client.
            _event(
                journal, "attempt_started", slot_id=slot["slot_id"], payload_sha256=_digest(payload)
            )
            record["attempted"] = True
            record["outcome"] = "transport_error"
            start = time.monotonic()
            record["timing_window"] = {
                "start_monotonic_seconds": start,
                "start_utc": _timestamp(),
                "terminal_monotonic_seconds": None,
                "end_monotonic_seconds": None,
            }
            deadline = start + timeout_seconds
            remote_terminal = False

            def sent() -> None:
                record["submit_to_sent_seconds"] = time.monotonic() - start

            try:
                reply = _json_request(
                    parts,
                    "POST",
                    api_path,
                    deadline=deadline,
                    credential=credential,
                    body=body,
                    content_type=content_type,
                    sent=sent,
                )
                record["submit_to_accepted_seconds"] = time.monotonic() - start
                identifier = reply.get("id")
                if not isinstance(identifier, str) or not re.fullmatch(
                    r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,199}", identifier
                ):
                    raise _RequestError("submission returned a missing or unsafe job identifier")
                record["job_id"] = identifier
                _event(journal, "job_submitted", slot_id=slot["slot_id"], job_id=identifier)
                for poll in range(MAX_POLLS + 1):
                    status = reply.get("status")
                    record["provider_status"] = (
                        status
                        if isinstance(status, str)
                        and status in _ACTIVE_STATUSES | _TERMINAL_STATUSES
                        else None
                    )
                    if status in _TERMINAL_STATUSES:
                        remote_terminal = True
                        record["submit_to_terminal_seconds"] = time.monotonic() - start
                        record["timing_window"]["terminal_monotonic_seconds"] = (
                            start + record["submit_to_terminal_seconds"]
                        )
                        if status in _FAILED_STATUSES:
                            record["outcome"] = (
                                "provider_failed" if status == "failed" else "provider_cancelled"
                            )
                            raise _RequestError(
                                f"provider job reported {status}; provider text omitted"
                            )
                        break
                    if status not in _ACTIVE_STATUSES:
                        raise _RequestError("provider job returned an unsupported status")
                    if poll == MAX_POLLS:
                        raise _RequestError("job exceeded maximum polling requests")
                    # Signals reach only the main thread; pool workers learn of the interrupt here.
                    if stop.wait(min(POLL_INTERVAL_SECONDS, _remaining(deadline))):
                        raise KeyboardInterrupt
                    reply = _json_request(
                        parts,
                        "GET",
                        api_path + "/" + identifier,
                        deadline=deadline,
                        credential=credential,
                    )
                artifact = directory / "artifacts" / (slot["slot_id"] + ".mp4")
                record["sha256"] = _transfer(
                    parts,
                    "GET",
                    api_path + "/" + identifier + "/content",
                    deadline=deadline,
                    credential=credential,
                    destination=artifact,
                )
                record["artifact_path"] = artifact.relative_to(directory).as_posix()
                record["submit_to_media_seconds"] = time.monotonic() - start
                record["outcome"] = "downloaded"
                if not defer_validation:
                    validate_media(record, deadline)
            except (Exception, KeyboardInterrupt) as exc:
                # The deadline watchdog closes sockets, which can surface as EOF instead of socket.timeout.
                if (
                    not isinstance(exc, KeyboardInterrupt)
                    and record["outcome"] not in {"provider_failed", "provider_cancelled"}
                    and time.monotonic() >= deadline
                ):
                    exc = TimeoutError("attempt deadline exceeded")
                record["error"] = _safe_error(exc)
                if isinstance(exc, (TimeoutError, socket.timeout)):
                    record["outcome"] = "timed_out"
                elif isinstance(exc, KeyboardInterrupt):
                    record["outcome"] = "interrupted"
                elif record["outcome"] == "downloaded":
                    record["outcome"] = "validation_error"
                if not remote_terminal:
                    abort_reason = record["error"]
                if isinstance(exc, KeyboardInterrupt):
                    interrupted = interrupted or exc
                    abort_reason = record["error"]
            finally:
                record["latency_seconds"] = max(0.0, time.monotonic() - start)
                record["timing_window"]["end_monotonic_seconds"] = start + record["latency_seconds"]
                if defer_validation:
                    record["timing_window"]["transport_end_monotonic_seconds"] = time.monotonic()
                    _event(journal, "transport_finished", slot_id=record["slot_id"])
        return record

    order = {slot["slot_id"]: index for index, slot in enumerate(slots)}

    def retain(record: dict) -> None:
        if record["attempted"]:
            _event(journal, "attempt_finished", record=record)
        run["records"].append(record)
        run["records"].sort(key=lambda item: order[item["slot_id"]])
        if not serving:
            run["measurement"]["wall_seconds"] = (
                max(0.0, time.monotonic() - measured_start) if measured_start is not None else 0.0
            )
        run["summary"] = _summary(run["records"], scheduled, run["measurement"]["wall_seconds"])
        _write_json(directory / "run.json", run)

    serial_slots = slots if not serving else [slot for slot in slots if slot["phase"] == "warmup"]
    for slot in serial_slots:
        if slot["phase"] == "measurement" and measured_start is None:
            measured_start = time.monotonic()
        record = attempt(slot)
        if slot["phase"] == "warmup" and not (
            record["status"] == "succeeded" and record["media"]["valid"] is True
        ):
            abort_reason = abort_reason or "warmup failed technical media contract"
            abort_code = "not_started_after_failed_warmup"
        retain(record)
    if serving:
        measured_start = time.monotonic()
        run["measurement"]["start_monotonic_seconds"] = measured_start
        # Keep CPU validation off transport workers so it cannot throttle offered concurrency.
        with ThreadPoolExecutor(max_workers=serving["concurrency"]) as pool:
            futures = [
                pool.submit(attempt, slot, defer_validation=True)
                for slot in slots
                if slot["phase"] == "measurement"
            ]
            pending = set(futures)
            while pending:
                try:
                    done, _ = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                    for future in done:
                        pending.remove(future)
                        record = future.result()
                        if record["outcome"] == "downloaded":
                            try:
                                _event(journal, "validation_started", slot_id=record["slot_id"])
                                validate_media(record, time.monotonic() + timeout_seconds)
                            except (Exception, KeyboardInterrupt) as exc:
                                record["error"] = _safe_error(exc)
                                record["outcome"] = (
                                    "timed_out"
                                    if isinstance(exc, (TimeoutError, socket.timeout))
                                    else "validation_error"
                                )
                                if isinstance(exc, KeyboardInterrupt):
                                    record["outcome"] = "interrupted"
                                    interrupted, abort_reason = exc, record["error"]
                                    stop.set()
                            finally:
                                record["latency_seconds"] = (
                                    time.monotonic()
                                    - record["timing_window"]["start_monotonic_seconds"]
                                )
                                record["timing_window"]["end_monotonic_seconds"] = (
                                    record["timing_window"]["start_monotonic_seconds"]
                                    + record["latency_seconds"]
                                )
                        retain(record)
                except KeyboardInterrupt as exc:
                    interrupted, abort_reason = exc, "interrupted by operator"
                    stop.set()
        end = max(
            (
                r.get("timing_window", {}).get("transport_end_monotonic_seconds", measured_start)
                for r in run["records"]
                if r["phase"] == "measurement"
            ),
            default=measured_start,
        )
        if end == measured_start:
            end = time.monotonic()
        run["measurement"].update(end_monotonic_seconds=end, wall_seconds=end - measured_start)
        run["serving"] = serving_summary(run)
    else:
        run["measurement"]["wall_seconds"] = (
            max(0.0, time.monotonic() - measured_start) if measured_start is not None else 0.0
        )
    wall = run["measurement"]["wall_seconds"]
    warmup_records = [record for record in run["records"] if record["phase"] == "warmup"]
    run["measurement"]["warmup_qualified"] = bool(warmup_records) and all(
        record["status"] == "succeeded" and record["media"]["valid"] is True
        for record in warmup_records
    )
    run["measurement"]["warmup_status"] = (
        "not_requested"
        if not warmup_records
        else ("qualified" if run["measurement"]["warmup_qualified"] else "failed")
    )
    run["summary"] = _summary(run["records"], scheduled, wall)
    run["finished_at"] = _timestamp()
    run["status"] = (
        "complete"
        if run["summary"]["valid"] == scheduled
        else ("partial" if run["summary"]["valid"] else "failed")
    )
    run["abort_reason"] = abort_reason
    _event(journal, "run_finished", status=run["status"], summary=run["summary"])
    _write_json(directory / "run.json", run)
    if interrupted:
        raise interrupted
    return run
