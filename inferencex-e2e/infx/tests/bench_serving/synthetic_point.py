"""Run the fixed-sequence client and aggregation against a scripted local server.

No GPU, model, or network: a local HTTP server replays a script of replies with known delays,
the real ``benchmark_serving`` client measures it, and the real ``infx.results.fixed_sequence``
and ``infx.results.collect_results`` aggregate it the way the workflows do. Everything written
is synthetic test data and says so in ``manifest.json``.

    uv run --project inferencex-e2e --locked --group test \
        python inferencex-e2e/infx/tests/bench_serving/synthetic_point.py --out <dir>
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from aiohttp import web

E2E_ROOT = Path(__file__).resolve().parents[3]
if str(E2E_ROOT) not in sys.path:
    sys.path.insert(0, str(E2E_ROOT))

from infx.bench import fixed_seq  # noqa: E402
from infx.bench_serving.request_records import COMPLETION_EVIDENCE, SCHEMA_VERSION  # noqa: E402

MODEL = "synthetic-test/not-a-real-model"
IMAGE = "synthetic-test-image:local"
ISL, OSL = 32, 8
TOKEN_TEXT = "w1 "
# How far a measured time may sit from its scripted value: sleeps never return early, and a
# loaded machine returns late.
TOLERANCE_MS = {"below": 5.0, "above": 400.0}


@dataclass(frozen=True)
class Reply:
    """One scripted reply. ``delay_s`` precedes the first byte and ``gap_s`` each later token.

    ``completed`` streams ``tokens`` tokens, a finish reason, usage, and ``[DONE]``.
    ``truncated`` streams ``tokens`` tokens and ends the response with none of the three, which
    the client fails as ``incomplete_stream``.
    ``http_status`` answers ``status``. ``stream_error`` and ``connection`` stream ``tokens``
    tokens, then send an error event or close the socket mid-body. ``empty_response`` sends
    usage and ``[DONE]`` without a token. ``hang`` streams one token and never finishes.
    """

    kind: str
    delay_s: float = 0.02
    tokens: int = 0
    gap_s: float = 0.0
    status: int = 200

    def expected(self) -> dict:
        """What the client should record for one attempt answered with this reply."""
        first = self.delay_s * 1000
        stream = first + max(self.tokens - 1, 0) * self.gap_s * 1000
        done, finish, usage = (COMPLETION_EVIDENCE[name] for name in ("done", "finish_reason", "usage"))
        if self.kind == "completed":
            return {
                "outcome": "completed",
                "ttft_ms": first,
                "tpot_ms": self.gap_s * 1000 if self.tokens > 1 else None,
                "e2el_ms": stream,
                "output_len": self.tokens,
                "completion_evidence": done | finish | usage,
            }
        streamed = self.kind in ("stream_error", "connection", "truncated")
        return {
            "outcome": "incomplete_stream" if self.kind == "truncated" else self.kind,
            "ttft_ms": first if streamed else None,
            "tpot_ms": None,
            "e2el_ms": stream if streamed else first,
            "output_len": None,
            "completion_evidence": usage | done if self.kind == "empty_response" else 0,
        }


WARMUP = Reply("completed", delay_s=0.01, tokens=2, gap_s=0.005)


def _event(payload: object) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(payload)
    return f"data: {data}\n\n".encode()


class ScriptedServer:
    """An OpenAI completions endpoint on 127.0.0.1 that answers arrivals from a script.

    The first ``warmups`` arrivals get ``WARMUP``. After them, each run of ``batch`` arrivals
    gets the next reply of ``script``: a client limited to ``batch`` requests in flight sends
    them in index order, so attempt ``i`` gets ``script[i // batch]`` however the requests of
    one batch race each other.
    """

    def __init__(self, script: list[Reply], *, batch: int, warmups: int) -> None:
        self.script, self.batch, self.warmups = script, batch, warmups
        self.arrivals = 0
        self.streaming = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=lambda: asyncio.run(self._serve()), daemon=True)

    def __enter__(self) -> "ScriptedServer":
        self._thread.start()
        self._ready.wait()
        return self

    def __exit__(self, *exc: object) -> None:
        self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join()

    async def _serve(self) -> None:
        self._loop, self._stop = asyncio.get_running_loop(), asyncio.Event()
        app = web.Application()
        app.router.add_post("/v1/completions", self._handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        self.base_url = f"http://127.0.0.1:{runner.addresses[0][1]}"
        self._ready.set()
        await self._stop.wait()
        await runner.cleanup()

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        measured = self.arrivals - self.warmups
        self.arrivals += 1
        reply = WARMUP if measured < 0 else self.script[measured // self.batch]
        await asyncio.sleep(reply.delay_s)
        if reply.kind == "http_status":
            return web.Response(status=reply.status, text="scripted failure")
        response = web.StreamResponse()
        await response.prepare(request)
        for index in range(reply.tokens):
            if index:
                await asyncio.sleep(reply.gap_s)
            last = reply.kind == "completed" and index == reply.tokens - 1
            choice = {"index": 0, "text": TOKEN_TEXT, "finish_reason": "length" if last else None}
            await response.write(_event({"choices": [choice]}))
        if reply.kind == "hang":
            await response.write(_event({"choices": [{"index": 0, "text": TOKEN_TEXT}]}))
            self.streaming.set()
            await self._stop.wait()
            return response
        elif reply.kind == "connection":
            request.transport.close()
            return response
        elif reply.kind == "stream_error":
            await response.write(_event({"error": {"message": "scripted failure", "code": 500}}))
        elif reply.kind in ("completed", "empty_response"):
            usage = {"prompt_tokens": len(body["prompt"].split()), "completion_tokens": reply.tokens}
            await response.write(_event({"choices": [], "usage": usage}))
            await response.write(_event("[DONE]"))
        await response.write_eof()
        return response


def write_tokenizer(directory: Path) -> Path:
    """Save a whitespace word tokenizer whose decode and encode round-trip any id sequence."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {f"w{index}": index for index in range(256)} | {"[UNK]": 256}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]").save_pretrained(directory)
    return directory


def client_command(base_url: str, tokenizer: Path, conc: int, num_prompts: int, result: Path) -> list[str]:
    """The command a fixed-sequence lane runs for one point, with this interpreter."""
    point = fixed_seq.Point(
        base_url=base_url,
        model=MODEL,
        backend="openai",
        tokenizer=str(tokenizer),
        isl=ISL,
        osl=OSL,
        random_range_ratio="1.0",
        conc=conc,
        num_prompts=num_prompts,
        result=result,
    )
    return [sys.executable, *fixed_seq.client_argv(point)[1:], "--metadata", "synthetic=true"]


def child_env(**extra: str) -> dict[str, str]:
    """This environment with the checkout importable and loopback kept off any proxy."""
    return {**os.environ, "PYTHONPATH": str(E2E_ROOT), "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost", **extra}


def aggregate_env(name: str, identity: dict[str, str]) -> dict[str, str]:
    """The environment the benchmark workflow gives ``infx.results.fixed_sequence``."""
    return child_env(
        RUNNER_TYPE=identity["hw"], FRAMEWORK=identity["framework"],
        PRECISION=identity["precision"], MODEL_PREFIX=identity["model_prefix"],
        SPEC_DECODING="none", DISAGG="false", IMAGE=IMAGE, RESULT_FILENAME=name,
        ISL=str(ISL), OSL=str(OSL), TP="1", EP_SIZE="1", DP_ATTENTION="false",
    )


LOWERED_LIMIT = (
    "import sys; from infx.results import fixed_sequence, request_records; "
    "request_records.MAX_AGGREGATE_ATTEMPTS = int(sys.argv.pop()); "
    "sys.exit(fixed_sequence.main())"
)


def run_point(
    out: Path, tokenizer: Path, identity: dict[str, str], conc: int, script: list[Reply],
    *, aggregate_limit: int | None = None,
) -> dict:
    """Measure ``script`` at concurrency ``conc``, aggregate it, and stage it as the job would.

    ``aggregate_limit`` lowers the aggregate's attempt limit for this point only, which the
    production entry point has no setting for.
    """
    name = f"synthetic-test_conc{conc}"
    audit = out / "power_audit" / f"power_audit_{name}"
    audit.mkdir(parents=True)
    raw = audit / f"{name}.json"
    with ScriptedServer(script, batch=conc, warmups=2 * conc) as server:
        command = client_command(server.base_url, tokenizer, conc, len(script) * conc, raw)
        client = subprocess.run(command, env=child_env(), capture_output=True, text=True)
    module = ["-m", "infx.results.fixed_sequence"]
    lowered = ["-c", LOWERED_LIMIT, str(aggregate_limit)]
    aggregate = subprocess.run(
        [sys.executable, "-P", *(module if aggregate_limit is None else lowered)],
        cwd=audit, env=aggregate_env(name, identity), capture_output=True, text=True,
    )
    if not (audit / f"agg_{name}.json").is_file():
        raise RuntimeError(f"{name}: no aggregate\n{client.stderr}\n{aggregate.stderr}")
    uploaded = None
    if client.returncode == 0 and aggregate.returncode == 0:
        uploaded = out / "results" / f"bmk_{name}"
        uploaded.mkdir(parents=True)
        shutil.copy(audit / f"agg_{name}.json", uploaded)
    attempts = [reply.expected() for reply in script for _ in range(conc)]
    failed = [attempt["outcome"] for attempt in attempts if attempt["outcome"] != "completed"]
    statuses = [str(reply.status) for reply in script if reply.kind == "http_status"] * conc
    scripted_s = sum(
        attempt["e2el_ms"] for attempt in attempts[::conc]
    ) / 1000
    return {
        "name": name,
        "conc": conc,
        "num_prompts": len(attempts),
        "warmups": 2 * conc,
        "client_exit_code": client.returncode,
        "aggregate_exit_code": aggregate.returncode,
        "raw": str(raw.relative_to(out)),
        "aggregate": str((audit / f"agg_{name}.json").relative_to(out)),
        "uploaded_as_bmk": None if uploaded is None else str((uploaded / f"agg_{name}.json").relative_to(out)),
        "aggregate_attempt_limit": aggregate_limit,
        "expected": {
            "gate_status": "failed" if len(failed) / len(attempts) > 0.05 else "passed",
            "counts": {
                "requested": len(attempts),
                "dispatched": len(attempts),
                "completed": len(attempts) - len(failed),
                "failed": len(failed),
                "failed_by_class": {kind: failed.count(kind) for kind in sorted(set(failed))},
                "http_status": {code: statuses.count(code) for code in sorted(set(statuses))},
                "warmup_excluded": 2 * conc,
            },
            "accounting_complete": True,
            "columns_omitted": aggregate_limit is not None and len(attempts) > aggregate_limit,
            "duration_s": {"min": scripted_s, "max": scripted_s + 0.5 * len(script) + 2},
            "attempts": attempts,
        },
    }


def fast(tokens: int = 4) -> Reply:
    return Reply("completed", delay_s=0.03, tokens=tokens, gap_s=0.01)


def slow(tokens: int = 4) -> Reply:
    return Reply("completed", delay_s=0.25, tokens=tokens, gap_s=0.05)


def curve() -> dict[int, list[Reply]]:
    """The scripted points by concurrency.

    1: 120 attempts in order, six failed (one per class the server can cause, two HTTP codes),
    one single-token; exactly at the 5% gate, so it passes.
    2: all completed, alternating fast and slow batches.
    4: all completed and slow; aggregated above a lowered limit so the columns are omitted.
    8: four of ten batches fail, so the 5% gate fails the point.
    """
    mixed = [fast()] * 120
    mixed[10] = Reply("http_status", delay_s=0.02, status=503)
    mixed[25] = Reply("stream_error", delay_s=0.03, tokens=2, gap_s=0.02)
    mixed[40] = Reply("empty_response", delay_s=0.03)
    mixed[55] = Reply("connection", delay_s=0.03, tokens=2, gap_s=0.02)
    mixed[70] = Reply("http_status", delay_s=0.02, status=429)
    mixed[85] = Reply("truncated", delay_s=0.03, tokens=3, gap_s=0.01)
    mixed[95] = fast(tokens=1)
    overloaded = [slow()] * 6 + [
        Reply("http_status", delay_s=0.05, status=503),
        Reply("stream_error", delay_s=0.25, tokens=2, gap_s=0.05),
        Reply("empty_response", delay_s=0.25),
        Reply("connection", delay_s=0.25, tokens=2, gap_s=0.05),
    ]
    return {1: mixed, 2: [fast(), slow()] * 5, 4: [slow()] * 10, 8: overloaded}


def produce(out: Path, identity: dict[str, str]) -> dict:
    """Run every scripted point into ``out`` and return the manifest written there."""
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    tokenizer = write_tokenizer(out / "tokenizer")
    points = [
        run_point(out, tokenizer, identity, conc, script, aggregate_limit=20 if conc == 4 else None)
        for conc, script in curve().items()
    ]
    collect = subprocess.run(
        [sys.executable, "-P", "-m", "infx.results.collect_results", "results/", "bmk"],
        cwd=out, env=child_env(), capture_output=True, text=True, check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(E2E_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    manifest = {
        "synthetic": True,
        "description": (
            "Synthetic test data from a scripted local server. No model or GPU ran; "
            "these are not benchmark results."
        ),
        "inferencex_commit": commit.stdout.strip() or None,
        "request_records_schema_version": SCHEMA_VERSION,
        "identity": {**identity, "model": MODEL, "image": IMAGE, "isl": ISL, "osl": OSL},
        "tolerance_ms": TOLERANCE_MS,
        "collected": "agg_bmk.json",
        "collect_stdout": collect.stdout,
        "points": points,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="empty or new output directory")
    for flag in ("hw", "framework", "precision", "model-prefix"):
        parser.add_argument(
            f"--{flag}", default="synthetic-test",
            help=f"the {flag} the aggregate rows carry",
        )
    args = parser.parse_args()
    identity = {name: getattr(args, name) for name in ("hw", "framework", "precision", "model_prefix")}
    manifest = produce(args.out.resolve(), identity)
    for point in manifest["points"]:
        counts = point["expected"]["counts"]
        print(
            f"{point['name']}: {counts['completed']}/{counts['requested']} completed, client exit "
            f"{point['client_exit_code']}, aggregate exit {point['aggregate_exit_code']}"
        )
    print(f"manifest: {args.out.resolve() / 'manifest.json'}")


if __name__ == "__main__":
    main()
