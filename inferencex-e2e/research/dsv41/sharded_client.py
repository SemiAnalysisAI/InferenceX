"""Independent per-DP stream readers and independently timed metrics collection."""

from __future__ import annotations

import asyncio
import gzip
import json
import multiprocessing as mp
import time
import urllib.request
from pathlib import Path


def metrics_worker(base, destination, stop):
    names = (
        "num_requests_running",
        "num_requests_waiting",
        "generation_tokens_total",
        "prefix_cache",
        "preempt",
        "kv_cache",
        "spec_decode",
        "spec_verify",
    )
    with gzip.open(destination, "wt") as output:
        while not stop.is_set():
            start = time.perf_counter()
            row = {"request_start_monotonic_s": start}
            try:
                with urllib.request.urlopen(base + "/metrics", timeout=10) as response:
                    raw = response.read().decode()
                row["metrics"] = [
                    line
                    for line in raw.splitlines()
                    if not line.startswith("#")
                    and any(n in line.split("{", 1)[0] for n in names)
                ]
            except Exception as error:  # noqa: BLE001 - preserve sampling failures in evidence
                row["error"] = str(error)
                row["metrics"] = []
            row["monotonic_s"] = time.perf_counter()
            row["request_duration_s"] = row["monotonic_s"] - start
            output.write(json.dumps(row) + "\n")
            output.flush()
            stop.wait(max(0, 0.5 - (time.perf_counter() - start)))


def stream_worker(base, prompts, output_length, rank, destination, ready, admit):
    import aiohttp

    from research.dsv41.vllm_cohort import completion_body, stream_request

    async def execute():
        bodies = [completion_body(prompt[0], output_length) for prompt in prompts]
        records = [
            {"events": [], "success": False, "meta": {}, "error": ""} for _ in prompts
        ]
        ready.value = 1
        while not admit.is_set():
            await asyncio.sleep(0.01)
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0, force_close=True),
            timeout=aiohttp.ClientTimeout(total=14400),
        ) as session:
            tasks = [
                asyncio.create_task(
                    stream_request(
                        session, base, p[0], output_length, r, rank, body=body
                    )
                )
                for p, r, body in zip(prompts, records, bodies)
            ]
            while not all("headers_received_at" in r for r in records):
                if any(t.done() for t in tasks):
                    raise RuntimeError(
                        "Shard admission failed: "
                        + repr([r["error"] for r in records if r["error"]])
                    )
                await asyncio.sleep(0.01)
            ready.value = 2
            await asyncio.gather(*tasks)
        with gzip.open(destination, "wt") as output:
            json.dump(records, output)
        ready.value = 3

    try:
        asyncio.run(execute())
    except BaseException as error:
        Path(str(destination) + ".error").write_text(repr(error))
        ready.value = -1
        raise


async def sharded_wave(
    session,
    base,
    prompts,
    output_length,
    *,
    output,
    dp_size,
    label,
    settle_seconds,
    timeout_seconds,
):
    from research.dsv41.long_context import post

    context = mp.get_context("spawn")
    admit, stop = context.Event(), context.Event()
    states = [context.Value("i", 0) for _ in range(dp_size)]
    files = [output / f"{label}-dp{rank}-events.json.gz" for rank in range(dp_size)]
    workers = [
        context.Process(
            target=stream_worker,
            args=(
                base,
                prompts[rank::dp_size],
                output_length,
                rank if dp_size > 1 else None,
                files[rank],
                states[rank],
                admit,
            ),
        )
        for rank in range(dp_size)
    ]
    metrics = context.Process(
        target=metrics_worker, args=(base, output / f"{label}-metrics.jsonl.gz", stop)
    )
    marker = {
        "label": label,
        "client_processes": dp_size,
        "metrics_process": True,
        "requests": len(prompts),
        "settle_seconds": settle_seconds,
    }
    paused = False

    async def wait_states(required, timeout):
        deadline = time.perf_counter() + timeout
        while not all(s.value >= required for s in states):
            if any(
                s.value < 0 or (p.exitcode is not None and p.exitcode != 0)
                for s, p in zip(states, workers)
            ):
                errors = [
                    Path(str(f) + ".error").read_text()
                    for f in files
                    if Path(str(f) + ".error").exists()
                ]
                raise RuntimeError("Stream worker failed: " + repr(errors))
            if time.perf_counter() > deadline:
                raise TimeoutError(f"Shard phase {required} timed out")
            await asyncio.sleep(0.05)

    try:
        for worker in workers:
            worker.start()
        await wait_states(1, timeout_seconds)
        await post(session, base, "/pause?mode=keep&clear_cache=false", {})
        paused = True
        async with session.get(base + "/is_paused") as response:
            if not (await response.json()).get("is_paused"):
                raise RuntimeError("Native pause was not established")
        metrics.start()
        admit.set()
        await wait_states(2, timeout_seconds)
        await asyncio.sleep(settle_seconds)
        # A completed request during the pause invalidates admission.
        if any(s.value != 2 for s in states):
            raise RuntimeError("Request shard completed during pause")
        marker["resume_start_monotonic_s"] = time.perf_counter()
        await post(session, base, "/resume", {})
        paused = False
        marker["resume_done_monotonic_s"] = time.perf_counter()
        await wait_states(3, 14400)
        records = [None] * len(prompts)
        for rank, file in enumerate(files):
            with gzip.open(file, "rt") as handle:
                shard = json.load(handle)
            for index, record in zip(range(rank, len(prompts), dp_size), shard):
                records[index] = record
        if any(r is None for r in records):
            raise RuntimeError("Missing shard records")
        if any(
            t < marker["resume_start_monotonic_s"]
            for r in records
            for t, _ in r["events"]
        ):
            raise RuntimeError("Output advanced before native resume")
        return records
    finally:
        if paused:
            await post(session, base, "/resume", {})
        stop.set()
        for process in [*workers, metrics]:
            if process.pid is not None:
                await asyncio.to_thread(process.join, 12)
                if process.is_alive():
                    process.terminate()
                    await asyncio.to_thread(process.join, 5)
        (output / f"admission-{label}.json").write_text(
            json.dumps(marker, indent=2) + "\n"
        )
