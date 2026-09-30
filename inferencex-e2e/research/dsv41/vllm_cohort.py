"""One full-concurrency serving wave, steady-decode evidence, then optional profiling."""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from research.dsv41.long_context import post, steady_decode_window


def completion_body(prompt, output_length):
    payload = {
        "model": os.environ["MODEL"],
        "prompt": prompt,
        "stream": True,
        "temperature": 0,
        "max_tokens": output_length,
        "ignore_eos": True,
        "stream_interval": 1,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }
    if os.environ.get("VLLM_COHORT_TOKEN_API") == "1":
        payload = {
            "model": os.environ["MODEL"],
            "token_ids": prompt,
            "stream": True,
            "return_token_ids": False,
            "sampling_params": {
                "temperature": 0,
                "max_tokens": output_length,
                "ignore_eos": True,
                "stream_interval": 1,
                "detokenize": False,
            },
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        }
    return json.dumps(payload, separators=(",", ":")).encode()


async def admit_wave(
    session,
    base,
    prompts,
    output_length,
    *,
    output,
    label,
    dp_size,
    barrier,
    settle_seconds,
    timeout_seconds,
):
    """Pre-encode requests, then hold generation until every HTTP body is accepted."""
    bodies = [completion_body(p[0], output_length) for p in prompts]
    records = [
        {"events": [], "success": False, "meta": {}, "error": ""} for _ in prompts
    ]
    marker = {
        "label": label,
        "requests": len(prompts),
        "body_bytes": sum(map(len, bodies)),
        "barrier": barrier,
        "settle_seconds": settle_seconds,
    }
    tasks, paused = [], False
    try:
        if barrier:
            paused = True
            await post(session, base, "/pause?mode=keep&clear_cache=false", {})
            async with session.get(base + "/is_paused") as response:
                state = json.loads(await response.text())
            if state.get("is_paused") is not True:
                raise RuntimeError("Native engine did not enter keep-mode pause")
        tasks = [
            asyncio.create_task(
                stream_request(
                    session, base, p[0], output_length, r, i % dp_size, body=body
                )
            )
            for i, (p, r, body) in enumerate(zip(prompts, records, bodies))
        ]
        if barrier:
            deadline = time.perf_counter() + timeout_seconds
            while not all("headers_received_at" in r for r in records):
                if any(t.done() for t in tasks):
                    raise RuntimeError(
                        "Request finished or failed during paused admission"
                    )
                if time.perf_counter() >= deadline:
                    raise RuntimeError(
                        "Timed out waiting for all HTTP admission headers"
                    )
                await asyncio.sleep(0.01)
            await asyncio.sleep(settle_seconds)
            if any(r["events"] for r in records) or any(t.done() for t in tasks):
                raise RuntimeError(
                    "Generation advanced or a request ended while paused"
                )
            marker["headers_received"] = len(records)
            marker["last_header_monotonic_s"] = max(
                r["headers_received_at"] for r in records
            )
            marker["resume_start_monotonic_s"] = time.perf_counter()
            await post(session, base, "/resume", {})
            paused = False
            marker["resume_done_monotonic_s"] = time.perf_counter()
        return records, tasks
    except BaseException as error:
        marker["error"] = f"{type(error).__name__}: {error}"
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        if paused:
            try:
                await post(session, base, "/resume", {})
            except Exception as error:  # noqa: BLE001 - retain cleanup evidence without hiding the original failure
                marker["resume_cleanup_error"] = str(error)
        (output / f"admission-{label}.json").write_text(
            json.dumps(marker, indent=2) + "\n"
        )


async def stream_request(
    session, base, prompt, output_length, record, rank, *, body=None
):
    body = completion_body(prompt, output_length) if body is None else body
    record["start"] = time.perf_counter()
    try:
        count = 0
        async with session.post(
            base
            + (
                "/inference/v1/generate"
                if os.environ.get("VLLM_COHORT_TOKEN_API") == "1"
                else "/v1/completions"
            ),
            headers={
                "X-data-parallel-rank": str(rank),
                "Content-Type": "application/json",
            },
            data=body,
        ) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"completion HTTP {response.status}: {(await response.text())[:500]}"
                )
            record["headers_received_at"] = time.perf_counter()
            async for raw in response.content:
                text = raw.decode().strip()
                if not text.startswith("data:"):
                    continue
                text = text[5:].strip()
                if text == "[DONE]":
                    continue
                item = json.loads(text)
                usage = item.get("usage")
                if usage:
                    count = int(usage.get("completion_tokens", 0))
                    if count and (
                        not record["events"] or count > record["events"][-1][1]
                    ):
                        record["events"].append([time.perf_counter(), count])
                    record["meta"] = usage
                if item.get("error"):
                    raise RuntimeError(str(item["error"]))
        record["success"] = bool(
            record["events"] and record["events"][-1][1] == output_length
        )
        if not record["success"]:
            record["error"] = "Incorrect completion length"
    except Exception as error:  # noqa: BLE001 - retain every failed request in the cohort artifact
        record["error"] = f"{type(error).__name__}: {error}"
        record["success"] = False
    record["end"] = time.perf_counter()


async def wave(
    session,
    base,
    prompts,
    output_length,
    *,
    output,
    profile_steps,
    dp_size,
    label,
    barrier,
    settle_seconds,
    timeout_seconds,
):
    records, tasks = await admit_wave(
        session,
        base,
        prompts,
        output_length,
        output=output,
        label=label,
        dp_size=dp_size,
        barrier=barrier,
        settle_seconds=settle_seconds,
        timeout_seconds=timeout_seconds,
    )
    profile_started = False
    marker = {}
    metric_names = (
        "num_requests_running",
        "num_requests_waiting",
        "spec_decode",
        "spec_verify_calls_total",
        "generation_tokens_total",
        "prefix_cache",
        "retract",
        "is_cuda_graph",
    )
    with gzip.open(
        output / f"{label}-metrics.jsonl.gz",
        "wt",
    ) as metrics_file:
        while not all(t.done() for t in tasks):
            if (
                profile_steps
                and not profile_started
                and all(r["events"] and r["events"][-1][1] >= 64 for r in records)
            ):
                if any(t.done() for t in tasks):
                    raise RuntimeError(
                        "Profile cohort drained before every request reached 64 tokens"
                    )
                trace_dir = output / "profiles"
                trace_dir.mkdir(exist_ok=True)
                marker = {
                    "start_monotonic_s": time.perf_counter(),
                    "completion_counts": [r["events"][-1][1] for r in records],
                }
                marker["response"] = await post(session, base, "/start_profile", {})
                profile_started = True
            async with session.get(base + "/metrics") as response:
                raw = await response.text()
                selected = [
                    line
                    for line in raw.splitlines()
                    if not line.startswith("#")
                    and any(name in line.split("{", 1)[0] for name in metric_names)
                ]
                metrics_file.write(
                    json.dumps(
                        {"monotonic_s": time.perf_counter(), "metrics": selected}
                    )
                    + "\n"
                )
            await asyncio.sleep(0.5)
    await asyncio.gather(*tasks)
    if profile_steps and not profile_started:
        raise RuntimeError("No full-concurrency profile window was captured")
    if profile_steps:
        await post(session, base, "/stop_profile", {})
        (output / "profile-start.json").write_text(json.dumps(marker, indent=2) + "\n")
    return records


def validate_cohort_profiles(
    paths: list[Path], gpu_count: int, dp_size: int, batch: int
) -> list[dict]:
    """Require GPU work and the requested logical batch on every DP/TP worker."""
    workers = set()
    rows = []
    for path in paths:
        match = re.search(r"dp(\d+)_pp0_tp(\d+)_", path.name)
        if match is None:
            raise RuntimeError(f"Unknown worker identity in trace {path.name}")
        identity = tuple(map(int, match.groups()))
        if identity in workers:
            raise RuntimeError("Duplicate worker trace")
        workers.add(identity)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as handle:
            events = json.load(handle)["traceEvents"]
        kernels = sum(e.get("cat") == "kernel" for e in events)
        counts = []
        for event in events:
            name = event.get("name", "")
            if (
                event.get("cat") == "user_annotation"
                and name.startswith("execute_")
                and "_context_0(" in name
            ):
                found = re.search(r"_generation_(\d+)", name)
                if found:
                    counts.append(int(found.group(1)))
        full = counts.count(batch // dp_size)
        if not kernels or not full:
            raise RuntimeError(
                f"No full-batch GPU execution evidence in {path.name}: {counts}"
            )
        rows.append(
            {
                "file": path.name,
                "dp": identity[0],
                "tp": identity[1],
                "kernel_events": kernels,
                "generation_counts": counts,
                "full_batch_execute_scopes": full,
            }
        )
    expected = {(dp, tp) for dp in range(dp_size) for tp in range(gpu_count // dp_size)}
    if workers != expected or len(paths) != gpu_count:
        raise RuntimeError(f"Missing worker traces: expected {expected}, got {workers}")
    return rows


async def run(args, prompts, tokenizer):
    import aiohttp

    from infx.bench_serving.backend_request_func import RequestFuncOutput
    from infx.bench_serving.benchmark_outcome import benchmark_outcome
    from infx.bench_serving.benchmark_serving import calculate_metrics

    base = f"http://{os.environ['SRT_FRONTEND_HOST']}:{os.environ['SRT_FRONTEND_PORT']}"
    concurrency = len(prompts)
    output_length = int(os.environ["OSL"])
    input_length = int(os.environ["ISL"])
    out = args.output / "research" / f"batch{concurrency}"
    out.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession(
        # Encoding a large next-wave body can outlast the server keepalive.
        # Open fresh sockets instead of reusing an idle connection across waves.
        connector=aiohttp.TCPConnector(limit=0, force_close=True),
        timeout=aiohttp.ClientTimeout(total=14400),
    ) as session:
        warm = await wave(
            session,
            base,
            prompts,
            args.warmup_output_tokens,
            output=out,
            profile_steps=0,
            dp_size=args.dp_size,
            label="warmup",
            barrier=args.admission_barrier,
            settle_seconds=args.admission_settle_seconds,
            timeout_seconds=args.admission_timeout_seconds,
        )
        if not all(
            r["success"] and r["meta"].get("prompt_tokens") == input_length
            for r in warm
        ):
            raise RuntimeError("Incomplete full-cohort prefix warmup")
        with gzip.open(out / "warmup-events.json.gz", "wt") as f:
            json.dump(warm, f)
        start_wall, start = time.time(), time.perf_counter()
        records = await wave(
            session,
            base,
            prompts,
            output_length,
            output=out,
            profile_steps=0,
            dp_size=args.dp_size,
            label="measured",
            barrier=args.admission_barrier,
            settle_seconds=args.admission_settle_seconds,
            timeout_seconds=args.admission_timeout_seconds,
        )
        duration = max(r["end"] for r in records) - start
        with gzip.open(out / "measured-events.json.gz", "wt") as f:
            json.dump(records, f)
        for r in records:
            if r["meta"].get("prompt_tokens") != input_length:
                r["success"], r["error"] = (
                    False,
                    "Server prompt-token count does not match the configured length",
                )
        outputs = [
            RequestFuncOutput(
                success=r["success"],
                prompt_len=input_length,
                output_tokens=r["events"][-1][1] if r["events"] else 0,
                ttft=r["events"][0][0] - r["start"] if r["events"] else 0,
                latency=r["events"][-1][0] - r["start"] if r["events"] else 0,
                itl=[b[0] - a[0] for a, b in zip(r["events"], r["events"][1:])],
                error=r["error"],
            )
            for r in records
        ]
        metrics, lengths = calculate_metrics(
            prompts,
            outputs,
            duration,
            tokenizer,
            ["ttft", "tpot", "itl", "e2el"],
            [50, 90, 99],
            {},
        )
        raw = asdict(metrics)
        result = {k: v for k, v in raw.items() if not k.startswith("percentiles_")}
        for field in ("ttft", "tpot", "itl", "e2el"):
            for percentile, value in raw[f"percentiles_{field}_ms"]:
                result[f"p{percentile:g}_{field}_ms"] = value
        result.update(
            duration=duration,
            benchmark_start_time_unix=start_wall,
            benchmark_end_time_unix=start_wall + duration,
            total_input_tokens=metrics.total_input,
            total_output_tokens=metrics.total_output,
            input_lens=[r.prompt_len for r in outputs],
            output_lens=lengths,
            num_prompts=concurrency,
            max_concurrency=concurrency,
            model_id=os.environ["MODEL"],
            backend="vllm",
            protocol="pre-encoded requests; native keep-mode HTTP admission barrier when enabled; fixed DP rank; prefix warmup; client steady window (not model timer)",
            admission_barrier=args.admission_barrier,
            token_api_no_detokenization=os.environ.get("VLLM_COHORT_TOKEN_API") == "1",
            request_rate="inf",
            benchmark_outcome=benchmark_outcome(concurrency, metrics.completed),
        )
        steady_error = None
        try:
            result["steady_decode"] = {
                "valid": True,
                **steady_decode_window(
                    records, burn=8, tail=8, gpu_count=args.gpu_count
                ),
            }
        except ValueError as error:
            steady_error = str(error)
            result["steady_decode"] = {"valid": False, "reason": steady_error}
        if args.result_layout == "single":
            result_path = args.output / f"{os.environ['RESULT_FILENAME']}.json"
        else:
            directory = args.output / f"sa-bench_isl_{input_length}_osl_{output_length}"
            directory.mkdir(exist_ok=True)
            result_path = directory / (
                f"results_concurrency_{concurrency}_gpus_{args.gpu_count}"
                f"_ctx_{args.gpu_count}_gen_0.json"
            )
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "completed": metrics.completed,
                    "steady_decode": result["steady_decode"],
                }
            ),
            flush=True,
        )
        if steady_error or metrics.completed != concurrency:
            raise RuntimeError(steady_error or "Incomplete request cohort")
        if args.profile_steps:
            trace_dir = args.trace_dir or (
                args.output / "research" / "vllm-cohort-profiles"
            )
            previous_traces = set(trace_dir.rglob("*.trace.json*"))
            # Reuse the original prompt prefixes; longer outputs leave a stable
            # cohort for capture. This wave is never used as the performance result.
            profiled = await wave(
                session,
                base,
                prompts,
                max(1024, output_length),
                output=out,
                profile_steps=args.profile_steps,
                dp_size=args.dp_size,
                label="profile",
                barrier=args.admission_barrier,
                settle_seconds=args.admission_settle_seconds,
                timeout_seconds=args.admission_timeout_seconds,
            )
            with gzip.open(out / "profile-events.json.gz", "wt") as f:
                json.dump(profiled, f)
            if not all(r["success"] for r in profiled):
                raise RuntimeError("Incomplete profiling cohort")
            traces = sorted(set(trace_dir.rglob("*.trace.json*")) - previous_traces)
            (out / "trace-files.json").write_text(
                json.dumps([str(p) for p in traces], indent=2) + "\n"
            )
            evidence = validate_cohort_profiles(
                traces, args.gpu_count, args.dp_size, concurrency
            )
            (out / "profile-validation.json").write_text(
                json.dumps(evidence, indent=2) + "\n"
            )
            if len(traces) != args.gpu_count:
                raise RuntimeError(
                    f"Expected {args.gpu_count} rank traces, got {len(traces)}"
                )


def main():
    import numpy as np

    from infx.bench_serving.benchmark_serving import (
        get_tokenizer,
        sample_random_requests,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-steps", type=int, required=True)
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument("--dp-size", type=int, required=True)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--warmup-output-tokens", type=int, default=1)
    parser.add_argument("--admission-barrier", action="store_true")
    parser.add_argument("--admission-settle-seconds", type=float)
    parser.add_argument("--admission-timeout-seconds", type=float)
    parser.add_argument("--result-layout", choices=["single", "multi"], required=True)
    args = parser.parse_args()
    if args.admission_barrier and (
        args.admission_settle_seconds is None
        or args.admission_settle_seconds < 0
        or args.admission_timeout_seconds is None
        or args.admission_timeout_seconds <= 0
    ):
        parser.error(
            "Admission barrier requires explicit nonnegative settle and positive timeout"
        )
    if (
        args.warmup_output_tokens <= 0
        or args.gpu_count <= 0
        or args.dp_size <= 0
        or args.gpu_count % args.dp_size
    ):
        parser.error(
            "Positive GPU/DP/warmup values are required; DP must divide GPU count"
        )
    concurrency = int(os.environ["CONC_LIST"])
    if args.dp_size <= 0 or concurrency % args.dp_size:
        raise ValueError("Positive DP size must divide global concurrency")
    tokenizer = get_tokenizer(os.environ["MODEL"], trust_remote_code=True)
    np.random.seed(12345)
    prompts, hashes = [], set()
    while len(prompts) < concurrency:
        generated = sample_random_requests(
            0,
            int(os.environ["ISL"]),
            int(os.environ["OSL"]),
            concurrency - len(prompts),
            1.0,
            tokenizer,
            use_chat_template=True,
            dsv4=True,
            tokenizer_id=os.environ["MODEL"],
            trust_remote_code=True,
            num_workers=8,
        )
        for prompt in generated:
            digest = hashlib.sha256(prompt[0].encode()).hexdigest()
            if digest not in hashes:
                prompts.append(prompt)
                hashes.add(digest)
    args.output.mkdir(exist_ok=True)
    (args.output / "cohort-inputs.json").write_text(
        json.dumps(
            {
                "seed": 12345,
                "unique_prompts": len(hashes),
                "prompt_sha256": sorted(hashes),
                "isl": int(os.environ["ISL"]),
                "osl": int(os.environ["OSL"]),
                "configured_mean_committed_length": os.environ[
                    "FIXED_SEQUENCE_ACCEPTANCE_LENGTH"
                ],
                "draft_tokens": os.environ["FIXED_SEQUENCE_DRAFT_TOKENS"],
                "profile_wave_osl": max(1024, int(os.environ["OSL"]))
                if args.profile_steps
                else None,
                "dp_size": args.dp_size,
                "prefix_warmup_tokens": args.warmup_output_tokens,
                "framework": "vllm",
            },
            indent=2,
        )
        + "\n"
    )
    tokenized = []
    for prompt, prompt_length, output_length, mm in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) != prompt_length:
            raise ValueError("Token-ID request differs from the rendered chat prompt")
        tokenized.append((ids, prompt_length, output_length, mm))
    asyncio.run(run(args, tokenized, tokenizer))


if __name__ == "__main__":
    main()
