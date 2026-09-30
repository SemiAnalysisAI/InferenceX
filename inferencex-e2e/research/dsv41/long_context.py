"""One full-concurrency serving wave, steady-decode evidence, then optional profiling."""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def steady_decode_window(
    records: list[dict], *, burn: int, tail: int, gpu_count: int
) -> dict:
    """Common interior interval in which every request has started decode and none finished."""
    if not records or not all(r["success"] for r in records):
        raise ValueError(
            "Every request must complete before validating a full-batch window"
        )
    decoded = [[e for e in r["events"] if e[1] > 1] for r in records]
    if min(map(len, decoded)) <= burn + tail + 1:
        raise ValueError("Too few decode chunks for the requested interior window")
    start = max(events[burn][0] for events in decoded)
    end = min(events[-tail - 1][0] for events in decoded)
    if end <= start:
        raise ValueError("No common full-batch decode interval")
    deltas = []
    for r in records:
        before = max(n for t, n in r["events"] if t <= start)
        after = max(n for t, n in r["events"] if t <= end)
        deltas.append(after - before)
    tokens = sum(deltas)
    if min(deltas) <= 0:
        raise ValueError("A request made no progress inside the full-batch window")
    seconds = end - start
    return {
        "start_monotonic_s": start,
        "end_monotonic_s": end,
        "duration_s": seconds,
        "batch": len(records),
        "gpu_count": gpu_count,
        "observed_tokens": tokens,
        "per_request_tokens": deltas,
        "output_tokens_per_second": tokens / seconds,
        "output_tokens_per_second_per_gpu": tokens / seconds / gpu_count,
        "equivalent_tpot_ms": len(records) * seconds / tokens * 1000,
        "discarded_leading_decode_chunks": burn,
        "discarded_trailing_decode_chunks": tail,
        "boundary": "client-observed cumulative token chunks; excludes admission/prefill and drain, includes serving overhead; not a model-only timer",
    }


async def post(session, base, path, payload):
    async with session.post(base + path, json=payload) as response:
        raw = await response.text()
        if response.status != 200:
            raise RuntimeError(f"{path} HTTP {response.status}: {raw[:500]}")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw


async def stream_request(session, base, prompt, output_length, record):
    record["start"] = time.perf_counter()
    try:
        async with session.post(
            base + "/generate",
            json={
                "input_ids": prompt,
                "stream": True,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": output_length,
                    "ignore_eos": True,
                },
            },
        ) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"generate HTTP {response.status}: {(await response.text())[:500]}"
                )
            async for raw in response.content:
                text = raw.decode().strip()
                if not text.startswith("data:"):
                    continue
                text = text[5:].strip()
                if text == "[DONE]":
                    continue
                item = json.loads(text)
                meta = item.get("meta_info", {})
                count = int(meta.get("completion_tokens", 0))
                if count and (not record["events"] or count > record["events"][-1][1]):
                    record["events"].append([time.perf_counter(), count])
                record["meta"] = meta
        record["success"] = bool(
            record["events"] and record["events"][-1][1] == output_length
        )
        if not record["success"]:
            record["error"] = "Incorrect completion length"
    except Exception as error:  # noqa: BLE001 - retain every failed request in the cohort artifact
        record["error"] = f"{type(error).__name__}: {error}"
        record["success"] = False
    record["end"] = time.perf_counter()


async def wave(session, base, prompts, output_length, *, output, profile_steps):
    records = [
        {"events": [], "success": False, "meta": {}, "error": ""} for _ in prompts
    ]
    tasks = [
        asyncio.create_task(stream_request(session, base, p[0], output_length, r))
        for p, r in zip(prompts, records)
    ]
    profile_started = False
    marker = {}
    metric_names = (
        "num_running_reqs",
        "num_queue_reqs",
        "spec_accept_length",
        "spec_verify_calls_total",
        "generation_tokens_total",
        "decode_sum_seq_lens",
        "retract",
        "is_cuda_graph",
    )
    with gzip.open(
        output
        / (
            "profile-metrics.jsonl.gz" if profile_steps else "measured-metrics.jsonl.gz"
        ),
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
                marker["response"] = await post(
                    session,
                    base,
                    "/start_profile",
                    {
                        "output_dir": str(trace_dir),
                        "start_step": 0,
                        "num_steps": profile_steps,
                        "activities": ["CPU", "GPU"],
                        "record_shapes": True,
                        "detailed_annotations": True,
                        "profile_prefix": f"dsv41_batch{len(prompts)}",
                    },
                )
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
        (output / "profile-start.json").write_text(json.dumps(marker, indent=2) + "\n")
    return records


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
        connector=aiohttp.TCPConnector(limit=0),
        timeout=aiohttp.ClientTimeout(total=14400),
    ) as session:
        await post(
            session,
            base,
            "/generate",
            {
                "input_ids": prompts[0][0],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                    "ignore_eos": True,
                },
            },
        )
        await post(session, base, "/flush_cache", {})
        start_wall, start = time.time(), time.perf_counter()
        records = await wave(
            session, base, prompts, output_length, output=out, profile_steps=0
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
            backend="sglang",
            request_rate="inf",
            benchmark_outcome=benchmark_outcome(concurrency, metrics.completed),
        )
        steady_error = None
        try:
            result["steady_decode"] = steady_decode_window(
                records, burn=8, tail=8, gpu_count=args.gpu_count
            )
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
            # Reuse the original prompt prefixes; longer outputs leave a stable
            # cohort for capture. This wave is never used as the performance result.
            profiled = await wave(
                session,
                base,
                prompts,
                1024,
                output=out,
                profile_steps=args.profile_steps,
            )
            with gzip.open(out / "profile-events.json.gz", "wt") as f:
                json.dump(profiled, f)
            if not all(r["success"] for r in profiled):
                raise RuntimeError("Incomplete profiling cohort")
            traces = list((out / "profiles").glob("*.trace.json*"))
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
    parser.add_argument("--result-layout", choices=["single", "multi"], required=True)
    args = parser.parse_args()
    concurrency = int(os.environ["CONC_LIST"])
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
                "profile_wave_osl": 1024 if args.profile_steps else None,
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
