# SPDX-License-Identifier: Apache-2.0
r"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    vLLM OpenAI API server
    vllm serve <your_model> \
        --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_tgi_server.sh <your_model> <max_batch_total_tokens>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --model <your_model> \
        --dataset-name random \
        --request-rate <request_rate> \ # By default <request_rate> is inf
        --num-prompts <num_prompts> # By default <num_prompts> is 1000

    when using tgi backend, add
        --endpoint /generate_stream
    to the end of the command above.
"""
import argparse
import asyncio
import base64
import gc
import io
import json
import math
import multiprocessing as mp
import os
import random
import time
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Collection, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
from backend_request_func import (AIOHTTP_TIMEOUT, ASYNC_REQUEST_FUNCS,
                                  RequestFuncInput, RequestFuncOutput)
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser

from benchmark_utils import (convert_to_pytorch_benchmark_format,
                             shard_round_robin)

MILLISECONDS_TO_SECONDS_CONVERSION = 1000


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    request_goodput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: List[Tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: List[Tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: List[Tuple[float, float]]
    # E2EL stands for end-to-end latency per request.
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: List[Tuple[float, float]]


def sample_random_requests(
    prefix_len: int,
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
    use_chat_template: bool = False,
) -> List[Tuple[str, int, int]]:
    prefix_token_ids = np.random.randint(0, tokenizer.vocab_size, size=prefix_len).tolist()

    if use_chat_template:
        chat_template_dummy = tokenizer.apply_chat_template(
            [{"role": "user", "content": "a"}],
            add_generation_prompt=True,
            tokenize=False,
        )
        tokenized_chat_template_dummy = tokenizer.encode(chat_template_dummy, add_special_tokens=False)
        chat_template_len = len(tokenized_chat_template_dummy) - 1
        input_len = input_len - chat_template_len

    def sample_uniform(seq_len):
        lower = int(seq_len * range_ratio)
        upper = seq_len
        seq_lens = np.random.randint(lower, upper+1, size=num_prompts).tolist()
        return seq_lens

    input_lens = sample_uniform(input_len)
    output_lens = sample_uniform(output_len)
    offsets = np.random.randint(0, tokenizer.vocab_size, size=num_prompts)

    input_requests = []
    mismatches = []
    for i in range(num_prompts):
        tgt_prompt_len = prefix_len + input_lens[i]
        prompt_token_ids = prefix_token_ids + [(offsets[i] + i + j) % tokenizer.vocab_size for j in range(input_lens[i])]
        prompt = tokenizer.decode(prompt_token_ids)

        max_retries = 10
        for _ in range(max_retries):
            prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if len(prompt_token_ids) < tgt_prompt_len:
                num_extras = tgt_prompt_len - len(prompt_token_ids)
                prompt_token_ids.extend(np.random.randint(0, tokenizer.vocab_size, size=num_extras).tolist())
            elif len(prompt_token_ids) > tgt_prompt_len:
                prompt_token_ids = prompt_token_ids[:tgt_prompt_len]
            else:
                break
            prompt = tokenizer.decode(prompt_token_ids)

        if use_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )

        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        mismatches.append(prompt_len - tgt_prompt_len)
        input_requests.append((prompt, prompt_len, output_lens[i], None))

    header_str = f'{"-"*19}  Input/Output Length Statistics  {"-"*19}'
    print(header_str)
    print(
        f' input_lens : '
        f'min={min(r[1] for r in input_requests):<4d}  '
        f'max={max(r[1] for r in input_requests):<4d}  '
        f'mean={np.mean([r[1] for r in input_requests]):<7.2f}  '
        f'avg_token_mismatch={np.mean(mismatches):<5.2f} '
    )
    print(
        f' output_lens: '
        f'min={min(r[2] for r in input_requests):<4d}  '
        f'max={max(r[2] for r in input_requests):<4d}  '
        f'mean={np.mean([r[2] for r in input_requests]):<7.2f} '
    )
    print('-' * len(header_str), '\n')

    return input_requests


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    """
    Asynchronously generates requests at a specified rate
    with OPTIONAL burstiness.

    Args:
        input_requests:
            A list of input requests, each represented as a tuple.
        request_rate:
            The rate at which requests are generated (requests/s).
        burstiness (optional):
            The burstiness factor of the request generation.
            Only takes effect when request_rate is not inf.
            Default value is 1, which follows a Poisson process.
            Otherwise, the request intervals follow a gamma distribution.
            A lower burstiness value (0 < burstiness < 1) results
            in more bursty requests, while a higher burstiness value
            (burstiness > 1) results in a more uniform arrival of requests.
    """
    input_requests = iter(input_requests)

    # Calculate scale parameter theta to maintain the desired request_rate.
    assert burstiness > 0, (
        f"A positive burstiness factor is expected, but given {burstiness}.")
    theta = 1.0 / (request_rate * burstiness)

    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the gamma distribution.
        # If burstiness is 1, it follows exponential distribution.
        interval = np.random.gamma(shape=burstiness, scale=theta)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[float],
    goodput_config_dict: Dict[str, float],
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens: List[int] = []
    total_input = 0
    completed = 0
    good_completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    all_tpots: List[float] = []
    ttfts: List[float] = []
    e2els: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_tokens

            if output_len is None:
                # We use the tokenizer to count the number of output tokens
                # for some serving backends instead of looking at
                # len(outputs[i].itl) since multiple output tokens may be
                # bundled together
                # Note : this may inflate the output token count slightly
                output_len = len(
                    tokenizer(outputs[i].generated_text,
                              add_special_tokens=False).input_ids)
            actual_output_lens.append(output_len)
            # Use outputs[i].prompt_len rather than input_requests[i][1] so
            # metrics don't depend on output order matching input order —
            # workers return outputs as they complete, not in dispatch order.
            total_input += outputs[i].prompt_len
            tpot = 0
            if output_len > 1:
                latency_minus_ttft = outputs[i].latency - outputs[i].ttft
                tpot = latency_minus_ttft / (output_len - 1)
                tpots.append(tpot)
            # Note: if output_len <= 1, we regard tpot as 0 for goodput
            all_tpots.append(tpot)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    if goodput_config_dict:
        valid_metrics = []
        slo_values = []

        if "ttft" in goodput_config_dict:
            valid_metrics.append(ttfts)
            slo_values.append(goodput_config_dict["ttft"] /
                              MILLISECONDS_TO_SECONDS_CONVERSION)
        if "tpot" in goodput_config_dict:
            valid_metrics.append(all_tpots)
            slo_values.append(goodput_config_dict["tpot"] /
                              MILLISECONDS_TO_SECONDS_CONVERSION)
        if "e2el" in goodput_config_dict:
            valid_metrics.append(e2els)
            slo_values.append(goodput_config_dict["e2el"] /
                              MILLISECONDS_TO_SECONDS_CONVERSION)

        for req_metric in zip(*valid_metrics):
            is_good_req = all([s >= r for s, r in zip(slo_values, req_metric)])
            if is_good_req:
                good_completed += 1

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2)
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        request_goodput=good_completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) *
        1000,  # ttfts is empty if streaming is not supported by backend
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[(p, np.percentile(ttfts or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[(p, np.percentile(tpots or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[(p, np.percentile(itls or 0, p) * 1000)
                            for p in selected_percentiles],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[(p, np.percentile(e2els or 0, p) * 1000)
                             for p in selected_percentiles],
    )

    return metrics, actual_output_lens


# Per-worker batch size for streaming RequestFuncOutput back to main via mp.Queue.
# Batching amortizes pickling/lock overhead so the queue isn't the bottleneck
# at high QPS (at ~10k req/s, per-request puts contend on the queue's lock).
_WORKER_QUEUE_BATCH_SIZE = 64


def _build_client_session(connector_limit: int) -> aiohttp.ClientSession:
    connector = aiohttp.TCPConnector(
        limit=connector_limit,
        limit_per_host=connector_limit,
        keepalive_timeout=300,
        enable_cleanup_closed=True,
    )
    return aiohttp.ClientSession(connector=connector,
                                 trust_env=True,
                                 timeout=AIOHTTP_TIMEOUT)


async def _run_warmup(request_func, test_input: RequestFuncInput,
                      num_warmups: int, max_concurrency: Optional[int],
                      disable_tqdm: bool):
    pbar = None if disable_tqdm else tqdm(total=num_warmups, desc="warmup")
    sem = asyncio.Semaphore(max_concurrency) if max_concurrency else None
    limit = max_concurrency or 256
    async with _build_client_session(connector_limit=limit) as session:
        async def _one():
            if sem is None:
                out = await request_func(request_func_input=test_input,
                                         session=session)
            else:
                async with sem:
                    out = await request_func(request_func_input=test_input,
                                             session=session)
            if pbar is not None:
                pbar.update(1)
            return out
        await asyncio.gather(*[_one() for _ in range(num_warmups)])
    if pbar is not None:
        pbar.close()


async def _one_off_request(request_func, req_input: RequestFuncInput
                           ) -> RequestFuncOutput:
    async with _build_client_session(connector_limit=4) as session:
        return await request_func(request_func_input=req_input, session=session)


def _worker_entry(worker_index: int, shard: List[Tuple], config: Dict[str, Any],
                  barrier: Any, result_queue: Any, seed: int) -> None:
    """Subprocess entrypoint. Runs an asyncio loop over a shard of requests."""
    try:
        random.seed(seed + worker_index)
        np.random.seed(seed + worker_index)
        asyncio.run(_worker_async(shard, config, barrier, result_queue))
    except BaseException:
        traceback.print_exc()
    finally:
        # Sentinel: main drains until it has received one sentinel per worker.
        # Must run even on exception so main doesn't hang.
        try:
            result_queue.put(None)
        except Exception:
            pass


async def _worker_async(shard: List[Tuple], config: Dict[str, Any],
                        barrier: Any, result_queue: Any) -> None:
    request_func = ASYNC_REQUEST_FUNCS[config["backend"]]
    request_rate = config["rate_per_worker"]
    burstiness = config["burstiness"]
    sem_size = config["max_concurrency_per_worker"]
    theta = (1.0 / (request_rate * burstiness)
             if request_rate != float("inf") else 0.0)

    batch_buffer: List[RequestFuncOutput] = []

    def flush_batch(force: bool = False) -> None:
        if not batch_buffer:
            return
        if force or len(batch_buffer) >= _WORKER_QUEUE_BATCH_SIZE:
            result_queue.put(batch_buffer.copy())
            batch_buffer.clear()

    async with _build_client_session(
            connector_limit=config["connector_limit"]) as session:
        # Barrier synchronizes worker start across processes so aggregate QPS
        # is measured from a single wall clock. Blocks the event loop briefly
        # (only at startup, once), which is fine.
        barrier.wait()

        sem = asyncio.Semaphore(sem_size) if sem_size else None
        in_flight: set = set()

        async def _fire(req_input: RequestFuncInput) -> None:
            try:
                out = await request_func(request_func_input=req_input,
                                         session=session)
                batch_buffer.append(out)
                flush_batch()
            finally:
                if sem is not None:
                    sem.release()

        for prompt, prompt_len, output_len, mm_content in shard:
            # Semaphore on the DISPATCH side (not inside the task) bounds
            # in-flight tasks. Acquiring here prevents task pileup when
            # requests complete slower than they arrive.
            if sem is not None:
                await sem.acquire()

            if request_rate != float("inf"):
                interval = np.random.gamma(shape=burstiness, scale=theta)
                await asyncio.sleep(interval)

            req_input = RequestFuncInput(
                model=config["model_id"],
                model_name=config["model_name"],
                prompt=prompt,
                api_url=config["api_url"],
                prompt_len=prompt_len,
                output_len=output_len,
                logprobs=config["logprobs"],
                best_of=config["best_of"],
                multi_modal_content=mm_content,
                ignore_eos=config["ignore_eos"],
            )

            task = asyncio.create_task(_fire(req_input))
            in_flight.add(task)
            task.add_done_callback(in_flight.discard)

        if in_flight:
            await asyncio.gather(*in_flight)
        flush_batch(force=True)


def run_benchmark(
    backend: str,
    api_url: str,
    base_url: str,
    model_id: str,
    model_name: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int, Any]],
    logprobs: Optional[int],
    best_of: int,
    request_rate: float,
    burstiness: float,
    disable_tqdm: bool,
    num_warmups: int,
    profile: bool,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[str],
    ignore_eos: bool,
    goodput_config_dict: Dict[str, float],
    max_concurrency: Optional[int],
    num_client_workers: int,
    client_connector_limit: int,
    seed: int,
):
    if backend not in ASYNC_REQUEST_FUNCS:
        raise ValueError(f"Unknown backend: {backend}")
    request_func = ASYNC_REQUEST_FUNCS[backend]

    print("Starting initial single prompt test run...")
    test_prompt, test_prompt_len, test_output_len, test_mm_content = (
        input_requests[0])
    if backend != "openai-chat" and test_mm_content is not None:
        raise ValueError(
            "Multi-modal content is only supported on 'openai-chat' backend.")
    test_input = RequestFuncInput(
        model=model_id,
        model_name=model_name,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=test_output_len,
        logprobs=logprobs,
        best_of=best_of,
        multi_modal_content=test_mm_content,
        ignore_eos=ignore_eos,
    )

    if num_warmups > 0:
        print(f"Warming up with {num_warmups} requests...")
        asyncio.run(
            _run_warmup(request_func, test_input, num_warmups, max_concurrency,
                        disable_tqdm))
        print("Warmup completed.")

    if profile:
        print("Starting profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            model_name=model_name,
            prompt=test_prompt,
            api_url=base_url + "/start_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            extra_body={"num_steps": 1, "merge_profiles": True, "profile_by_stage": True},
            logprobs=logprobs,
            best_of=best_of,
            multi_modal_content=test_mm_content,
            ignore_eos=ignore_eos,
        )
        profile_output = asyncio.run(_one_off_request(request_func, profile_input))
        if profile_output.success:
            print("Profiler started")

    if burstiness == 1.0:
        distribution = "Poisson process"
    else:
        distribution = "Gamma distribution"

    print(f"Traffic request rate: {request_rate}")
    print(f"Burstiness factor: {burstiness} ({distribution})")
    print(f"Maximum request concurrency: {max_concurrency}")
    print(f"Client worker processes: {num_client_workers}")

    shards = shard_round_robin(input_requests, num_client_workers)

    rate_per_worker = (request_rate / num_client_workers
                       if request_rate != float("inf") else float("inf"))
    mc_per_worker = (math.ceil(max_concurrency / num_client_workers)
                     if max_concurrency else None)
    # Auto-size the per-worker aiohttp connector: must be at least as large as
    # the in-flight cap, with a floor so unconstrained runs still get pooling.
    if client_connector_limit and client_connector_limit > 0:
        connector_limit = client_connector_limit
    else:
        connector_limit = max(256, mc_per_worker or 256)

    config = {
        "backend": backend,
        "api_url": api_url,
        "model_id": model_id,
        "model_name": model_name,
        "logprobs": logprobs,
        "best_of": best_of,
        "ignore_eos": ignore_eos,
        "rate_per_worker": rate_per_worker,
        "burstiness": burstiness,
        "max_concurrency_per_worker": mc_per_worker,
        "connector_limit": connector_limit,
    }

    ctx = mp.get_context("spawn")
    # +1 for main so workers don't start timing the run before main's pbar
    # and stopwatch are ready.
    barrier = ctx.Barrier(num_client_workers + 1)
    # Bounded queue provides back-pressure: if main can't drain fast enough,
    # workers block on put rather than buffering unbounded results in memory.
    result_queue = ctx.Queue(maxsize=num_client_workers * 32)

    processes = []
    for i, shard in enumerate(shards):
        p = ctx.Process(
            target=_worker_entry,
            args=(i, shard, config, barrier, result_queue, seed),
        )
        p.start()
        processes.append(p)

    print("Starting main benchmark run...")
    total = len(input_requests)
    pbar = None if disable_tqdm else tqdm(total=total, desc="bench")

    # Wait on the barrier too so our stopwatch starts in sync with workers.
    barrier.wait()
    benchmark_start_time = time.perf_counter()

    outputs: List[RequestFuncOutput] = []
    sentinels = 0
    while sentinels < num_client_workers:
        item = result_queue.get()
        if item is None:
            sentinels += 1
            continue
        outputs.extend(item)
        if pbar is not None:
            pbar.update(len(item))

    benchmark_duration = time.perf_counter() - benchmark_start_time

    if pbar is not None:
        pbar.close()

    for p in processes:
        p.join(timeout=30)
        if p.is_alive():
            warnings.warn(
                f"Worker {p.pid} did not exit cleanly; terminating.",
                stacklevel=2)
            p.terminate()

    for p in processes:
        if p.exitcode not in (0, None):
            warnings.warn(
                f"Worker {p.pid} exited with code {p.exitcode}; "
                "aggregated results may be incomplete.",
                stacklevel=2)

    if profile:
        print("Stopping profiler...")
        stop_profile_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=base_url + "/stop_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
            best_of=best_of,
        )
        stop_output = asyncio.run(_one_off_request(request_func, stop_profile_input))
        if stop_output.success:
            print("Profiler stopped")

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentile_metrics=selected_percentile_metrics,
        selected_percentiles=selected_percentiles,
        goodput_config_dict=goodput_config_dict,
    )

    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):",
                                    benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:",
                                 metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):",
                                    metrics.request_throughput))
    if goodput_config_dict:
        print("{:<40} {:<10.2f}".format("Request goodput (req/s):",
                                        metrics.request_goodput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):",
                                    metrics.output_throughput))
    print("{:<40} {:<10.2f}".format("Total Token throughput (tok/s):",
                                    metrics.total_token_throughput))

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "request_goodput:":
        metrics.request_goodput if goodput_config_dict else None,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }

    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function prints and adds statistics of the specified
        # metric.
        if metric_attribute_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c='-'))
        print("{:<40} {:<10.2f}".format(
            f"Mean {metric_name} (ms):",
            getattr(metrics, f"mean_{metric_attribute_name}_ms")))
        print("{:<40} {:<10.2f}".format(
            f"Median {metric_name} (ms):",
            getattr(metrics, f"median_{metric_attribute_name}_ms")))
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms")
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms")
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms")
        for p, value in getattr(metrics,
                                f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):",
                                            value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT",
                       "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    print("=" * 50)

    return result


def check_goodput_args(args):
    # Check and parse goodput arguments
    goodput_config_dict = {}
    VALID_NAMES = ["ttft", "tpot", "e2el"]
    if args.goodput:
        goodput_config_dict = parse_goodput(args.goodput)
        for slo_name, slo_val in goodput_config_dict.items():
            if slo_name not in VALID_NAMES:
                raise ValueError(
                    f"Invalid metric name found, {slo_name}: {slo_val}. "
                    "The service level objective name should be one of "
                    f"{str(VALID_NAMES)}. ")
            if slo_val < 0:
                raise ValueError(
                    f"Invalid value found, {slo_name}: {slo_val}. "
                    "The service level objective value should be "
                    "non-negative.")
    return goodput_config_dict


def parse_goodput(slo_pairs):
    goodput_config_dict = {}
    try:
        for slo_pair in slo_pairs:
            slo_name, slo_val = slo_pair.split(":")
            goodput_config_dict[slo_name] = float(slo_val)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            "Invalid format found for service level objectives. "
            "Specify service level objectives for goodput as \"KEY:VALUE\" "
            "pairs, where the key is a metric name, and the value is a "
            "number in milliseconds.") from err
    return goodput_config_dict


def save_to_pytorch_benchmark_format(args: argparse.Namespace,
                                     results: Dict[str, Any],
                                     file_name: str) -> None:
    metrics = [
        "median_ttft_ms", "mean_ttft_ms", "std_ttft_ms", "p99_ttft_ms",
        "mean_tpot_ms", "median_tpot_ms", "std_tpot_ms", "p99_tpot_ms",
        "median_itl_ms", "mean_itl_ms", "std_itl_ms", "p99_itl_ms"
    ]
    # These raw data might be useful, but they are rather big. They can be added
    # later if needed
    ignored_metrics = ["ttfts", "itls", "generated_texts", "errors"]
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={k: [results[k]]
                 for k in metrics},
        extra_info={
            k: results[k]
            for k in results if k not in metrics and k not in ignored_metrics
        })
    if pt_records:
        # Don't use json suffix here as we don't want CI to pick it up
        pt_file = f"{os.path.splitext(file_name)[0]}.pytorch.json"
        with open(pt_file, "w") as f:
            json.dump(pt_records, f)


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    model_name = args.served_model_name
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer_mode = args.tokenizer_mode

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"

    tokenizer = get_tokenizer(tokenizer_id,
                              tokenizer_mode=tokenizer_mode,
                              trust_remote_code=args.trust_remote_code)


    if args.dataset_name != "random":
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    input_requests = sample_random_requests(
        prefix_len=args.random_prefix_len,
        input_len=args.random_input_len,
        output_len=args.random_output_len,
        num_prompts=args.num_prompts,
        range_ratio=args.random_range_ratio,
        tokenizer=tokenizer,
        use_chat_template=args.use_chat_template,
    )

    goodput_config_dict = check_goodput_args(args)

    # Avoid GC processing "static" data - reduce pause times.
    gc.collect()
    gc.freeze()

    benchmark_result = run_benchmark(
        backend=backend,
        api_url=api_url,
        base_url=base_url,
        model_id=model_id,
        model_name=model_name,
        tokenizer=tokenizer,
        input_requests=input_requests,
        logprobs=args.logprobs,
        best_of=args.best_of,
        request_rate=args.request_rate,
        burstiness=args.burstiness,
        disable_tqdm=args.disable_tqdm,
        num_warmups=args.num_warmups,
        profile=args.profile,
        selected_percentile_metrics=args.percentile_metrics.split(","),
        selected_percentiles=[
            float(p) for p in args.metric_percentiles.split(",")
        ],
        ignore_eos=args.ignore_eos,
        goodput_config_dict=goodput_config_dict,
        max_concurrency=args.max_concurrency,
        num_client_workers=args.num_client_workers,
        client_connector_limit=args.client_connector_limit,
        seed=args.seed,
    )

    # Save config and results to json
    if args.save_result:
        result_json: Dict[str, Any] = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["best_of"] = args.best_of
        result_json["num_prompts"] = args.num_prompts

        # Metadata
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=")
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError(
                        "Invalid metadata format. Please use KEY=VALUE format."
                    )

        # Traffic
        result_json["request_rate"] = (args.request_rate if args.request_rate
                                       < float("inf") else "inf")
        result_json["burstiness"] = args.burstiness
        result_json["max_concurrency"] = args.max_concurrency

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}
        
        if not args.save_detailed:
            # Remove fields with too many data points
            for field in [
                "input_lens",
                "output_lens",
                "ttfts",
                "itls",
                "generated_texts",
                "errors",
            ]:
                if field in result_json:
                    del result_json[field]
                if field in benchmark_result:
                    del benchmark_result[field]

        # Save to file
        base_model_id = model_id.split("/")[-1]
        max_concurrency_str = (f"-concurrency{args.max_concurrency}"
                               if args.max_concurrency is not None else "")
        file_name = f"{backend}-{args.request_rate}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json"  #noqa
        if args.result_filename:
            file_name = args.result_filename
        if args.result_dir:
            file_name = os.path.join(args.result_dir, file_name)
        with open(file_name, "w", encoding='utf-8') as outfile:
            json.dump(result_json, outfile)
        save_to_pytorch_benchmark_format(args, result_json, file_name)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    # Use 127.0.0.1 here instead of localhost to force the use of ipv4
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="random",
        choices=["random"],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help=
        "Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and "
        "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help=("Number of logprobs-per-token to compute & return as part of "
              "the request. If unspecified, then either (1) if beam search "
              "is disabled, no logprobs are computed & a single dummy "
              "logprob is returned for each token; or (2) if beam search "
              "is enabled 1 logprob per token is computed"),
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process or gamma distribution "
        "to synthesize the request arrival times.",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor of the request generation. "
        "Only take effect when request_rate is not inf. "
        "Default value is 1, which follows Poisson process. "
        "Otherwise, the request intervals follow a gamma distribution. "
        "A lower burstiness value (0 < burstiness < 1) results in more "
        "bursty requests. A higher burstiness value (burstiness > 1) "
        "results in a more uniform arrival of requests.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use Torch Profiler. The endpoint must be launched with "
        "VLLM_TORCH_PROFILER_DIR to enable profiler.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--save-detailed",
        action="store_true",
        default=False,
        help="When saving results, include detailed per-request data "
        "(input_lens, output_lens, ttfts, itls, generated_texts, errors). "
        "By default, only aggregated metrics are saved to reduce file size.",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Specify the filename to save benchmark json results."
        "If not specified, results will be saved in "
        "{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
        " format.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Set ignore_eos flag when sending the benchmark request."
        "Warning: ignore_eos is not supported in deepspeed_mii and tgi.")
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl,e2el",
        help="Comma-seperated list of selected metrics to report percentils. "
        "This argument specifies the metrics to report percentiles. "
        "Allowed metric names are \"ttft\", \"tpot\", \"itl\", \"e2el\". "
        "Default value is \"ttft,tpot,itl,e2el\".")
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="90,99,99.9",
        help="Comma-seperated list of percentiles for selected metrics. "
        "To report 25-th, 50-th, and 75-th percentiles, use \"25,50,75\". "
        "Default value is \"90,99,99.9\". "
        "Use \"--percentile-metrics\" to select metrics.",
    )
    parser.add_argument(
        "--goodput",
        nargs="+",
        required=False,
        help="Specify service level objectives for goodput as \"KEY:VALUE\" "
        "pairs, where the key is a metric name, and the value is in "
        "milliseconds. Multiple \"KEY:VALUE\" pairs can be provided, "
        "separated by spaces. Allowed request level metric names are "
        "\"ttft\", \"tpot\", \"e2el\". For more context on the definition of "
        "goodput, refer to DistServe paper: https://arxiv.org/pdf/2401.09670 "
        "and the blog: https://hao-ai-lab.github.io/blogs/distserve")

    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help=
        "Number of input tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help=
        "Number of output tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-range-ratio",
        type=float,
        default=1.0,
        help="Range of sampled ratio of input/output length, "
        "used only for random sampling.",
    )
    random_group.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help="Number of fixed prefix tokens before random "
        " context. The length range of context in a random "
        " request is [random-prefix-len, "
        " random-prefix-len + random-prefix-len * random-range-ratio).")
    random_group.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Use chat template to format the prompt.",
    )

    parser.add_argument(
        '--tokenizer-mode',
        type=str,
        default="auto",
        choices=['auto', 'slow', 'mistral', 'custom'],
        help='The tokenizer mode.\n\n* "auto" will use the '
        'fast tokenizer if available.\n* "slow" will '
        'always use the slow tokenizer. \n* '
        '"mistral" will always use the `mistral_common` tokenizer. \n*'
        '"custom" will use --tokenizer to select the preregistered tokenizer.')

    parser.add_argument("--served-model-name",
                        type=str,
                        default=None,
                        help="The model name used in the API. "
                        "If not specified, the model name will be the "
                        "same as the ``--model`` argument. ")

    parser.add_argument('--num-warmups', type=int, default=0)

    # Cap the auto-detected default so a 128-vCPU host doesn't spawn 128
    # workers by accident. Override via BENCH_CLIENT_WORKERS_CAP env var when
    # you actually want more (or fewer) by default.
    _default_workers_cap = int(os.environ.get("BENCH_CLIENT_WORKERS_CAP", "8"))
    parser.add_argument(
        "--num-client-workers",
        type=int,
        default=min(os.cpu_count() or 1, _default_workers_cap),
        help="Number of client worker processes. Each runs its own asyncio "
        f"loop and one shared aiohttp session. Defaults to min(cpu_count, "
        f"{_default_workers_cap}) — the cap is set via "
        "BENCH_CLIENT_WORKERS_CAP (default 8). Raise to drive higher QPS; "
        "single-process Python maxes out around a few hundred QPS due to "
        "GIL/event-loop contention.")

    parser.add_argument(
        "--client-connector-limit",
        type=int,
        default=0,
        help="Per-worker aiohttp TCPConnector limit. 0 = auto (max(256, "
        "max_concurrency/num_workers)). Raise if the client runs out of "
        "sockets before the server is saturated.")

    args = parser.parse_args()
    main(args)
