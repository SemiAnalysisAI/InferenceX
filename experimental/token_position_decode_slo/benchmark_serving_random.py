# SPDX-License-Identifier: Apache-2.0
"""Minimal benchmark for online serving throughput with random prompts."""
import argparse
import asyncio
import contextlib
import gc
import json
import random
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    logprobs: Optional[int] = None
    ignore_eos: bool = False


@dataclass
class RequestFuncOutput:
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0
    prompt_len: int = 0
    error: str = ""


async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)

    async with aiohttp.ClientSession(trust_env=True, timeout=timeout) as session:
        payload = {
            "model": request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": 0.0,
            "max_tokens": request_func_input.output_len,
            "logprobs": request_func_input.logprobs,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        }
        if request_func_input.ignore_eos:
            payload["ignore_eos"] = request_func_input.ignore_eos

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    first_chunk_received = False
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                        if chunk != "[DONE]":
                            data = json.loads(chunk)

                            if choices := data.get("choices"):
                                timestamp = time.perf_counter()
                                if not first_chunk_received:
                                    first_chunk_received = True
                                    output.ttft = timestamp - st
                                most_recent_timestamp = timestamp
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get("completion_tokens")
                    if first_chunk_received:
                        output.success = True
                    else:
                        output.success = False
                        output.error = (
                            "Never received a valid chunk to calculate TTFT. "
                            "This response will be marked as failed!"
                        )
                    output.latency = most_recent_timestamp - st
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


@dataclass
class BenchmarkMetrics:
    # Basic throughput metrics
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    # Latency metrics (median + p99): ttft, e2el, tpot
    median_ttft: float
    p99_ttft: float
    median_e2el: float
    p99_e2el: float
    median_tpot: float
    p99_tpot: float
    # Goodput-related
    good_completed: int
    request_goodput: float


def sample_random_requests(
    prefix_len: int,
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int, None]]:
    prefix_token_ids = np.random.randint(0, tokenizer.vocab_size, size=prefix_len).tolist()

    def sample_uniform(seq_len):
        lower = int(seq_len * (1.0 - range_ratio))
        upper = int(seq_len * (1.0 + range_ratio))
        seq_lens = np.random.randint(lower, upper + 1, size=num_prompts).tolist()
        return seq_lens

    input_lens = sample_uniform(input_len)
    output_lens = sample_uniform(output_len)
    offsets = np.random.randint(0, tokenizer.vocab_size, size=num_prompts)

    input_requests = []
    mismatches = []
    for i in range(num_prompts):
        tgt_prompt_len = prefix_len + input_lens[i]
        prompt_token_ids = prefix_token_ids + [
            (offsets[i] + i + j) % tokenizer.vocab_size for j in range(input_lens[i])
        ]
        prompt = tokenizer.decode(prompt_token_ids)

        max_retries = 10
        for _ in range(max_retries):
            prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if len(prompt_token_ids) < tgt_prompt_len:
                num_extras = tgt_prompt_len - len(prompt_token_ids)
                prompt_token_ids.extend(
                    np.random.randint(0, tokenizer.vocab_size, size=num_extras).tolist()
                )
            elif len(prompt_token_ids) > tgt_prompt_len:
                prompt_token_ids = prompt_token_ids[:tgt_prompt_len]
            else:
                break
            prompt = tokenizer.decode(prompt_token_ids)

        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        mismatches.append(prompt_len - tgt_prompt_len)
        input_requests.append((prompt, prompt_len, output_lens[i], None))

    header_str = f'{"-"*19}  Input/Output Length Statistics  {"-"*19}'
    print(header_str)
    print(
        f" input_lens : "
        f"min={min(r[1] for r in input_requests):<4d}  "
        f"max={max(r[1] for r in input_requests):<4d}  "
        f"mean={np.mean([r[1] for r in input_requests]):<7.2f}  "
        f"avg_token_mismatch={np.mean(mismatches):<5.2f} "
    )
    print(
        f" output_lens: "
        f"min={min(r[2] for r in input_requests):<4d}  "
        f"max={max(r[2] for r in input_requests):<4d}  "
        f"mean={np.mean([r[2] for r in input_requests]):<7.2f} "
    )
    print("-" * len(header_str), "\n")

    return input_requests


async def get_request(
    input_requests: List[Tuple[str, int, int, None]],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[Tuple[str, int, int, None], None]:
    input_requests = iter(input_requests)
    assert burstiness > 0, f"A positive burstiness factor is expected, but given {burstiness}."
    theta = 1.0 / (request_rate * burstiness)

    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            continue

        interval = np.random.gamma(shape=burstiness, scale=theta)
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: List[Tuple[str, int, int, None]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    goodput_config_dict: Dict[str, float],
) -> Tuple[BenchmarkMetrics, List[int]]:
    ms_to_s = 1000

    actual_output_lens: List[int] = []
    total_input = 0
    completed = 0
    good_completed = 0
    tpots: List[float] = []
    all_tpots: List[float] = []
    ttfts: List[float] = []
    e2els: List[float] = []

    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_tokens
            actual_output_lens.append(output_len)
            total_input += input_requests[i][1]
            tpot = 0
            if output_len > 1:
                latency_minus_ttft = outputs[i].latency - outputs[i].ttft
                tpot = latency_minus_ttft / (output_len - 1)
                tpots.append(tpot)
            all_tpots.append(tpot)
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
            slo_values.append(goodput_config_dict["ttft"] / ms_to_s)
        if "tpot" in goodput_config_dict:
            valid_metrics.append(all_tpots)
            slo_values.append(goodput_config_dict["tpot"] / ms_to_s)
        if "e2el" in goodput_config_dict:
            valid_metrics.append(e2els)
            slo_values.append(goodput_config_dict["e2el"] / ms_to_s)

        for req_metric in zip(*valid_metrics):
            is_good_req = all([s >= r for s, r in zip(slo_values, req_metric)])
            if is_good_req:
                good_completed += 1

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )

    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        median_ttft=np.median(ttfts or 0),
        p99_ttft=np.percentile(ttfts or 0, 99),
        median_e2el=np.median(e2els or 0),
        p99_e2el=np.percentile(e2els or 0, 99),
        median_tpot=np.median(tpots or 0),
        p99_tpot=np.percentile(tpots or 0, 99),
        good_completed=good_completed,
        request_goodput=good_completed / dur_s,
    )

    return metrics, actual_output_lens


async def benchmark(
    api_url: str,
    base_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int, None]],
    logprobs: Optional[int],
    request_rate: float,
    burstiness: float,
    disable_tqdm: bool,
    num_warmups: int,
    profile: bool,
    ignore_eos: bool,
    goodput_config_dict: Dict[str, float],
    max_concurrency: Optional[int],
):
    test_prompt, test_prompt_len, test_output_len, _ = input_requests[0]
    test_input = RequestFuncInput(
        model=model_id,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=test_output_len,
        logprobs=logprobs,
        ignore_eos=ignore_eos,
    )

    if num_warmups > 0:
        print(f"Warming up with {num_warmups} requests...")
        warmup_pbar = None if disable_tqdm else tqdm(total=num_warmups)
        warmup_semaphore = (
            asyncio.Semaphore(max_concurrency) if max_concurrency else contextlib.nullcontext()
        )

        async def warmup_limited_req_fn():
            async with warmup_semaphore:
                return await async_request_openai_completions(
                    request_func_input=test_input, pbar=warmup_pbar
                )

        warmup_tasks = []
        for _ in range(num_warmups):
            task = asyncio.create_task(warmup_limited_req_fn())
            warmup_tasks.append(task)
        _ = await asyncio.gather(*warmup_tasks)

        if warmup_pbar is not None:
            warmup_pbar.close()
        print("Warmup completed.")

    if profile:
        print("Starting profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=base_url + "/start_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
            ignore_eos=ignore_eos,
        )
        profile_output = await async_request_openai_completions(
            request_func_input=profile_input
        )
        if profile_output.success:
            print("Profiler started")

    if burstiness == 1.0:
        distribution = "Poisson process"
    else:
        distribution = "Gamma distribution"

    print(f"Traffic request rate: {request_rate}")
    print(f"Burstiness factor: {burstiness} ({distribution})")
    print(f"Maximum request concurrency: {max_concurrency}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else contextlib.nullcontext()

    async def limited_request_func(request_func_input, pbar):
        async with semaphore:
            return await async_request_openai_completions(
                request_func_input=request_func_input, pbar=pbar
            )

    print("Starting main benchmark run...")

    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate, burstiness):
        prompt, prompt_len, output_len, _ = request
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            logprobs=logprobs,
            ignore_eos=ignore_eos,
        )
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input, pbar=pbar)
            )
        )
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if profile:
        print("Stopping profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=base_url + "/stop_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
        )
        profile_output = await async_request_openai_completions(
            request_func_input=profile_input
        )
        if profile_output.success:
            print("Profiler stopped")

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        goodput_config_dict=goodput_config_dict,
    )

    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    if goodput_config_dict:
        print("{:<40} {:<10}".format("Good requests:", metrics.good_completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", metrics.request_throughput))
    if goodput_config_dict:
        print("{:<40} {:<10.2f}".format("Request goodput (req/s):", metrics.request_goodput))
    print("{:<40} {:<10.2f}".format("Input token throughput (tok/s):", metrics.input_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", metrics.output_throughput))

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "good_completed": metrics.good_completed if goodput_config_dict else None,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "request_goodput": metrics.request_goodput if goodput_config_dict else None,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
    }

    def process_one_metric(
        metric_attribute_name: str,
        metric_name: str,
        metric_header: str,
    ):
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
        median = getattr(metrics, f"median_{metric_attribute_name}")
        p99 = getattr(metrics, f"p99_{metric_attribute_name}")
        print("{:<40} {:<10.4f}".format(f"Median {metric_name} (s):", median))
        print("{:<40} {:<10.4f}".format(f"P99 {metric_name} (s):", p99))
        result[f"median_{metric_attribute_name}"] = median
        result[f"p99_{metric_attribute_name}"] = p99

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")
    process_one_metric("tpot", "TPOT", "Time per Output Token (excl. 1st token)")

    print("=" * 50)

    return result


def check_goodput_args(args):
    goodput_config_dict = {}
    VALID_NAMES = ["ttft", "tpot", "e2el"]
    if args.goodput:
        goodput_config_dict = parse_goodput(args.goodput)
        for slo_name, slo_val in goodput_config_dict.items():
            if slo_name not in VALID_NAMES:
                raise ValueError(
                    f"Invalid metric name found, {slo_name}: {slo_val}. "
                    "The service level objective name should be one of "
                    f"{str(VALID_NAMES)}. "
                )
            if slo_val < 0:
                raise ValueError(
                    f"Invalid value found, {slo_name}: {slo_val}. "
                    "The service level objective value should be "
                    "non-negative."
                )
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
            "number in milliseconds."
        ) from err
    return goodput_config_dict


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model_id = args.model
    tokenizer_id = args.model

    api_url = f"{args.base_url}/v1/completions"
    base_url = args.base_url

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        trust_remote_code=True,
    )

    input_requests = sample_random_requests(
        prefix_len=args.random_prefix_len,
        input_len=args.random_input_len,
        output_len=args.random_output_len,
        num_prompts=args.num_prompts,
        range_ratio=args.random_range_ratio,
        tokenizer=tokenizer,
    )

    goodput_config_dict = check_goodput_args(args)

    gc.collect()
    gc.freeze()

    benchmark_result = asyncio.run(
        benchmark(
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            logprobs=args.logprobs,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            disable_tqdm=args.disable_tqdm,
            num_warmups=args.num_warmups,
            profile=args.profile,
            ignore_eos=args.ignore_eos,
            goodput_config_dict=goodput_config_dict,
            max_concurrency=args.max_concurrency,
        )
    )

    if args.result_filepath:
        result_json: Dict[str, Any] = {}

        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["num_prompts"] = args.num_prompts

        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=")
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError(
                        "Invalid metadata format. Please use KEY=VALUE format."
                    )

        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf"
        )
        result_json["burstiness"] = args.burstiness
        result_json["max_concurrency"] = args.max_concurrency

        result_json = {**result_json, **benchmark_result}

        with open(args.result_filepath, "w", encoding="utf-8") as outfile:
            json.dump(result_json, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument(
        "--base-url",
        type=str,
        required=True,
        help="Server or API base url (e.g., http://localhost:8000).",
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
        "if the server is not processing requests fast enough to keep up.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
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
        help="Number of logprobs-per-token to compute & return as part of "
        "the request.",
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
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-filepath",
        type=str,
        default=None,
        help="Specify the filepath to save benchmark json results. "
        "If not specified, results are not saved to file.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Set ignore_eos flag when sending the benchmark request.",
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
        "and the blog: https://hao-ai-lab.github.io/blogs/distserve",
    )
    parser.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Number of input tokens per request, used only for random sampling.",
    )
    parser.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help="Number of output tokens per request, used only for random sampling.",
    )
    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=1.0,
        help="Range of sampled ratio of input/output length, "
        "used only for random sampling.",
    )
    parser.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help="Number of fixed prefix tokens before random "
        " context. The length range of context in a random "
        " request is [random-prefix-len, "
        " random-prefix-len + random-prefix-len * random-range-ratio).",
    )
    parser.add_argument("--num-warmups", type=int, default=0)

    args = parser.parse_args()
    main(args)