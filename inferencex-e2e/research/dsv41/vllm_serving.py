"""Unprofiled serving measurement followed by native vLLM profile capture."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
from pathlib import Path

from experiment import ROOT, request


def main() -> None:
    import numpy as np

    from infx.bench_serving.benchmark_serving import (
        get_tokenizer,
        sample_random_requests,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-count", type=int, required=True)
    args = parser.parse_args()
    if args.gpu_count <= 0:
        parser.error("--gpu-count must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "bash",
            str(ROOT / "benchmarks/single_node/srt_fixed_sequence.sh"),
            "--trust-remote-code",
            "--dsv4",
        ],
        check=True,
    )
    base = f"http://{os.environ['SRT_FRONTEND_HOST']}:{os.environ['SRT_FRONTEND_PORT']}"
    tokenizer = get_tokenizer(os.environ["MODEL"], trust_remote_code=True)
    np.random.seed(12345)
    prompt, length, output_length, _ = sample_random_requests(
        prefix_len=0,
        input_len=int(os.environ["ISL"]),
        output_len=int(os.environ["OSL"]),
        num_prompts=1,
        range_ratio=1.0,
        tokenizer=tokenizer,
        use_chat_template=True,
        dsv4=True,
        num_workers=1,
    )[0]
    payload = {
        "model": os.environ["MODEL"],
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": output_length,
        "ignore_eos": True,
        "stream": False,
    }
    request(base, "/v1/completions", payload)
    before = request(base, "/metrics")
    start = request(base, "/start_profile", {})
    try:
        response = request(base, "/v1/completions", payload)
    finally:
        stop = request(base, "/stop_profile", {})
    usage = response.get("usage", {})
    if (
        usage.get("prompt_tokens") != length
        or usage.get("completion_tokens") != output_length
    ):
        raise RuntimeError(
            f"Profile token mismatch: {usage}, expected {length}/{output_length}"
        )
    gpu_traces = []
    for path in (args.output / "serving").rglob("*.json*"):
        if not path.name.endswith((".json", ".json.gz")):
            continue
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as handle:
            trace = json.load(handle)
        count = sum(
            event.get("cat") == "kernel" for event in trace.get("traceEvents", [])
        )
        if count:
            gpu_traces.append(
                {"file": str(path.relative_to(args.output)), "kernel_events": count}
            )
    if len(gpu_traces) != args.gpu_count:
        raise RuntimeError(f"Expected {args.gpu_count} GPU traces, got {gpu_traces}")
    (args.output / "profile-request.json").write_text(
        json.dumps(
            {
                "framework": "vllm",
                "usage": usage,
                "gpu_count": args.gpu_count,
                "configured_committed_al": os.environ[
                    "FIXED_SEQUENCE_ACCEPTANCE_LENGTH"
                ],
                "draft_tokens": os.environ["FIXED_SEQUENCE_DRAFT_TOKENS"],
                "prefix_caching": False,
                "start_response": start,
                "stop_response": stop,
                "traces": gpu_traces,
            },
            indent=2,
        )
        + "\n"
    )
    (args.output / "metrics-before-profile.txt").write_text(str(before))
    (args.output / "metrics-after-profile.txt").write_text(
        str(request(base, "/metrics"))
    )
    print(f"Validated {len(gpu_traces)} vLLM GPU traces", flush=True)


if __name__ == "__main__":
    main()
