"""Research-only serving traces and standalone Engram timings in a pinned image.

Run the normal serving measurement first. Profiling and microbenchmarks happen
later and must never be used as the unprofiled end-to-end performance result.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def request(base: str, path: str, payload: dict | None = None):
    req = urllib.request.Request(
        base + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as response:
        raw = response.read().decode()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def serving_profile(output: Path, expected_ranks: int) -> None:
    import numpy as np

    from infx.bench_serving.benchmark_serving import (
        get_tokenizer,
        sample_random_requests,
    )

    base = f"http://{os.environ['SRT_FRONTEND_HOST']}:{os.environ['SRT_FRONTEND_PORT']}"
    model = os.environ["MODEL"]
    tokenizer = get_tokenizer(model, trust_remote_code=True)
    np.random.seed(12345)
    prompt, length, out_length, _ = sample_random_requests(
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
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": out_length,
        "ignore_eos": True,
        "stream": False,
    }
    request(base, "/v1/completions", payload)
    request(base, "/flush_cache", {})
    before = request(base, "/metrics")
    profile_dir = output / "serving"
    profile_dir.mkdir(parents=True, exist_ok=True)
    start = request(
        base,
        "/start_profile",
        {
            "output_dir": str(profile_dir),
            "start_step": 0,
            "num_steps": 16,
            "activities": ["CPU", "GPU"],
            "record_shapes": True,
            "with_stack": False,
            "detailed_annotations": True,
            "profile_prefix": "dsv41_8k256",
        },
    )
    if isinstance(start, dict) and start.get("success") is False:
        raise RuntimeError(f"Profiler refused capture: {start}")
    response = request(base, "/v1/completions", payload)
    # The pinned API stops and exports automatically after num_steps.
    stop = "automatic after 16 scheduler steps"
    after = request(base, "/metrics")
    usage = response.get("usage", {})
    if (
        usage.get("prompt_tokens") != length
        or usage.get("completion_tokens") != out_length
    ):
        raise RuntimeError(
            f"Profile request length mismatch: {usage}, expected {length}/{out_length}"
        )
    (output / "profile-request.json").write_text(
        json.dumps(
            {
                "model": model,
                "input_tokens": length,
                "output_tokens": out_length,
                "seed": 12345,
                "profile_steps": 16,
                "start_response": start,
                "stop_response": stop,
                "usage": usage,
                "configured_mean_committed_length": os.environ[
                    "FIXED_SEQUENCE_ACCEPTANCE_LENGTH"
                ],
                "draft_tokens": os.environ["FIXED_SEQUENCE_DRAFT_TOKENS"],
                "timing_qualification": "Profiled request; not a serving-latency measurement",
            },
            indent=2,
        )
        + "\n"
    )
    (output / "metrics-before-profile.txt").write_text(str(before))
    (output / "metrics-after-profile.txt").write_text(str(after))
    traces = list(profile_dir.rglob("*.json")) + list(profile_dir.rglob("*.json.gz"))
    if len(traces) != expected_ranks:
        raise RuntimeError(
            f"Expected {expected_ranks} rank traces, got {len(traces)} in {profile_dir}"
        )
    print(f"Serving profile: {len(traces)} trace files", flush=True)


def kernel_samples(trace: dict, count: int) -> list[float]:
    """Sum device kernels within each GPU annotation, excluding launch gaps."""
    events = trace["traceEvents"]
    scopes = [
        e
        for e in events
        if e.get("cat") == "gpu_user_annotation"
        and e.get("name", "").startswith("target_sample_")
    ]
    if len(scopes) != count or len({e["name"] for e in scopes}) != count:
        raise RuntimeError("Missing or duplicate GPU sample annotations")
    kernels = [e for e in events if e.get("cat") == "kernel"]
    samples = []
    for scope in sorted(scopes, key=lambda e: int(e["name"].rsplit("_", 1)[1])):
        start, end = scope["ts"], scope["ts"] + scope["dur"]
        duration = sum(
            e["dur"]
            for e in kernels
            if e["pid"] == scope["pid"]
            and e["tid"] == scope["tid"]
            and e["ts"] >= start - 0.001
            and e["ts"] + e["dur"] <= end + 0.001
        )
        if duration <= 0:
            raise RuntimeError(f"No GPU kernels in {scope['name']}")
        samples.append(duration)
    return samples


def engram_bench(output: Path, *, resident_server: bool) -> None:
    import torch
    from sglang.kernels.ops.embeddings.engram_gate import fused_engram_gate

    torch.manual_seed(12345)
    torch.cuda.set_device(0)
    cases = [
        (1, False),
        (72, False),
        (128, False),
        (512, True),
        (1024, False),
        (4096, False),
        (8192, True),
        (8192, False),
        (16384, False),
    ]
    eviction = torch.empty(256 * 1024 * 1024 // 2, device="cuda", dtype=torch.float16)
    eviction.fill_(1)
    records = []
    output.mkdir(parents=True, exist_ok=True)
    for tokens, masked in cases:
        x = torch.randn(tokens, 4, 5120, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(tokens, 5 * 5120, device="cuda", dtype=torch.bfloat16)
        qw = torch.randn(4, 5120, device="cuda", dtype=torch.float32)
        kw = torch.randn_like(qw)
        image_mask = (torch.arange(tokens, device="cuda") % 2 == 1)[:, None, None]

        def reference(
            x=x,
            kv=kv,
            qw=qw,
            kw=kw,
            image_mask=image_mask,
            tokens=tokens,
            masked=masked,
        ):
            h = x.float()
            key = kv[:, : 4 * 5120].float().reshape(tokens, 4, 5120)
            value = kv[:, 4 * 5120 :].float()
            rstd = torch.rsqrt(h.square().mean(-1) + 1e-20)
            rstd = rstd * torch.rsqrt(key.square().mean(-1) + 1e-20)
            dot = (h * (qw * kw) * key).sum(-1) * rstd * (5120**-0.5)
            gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
            result = (h + gate.unsqueeze(-1) * value.unsqueeze(-2)).to(x.dtype)
            return torch.where(image_mask, x, result) if masked else result

        def native(x=x, kv=kv, qw=qw, kw=kw, image_mask=image_mask, masked=masked):
            result = fused_engram_gate(x, kv, qw, kw, 1e-20, 1e-6)
            return torch.where(image_mask, x, result) if masked else result

        expected, actual = reference(), native()
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
        max_error = (actual.float() - expected.float()).abs().max().item()
        if masked and not torch.equal(actual[1::2], x[1::2]):
            raise RuntimeError("Masked rows must preserve the input exactly")
        del expected, actual
        for provider, fn in [("native_triton", native), ("pytorch", reference)]:
            for _ in range(3):
                result = fn()
            torch.cuda.synchronize()
            events = []
            label = f"gate_T{tokens}_{'partial' if masked else 'none'}_{provider}"
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            ) as profiler:
                for i in range(10):
                    torch.argmax(eviction)
                    torch.cuda.synchronize()
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    with torch.profiler.record_function(f"target_sample_{i}"):
                        begin.record()
                        result = fn()
                        end.record()
                    torch.cuda.synchronize()
                    events.append(begin.elapsed_time(end) * 1000)
            trace_path = output / f"{label}.json"
            profiler.export_chrome_trace(str(trace_path))
            samples = kernel_samples(json.loads(trace_path.read_text()), 10)
            row = {
                "operator": "engram_gate",
                "tokens": tokens,
                "hidden": 5120,
                "hc": 4,
                "mask": "odd_positions" if masked else "none",
                "provider": provider,
                "input_dtype": "bf16",
                "weight_dtype": "fp32",
                "epsilon": 1e-20,
                "clamp": 1e-6,
                "warmups": 3,
                "samples": 10,
                "eviction": "256 MiB FP16 argmax before each target, excluded",
                "kernel_sum_samples_us": samples,
                "kernel_sum_median_us": statistics.median(samples),
                "cuda_event_samples_us": events,
                "cuda_event_median_us": statistics.median(events),
                "max_abs_error_vs_pytorch": max_error,
                "mask_implementation": "native gate plus torch.where"
                if masked
                else "native gate",
                "resident_server": resident_server,
            }
            records.append(row)
            (output / "engram-results.json").write_text(
                json.dumps(
                    {
                        "gpu": torch.cuda.get_device_name(),
                        "torch": torch.__version__,
                        "seed": 12345,
                        "rows": records,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(
                f"{label}: {row['kernel_sum_median_us']:.3f} us GPU kernel sum",
                flush=True,
            )
        del x, kv, qw, kw, image_mask, result, fn, reference, native
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", required=True, choices=["serving", "operators", "both"]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-count", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode in {"serving", "both"}:
        if args.gpu_count is None or args.gpu_count <= 0:
            parser.error("--gpu-count must be positive for serving profiles")
        subprocess.run(
            [
                "bash",
                str(ROOT / "benchmarks/single_node/srt_fixed_sequence.sh"),
                "--trust-remote-code",
                "--dsv4",
            ],
            check=True,
        )
        serving_profile(args.output, args.gpu_count)
    if args.mode in {"operators", "both"}:
        engram_bench(args.output / "operators", resident_server=args.mode == "both")


if __name__ == "__main__":
    main()
