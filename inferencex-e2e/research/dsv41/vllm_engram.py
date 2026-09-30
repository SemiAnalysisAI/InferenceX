"""Measure the installed vLLM production post-WKV gate without engine patches."""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import statistics
from pathlib import Path

from experiment import kernel_samples


def main() -> None:
    import torch
    import triton
    import vllm
    from vllm.models.deepseek_v41.common.engram import _fused_engram_post_wkv_kernel

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--kernel-sha256", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = inspect.getsource(_fused_engram_post_wkv_kernel.fn)
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
    body = ast.get_source_segment(source, node)
    digest = hashlib.sha256(body.encode()).hexdigest()
    if digest != args.kernel_sha256:
        raise RuntimeError(f"Unexpected installed production kernel: {digest}")
    torch.cuda.set_device(0)
    eviction = torch.ones(256 * 1024 * 1024 // 2, dtype=torch.float16, device="cuda")
    rows = []
    metadata = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "vllm": vllm.__version__,
        "image": args.image,
        "kernel_sha256": digest,
        "seed": 12345,
        "resident_server": False,
        "scope": "post-WKV gate only; excludes embedding, WKV projection and allocation",
        "rows": rows,
    }
    for weight_dtype in (torch.bfloat16, torch.float32):
        for tokens in (1, 72, 128, 512, 1024, 4096, 8192, 16384):
            torch.manual_seed(12345 + tokens)
            x = torch.randn(tokens, 4, 5120, device="cuda", dtype=torch.bfloat16)
            kv = torch.randn(tokens, 5 * 5120, device="cuda", dtype=torch.bfloat16)
            qw = torch.randn(4, 5120, device="cuda", dtype=weight_dtype)
            kw = torch.randn_like(qw)
            out = torch.empty_like(x)
            modes = ["none", "all_active"]
            if tokens in (512, 8192):
                modes.append("odd_positions")
            for mode in modes:
                mask = (
                    None
                    if mode == "none"
                    else torch.ones(tokens, dtype=torch.bool, device="cuda")
                )
                if mode == "odd_positions":
                    mask[1::2] = False

                def run(x=x, kv=kv, qw=qw, kw=kw, out=out, mask=mask, tokens=tokens):
                    _fused_engram_post_wkv_kernel[(tokens * 4,)](
                        x,
                        kv,
                        qw,
                        kw,
                        x if mask is None else mask,
                        out,
                        tokens,
                        *x.stride(),
                        *kv.stride(),
                        *qw.stride(),
                        *kw.stride(),
                        0 if mask is None else mask.stride(0),
                        *out.stride(),
                        1e-20,
                        1e-6,
                        DIM=5120,
                        HC_MULT=4,
                        BLOCK_SIZE=8192,
                        HAS_MASK=mask is not None,
                        num_warps=8,
                    )
                    return out

                h = x.float()
                key = kv[:, : 4 * 5120].float().reshape(tokens, 4, 5120)
                value = kv[:, 4 * 5120 :].float()
                dot = (h * qw.float() * kw.float() * key).sum(-1)
                dot *= torch.rsqrt(h.square().mean(-1) + 1e-20)
                dot *= torch.rsqrt(key.square().mean(-1) + 1e-20) * 5120**-0.5
                gate = torch.sigmoid(
                    torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot)
                )
                if mask is not None:
                    gate = gate.masked_fill(~mask[:, None], 0)
                expected = (h + gate[..., None] * value[:, None, :]).to(x.dtype)
                actual = run()
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
                error = (actual.float() - expected.float()).abs().max().item()
                if mask is not None and not torch.equal(actual[~mask], x[~mask]):
                    raise RuntimeError("Masked rows were modified")
                del h, key, value, dot, gate, expected, actual
                for _ in range(3):
                    run()
                torch.cuda.synchronize()
                elapsed = []
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    for i in range(10):
                        torch.argmax(eviction)
                        torch.cuda.synchronize()
                        begin, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        with torch.profiler.record_function(f"target_sample_{i}"):
                            begin.record()
                            run()
                            end.record()
                        torch.cuda.synchronize()
                        elapsed.append(begin.elapsed_time(end) * 1000)
                label = f"vllm_T{tokens}_{mode}_{str(weight_dtype).split('.')[-1]}"
                path = args.output / f"{label}.trace.json"
                prof.export_chrome_trace(str(path))
                trace = json.loads(path.read_text())
                samples = kernel_samples(trace, 10)
                kernels = [
                    e
                    for e in trace["traceEvents"]
                    if e.get("cat") == "kernel"
                    and "_fused_engram_post_wkv_kernel" in e.get("name", "")
                ]
                if len(kernels) != 10:
                    raise RuntimeError(
                        f"Expected ten production kernel launches, found {len(kernels)}"
                    )
                rows.append(
                    {
                        "tokens": tokens,
                        "hidden": 5120,
                        "hc": 4,
                        "mask": mode,
                        "weight_dtype": str(weight_dtype),
                        "input_dtype": "bf16",
                        "epsilon": 1e-20,
                        "clamp": 1e-6,
                        "warmups": 3,
                        "samples": 10,
                        "eviction": "256 MiB FP16 ArgMax, excluded",
                        "kernel_sum_samples_us": samples,
                        "kernel_sum_median_us": statistics.median(samples),
                        "cuda_event_samples_us": elapsed,
                        "max_abs_error_vs_pytorch": error,
                        "production_kernel_launches": len(kernels),
                        "trace": path.name,
                    }
                )
                (args.output / "results.json").write_text(
                    json.dumps(metadata, indent=2) + "\n"
                )
                print(f"{label}: {statistics.median(samples):.3f} us", flush=True)


if __name__ == "__main__":
    main()
