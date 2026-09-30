"""Run explicit global batches sequentially against one native vLLM server."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def capacity_bounds(
    log_texts: list[str], dp_size: int, input_length: int
) -> dict[int, float] | None:
    """Optimistic input-only capacity bounds; require evidence from every DP engine."""
    bounds = {}
    pattern = re.compile(
        r"EngineCore_DP(\d+).*Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x"
    )
    for text in log_texts:
        for line in text.splitlines():
            match = pattern.search(line)
            if match:
                rank, length, concurrency = match.groups()
                length = int(length.replace(",", ""))
                if length >= input_length:
                    bound = float(concurrency) * length / input_length
                    rank = int(rank)
                    bounds[rank] = min(bounds.get(rank, bound), bound)
    return bounds if set(bounds) == set(range(dp_size)) else None


async def protocol_probe(
    output: Path, dp_size: int, settle_seconds: float, timeout_seconds: float
) -> None:
    """Exercise native admission pause and real streaming progress on every rank."""
    import aiohttp
    from vllm_cohort import admit_wave

    base = f"http://{os.environ['SRT_FRONTEND_HOST']}:{os.environ['SRT_FRONTEND_PORT']}"
    prompts = [([1] * 64, 64, 128, None) for _ in range(dp_size)]
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=120)
    ) as session:
        records, tasks = await admit_wave(
            session,
            base,
            prompts,
            128,
            output=output,
            label="probe",
            dp_size=dp_size,
            barrier=True,
            settle_seconds=settle_seconds,
            timeout_seconds=timeout_seconds,
        )
        await asyncio.gather(*tasks)
    (output / "protocol-probe.json").write_text(json.dumps(records, indent=2) + "\n")
    if not all(
        r["success"] and r["meta"].get("prompt_tokens") == 64 and len(r["events"]) >= 8
        for r in records
    ):
        raise RuntimeError(
            "Admission/streaming probe failed: need multiple progress chunks on every DP rank"
        )
    print(
        f"Native paused admission and streaming verified on all {dp_size} DP ranks",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument("--dp-size", type=int, required=True)
    parser.add_argument("--profile-steps", type=int, required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--warmup-output-tokens", type=int, required=True)
    parser.add_argument("--admission-settle-seconds", type=float, required=True)
    parser.add_argument("--admission-timeout-seconds", type=float, required=True)
    parser.add_argument("--concurrencies", nargs="+", type=int, required=True)
    args = parser.parse_args()
    if args.dp_size <= 0 or args.gpu_count <= 0:
        parser.error("GPU and DP counts must be positive")
    if (
        args.warmup_output_tokens <= 0
        or args.admission_settle_seconds < 0
        or args.admission_timeout_seconds <= 0
        or args.gpu_count % args.dp_size
        or any(c <= 0 or c % args.dp_size for c in args.concurrencies)
    ):
        parser.error("Invalid warmup/admission budget or GPU/concurrency divisibility")
    if args.concurrencies != sorted(set(args.concurrencies)):
        parser.error("Concurrencies must be unique and ascending")
    if int(os.environ["CONC"]) != args.concurrencies[0]:
        parser.error("The matrix concurrency must match the first sweep case")
    root_name = os.environ["RESULT_FILENAME"]
    out = args.output / "research" / "cohort-sweep"
    out.mkdir(parents=True, exist_ok=True)
    if args.probe:
        asyncio.run(
            protocol_probe(
                out,
                args.dp_size,
                args.admission_settle_seconds,
                args.admission_timeout_seconds,
            )
        )
    manifest = []
    bounds = capacity_bounds(
        [p.read_text(errors="replace") for p in args.output.glob("*_agg_w0.out")],
        args.dp_size,
        int(os.environ["ISL"]),
    )
    (out / "runtime-capacity-bounds.json").write_text(
        json.dumps(
            {
                "per_dp_input_only_upper_bound": bounds,
                "qualification": "Native runtime KV-capacity estimates, scaled optimistically to input length only. Not proof that a batch fits; full-window validation still required.",
            },
            indent=2,
        )
        + "\n"
    )
    for i, concurrency in enumerate(args.concurrencies):
        if bounds is not None and concurrency / args.dp_size > min(bounds.values()):
            manifest.append(
                {
                    "global_batch": concurrency,
                    "status": "not_run_capacity_bound",
                    "per_dp_requested": concurrency // args.dp_size,
                    "per_dp_input_only_upper_bound": min(bounds.values()),
                }
            )
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(
                f"Skipping global batch {concurrency}: exceeds native runtime KV capacity bound",
                flush=True,
            )
            if i == 0:
                raise RuntimeError(
                    "First requested batch exceeds native runtime KV capacity"
                )
            continue
        case_dir = args.output if i == 0 else out / f"batch{concurrency}"
        case_dir.mkdir(parents=True, exist_ok=True)
        name = root_name if i == 0 else f"research_batch{concurrency}"
        environment = {
            **os.environ,
            "CONC": str(concurrency),
            "CONC_LIST": str(concurrency),
            "RESULT_FILENAME": name,
        }
        command = [
            sys.executable,
            str(Path(__file__).with_name("vllm_cohort.py")),
            "--output",
            str(case_dir),
            "--gpu-count",
            str(args.gpu_count),
            "--dp-size",
            str(args.dp_size),
            "--profile-steps",
            str(args.profile_steps),
            "--result-layout",
            "single",
            "--warmup-output-tokens",
            str(args.warmup_output_tokens),
            "--admission-barrier",
            "--admission-settle-seconds",
            str(args.admission_settle_seconds),
            "--admission-timeout-seconds",
            str(args.admission_timeout_seconds),
            "--trace-dir",
            str(args.output / "research" / "vllm-cohort-profiles"),
        ]
        print(
            f"Starting global batch {concurrency} on the same vLLM server", flush=True
        )
        result = subprocess.run(command, env=environment, check=False)
        result_path = case_dir / f"{name}.json"
        if result_path.exists():
            shutil.copyfile(result_path, out / f"result_batch{concurrency}.json")
        manifest.append(
            {
                "global_batch": concurrency,
                "returncode": result.returncode,
                "result_present": result_path.exists(),
                "result_path": str(result_path),
            }
        )
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command)


if __name__ == "__main__":
    main()
