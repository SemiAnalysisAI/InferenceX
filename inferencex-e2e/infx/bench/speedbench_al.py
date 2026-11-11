from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from infx.bench import env, proc, server, speedbench


def levels(value: str) -> list[int]:
    parsed = [env.parse_positive_int("MTP_LIST", item) for item in value.split()]
    if not parsed or len(parsed) != len(set(parsed)):
        raise env.InputError("MTP_LIST must contain distinct positive integers")
    return parsed


def modes(value: str, on_kwargs: str) -> dict[str, dict[str, bool]]:
    names = value.split()
    if not names or len(names) != len(set(names)) or set(names) - {"on", "off"}:
        raise env.InputError("THINKING_MODES must contain distinct on/off values")
    if json.loads(on_kwargs) != {"enable_thinking": True}:
        raise env.InputError('Thinking-on requires {"enable_thinking": true}')
    return {name: {"enable_thinking": name == "on"} for name in names}


def read_acceptance(path: Path, *, expected: int, mtp: int) -> float:
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("completed") != expected
        or result.get("failed", 0) != 0
        or result.get("total_output_tokens", 0) <= 0
    ):
        raise env.BenchError(f"Incomplete SPEED-Bench result: {result.get('completed')}/{expected}")
    value = result.get("spec_decode_acceptance_length")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 1 <= value <= mtp + 1
    ):
        raise env.BenchError(f"Invalid acceptance length for MTP={mtp}: {value!r}")
    drafts = result.get("spec_decode_num_drafts")
    accepted = result.get("spec_decode_accepted_tokens")
    if (
        type(drafts) is not int
        or drafts <= 0
        or type(accepted) is not int
        or not 0 <= accepted <= mtp * drafts
        or not math.isclose(value, 1 + accepted / drafts, rel_tol=1e-9)
    ):
        raise env.BenchError("Missing or inconsistent speculative decoding counter deltas")
    return float(value)


def run(command: Sequence[str]) -> None:
    code = proc.call(command)
    if code:
        raise env.BenchError(f"Command exited with status {code}: {command[0]}")


def prepare_dataset(directory: Path) -> tuple[Path, int]:
    constraints = directory / "constraints.txt"
    constraints.write_text(
        "".join(
            f"{name}=={importlib.metadata.version(name)}\n"
            for name in ("huggingface-hub", "transformers", "tokenizers")
        ),
        encoding="utf-8",
    )
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-c",
            str(constraints),
            "datasets",
            "tiktoken",
        ]
    )
    speedbench.main(["--output-dir", str(directory)])
    prompts = speedbench.load(directory / speedbench.PROMPTS_FILE)
    return write_dataset(directory, prompts), len(prompts)


def write_dataset(directory: Path, prompts: list[str]) -> Path:
    if speedbench.digest(prompts) != speedbench.PROMPTS_SHA256:
        raise env.BenchError("SPEED-Bench prompt digest differs from the golden coding corpus")
    path = directory / "qualitative.jsonl"
    path.write_text(
        "".join(
            json.dumps({"category": "coding", "messages": [{"role": "user", "content": text}]})
            + "\n"
            for text in prompts
        ),
        encoding="utf-8",
    )
    return path


def free_port(requested: int) -> int:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", requested))
        except OSError:
            sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def stop(child: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # The launcher can exit before its GPU workers.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    deadline = time.monotonic() + 120
    while True:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
        used = [int(line.strip()) for line in output.splitlines() if line.strip()]
        if used and max(used) < 2000:
            return
        if time.monotonic() >= deadline:
            raise env.BenchError(f"GPU memory did not drain after server shutdown: {used} MiB")
        time.sleep(3)


def benchmark_command(
    model: str,
    port: int,
    dataset: Path,
    output_len: int,
    result: Path,
    kwargs: Mapping[str, bool],
    sampling: Mapping[str, Any],
    args: argparse.Namespace,
) -> list[str]:
    return [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        model,
        "--dataset-name",
        "speed_bench",
        "--dataset-path",
        str(dataset.parent),
        "--speed-bench-category",
        "coding",
        "--speed-bench-output-len",
        str(output_len),
        "--num-prompts",
        "-1",
        "--max-concurrency",
        str(args.concurrency),
        "--temperature",
        str(sampling["temperature"]),
        "--top-p",
        str(sampling["top_p"]),
        "--extra-body",
        json.dumps(dict(sampling)),
        "--chat-template-kwargs",
        json.dumps(dict(kwargs)),
        "--num-warmups",
        "0",
        "--ready-check-timeout-sec",
        "0",
        "--seed",
        str(args.seed),
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(result.parent),
        "--result-filename",
        result.name,
    ]


def collect_depth(
    *,
    model: str,
    mtp: int,
    dataset: Path,
    count: int,
    output_len: int,
    results: Path,
    thinking: Mapping[str, dict[str, bool]],
    args: argparse.Namespace,
    requested_port: int,
) -> dict[str, float]:
    port = free_port(requested_port)
    command = [
        "vllm",
        "serve",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        *args.server_args,
        "--speculative-config",
        json.dumps({"method": args.spec_method, "num_speculative_tokens": mtp}),
    ]
    log = results / f"server_mtp{mtp}.log"
    values = {}
    proc.echo(command)
    with log.open("wb") as sink:
        child = subprocess.Popen(
            command, stdout=sink, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            server.wait_ready(f"http://127.0.0.1:{port}/health", pid=child.pid, log=log)
            for mode, kwargs in thinking.items():
                result = results / f"speedbench_{mode}_mtp{mtp}.json"
                result.unlink(missing_ok=True)
                command = benchmark_command(
                    model, port, dataset, output_len, result, kwargs, args.sampling[mode], args
                )
                (results / f"command_{mode}_mtp{mtp}.json").write_text(
                    json.dumps(command, indent=2) + "\n", encoding="utf-8"
                )
                run(command)
                values[mode] = read_acceptance(result, expected=count, mtp=mtp)
                print(f"thinking={mode} MTP={mtp}: AL={values[mode]:.6f}", flush=True)
        finally:
            stop(child)
    return values


def write_matrix(path: Path, model: str, values: Mapping[str, Mapping[int, float]]) -> None:
    text = f"{json.dumps(model.rsplit('/', 1)[-1].lower())}:\n"
    for mode, cells in values.items():
        text += f"  thinking_{mode}:\n"
        text += "".join(f"    {level}: {value:.6f}\n" for level, value in cells.items())
    path.write_text(text, encoding="utf-8")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--spec-method", required=True)
    parser.add_argument("--sampling", type=json.loads, required=True)
    parser.add_argument("server_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.server_args[:1] == ["--"]:
        args.server_args = args.server_args[1:]
    inputs = env.require(
        "CATEGORY",
        "CHAT_TEMPLATE_KWARGS_ON",
        "IMAGE",
        "MODEL",
        "MODEL_PATH",
        "MODEL_REVISION",
        "MTP_LIST",
        "OUT_YAML",
        "PORT",
        "SPEEDBENCH_OUTPUT_LEN",
        "THINKING_MODES",
    )
    if inputs["CATEGORY"] != "coding" or args.concurrency < 1:
        raise env.InputError("Collection requires CATEGORY=coding and positive concurrency")
    if not re.fullmatch(r"[0-9a-f]{40}", inputs["MODEL_REVISION"]):
        raise env.InputError("MODEL_REVISION must be a full Hugging Face commit SHA")
    thinking = modes(inputs["THINKING_MODES"], inputs["CHAT_TEMPLATE_KWARGS_ON"])
    for mode in thinking:
        sample = args.sampling[mode]
        if not sample["temperature"] > 0 or not 0 < sample["top_p"] <= 1:
            raise env.InputError(f"Invalid sampling for {mode}")
    mtp_levels = levels(inputs["MTP_LIST"])
    output_len = env.positive_int("SPEEDBENCH_OUTPUT_LEN")
    port = env.positive_int("PORT")
    out = Path(inputs["OUT_YAML"])
    out.unlink(missing_ok=True)
    results = out.parent / "speedbench_results"
    results.mkdir(exist_ok=True)
    # Isolate the pinned revision from any other checkpoint already in the shared model root.
    model = Path(inputs["MODEL_PATH"]) / inputs["MODEL_REVISION"]
    run(
        [
            "hf",
            "download",
            inputs["MODEL"],
            "--revision",
            inputs["MODEL_REVISION"],
            "--local-dir",
            str(model),
        ]
    )
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    if (
        config.get("quantization_config", {}).get("quant_method") != "fp8"
        or config.get("text_config", {}).get("mtp_num_hidden_layers") != 1
    ):
        raise env.BenchError("Expected the FP8 checkpoint with its embedded one-layer MTP head")
    metadata = results / "speedbench_provenance.json"
    with tempfile.TemporaryDirectory(prefix="infx-speedbench-") as temp:
        dataset, count = prepare_dataset(Path(temp))
        provenance = {
            "inputs": inputs,
            "server_args": args.server_args,
            "spec_method": args.spec_method,
            "concurrency": args.concurrency,
            "seed": args.seed,
            "sampling": args.sampling,
            "thinking": thinking,
            "num_prompts": count,
            "prompt_selection": "first user turn of every coding row; not multi-turn replay",
            "prompts_sha256": speedbench.PROMPTS_SHA256,
            "dataset_revision": speedbench.SPEED_BENCH_REVISION,
            "prepare_url": speedbench.PREPARE_PY,
            "prepare_sha256": speedbench.PREPARE_PY_SHA256,
            "config_sha256": hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
            "quantization_config": config["quantization_config"],
            "draft": "embedded MTP; upstream checkpoint-default precision; no draft overrides",
            "verification": "natural acceptance; no synthetic or block-rejection override",
            "vllm_version": importlib.metadata.version("vllm"),
            "frontend": os.environ.get("VLLM_USE_RUST_FRONTEND"),
            "complete": False,
        }
        metadata.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        values: dict[str, dict[int, float]] = {mode: {} for mode in thinking}
        for mtp in mtp_levels:
            measured = collect_depth(
                model=str(model),
                mtp=mtp,
                dataset=dataset,
                count=count,
                output_len=output_len,
                results=results,
                thinking=thinking,
                args=args,
                requested_port=port,
            )
            for mode, value in measured.items():
                values[mode][mtp] = value
        # Only a complete matrix is eligible for the workflow's golden-curve PR step.
        write_matrix(out, inputs["MODEL"], values)
        provenance["complete"] = True
        metadata.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return 0
