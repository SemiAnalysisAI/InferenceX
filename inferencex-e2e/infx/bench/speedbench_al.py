from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path

from infx.bench import env, proc, server

PREPARE_URL = (
    "https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/"
    "bcf059af55c20a89f797724598f9908d126153e6/"
    "nemo_skills/dataset/speed-bench/prepare.py"
)


def levels(value: str) -> list[int]:
    parsed = [env.parse_positive_int("MTP_LIST", item) for item in value.split()]
    if not parsed or len(parsed) != len(set(parsed)):
        raise env.InputError("MTP_LIST must contain distinct positive integers")
    return parsed


def select_category(source: Path, destination: Path, category: str) -> int:
    count = 0
    with source.open(encoding="utf-8") as rows, destination.open("w", encoding="utf-8") as out:
        for line in rows:
            row = json.loads(line)
            if row.get("category") != category:
                continue
            messages = row["messages"]
            # Coding collection is single-turn; do not silently flatten conversations.
            if (
                len(messages) != 1
                or messages[0]["role"] != "user"
                or not isinstance(messages[0]["content"], str)
                or not messages[0]["content"]
            ):
                raise env.BenchError("SPEED-Bench row must contain exactly one nonempty user turn")
            out.write(json.dumps(row) + "\n")
            count += 1
    if count == 0:
        raise env.BenchError(f"No SPEED-Bench rows for category {category!r}")
    return count


def read_acceptance(path: Path, *, expected: int, mtp: int) -> float:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise env.BenchError(f"Expected one benchmark result in {path}") from error
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


def prepare_dataset(directory: Path, category: str) -> tuple[Path, int]:
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
    prepare = directory / "prepare.py"
    with urllib.request.urlopen(PREPARE_URL, timeout=60) as response:
        prepare.write_bytes(response.read())
    run([sys.executable, str(prepare), "--config", "qualitative", "--output_dir", str(directory)])
    selected = directory / "selected.jsonl"
    dataset = directory / "qualitative.jsonl"
    count = select_category(dataset, selected, category)
    selected.replace(dataset)
    return dataset, count


def run(command: Sequence[str]) -> None:
    code = proc.call(command)
    if code:
        raise env.BenchError(f"Command exited with status {code}: {command[0]}")


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


def collect_cell(
    *,
    model_path: str,
    mtp: int,
    port: int,
    dataset: Path,
    count: int,
    output_len: int,
    results: Path,
    args: argparse.Namespace,
) -> float:
    command = [
        "vllm",
        "serve",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        *args.server_args,
        "--speculative-config",
        json.dumps({"method": "mtp", "num_speculative_tokens": mtp}),
    ]
    log = results / f"server_on_mtp{mtp}.log"
    result = results / f"speedbench_on_mtp{mtp}.json"
    result.unlink(missing_ok=True)
    proc.echo(command)
    with log.open("wb") as sink:
        child = subprocess.Popen(
            command, stdout=sink, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            server.wait_ready(f"http://127.0.0.1:{port}/health", pid=child.pid, log=log)
            run(
                [
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
                    model_path,
                    "--dataset-name",
                    "speed_bench",
                    "--dataset-path",
                    str(dataset.parent),
                    "--speed-bench-output-len",
                    str(output_len),
                    "--num-prompts",
                    "-1",
                    "--max-concurrency",
                    str(args.concurrency),
                    "--temperature",
                    str(args.temperature),
                    "--top-p",
                    str(args.top_p),
                    "--chat-template-kwargs",
                    os.environ["CHAT_TEMPLATE_KWARGS_ON"],
                    "--trust-remote-code",
                    # vLLM records counter deltas around the formal request window.
                    "--num-warmups",
                    "0",
                    "--ready-check-timeout-sec",
                    "0",
                    "--seed",
                    "42",
                    "--save-result",
                    "--save-detailed",
                    "--result-dir",
                    str(results),
                    "--result-filename",
                    result.name,
                ]
            )
            return read_acceptance(result, expected=count, mtp=mtp)
        finally:
            stop(child)


def write_matrix(path: Path, model: str, values: Mapping[int, float], metadata: Path) -> None:
    text = f"# Collection provenance: {metadata.name}\n{json.dumps(model.rsplit('/', 1)[-1].lower())}:\n"
    text += "  thinking_on:\n"
    text += "".join(f"    {level}: {value:.6f}\n" for level, value in values.items())
    path.write_text(text, encoding="utf-8")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("server_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.server_args[:1] == ["--"]:
        args.server_args = args.server_args[1:]
    if args.concurrency < 1 or not args.temperature > 0 or not 0 < args.top_p <= 1:
        raise env.InputError("Invalid concurrency or sampling parameters")
    inputs = env.require(
        "CATEGORY",
        "CHAT_TEMPLATE_KWARGS_ON",
        "IMAGE",
        "MODEL",
        "MODEL_PATH",
        "MTP_LIST",
        "OUT_YAML",
        "PORT",
        "SPEEDBENCH_OUTPUT_LEN",
        "THINKING_MODES",
    )
    # GLM-5.3-Flash always thinks; enable_thinking=false does not disable it.
    if inputs["THINKING_MODES"] != "on" or json.loads(inputs["CHAT_TEMPLATE_KWARGS_ON"]) not in (
        {},
        {"reasoning_effort": "max"},
    ):
        raise env.InputError(
            "GLM-5.3-Flash collection requires THINKING_MODES=on and max reasoning effort"
        )
    mtp_levels = levels(inputs["MTP_LIST"])
    output_len = env.positive_int("SPEEDBENCH_OUTPUT_LEN")
    requested_port = env.positive_int("PORT")
    model = Path(inputs["MODEL_PATH"])
    if not (model / "config.json").is_file():
        raise env.BenchError(
            f"Staged checkpoint missing: {model / 'config.json'}; no download attempted"
        )
    out = Path(inputs["OUT_YAML"])
    out.unlink(missing_ok=True)
    results = out.parent / "speedbench_results"
    results.mkdir(exist_ok=True)
    metadata = results / "speedbench_provenance.json"
    with tempfile.TemporaryDirectory(prefix="infx-speedbench-") as temp:
        dataset, count = prepare_dataset(Path(temp), inputs["CATEGORY"])
        provenance = {
            "inputs": inputs,
            "server_args": args.server_args,
            "concurrency": args.concurrency,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "thinking": "reasoning_effort=max for GLM-5.3-Flash",
            "draft": "embedded MTP, upstream checkpoint-default precision",
            "verification": "standard (no block or synthetic acceptance flags)",
            "prepare_url": PREPARE_URL,
            "num_prompts": count,
            "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
            "vllm_version": importlib.metadata.version("vllm"),
            "frontend": os.environ.get("VLLM_USE_RUST_FRONTEND"),
            "complete": False,
        }
        metadata.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        values: dict[int, float] = {}
        for mtp in mtp_levels:
            print(f"Collecting thinking=on MTP={mtp}, {count} prompts", flush=True)
            values[mtp] = collect_cell(
                model_path=str(model),
                mtp=mtp,
                port=free_port(requested_port),
                dataset=dataset,
                count=count,
                output_len=output_len,
                results=results,
                args=args,
            )
            write_matrix(out, inputs["MODEL"], values, metadata)
            print(f"MTP={mtp}: AL={values[mtp]:.6f}", flush=True)
        provenance["complete"] = True
        metadata.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return 0
