from __future__ import annotations

import contextlib
import json
import math
import os
import re
import signal
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from infx.bench import env, server

COUNTERS = ("vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_drafts_total")


def read_counters(text: str) -> tuple[float, float]:
    values = []
    for name in COUNTERS:
        samples = re.findall(
            r"^" + re.escape(name) + r"(?:\{[^\n]*\})?\s+(\S+)(?:\s+\S+)?$",
            text,
            re.MULTILINE,
        )
        if not samples:
            raise env.BenchError(f"Missing acceptance counter: {name}")
        numbers = [float(sample) for sample in samples]
        if any(not math.isfinite(n) or n < 0 for n in numbers):
            raise env.BenchError(f"Invalid acceptance counter: {name}")
        values.append(sum(numbers))
    return values[0], values[1]


def measure_al(before: str, after: str, benchmark: dict[str, object], draft_length: int) -> float:
    if benchmark.get("completed") != 80 or benchmark.get("failed", 0) != 0:
        raise env.BenchError("Expected all 80 coding prompts to complete successfully")
    old_acc, old_drafts = read_counters(before)
    new_acc, new_drafts = read_counters(after)
    accepted, drafts = new_acc - old_acc, new_drafts - old_drafts
    if drafts <= 0 or accepted < 0 or accepted > draft_length * drafts:
        raise env.BenchError("Invalid acceptance deltas or counter reset")
    return 1 + accepted / drafts


def metrics(port: str, destination: Path) -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=30) as response:
        text = response.read().decode()
    destination.write_text(text)
    return text


def stop_server(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def collect(settings: dict[str, str], results: Path, dataset: Path) -> None:
    lengths = [env.parse_positive_int("MTP_LIST", item) for item in settings["MTP_LIST"].split()]
    modes = settings["THINKING_MODES"].split()
    if not lengths or any(length > 5 for length in lengths):
        raise env.InputError("DSv4.1 Flash draft lengths must be in 1..5")
    if not modes or any(mode not in {"on", "off"} for mode in modes):
        raise env.InputError("THINKING_MODES must contain on and/or off")
    if settings["CATEGORY"] != "coding":
        raise env.InputError("This collector requires the SPEED-Bench coding category")
    model, model_path, port = (settings[key] for key in ("MODEL", "MODEL_PATH", "PORT"))
    subprocess.run(
        [
            "hf",
            "download",
            model,
            "--revision",
            settings["MODEL_REVISION"],
            "--local-dir",
            model_path,
        ],
        check=True,
    )
    subprocess.run(["pip", "install", "-q", "datasets", "tiktoken"], check=True)
    prepare = dataset / "prepare.py"
    subprocess.run(
        [
            "curl",
            "-fLsS",
            "https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py",
            "-o",
            str(prepare),
        ],
        check=True,
    )
    subprocess.run(
        ["python3", str(prepare), "--config", "qualitative", "--output_dir", str(dataset)],
        check=True,
    )
    if not (dataset / "qualitative.jsonl").is_file():
        raise env.BenchError("SPEED-Bench preparation produced no qualitative.jsonl")
    bench = [
        "vllm",
        "bench",
        "serve",
        "--model",
        model,
        "--tokenizer",
        model_path,
        "--port",
        port,
        "--dataset-name",
        "speed_bench",
        "--dataset-path",
        str(dataset),
        "--speed-bench-category",
        settings["CATEGORY"],
        "--speed-bench-output-len",
        settings["SPEEDBENCH_OUTPUT_LEN"],
        "--num-prompts",
        "-1",
        "--max-concurrency",
        "32",
        "--tokenizer-mode",
        "deepseek_v41",
        "--temperature",
        "1.0",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(results),
    ]
    values: dict[str, dict[int, float]] = {mode: {} for mode in modes}
    for length in lengths:
        spec = {
            "method": "dspark",
            "num_speculative_tokens": length,
            "draft_sample_method": "probabilistic",
            "rejection_sample_method": "block",
            "enable_adaptive_verification": False,
        }
        command = [
            "vllm",
            "serve",
            model_path,
            "--served-model-name",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            port,
            "--tensor-parallel-size",
            settings["TP"],
            "--language-model-only",
            "--tokenizer-mode",
            "deepseek_v41",
            "--tool-call-parser",
            "deepseek_v41",
            "--enable-auto-tool-choice",
            "--reasoning-parser",
            "deepseek_v41",
            "--engram-config",
            '{"cpu_offload":true}',
            "--speculative-config",
            json.dumps(spec),
            "--max-model-len",
            "16384",
            "--no-enable-prefix-caching",
            "--max-cudagraph-capture-size",
            "256",
            "--max-num-seqs",
            "64",
            "--disable-uvicorn-access-log",
        ]
        log = results / f"server_dspark{length}.log"
        with log.open("wb") as stream:
            process = subprocess.Popen(
                command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                server.wait_ready(f"http://127.0.0.1:{port}/health", pid=process.pid, log=log)
                for mode in modes:
                    cell = f"{mode}_mtp{length}"
                    kwargs = (
                        settings["CHAT_TEMPLATE_KWARGS_ON"]
                        if mode == "on"
                        else '{"thinking":false}'
                    )
                    before = metrics(port, results / f"speedbench_before_{cell}.prom")
                    subprocess.run(
                        [
                            *bench,
                            "--chat-template-kwargs",
                            kwargs,
                            "--result-filename",
                            f"speedbench_{cell}.json",
                        ],
                        check=True,
                    )
                    after = metrics(port, results / f"speedbench_after_{cell}.prom")
                    detail = json.loads((results / f"speedbench_{cell}.json").read_text())
                    value = measure_al(before, after, detail, length)
                    values[mode][length] = value
                    print(f"thinking={mode} DSpark={length} AL={value:.4f}", flush=True)
                    (results / "speedbench_partial_al.json").write_text(
                        json.dumps(values, indent=2)
                    )
            finally:
                stop_server(process)
    lines = [
        f"# revision: {settings['MODEL_REVISION']}",
        f"# model: {model} | image: {settings['IMAGE']} | TP: {settings['TP']}",
        f"# coding | temperature: 1.0 | output_len: {settings['SPEEDBENCH_OUTPUT_LEN']}",
        f"# thinking_on: {settings['CHAT_TEMPLATE_KWARGS_ON']}",
        "# DSpark probabilistic drafting, block rejection, adaptive verification disabled.",
        "# AL includes the target verification token.",
        f"{model.rsplit('/', 1)[-1].lower()}:",
    ]
    for mode in modes:
        lines.append(f"  thinking_{mode}:")
        lines.extend(f"    {length}: {values[mode][length]:.2f}" for length in lengths)
    output = "\n".join(lines) + "\n"
    Path(settings["OUT_YAML"]).write_text(output)
    print(output, flush=True)


def main(argv: list[str]) -> int:
    if argv:
        raise env.InputError("speedbench-dsv41 takes configuration from the environment")
    settings = env.require(
        "MODEL",
        "MODEL_PATH",
        "MODEL_REVISION",
        "IMAGE",
        "TP",
        "PORT",
        "MTP_LIST",
        "THINKING_MODES",
        "CATEGORY",
        "SPEEDBENCH_OUTPUT_LEN",
        "CHAT_TEMPLATE_KWARGS_ON",
        "OUT_YAML",
    )
    results = Path("benchmarks").resolve()
    if not results.is_dir():
        raise env.InputError("Run the collector from the inferencex-e2e checkout")
    Path("speedbench_results").symlink_to("benchmarks", target_is_directory=True)
    with tempfile.TemporaryDirectory(prefix="infx-speedbench-", dir="/tmp") as scratch:
        collect(settings, results, Path(scratch))
    return 0
