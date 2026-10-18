"""Run a selected YAML video workload through the native srt-slurm service."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

from specs import h3_spec

from .mvp_runner import run_plan, validate_plan
from .mvp_srt import BENCHMARK_STEM, CELL_DIR, _write, benchmark_window


def selected_plan(path: Path, set_id: str, expected_sha256: str) -> dict:
    spec = h3_spec.load_spec(path)
    if set_id not in spec["sets"]:
        raise ValueError(f"unknown selected workload set: {set_id}")
    plan = validate_plan(h3_spec.plan_for(spec, set_id))
    if h3_spec.plan_sha256(plan) != expected_sha256:
        raise ValueError("selected workload plan does not match the bound plan_sha256")
    return plan


def h3_vllm_model_view(snapshot: Path, output: Path) -> Path:
    """Expose only the tested FL2VA files without copying weights or mutating the cache."""
    required = ("FL2VA", "model_index.json", "LICENSE", "README.md", "Ref2VA/model_index.json")
    for name in required:
        if not (snapshot / name).exists():
            raise ValueError(f"H3 model snapshot lacks {name}")
    output.mkdir(parents=True, exist_ok=False)
    for name in required:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(snapshot / name)
    return output


def server_argv(plan: dict, runtime: str, model_path: Path, port: int, gpus: int) -> list[str]:
    if plan["model_id"] != "MiniMaxAI/MiniMax-H3" or gpus != 2:
        raise ValueError("this H3 native profile requires two GPUs")
    if runtime == "sglang":
        from .mvp_gpu_job import _server_argv

        spec = {
            "baseline": {"python": "/usr/bin/python3"},
            "model": {"path": str(model_path), "revision": plan["model_revision"]},
            "plan": plan,
            "server": {"ulysses_degree": 2, "tp_size": 1, "encoder_parallel": "auto",
                       "performance_mode": "speed", "dit_cpu_offload": False},
            "gpu_uuids": ["assigned"] * gpus,
            "port": port,
        }
        return _server_argv(spec, "baseline", host="0.0.0.0")
    if runtime == "vllm-omni":
        return [
            "vllm", "serve", str(model_path), "--served-model-name", plan["model_id"],
            "--omni", "--host", "0.0.0.0", "--port", str(port),
            "--trust-remote-code", "--task-type", "fl2va", "--num-gpus", "2",
            "--tensor-parallel-size", "2", "--usp", "1", "--ring", "1",
            "--text-encoder-tp-size", "2", "--vae-patch-parallel-size", "2",
            "--vae-parallel-mode", "tile", "--vae-use-tiling", "--enforce-eager",
        ]
    raise ValueError(f"unsupported native video runtime: {runtime}")


def serve(
    plan: dict, *, runtime: str, runtime_revision: str, model_path: Path,
    port: int, gpus: int, output: Path,
) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", runtime_revision):
        raise ValueError("runtime revision must be an immutable commit")
    if not model_path.is_dir():
        raise ValueError("the pinned model snapshot is unavailable")
    if runtime == "vllm-omni":
        model_path = h3_vllm_model_view(model_path, output / "model-view")
    argv = server_argv(plan, runtime, model_path, port, gpus)
    from .mvp_gpu_job import cuda_devices

    uuids = cuda_devices()
    if len(uuids) != gpus:
        raise RuntimeError("visible GPU count differs from the native recipe")
    env = os.environ.copy()
    env["VGBENCH_GPU_UUIDS"] = ",".join(uuids)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    if runtime == "vllm-omni":
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        env["VLLM_OMNI_VIDEO_SYNC_TIMEOUT"] = "14400"
        storage = output / "storage"
        storage.mkdir(parents=True, exist_ok=True)
        env["VLLM_OMNI_SERVER_STORAGE__PATH"] = str(storage)
    _write(output / "launch.json", {
        "runtime": runtime, "runtime_revision": runtime_revision,
        "model_id": plan["model_id"], "model_revision": plan["model_revision"],
        "plan_sha256": h3_spec.plan_sha256(plan), "gpu_uuids": uuids,
        "argv": argv, "slurm_job_id": env.get("SLURM_JOB_ID"),
    })
    os.execvpe(argv[0], argv, env)


def run_matrix(
    plan: dict, *, runtime: str, runtime_revision: str, endpoint: str,
    output: Path, concurrencies: list[int], gpus: int, window_dir: Path,
    request_seconds: float = 900,
) -> dict[int, dict]:
    if runtime not in {"sglang", "vllm-omni"}:
        raise ValueError("unsupported native video runtime")
    if not concurrencies or len(set(concurrencies)) != len(concurrencies) or any(c < 1 for c in concurrencies):
        raise ValueError("concurrencies must be distinct positive integers")
    results = {}
    for concurrency in concurrencies:
        cell = output / CELL_DIR.format(concurrency=concurrency)
        before = time.time() - time.monotonic()
        run = run_plan(
            plan, cell / "baseline", endpoint=endpoint, runtime=runtime,
            runtime_revision=runtime_revision,
            hardware_label=f"native srt-slurm {gpus} GPUs",
            model_revision=plan["model_revision"], timeout_seconds=request_seconds,
            serving_concurrency=concurrency,
        )
        after = time.time() - time.monotonic()
        contract = benchmark_window(run, utc_minus_monotonic=(before + after) / 2)
        contract["clock"]["offset_spread_seconds"] = abs(after - before)
        path = cell / f"{BENCHMARK_STEM.format(concurrency=concurrency)}.json"
        _write(path, contract)
        from infx.results.power.window import write_window

        write_window(path, concurrency, window_dir)
        results[concurrency] = run
    return results
