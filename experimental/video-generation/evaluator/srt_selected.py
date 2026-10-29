from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

from specs import h3_spec

from .runner import run_plan, validate_plan

RECORDED_ENGINE_ENV = (
    "PYTHONPATH", "PYTHONUNBUFFERED", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
    "VLLM_WORKER_MULTIPROC_METHOD", "VLLM_OMNI_VIDEO_SYNC_TIMEOUT",
    "VLLM_OMNI_SERVER_STORAGE__PATH", "VGBENCH_GPU_UUIDS",
)

BENCHMARK_STEM = "benchmark_concurrency_{concurrency}"
CELL_DIR = "gpu/c{concurrency}"
_CREDENTIAL_ENV = re.compile(
    r"(?:^|_)(?:TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|AUTH|BEARER|"
    r"API_KEY|ACCESS_KEY|PRIVATE_KEY|SSH_KEY)(?:_|$)",
    re.IGNORECASE,
)


def _recipe_environment(env: dict[str, str]) -> dict[str, str]:
    raw = env.pop("H3_RECIPE_ENV_KEYS", None)
    if raw is None:
        return {}
    try:
        names = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("invalid H3_RECIPE_ENV_KEYS") from error
    if not isinstance(names, list) or any(
        not isinstance(name, str)
        or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
        or _CREDENTIAL_ENV.search(name)
        for name in names
    ):
        raise ValueError("invalid H3_RECIPE_ENV_KEYS")
    if len(names) != len(set(names)) or any(name not in env for name in names):
        raise ValueError("H3_RECIPE_ENV_KEYS names must be unique and present at launch")
    return {name: env[name] for name in names}


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def benchmark_window(run: dict, *, utc_minus_monotonic: float) -> dict:
    """Map a measured monotonic block to the host's Unix telemetry clock."""
    measurement, summary = run["measurement"], run["summary"]
    start = measurement["start_monotonic_seconds"] + utc_minus_monotonic
    end = measurement["end_monotonic_seconds"] + utc_minus_monotonic
    if not end > start:
        raise ValueError("measured block has no positive duration")
    return {
        "schema_version": 1,
        "benchmark_type": "video-generation",
        "max_concurrency": run["measurement"]["concurrency"],
        "benchmark_start_time_unix": start,
        "benchmark_end_time_unix": end,
        "duration": end - start,
        "completed": summary["valid"],
        "scheduled": summary["scheduled"],
        "valid": summary["valid"],
        "failed": summary["failed"],
        "energy_normalization": "clips",
        "clock": {"utc_minus_monotonic_seconds": utc_minus_monotonic},
        "run_path": "baseline/run.json",
        "run_sha256": hashlib.sha256(
            json.dumps(run, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


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


def serve(
    plan: dict, *, runtime: str, runtime_revision: str, model_path: Path,
    port: int, gpus: int, output: Path, command: list[str], model_view: str,
) -> None:
    if plan["model_id"] != "MiniMaxAI/MiniMax-H3":
        raise ValueError("this H3 native profile requires MiniMaxAI/MiniMax-H3")
    if runtime == "sglang":
        if gpus not in (2, 4):
            raise ValueError("the H3 SGLang native profile requires two or four GPUs")
    elif runtime == "vllm-omni":
        if gpus != 2:
            raise ValueError("the H3 vLLM-Omni native profile requires two GPUs")
    else:
        raise ValueError(f"unsupported native video runtime: {runtime}")
    if not re.fullmatch(r"[0-9a-f]{40}", runtime_revision):
        raise ValueError("runtime revision must be an immutable commit")
    if not model_path.is_dir():
        raise ValueError("the pinned model snapshot is unavailable")
    if model_view not in {"snapshot", "h3-fl2va"} or (model_view == "h3-fl2va" and runtime != "vllm-omni"):
        raise ValueError(f"unsupported model view {model_view!r} for {runtime}")
    if not command or any(not isinstance(part, str) or not part for part in command):
        raise ValueError("the video server requires a complete engine command")
    snapshot = str(model_path)
    if command.count(snapshot) != 1:
        raise ValueError("the engine command must contain the model snapshot path exactly once")
    from .gpu_job import cuda_devices

    uuids = cuda_devices()
    if len(uuids) != gpus:
        raise RuntimeError("visible GPU count differs from the native recipe")
    env = os.environ.copy()
    env["VGBENCH_GPU_UUIDS"] = ",".join(uuids)
    recorded_env = _recipe_environment(env)
    served_path = model_path
    if model_view == "h3-fl2va":
        storage = env.get("VLLM_OMNI_SERVER_STORAGE__PATH")
        if not storage:
            raise ValueError("VLLM_OMNI_SERVER_STORAGE__PATH must be set by the recipe")
        served_path = h3_vllm_model_view(model_path, output / "model-view")
        Path(storage).mkdir(parents=True, exist_ok=True)
    argv = [str(served_path) if part == snapshot else part for part in command]
    recorded_env.update({name: env[name] for name in RECORDED_ENGINE_ENV if name in env})
    _write(output / "spec.json", {
        "schema_version": "0.1.0",
        "plan": plan,
        "gpu_uuids": uuids,
        "baseline": {
            "name": runtime,
            "revision": runtime_revision,
            "model_path": str(served_path),
            "model_revision": plan["model_revision"],
            "argv": argv,
        },
        "serving": {"mode": "closed_loop"},
    })
    _write(output / "launch.json", {
        "runtime": runtime, "runtime_revision": runtime_revision,
        "model_id": plan["model_id"], "model_revision": plan["model_revision"],
        "plan_sha256": h3_spec.plan_sha256(plan), "gpu_uuids": uuids,
        "argv": argv, "environment": recorded_env, "slurm_job_id": env.get("SLURM_JOB_ID"),
    })
    os.execvpe(argv[0], argv, env)


def _sglang_main() -> None:
    """Check the assigned device order before importing the engine in the child."""
    from .gpu_job import cuda_devices

    expected = os.environ.get("VGBENCH_GPU_UUIDS")
    if not expected:
        raise RuntimeError("VGBENCH_GPU_UUIDS is required for the native SGLang server")
    if cuda_devices() != expected.split(","):
        raise RuntimeError("Runtime device UUIDs differ from assigned GPUs")
    from sglang.cli.main import main

    main()


def run_matrix(
    plan: dict, *, runtime: str, runtime_revision: str, endpoint: str,
    output: Path, concurrencies: list[int], gpus: int, window_dir: Path,
    request_seconds: float,
) -> dict[int, dict]:
    if runtime not in {"sglang", "vllm-omni"}:
        raise ValueError("unsupported native video runtime")
    if not math.isfinite(request_seconds) or request_seconds <= 0:
        raise ValueError("request_seconds must be a positive finite number")
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


if __name__ == "__main__":
    _sglang_main()
