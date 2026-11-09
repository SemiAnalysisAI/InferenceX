from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path

from infx.bench.video import CREDENTIAL_ENV, cell_dir, read_object, window_file, write_json
from specs import video_spec

from .runner import RUNTIMES, run_plan, validate_plan

RECORDED_ENGINE_ENV = (
    "PYTHONPATH",
    "PYTHONUNBUFFERED",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "VLLM_WORKER_MULTIPROC_METHOD",
    "VLLM_OMNI_VIDEO_SYNC_TIMEOUT",
    "VLLM_OMNI_SERVER_STORAGE__PATH",
    "VGBENCH_GPU_UUIDS",
)
_BOOLEAN = {"true": True, "false": False}
KFD = Path("/dev/kfd")
PCI_DEVICES = Path("/sys/bus/pci/devices")
# Deployment keys only: the dashboard treats any other spec.json server key as workload identity.
SERVER_FLAGS = {
    "sglang": {
        "--tp-size": ("tp_size", int),
        "--ulysses-degree": ("ulysses_degree", int),
        "--encoder-parallel": ("encoder_parallel", str),
        "--dit-cpu-offload": ("dit_cpu_offload", lambda value: _BOOLEAN[value.lower()]),
        "--attention-backend": ("attention_backend", str),
    },
    "vllm-omni": {
        "--tensor-parallel-size": ("tp_size", int),
        "--usp": ("ulysses_degree", int),
    },
}


def cuda_devices() -> list[str]:
    # Resolve the current mask through the driver without creating a CUDA context;
    # nvidia-smi ordinals alone do not establish CUDA device identity.
    cuda = ctypes.CDLL("libcuda.so.1")

    def ok(code):
        if code != 0:
            raise RuntimeError("CUDA driver inventory failed: " + str(code))

    ok(cuda.cuInit(0))
    count = ctypes.c_int()
    ok(cuda.cuDeviceGetCount(ctypes.byref(count)))
    values = []
    for ordinal in range(count.value):
        device, raw = ctypes.c_int(), (ctypes.c_ubyte * 16)()
        ok(cuda.cuDeviceGet(ctypes.byref(device), ordinal))
        ok(cuda.cuDeviceGetUuid(ctypes.byref(raw), device))
        values.append("GPU-" + str(uuid.UUID(bytes=bytes(raw))))
    return values


def rocm_devices() -> list[str]:
    # AMD power exporters label each GPU by its board serial, not by a HIP UUID.
    hip = ctypes.CDLL("libamdhip64.so")

    def ok(code):
        if code != 0:
            raise RuntimeError("HIP device inventory failed: " + str(code))

    count = ctypes.c_int()
    ok(hip.hipGetDeviceCount(ctypes.byref(count)))
    values = []
    for ordinal in range(count.value):
        bus = ctypes.create_string_buffer(32)
        ok(hip.hipDeviceGetPCIBusId(bus, len(bus), ordinal))
        values.append((PCI_DEVICES / bus.value.decode() / "serial_number").read_text().strip())
    return values


def gpu_devices() -> list[str]:
    """The visible GPUs in engine order, named as the cluster's power exporter labels them."""
    return rocm_devices() if KFD.exists() else cuda_devices()


def _recipe_environment(env: dict[str, str]) -> dict[str, str]:
    raw = env.pop("VIDEO_RECIPE_ENV_KEYS", None)
    if raw is None:
        return {}
    try:
        names = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("invalid VIDEO_RECIPE_ENV_KEYS") from error
    if not isinstance(names, list) or any(
        not isinstance(name, str)
        or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
        or CREDENTIAL_ENV.search(name)
        for name in names
    ):
        raise ValueError("invalid VIDEO_RECIPE_ENV_KEYS")
    if len(names) != len(set(names)) or any(name not in env for name in names):
        raise ValueError("VIDEO_RECIPE_ENV_KEYS names must be unique and present at launch")
    return {name: env[name] for name in names}


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
        "run_sha256": hashlib.sha256(video_spec.canonical_json_bytes(run)).hexdigest(),
    }


def selected_plan(path: Path, set_id: str, expected_sha256: str) -> dict:
    spec = video_spec.load_spec(path)
    if set_id not in spec["sets"]:
        raise ValueError(f"unknown selected workload set: {set_id}")
    plan = validate_plan(video_spec.plan_for(spec, set_id))
    if video_spec.plan_sha256(plan) != expected_sha256:
        raise ValueError("selected workload plan does not match the bound plan_sha256")
    return plan


def vllm_model_view(snapshot: Path, output: Path) -> Path:
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


def server_parallelism(runtime: str, argv: list[str]) -> dict:
    """The explicit engine parallelism flags, under the keys the dashboard reads."""
    flags = SERVER_FLAGS[runtime]
    return {
        flags[flag][0]: flags[flag][1](value)
        for flag, value in zip(argv, argv[1:])
        if flag in flags
    }


def serve(
    plan: dict,
    *,
    runtime: str,
    runtime_revision: str,
    model_path: Path,
    gpus: int,
    output: Path,
    command: list[str],
    model_view: str,
) -> None:
    if runtime not in RUNTIMES:
        raise ValueError(f"unsupported native video runtime: {runtime}")
    if not re.fullmatch(r"[0-9a-f]{40}", runtime_revision):
        raise ValueError("runtime revision must be an immutable commit")
    if not model_path.is_dir():
        raise ValueError("the pinned model snapshot is unavailable")
    if model_view == "h3-fl2va" and runtime != "vllm-omni":
        raise ValueError(f"model view {model_view!r} is for vllm-omni, not {runtime}")
    if not command or any(not isinstance(part, str) or not part for part in command):
        raise ValueError("the video server requires a complete engine command")
    snapshot = str(model_path)
    if command.count(snapshot) != 1:
        raise ValueError("the engine command must contain the model snapshot path exactly once")

    uuids = gpu_devices()
    if len(uuids) != gpus:
        raise RuntimeError("visible GPU count differs from the native recipe")
    env = os.environ.copy()
    env["VGBENCH_GPU_UUIDS"] = ",".join(uuids)
    recorded_env = _recipe_environment(env)
    image = re.fullmatch(r"[^\s@]+@sha256:([0-9a-f]{64})", env.get("VIDEO_CONTAINER_IMAGE", ""))
    if image is None:
        raise ValueError("VIDEO_CONTAINER_IMAGE must be an image pinned by a SHA256 digest")
    served_path = model_path
    if model_view == "h3-fl2va":
        storage = env.get("VLLM_OMNI_SERVER_STORAGE__PATH")
        if not storage:
            raise ValueError("VLLM_OMNI_SERVER_STORAGE__PATH must be set by the recipe")
        served_path = vllm_model_view(model_path, output / "model-view")
        Path(storage).mkdir(parents=True, exist_ok=True)
    argv = [str(served_path) if part == snapshot else part for part in command]
    server = server_parallelism(runtime, argv)
    recorded_env.update({name: env[name] for name in RECORDED_ENGINE_ENV if name in env})
    write_json(
        output / "spec.json",
        {
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
            "server": server,
        },
    )
    launch = {
        "runtime": runtime,
        "runtime_revision": runtime_revision,
        "model_id": plan["model_id"],
        "model_revision": plan["model_revision"],
        "plan_sha256": video_spec.plan_sha256(plan),
        "gpu_uuids": uuids,
        "argv": argv,
        "environment": recorded_env,
        "image_digest": image.group(1),
        "slurm_job_id": env.get("SLURM_JOB_ID"),
    }
    write_json(output / "launch.json", launch)
    os.execvpe(argv[0], argv, env)


def _sglang_main() -> None:
    """Check the assigned device order before importing the engine in the child."""

    expected = os.environ.get("VGBENCH_GPU_UUIDS")
    if not expected:
        raise RuntimeError("VGBENCH_GPU_UUIDS is required for the native SGLang server")
    if gpu_devices() != expected.split(","):
        raise RuntimeError("Runtime device UUIDs differ from assigned GPUs")
    from sglang.cli.main import main

    main()


def _sealed_identity(path: Path, plan: dict, *, runtime: str, gpus: int) -> dict:
    """The sealed server launch as the run's numerical identity, which run_plan validates."""
    try:
        launch = read_object(path)
    except (OSError, ValueError) as error:
        raise ValueError("the client requires the sealed server launch before requests") from error
    expected = {
        "runtime": runtime,
        "model_revision": plan["model_revision"],
        "plan_sha256": video_spec.plan_sha256(plan),
    }
    for key, value in expected.items():
        if launch.get(key) != value:
            raise ValueError(f"sealed server launch {key} differs from the selected plan")
    uuids = launch.get("gpu_uuids")
    if not isinstance(uuids, list) or len(uuids) != gpus:
        raise ValueError(f"sealed server launch must record {gpus} GPU UUIDs")
    return {
        "runtime_revision": launch.get("runtime_revision"),
        "image_digest": launch.get("image_digest"),
        "server_argv": launch.get("argv"),
        "server_environment": launch.get("environment"),
        "gpu_topology": {"gpu_uuids": uuids},
    }


def run_matrix(
    plan: dict,
    *,
    runtime: str,
    runtime_revision: str,
    endpoint: str,
    output: Path,
    concurrencies: list[int],
    gpus: int,
    window_dir: Path,
    request_seconds: float,
    optimizations: list[str] | None,
) -> dict[int, dict]:
    numerical_identity = _sealed_identity(
        output / "server" / "launch.json", plan, runtime=runtime, gpus=gpus
    )
    results = {}
    for concurrency in concurrencies:
        cell = output / cell_dir(concurrency)
        before = time.time() - time.monotonic()
        run = run_plan(
            plan,
            cell / "baseline",
            endpoint=endpoint,
            runtime=runtime,
            runtime_revision=runtime_revision,
            hardware_label=f"native srt-slurm {gpus} GPUs",
            timeout_seconds=request_seconds,
            serving_concurrency=concurrency,
            numerical_identity=numerical_identity,
            optimizations=optimizations,
        )
        after = time.time() - time.monotonic()
        contract = benchmark_window(run, utc_minus_monotonic=(before + after) / 2)
        contract["clock"]["offset_spread_seconds"] = abs(after - before)
        path = cell / window_file(concurrency)
        write_json(path, contract)
        from infx.results.power.window import write_window

        write_window(path, concurrency, window_dir)
        results[concurrency] = run
    return results


if __name__ == "__main__":
    _sglang_main()
