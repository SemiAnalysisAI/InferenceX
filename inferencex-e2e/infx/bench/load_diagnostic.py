from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import socket
import subprocess
import time
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any


def emit(event: str, **fields: Any) -> None:
    print(
        json.dumps({"time": time.time(), "host": socket.gethostname(), "event": event, **fields}),
        flush=True,
    )


def read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError as error:
        return f"unavailable: {error}"


def gpu_snapshot(root: Path = Path("/sys/class/drm")) -> dict[str, Any]:
    result = {}
    for card in sorted(root.glob("card[0-9]*")):
        if not card.name.removeprefix("card").isdigit():
            continue
        device = card / "device"
        if not (device / "mem_info_vram_total").exists():
            continue
        result[card.name] = {
            "pci": device.resolve().name,
            **{
                name: read(device / name)
                for name in (
                    "mem_info_vram_total",
                    "mem_info_vram_used",
                    "mem_info_vis_vram_used",
                    "mem_info_gtt_used",
                    "gpu_busy_percent",
                    "mem_busy_percent",
                )
            },
        }
    return result


def gpu_processes(root: Path = Path("/proc")) -> dict[str, Any]:
    processes = []
    denied = 0
    for process in root.glob("[0-9]*"):
        try:
            descriptors = list((process / "fd").iterdir())
        except PermissionError:
            denied += 1
            continue
        except OSError:
            continue
        gpu_fds = {}
        for fd in descriptors:
            try:
                target = str(fd.readlink())
            except OSError:
                continue
            if target == "/dev/kfd" or target.startswith("/dev/dri/"):
                gpu_fds[fd.name] = {"device": target, "info": read(process / "fdinfo" / fd.name)}
        if gpu_fds:
            processes.append(
                {
                    "pid": int(process.name),
                    "comm": read(process / "comm"),
                    "status": read(process / "status"),
                    "cgroup": read(process / "cgroup"),
                    "fds": gpu_fds,
                }
            )
    return {"processes": processes, "inaccessible_processes": denied}


def command_snapshot(argv: list[str]) -> None:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=10, check=False)
        emit(
            "command",
            argv=argv,
            returncode=result.returncode,
            stdout=result.stdout[-24000:],
            stderr=result.stderr[-4000:],
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        emit("command_unavailable", argv=argv, error=str(error))


def snapshot(phase: str, *, detailed: bool) -> None:
    emit(
        "snapshot",
        phase=phase,
        gpus=gpu_snapshot(),
        meminfo=read(Path("/proc/meminfo")),
        memory_pressure=read(Path("/proc/pressure/memory")),
        **gpu_processes(),
    )
    if detailed:
        emit(
            "system",
            kernel=platform.release(),
            limits=read(Path("/proc/self/limits")),
            cgroup=read(Path("/proc/self/cgroup")),
            amdgpu_version=read(Path("/sys/module/amdgpu/version")),
            job_id=os.environ.get("SLURM_JOB_ID"),
        )
        command_snapshot(["rocm-smi", "--showmeminfo", "vram", "--showpids", "--showuniqueid"])
        command_snapshot(["rocm-smi", "--showrasinfo", "all", "--showretiredpages"])
        command_snapshot(["dmesg", "--ctime", "--level=err,warn"])


def watch(seconds: int) -> None:
    start = time.monotonic()
    tick = 0
    while time.monotonic() - start < seconds:
        emit("vram", gpus=gpu_snapshot())
        if tick % 10 == 0:
            snapshot("loading", detailed=False)
        tick += 1
        time.sleep(1)


def torch_memory(torch: Any) -> dict[str, Any]:
    # Query only the worker's initialized device; do not synchronize or empty its cache.
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    stats = torch.cuda.memory_stats(device)
    return {
        "device": device,
        "free": free,
        "total": total,
        **{
            key: stats.get(key)
            for key in (
                "allocated_bytes.all.current",
                "reserved_bytes.all.current",
                "allocated_bytes.all.peak",
                "reserved_bytes.all.peak",
                "inactive_split_bytes.all.current",
                "num_alloc_retries",
                "num_ooms",
            )
        },
    }


def trace_weights_iterator(
    original: Callable[..., Iterator[Any]], *, logdir: Path = Path("/logs")
) -> Callable[..., Iterator[Any]]:
    @functools.wraps(original)
    def traced(files: list[str], *args: Any, **kwargs: Any) -> Iterator[Any]:
        import torch

        pid = os.getpid()
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        log_path = logdir / f"load-memory-{socket.gethostname()}-rank{rank}-pid{pid}.jsonl"
        stream = log_path.open("a", buffering=1)

        def record(event: str, **fields: Any) -> None:
            try:
                stream.write(
                    json.dumps(
                        {
                            "time": time.time(),
                            "event": event,
                            "pid": pid,
                            "rank": rank,
                            **torch_memory(torch),
                            **fields,
                        }
                    )
                    + "\n"
                )
            except (OSError, RuntimeError) as error:
                print(f"load diagnostic unavailable: {error}", flush=True)

        iterator = None
        count = 0
        last_name = None
        try:
            manifest = [{"path": f, "bytes": Path(f).stat().st_size} for f in files]
            record("iterator_start", files=manifest)
            iterator = original(files, *args, **kwargs)
            last_sample = 0.0
            for name, tensor in iterator:
                count += 1
                last_name = name
                now = time.monotonic()
                if now - last_sample >= 0.5:
                    record(
                        "tensor_yield",
                        count=count,
                        name=name,
                        shape=list(tensor.shape),
                        dtype=str(tensor.dtype),
                        tensor_bytes=tensor.numel() * tensor.element_size(),
                    )
                    last_sample = now
                yield name, tensor
                del tensor
            record("iterator_complete", count=count)
        except BaseException as error:
            record("iterator_error", count=count, last_name=last_name, error=repr(error))
            raise
        finally:
            if iterator is not None:
                iterator.close()
            record("iterator_closed", count=count)
            stream.close()

    return traced


def checkpoint_manifest(folder: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(folder.glob("*.safetensors")):
        record = {"path": str(path), "bytes": path.stat().st_size}
        with path.open("rb") as source:
            size = int.from_bytes(source.read(8), "little")
            # Hash metadata only, never scan multi-terabyte weight payloads.
            if 0 < size <= 16 * 1024 * 1024:
                header = source.read(size)
                if len(header) != size:
                    raise ValueError(f"truncated safetensors header: {path}")
                record["header_sha256"] = hashlib.sha256(header).hexdigest()
            else:
                record["header_skipped_bytes"] = size
        records.append(record)
    return records


def install(logdir: Path = Path("/logs")) -> None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("vLLM source not found")
    folder = Path(spec.origin).parent / "model_executor/model_loader"
    target = folder / "weight_utils.py"
    source = target.read_text()
    marker = "# K3 one-shot loader diagnostics"
    if marker in source or "def fastsafetensors_weights_iterator(" not in source:
        raise RuntimeError("unexpected weight_utils source; refusing to instrument")
    host = socket.gethostname()
    (logdir / f"weight-utils-original-{host}.py").write_text(source)
    for model in (Path("/model"), Path("/models/Inferact-Kimi-K3-DSpark")):
        emit("checkpoint_manifest", model=str(model), files=checkpoint_manifest(model))
        for name in ("config.json", "model.safetensors.index.json"):
            path = model / name
            if path.is_file():
                emit(
                    "checkpoint_metadata",
                    path=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
    for package in ("torch", "vllm", "fastsafetensors", "aiter", "mori"):
        try:
            emit("package", package=package, version=importlib.metadata.version(package))
        except importlib.metadata.PackageNotFoundError:  # noqa: PERF203
            emit("package", package=package, version="metadata unavailable")
    loader = importlib.util.find_spec("fastsafetensors")
    if loader is None or loader.origin is None:
        raise RuntimeError("fastsafetensors source not found")
    for name in ("parallel_loader.py", "loader.py", "copier.py"):
        path = Path(loader.origin).parent / name
        if path.exists():
            data = path.read_bytes()
            (logdir / f"fastsafetensors-{name}-{host}.txt").write_bytes(data)
            emit("source", path=str(path), sha256=hashlib.sha256(data).hexdigest())
    helper = folder / "_k3_load_diagnostic.py"
    if helper.exists():
        raise RuntimeError(f"refusing to overwrite {helper}")
    helper.write_bytes(Path(__file__).read_bytes())
    suffix = (
        f"\n{marker}\nfrom ._k3_load_diagnostic import trace_weights_iterator as _k3_trace\n"
        "fastsafetensors_weights_iterator = _k3_trace(fastsafetensors_weights_iterator)\n"
    )
    compile(source + suffix, str(target), "exec")
    target.write_text(source + suffix)
    emit(
        "instrumented",
        path=str(target),
        original_sha256=hashlib.sha256(source.encode()).hexdigest(),
    )


def smoke() -> None:
    base = f"http://{os.environ['SRT_FRONTEND_HOST']}:{int(os.environ['SRT_FRONTEND_PORT'])}"
    payload = {
        "model": os.environ["MODEL"],
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 16,
        "temperature": 0,
        "stream": False,
    }
    request = urllib.request.Request(  # noqa: S310
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        body = json.load(response)
        if response.status != 200 or not body.get("choices"):
            raise RuntimeError(f"Invalid routed response: {body}")
        emit("routed_request", status=response.status, response=body)
    emit(
        "diagnostic_complete",
        note="Startup and routed-request diagnostic only; no performance result",
    )


def preserve(workspace: Path, run_id: str) -> None:
    if not run_id.isdigit():
        raise ValueError("run ID must be numeric")
    manifest = json.loads((workspace / "srt-submission.json").read_text())
    if isinstance(manifest, list):
        manifest = manifest[0]
    output = Path(manifest["output_dir"])
    destination = output.parent.parent / "k3-startup-diagnostics" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    for name in (
        "LOGS",
        "multinode_server_logs.tar.gz",
        "srt-submission.json",
        "srt-bind.log",
        "srt-setup.log",
    ):
        path = workspace / name
        if path.is_dir():
            shutil.copytree(path, destination / name)
        elif path.is_file():
            shutil.copy2(path, destination / name)
    reports = []
    for log in sorted(destination.rglob("load-memory-*.jsonl")):
        records = []
        for line in log.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:  # noqa: PERF203
                continue
        if records:
            item = {
                "file": str(log.relative_to(destination)),
                "samples": len(records),
                "min_free_bytes": min(row["free"] for row in records),
                "last": records[-1],
            }
            reports.append(item)
            emit("loader_summary", **item)
    (destination / "loader-summary.json").write_text(json.dumps(reports, indent=2))
    report = (
        "# K3 c70 startup diagnostic\n\n"
        f"CI run: {run_id}. Baseline: ec02c03075854a0bb0688e31afee2001cc39e5eb.\n\n"
        "1P1D, 16 MI355X GPUs, unchanged c70 engine settings and image digest. "
        "Only startup and one short routed request; no AgentX profiling or GSM8K. "
        "TTFT/ITL percentiles, throughput and Pareto metrics: not measured.\n\n"
        "Evidence: load-memory-*.jsonl (worker allocator), load-watch-*.jsonl "
        "(per-device VRAM/processes), host_setup_*.out and host_teardown_*.out "
        "(prior occupation and release), P/D worker logs, installed loader source copies. "
        "Inaccessible process details are counted, not interpreted as absence. "
        "See loader-summary.json for minimum observed free memory and last samples; "
        "the final sample need not be the absolute failure peak.\n\n"
        "K3 validation and Slurm job-sizing guidance were used to preserve engine controls "
        "and bound traffic. No GPU reset, process killing or allocator synchronization "
        "was added. Root-cause assessment remains pending cross-log analysis.\n"
    )
    (destination / "perf-analysis.md").write_text(report)
    if (output / "logs").is_dir():
        (output / "logs/perf-analysis.md").write_text(report)
    emit("preserved", directory=str(destination), reports=len(reports))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("snapshot", "watch", "install", "smoke", "preserve"))
    parser.add_argument("--phase", default="unspecified")
    parser.add_argument("--seconds", type=int, default=1)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    if args.mode == "snapshot":
        snapshot(args.phase, detailed=True)
    elif args.mode == "watch":
        watch(args.seconds)
    elif args.mode == "install":
        install()
    elif args.mode == "smoke":
        smoke()
    else:
        if args.workspace is None or args.run_id is None:
            parser.error("preserve requires --workspace and --run-id")
        preserve(args.workspace, args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
