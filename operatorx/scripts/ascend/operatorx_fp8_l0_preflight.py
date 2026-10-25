from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_health_report(health, device):
    identity = health.get("identity", {})
    if identity.get("device") != device:
        raise ValueError("Health report belongs to a different logical device")
    expected = hashlib.sha256(
        json.dumps(
            identity, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    if (
        len(health.get("idle_observations", [])) != 2
        or health.get("identity_sha256") != expected
        or set(health.get("diagnostic_gate", {}))
        != {"health", "current-fault-event", "ecc", "ecc-enable", "pcie-err"}
    ):
        raise ValueError(
            "Missing or inconsistent preflight health and hardware identity"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path)
    p.add_argument("--shape", nargs=3, type=int, metavar=("M", "N", "K"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--guard-so", type=Path, required=True)
    p.add_argument("--worker-dir", type=Path, required=True)
    p.add_argument("--health-report", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--memory-budget-bytes", type=int, required=True)
    args = p.parse_args()
    if (
        args.device < 0
        or args.memory_budget_bytes <= 0
        or not args.guard_so.is_absolute()
    ):
        p.error(
            "Require a nonnegative device, positive memory budget and absolute guard-so path"
        )
    if bool(args.plan) == bool(args.shape):
        p.error("Choose exactly one of --plan and --shape")
    if os.environ.get("TASK_QUEUE_ENABLE") != "0" or str(
        args.guard_so
    ) not in os.environ.get("LD_PRELOAD", "").split(":"):
        p.error("Start with TASK_QUEUE_ENABLE=0 and absolute guard-so in LD_PRELOAD")
    path = args.worker_dir / "operatorx_fp8_block_worker.py"
    expected = sha(path)
    spec = importlib.util.spec_from_file_location("original_fp8", path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    if args.shape:
        m, n, k = args.shape
        cases = [
            {
                "case_id": f"guard-smoke-{m}-{n}-{k}",
                "opType": "gemm",
                "args": {
                    "m": m,
                    "n": n,
                    "k": k,
                    "out": "bf16",
                    "a": {
                        "dtype": "e4m3",
                        "scale": {"dtype": "fp32", "group": [1, 128], "static": False},
                    },
                    "b": {
                        "dtype": "e4m3",
                        "scale": {"dtype": "fp32", "group": [128, 128], "static": True},
                    },
                },
            }
        ]
    else:
        cases = [
            c
            for c in json.loads(args.plan.read_text())["cases"]
            if c.get("phase") in ("new", "remeasure")
            and worker.classify_case(c)["supported"]
        ]
    if args.limit:
        cases = cases[: args.limit]
    if args.output.exists():
        raise ValueError(
            "Use a new output file; do not overwrite existing observations"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    health = json.loads(args.health_report.read_text())["before"]
    validate_health_report(health, args.device)
    guard = ctypes.CDLL(str(args.guard_so))
    guard.operatorx_fp8_guard_configure.argtypes = [ctypes.c_uint32] * 3 + [
        ctypes.c_int
    ]
    guard.operatorx_fp8_guard_configure.restype = ctypes.c_int
    guard.operatorx_fp8_guard_last_json.restype = ctypes.c_char_p
    import torch
    import torch_npu

    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    torch.npu.synchronize()
    _, mm, schemas = worker.native_parameters(torch)
    provenance = {
        "script_sha256": sha(__file__),
        "guard_sha256": sha(args.guard_so),
        "worker_sha256": expected,
        "torch": str(torch.__version__),
        "torch_npu": str(torch_npu.__version__),
        "schemas": schemas,
        "task_queue_enable": "0",
        "diagnostic_only": True,
        "inputs": "Uninitialized tensors with original quantized shape, dtype and strides; GEMM is blocked before launch.",
    }
    for index, case in enumerate(cases):
        watchdog = threading.Timer(300, lambda: os._exit(124))
        watchdog.daemon = True
        watchdog.start()
        a = case["args"]
        m, n, k = (a[x] for x in ("m", "n", "k"))
        record = {
            "case_id": case["case_id"],
            "opType": "gemm",
            "args": a,
            "index": index,
            "total": len(cases),
            "hardware_identity": health["identity_sha256"],
            "utc": datetime.now(timezone.utc).isoformat(),
            "provenance": provenance,
        }
        try:
            free, _ = torch.npu.mem_get_info(args.device)
            required = worker.estimate_memory(case)["estimated_required_bytes"]
            if required > min(int(free), args.memory_budget_bytes):
                raise ValueError(
                    "Preflight tensors exceed the available/configured memory budget"
                )
            q = torch.empty((m, k), device=device, dtype=torch.float8_e4m3fn)
            w = torch.empty((n, k), device=device, dtype=torch.float8_e4m3fn)
            ws = torch.empty(
                ((n + 127) // 128, k // 128), device=device, dtype=torch.float32
            )
            scales = torch.empty((m, k // 128), device=device, dtype=torch.float32)
            torch.npu.synchronize()
            if guard.operatorx_fp8_guard_configure(m, n, k, 1):
                raise RuntimeError("Guard configuration rejected")
            error = None
            try:
                output = torch_npu.npu_quant_matmul(
                    q, w.t(), ws.t(), pertoken_scale=scales, **mm
                )
                torch.npu.synchronize()
                del output
            except RuntimeError as exc:
                error = str(exc)
            finally:
                record["guard"] = json.loads(guard.operatorx_fp8_guard_last_json())
                guard.operatorx_fp8_guard_configure(0, 0, 0, 0)
            record["launch_exception"] = error
            if not record["guard"].get("observed") or error is None:
                raise RuntimeError(
                    "Prelaunch interception not proved; stop before checking further shapes"
                )
            if (
                "100000" not in error
                or record["guard"].get("mode") != 1
                or record["guard"].get("forward_allowed") is not False
            ):
                raise RuntimeError(
                    "Expected deliberate prelaunch rejection not verified"
                )
            if "507015" in error or "507035" in error:
                raise RuntimeError("Unexpected device fault despite prelaunch guard")
            torch.npu.synchronize()
            del q, w, ws, scales
            torch.npu.empty_cache()
            record["status"] = "inspected"
        except Exception:
            record["status"] = "error"
            record["error"] = traceback.format_exc()
        finally:
            watchdog.cancel()
        with args.output.open("a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        print(
            json.dumps(
                {k: record[k] for k in ("case_id", "index", "total", "status")},
                default=str,
            ),
            flush=True,
        )
        if record["status"] != "inspected":
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
