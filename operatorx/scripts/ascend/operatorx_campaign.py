#!/usr/bin/env python3

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
GIB = 1024**3


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def idle_helper_path():
    return HERE / "operatorx_npu_health.py"


def select_cases(plan, phase):
    if phase == "smoke":
        wanted = set(plan["smoke_case_ids"])
        return [case for case in plan["cases"] if case["case_id"] in wanted]
    if phase == "all-ready":
        return [case for case in plan["cases"] if case["phase"] in ("new", "remeasure")]
    return [case for case in plan["cases"] if case["phase"] == phase]


def command(argv, timeout=15):
    start = time.monotonic()
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return {
            "argv": argv,
            "utc": utc(),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "elapsed_s": time.monotonic() - start,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "utc": utc(), "returncode": None, "error": str(exc)}


def idle_snapshot(args, helper, mapped):
    capture = command(["npu-smi", "info"])
    if capture["returncode"] != 0:
        raise RuntimeError("npu-smi info failed: " + canonical(capture))
    parsed = helper.status_rows(capture["stdout"], args.board_id)
    keywords = {}
    if getattr(args, "acl_runtime_mapping", None) is not None:
        keywords["acl_runtime_mapping"] = args.acl_runtime_mapping
    selected = helper.validate_identity(
        parsed,
        mapped,
        args.board_id,
        args.chip_id,
        args.device,
        args.expect_pci,
        **keywords,
    )
    capture["parsed"] = parsed
    capture["selected"] = selected
    if parsed["processes"] or any(
        row["aicore_percent"] != 0 for row in parsed["devices"]
    ):
        raise RuntimeError("Selected board is not idle: " + canonical(capture))
    return capture


def preflight(args):
    helper = load_module(idle_helper_path(), "operatorx_idle_probe")
    visibility = {
        k: os.environ[k] for k in helper.VISIBILITY_VARIABLES if os.environ.get(k)
    }
    if visibility:
        raise RuntimeError(
            "Device remapping must be resolved before running: " + canonical(visibility)
        )
    mapping = command(["npu-smi", "info", "-m"])
    if mapping["returncode"] != 0:
        raise RuntimeError("Cannot obtain NPU mapping: " + canonical(mapping))
    mapped = helper.mapping_rows(mapping["stdout"])
    if any(
        row["logical_device_id"] is None
        for row in mapped
        if row["board"] == args.board_id
    ):
        args.acl_runtime_mapping = helper.collect_acl_runtime_mapping(args.device)
    observations = [idle_snapshot(args, helper, mapped)]
    time.sleep(1)
    observations.append(idle_snapshot(args, helper, mapped))
    queries = {
        kind: command(["npu-smi", "info", "-t", kind, "-i", str(args.board_id)])
        for kind in ("board", "common")
    }
    diagnostic_gate = None
    if hasattr(helper, "validate_diagnostic_queries"):
        for kind in ("health", "current-fault-event", "ecc", "ecc-enable", "pcie-err"):
            queries[kind] = command(
                ["npu-smi", "info", "-t", kind, "-i", str(args.board_id)]
            )
        diagnostic_gate = helper.validate_diagnostic_queries(
            list(queries.values()), args.board_id
        )
    raw_identity = canonical(
        {"mapping": mapped, "board": queries["board"].get("stdout")}
    )
    if args.expected_name.lower() not in (raw_identity + canonical(queries)).lower():
        raise RuntimeError(
            "Expected chip name not present in npu-smi identity captures"
        )
    if queries["board"]["returncode"] != 0:
        raise RuntimeError(
            "A successful board identity capture is required for resumable results"
        )
    identity = {
        "board_id": args.board_id,
        "chip_id": args.chip_id,
        "device": args.device,
        "pci": observations[-1]["selected"]["pci"],
        "mapping": mapped,
        "board_text": queries["board"]["stdout"],
    }
    report = {
        "utc": utc(),
        "mapping": mapping,
        "idle_observations": observations,
        "queries": queries,
        "identity": identity,
        "diagnostic_gate": diagnostic_gate,
        "acl_runtime_mapping": getattr(args, "acl_runtime_mapping", None),
        "identity_sha256": hashlib.sha256(canonical(identity).encode()).hexdigest(),
    }
    return report, helper, mapped


def open_journal(path, metadata):
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS results (seq INTEGER PRIMARY KEY, case_id TEXT NOT NULL, status TEXT NOT NULL, record TEXT NOT NULL)"
    )
    fingerprint = canonical(metadata)
    existing = connection.execute(
        "SELECT value FROM metadata WHERE key='fingerprint'"
    ).fetchone()
    if existing and existing[0] != fingerprint:
        connection.close()
        raise ValueError(
            "Resume refused: plan, scripts, hardware identity or timing configuration changed; use a new run directory"
        )
    if not existing:
        with connection:
            connection.execute(
                "INSERT INTO metadata VALUES ('fingerprint', ?)", (fingerprint,)
            )
    return connection


def record_result(connection, case_id, record):
    with connection:
        connection.execute(
            "INSERT INTO results(case_id,status,record) VALUES (?,?,?)",
            (case_id, record["status"], canonical(record)),
        )


def export_journal(connection, path):
    latest = {}
    for case_id, record in connection.execute(
        "SELECT case_id,record FROM results ORDER BY seq"
    ):
        latest[case_id] = json.loads(record)
    summary = {
        "exported_at": utc(),
        "counts": dict(Counter(x["status"] for x in latest.values())),
        "results": list(latest.values()),
        "qualification": "Eager NPU events with a cache-write preparation. This is not proof of cache eviction or public-run protocol parity. Correctness is sampled CPU FP64 reference validation, not full output certification.",
    }
    write_json(path, summary)
    return summary


def validate_worker_result(outcome, case, config, worker_sha):
    if not isinstance(outcome, dict) or outcome.get("status") not in (
        "ok",
        "error",
        "unsupported",
        "skipped_memory",
    ):
        raise ValueError("Worker did not return a recognized terminal status")
    if (
        outcome.get("case_id") != case["case_id"]
        or outcome.get("opType") != case["opType"]
        or outcome.get("args") != case["args"]
    ):
        raise ValueError("Worker result identity differs from the requested case")
    semantic = {"opType": case["opType"], "args": case["args"]}
    provenance = outcome.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("Worker provenance is not an object")
    if (
        provenance.get("worker_sha256") != worker_sha
        or provenance.get("case_semantic_sha256")
        != hashlib.sha256(canonical(semantic).encode()).hexdigest()
    ):
        raise ValueError("Worker/source or semantic hash mismatch")
    if outcome["status"] != "ok":
        if outcome.get("metrics") is not None:
            raise ValueError("Unsuccessful result must not contain performance metrics")
        return
    for key in ("config", "correctness", "timing", "metrics", "tensors"):
        if not isinstance(outcome.get(key), dict):
            raise ValueError("Successful result requires an object: " + key)
    for key in (
        "samples",
        "warmups",
        "warmup_ms",
        "seed",
        "device",
        "cache_write_bytes",
        "workspace_reserve_bytes",
        "memory_budget_bytes",
    ):
        if outcome.get("config", {}).get(key) != config[key]:
            raise ValueError("Worker configuration mismatch: " + key)
    for when in ("before_timing", "after_timing"):
        check = outcome.get("correctness", {}).get(when, {})
        if (
            not isinstance(check, dict)
            or check.get("passed") is not True
            or check.get("all_output_elements_finite") is not True
            or not check.get("samples")
        ):
            raise ValueError("Missing successful correctness/finite gate: " + when)
    samples = outcome.get("timing", {}).get("device_event_samples_us", [])
    if (
        not isinstance(samples, list)
        or len(samples) != config["samples"]
        or any(
            type(x) not in (int, float) or not math.isfinite(x) or x <= 0
            for x in samples
        )
    ):
        raise ValueError("Invalid or incomplete raw event timing samples")
    median = sorted(samples)[len(samples) // 2]
    metrics = outcome.get("metrics", {})
    if metrics.get("latency_us") != median or metrics.get("npu_graph") is not False:
        raise ValueError("Reported latency or graph mode disagrees with the raw timing")
    a = case["args"]
    expected = 2 * a["m"] * a["n"] * a["k"] / (median * 1e6)
    if not math.isclose(metrics.get("tflops", float("nan")), expected, rel_tol=1e-12):
        raise ValueError(
            "Throughput disagrees with requested work and measured latency"
        )
    output = outcome.get("tensors", {}).get("output", {})
    dtype = {
        "bf16": "torch.bfloat16",
        "fp16": "torch.float16",
        "fp32": "torch.float32",
    }[a.get("out", "bf16")]
    if (
        not isinstance(output, dict)
        or output.get("shape") != [a["m"], a["n"]]
        or output.get("dtype") != dtype
    ):
        raise ValueError("Output descriptor differs from the requested shape/precision")


def monitor(stop, path, board_id, helper, state):
    with path.open("a") as handle:
        while not stop.is_set():
            sample_start = time.monotonic()
            row = {
                "utc": utc(),
                "status": command(["npu-smi", "info"], timeout=5),
                "common": command(
                    ["npu-smi", "info", "-t", "common", "-i", str(board_id)], timeout=5
                ),
            }
            try:
                if row["status"]["returncode"] != 0:
                    raise ValueError("npu-smi status query failed")
                parsed = helper.status_rows(row["status"]["stdout"], board_id)
                with state["lock"]:
                    foreign = [
                        p
                        for p in parsed["processes"]
                        if p["pid"] not in state["own_pids"]
                    ]
                if foreign:
                    row["foreign_processes"] = foreign
                    with state["lock"]:
                        state["issues"].append(
                            {
                                "start": sample_start,
                                "end": time.monotonic(),
                                "utc": row["utc"],
                                "foreign_processes": foreign,
                            }
                        )
            except ValueError as exc:
                row["process_check_error"] = str(exc)
                with state["lock"]:
                    state["issues"].append(
                        {
                            "start": sample_start,
                            "end": time.monotonic(),
                            "utc": row["utc"],
                            "process_check_error": str(exc),
                        }
                    )
            handle.write(canonical(row) + "\n")
            handle.flush()
            stop.wait(1)
