#!/usr/bin/env python3
"""GPU-free validation of the ExecutionTrace capture shim.

Exercises benchmarks/patches/execution_trace_shim/sitecustomize.py exactly the
way the engines do -- construct torch.profiler.profile, call .start(), run
forwards, call .stop(), export (sglang scheduler pattern; vllm's
TorchProfilerWrapper reuses one profile object across sessions, covered below)
-- and then proves the two outputs join:

  1. the ET JSON has operator nodes with rf_id and tensor inputs/outputs
     (the dataflow edges kineto lacks), and
  2. those rf_ids match the kineto trace's cpu_op "Record function id" args,
     with identical op names on matched pairs.

Run (any machine, CPU is fine):
    uv run --with torch python utils/validate_execution_trace_join.py
or with an existing torch install:
    python3 utils/validate_execution_trace_join.py
"""

import importlib.util
import json
import os
import sys
import tempfile


def main() -> int:
    out_dir = tempfile.mkdtemp(prefix="et-validate-")
    os.environ["PROFILE_EXECUTION_TRACE_DIR"] = out_dir

    # Load the shim the same way sitecustomize would (env var already set, so
    # its import hook arms itself before torch.profiler is imported).
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shim_path = os.path.join(
        repo_root, "benchmarks", "patches", "execution_trace_shim", "sitecustomize.py"
    )
    assert "torch" not in sys.modules, "import torch after the shim, not before"
    spec = importlib.util.spec_from_file_location("sitecustomize", shim_path)
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)

    import torch
    from torch import nn

    assert getattr(
        torch.profiler.profile, "_infx_et_patched", False
    ), "import hook did not patch torch.profiler.profile"

    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
    x = torch.randn(4, 16)

    # sglang's SchedulerProfilerManager pattern: no schedule, explicit
    # start/stop, export after stop.
    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        with_stack=True,
        record_shapes=True,
    )
    prof.start()
    for _ in range(3):
        model(x)
    prof.stop()
    kineto_path = os.path.join(out_dir, "validate.trace.json")
    prof.export_chrome_trace(kineto_path)

    et_files = [f for f in os.listdir(out_dir) if f.startswith("et-") and f.endswith(".json")]
    assert len(et_files) == 1, f"expected one finalized ET JSON, got {et_files}"
    et_path = os.path.join(out_dir, et_files[0])

    with open(et_path) as fh:
        et = json.load(fh)
    nodes = et["nodes"]

    # ET node schema (torch 2.x): rf_id is an entry in the "attrs" list;
    # inputs/outputs are {"values", "shapes", "types", "strides"} where tensor
    # values are [tensor_id, storage_id, offset, numel, elem_bytes, device].
    def rf_id_of(node):
        for attr in node.get("attrs", []):
            if attr.get("name") == "rf_id":
                return attr.get("value")
        return node.get("rf_id")  # pre-2.x layout

    def has_tensor_io(node, key):
        io = node.get(key, {})
        types = io["types"] if isinstance(io, dict) else []
        return any(t.startswith("Tensor") for t in types)

    et_by_rfid = {}
    tensor_nodes = 0
    for node in nodes:
        rf_id = rf_id_of(node)
        if rf_id is not None:
            et_by_rfid[rf_id] = node
        if has_tensor_io(node, "inputs") or has_tensor_io(node, "outputs"):
            tensor_nodes += 1
    assert et_by_rfid, "ET JSON has no nodes with rf_id"
    assert tensor_nodes, "ET JSON has no nodes with tensor inputs/outputs (no dataflow)"

    with open(kineto_path) as fh:
        kineto = json.load(fh)
    cpu_ops = [
        e
        for e in kineto["traceEvents"]
        if e.get("cat") == "cpu_op" and "Record function id" in e.get("args", {})
    ]
    assert cpu_ops, "kineto trace has no cpu_op events with 'Record function id'"

    joined = mismatched = 0
    for ev in cpu_ops:
        node = et_by_rfid.get(ev["args"]["Record function id"])
        if node is None:
            continue
        joined += 1
        if node["name"] != ev["name"]:
            mismatched += 1
    assert joined, "no kineto cpu_op joined an ET node on rf_id"
    assert mismatched == 0, f"{mismatched}/{joined} joined pairs disagree on op name"

    kineto_rfids = {e["args"]["Record function id"] for e in cpu_ops}
    sample = next(
        n
        for n in nodes
        if rf_id_of(n) in kineto_rfids and has_tensor_io(n, "inputs")
    )
    print(f"ET file: {et_path}")
    print(f"ET nodes: {len(nodes)} ({len(et_by_rfid)} with rf_id, {tensor_nodes} with tensor IO)")
    print(f"kineto cpu_ops with 'Record function id': {len(cpu_ops)}")
    print(f"joined on rf_id: {joined} (op names all match)")
    print(f"sample joined node: {sample['name']} rf_id={rf_id_of(sample)}")
    print("OK: ET <-> kineto rf_id join validated")

    # sglang's profile_by_stage runs two sequential sessions (EXTEND, DECODE)
    # in the same scheduler process with fresh profile objects; prove a second
    # observer registers cleanly after the first unregistered.
    prof2 = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True
    )
    prof2.start()
    model(x)
    prof2.stop()
    et_files2 = sorted(
        f for f in os.listdir(out_dir) if f.startswith("et-") and f.endswith(".json")
    )
    assert len(et_files2) == 2, f"expected two ET JSONs after second session, got {et_files2}"
    with open(os.path.join(out_dir, et_files2[-1])) as fh:
        assert json.load(fh)["nodes"], "second session ET JSON is empty"
    print(f"OK: second (sglang by-stage) session produced {et_files2[-1]}")

    # vllm's TorchProfilerWrapper keeps ONE profile object per worker and
    # calls start/stop on it for every capture session; prove the shim
    # produces a fresh ET file when the same object is restarted.
    prof2.start()
    model(x)
    prof2.stop()
    et_files3 = sorted(
        f for f in os.listdir(out_dir) if f.startswith("et-") and f.endswith(".json")
    )
    assert len(et_files3) == 3, f"expected three ET JSONs after object reuse, got {et_files3}"
    with open(os.path.join(out_dir, et_files3[-1])) as fh:
        assert json.load(fh)["nodes"], "reused-object session ET JSON is empty"
    print(f"OK: reused profile object (vllm worker pattern) produced {et_files3[-1]}")

    # Per-window control: a .et-disabled marker (in the output dir or its
    # parent) suppresses the observer for sessions starting while it exists
    # -- how the agentic trigger keeps ET off the stacked ramp window -- and
    # removal re-enables it for the next session.
    def _count_et():
        return sum(
            1 for f in os.listdir(out_dir)
            if f.startswith("et-") and f.endswith(".json")
        )

    marker = os.path.join(out_dir, ".et-disabled")
    open(marker, "w").close()
    prof2.start()
    model(x)
    prof2.stop()
    assert _count_et() == 3, "marker did not suppress the ET observer"
    os.remove(marker)
    prof2.start()
    model(x)
    prof2.stop()
    assert _count_et() == 4, "removing the marker did not re-enable ET"
    print("OK: .et-disabled marker suppresses and re-enables per session")
    return 0


if __name__ == "__main__":
    sys.exit(main())
