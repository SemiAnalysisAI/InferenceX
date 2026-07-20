"""Load and flatten InferenceX matrix entries from master YAML configs.

Delegates the heavy expansion (conc-range -> per-conc rows, search-space
fan-out, etc.) to InferenceX's own `generate_sweep_configs.py full-sweep` so
that this module stays in sync with InferenceX semantics automatically.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WorkloadRow:
    """One InferenceX matrix entry, normalised for shape enumeration."""

    model: str             # HF id, e.g. "deepseek-ai/DeepSeek-R1-0528"
    model_prefix: str      # InferenceX canonical model family, e.g. "dsr1"
    framework: str         # sglang / trt / vllm / dynamo-trt / ...
    precision: str         # fp4 / fp8 / bf16 / int4 / mxfp4 / nvfp4
    runner: str            # b200 / b300 / h200 / mi300x / ...
    isl: int
    osl: int
    conc: int              # single-node: scalar; multi-node: representative value
    tp: int
    ep: int
    dp_attn: bool
    spec_decoding: str     # "none" | "mtp" | "draft_model"
    disagg: bool
    multinode: bool
    exp_name: str
    # For multinode disagg rows: which phase this row represents and how many
    # worker instances exist for that phase. Single-node rows leave phase=None
    # (both prefill+decode ops emitted) and num_workers=1.
    phase: str | None = None         # "prefill" | "decode" | None
    num_workers: int = 1

    @property
    def world_size(self) -> int:
        # Single-node: world_size == tp * ep_dim. EP and TP can overlap in
        # InferenceX configs (e.g. tp=8, ep=4 means EP shards across 4 of the 8
        # ranks). For collective sizing we use max(tp, ep) as the dominant
        # group size — refined per-collective in collectives.py.
        return max(self.tp, self.ep)


def _run_inferencex_generator(
    inferencex_dir: str,
    config_files: list[str],
    extra_flags: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Invoke InferenceX's matrix generator and return the JSON entries."""
    if not os.path.isdir(inferencex_dir):
        raise FileNotFoundError(f"InferenceX directory not found: {inferencex_dir}")

    script = os.path.join(
        inferencex_dir, "utils", "matrix_logic", "generate_sweep_configs.py"
    )
    if not os.path.isfile(script):
        raise FileNotFoundError(f"InferenceX generator not found: {script}")

    args = [sys.executable, script, "full-sweep", "--config-files", *config_files]
    if extra_flags:
        args.extend(extra_flags)

    # Run from the InferenceX root so relative paths in the master config resolve.
    proc = subprocess.run(
        args, capture_output=True, text=True, cwd=inferencex_dir, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"generate_sweep_configs.py failed (exit {proc.returncode}):\n"
            f"stderr:\n{proc.stderr}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Could not parse generator output as JSON: {e}\n"
            f"stdout head:\n{proc.stdout[:1000]}"
        ) from e


_AGENTIC_DEFAULT_ISL = 8192   # representative for agentic-coding traces
_AGENTIC_DEFAULT_OSL = 1024


def _normalize_entry(entry: dict[str, Any]) -> list[WorkloadRow]:
    """Turn one matrix entry (single-node or multi-node) into one+ WorkloadRows.

    Multi-node entries have nested prefill/decode worker configs and a list of
    concs; we emit one row per (worker phase, conc) so prefill and decode
    shapes get enumerated independently.

    Agentic-coding scenarios don't carry isl/osl in the matrix entry — the
    actual lengths come from the trace replay. We substitute representative
    defaults so the canonical-shape enumerator still emits useful entries.
    """
    multinode = "prefill" in entry and "decode" in entry
    base = dict(
        model=entry["model"],
        model_prefix=entry["model-prefix"],
        framework=entry["framework"],
        precision=entry["precision"],
        runner=entry["runner"],
        isl=int(entry.get("isl", _AGENTIC_DEFAULT_ISL)),
        osl=int(entry.get("osl", _AGENTIC_DEFAULT_OSL)),
        spec_decoding=entry.get("spec-decoding", "none"),
        disagg=bool(entry.get("disagg", False)),
        multinode=multinode,
        exp_name=entry["exp-name"],
    )

    if multinode:
        concs = entry["conc"]
        if isinstance(concs, int):
            concs = [concs]
        rows: list[WorkloadRow] = []
        # Each multinode entry yields one row per (phase, conc). The row is
        # tagged with `phase` so the enumerator only emits that phase's ops,
        # and with `num_workers` so per-step batch sizes get divided by the
        # number of worker instances handling that phase.
        for phase in ("prefill", "decode"):
            worker = entry[phase]
            num_workers = int(worker.get("num-worker", 1))
            for c in concs:
                rows.append(WorkloadRow(
                    conc=int(c),
                    tp=int(worker["tp"]),
                    ep=int(worker.get("ep", 1)),
                    dp_attn=bool(worker.get("dp-attn", False)),
                    phase=phase,
                    num_workers=num_workers,
                    **base,
                ))
        return rows

    concs = entry["conc"]
    if isinstance(concs, int):
        concs = [concs]
    return [
        WorkloadRow(
            conc=int(c),
            tp=int(entry["tp"]),
            ep=int(entry.get("ep", 1)),
            dp_attn=bool(entry.get("dp-attn", False)),
            **base,
        )
        for c in concs
    ]


def load_rows(
    inferencex_dir: str,
    config_files: list[str],
    extra_flags: list[str] | None = None,
) -> list[WorkloadRow]:
    raw = _run_inferencex_generator(inferencex_dir, config_files, extra_flags)
    rows: list[WorkloadRow] = []
    for entry in raw:
        rows.extend(_normalize_entry(entry))
    return rows
