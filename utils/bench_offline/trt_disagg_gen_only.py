"""Helpers for the DSV4 TensorRT-LLM disaggregated offline benchmark.

The deployment remains CTX/GEN disaggregated.  Only generation-worker device
iterations are scored, using decode steps as the primary throughput unit.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ACCEPTANCE_LENGTHS = {1: 1.70, 3: 2.44}


@dataclass(frozen=True)
class CaseMetadata:
    concurrency: int
    ctx_num_workers: int
    ctx_tp: int
    ctx_ep: int
    ctx_dp_attn: bool
    gen_num_workers: int
    gen_tp: int
    gen_ep: int
    gen_dp_attn: bool
    mtp: int
    node_count: int


@dataclass(frozen=True)
class IterationSummary:
    raw_exact_batch_samples: int
    retained_samples: int
    mean_device_step_ms: float
    median_device_step_ms: float
    p90_device_step_ms: float
    p99_device_step_ms: float


class NoExactFullBatchError(ValueError):
    """Raised when an iterlog has no exact-full-batch generation rows."""


_ITERLOG_RE = re.compile(
    r"iter = (?P<iter>\d+), global_rank = (?P<global_rank>\d+), "
    r"rank = \d+, currank_total_requests = \d+/\d+, "
    r"host_step_time = [\d.]+ms, prev_device_step_time = "
    r"(?P<device_ms>[\d.]+)ms, timestamp = .*?, "
    r"num_scheduled_requests: (?P<scheduled>\d+), "
    r"states = \{'num_ctx_requests': \d+, 'num_ctx_tokens': "
    r"(?P<ctx_tokens>\d+), 'num_generation_tokens': "
    r"(?P<generation_tokens>\d+)\}"
)


def timed_max_tokens(decode_steps: int, mtp: int) -> int:
    """Return the output-token cap targeting ``decode_steps`` model rounds."""
    if decode_steps <= 0:
        raise ValueError("decode_steps must be positive")
    try:
        acceptance = DEFAULT_ACCEPTANCE_LENGTHS[mtp]
    except KeyError as exc:
        raise ValueError(f"No fixed acceptance length for MTP{mtp}") from exc
    return 1 + round((decode_steps - 1) * acceptance)


def _world_size(worker: dict[str, Any]) -> int:
    return (
        int(worker.get("tensor_parallel_size", 1))
        * int(worker.get("context_parallel_size", 1))
        * int(worker.get("pipeline_parallel_size", 1))
    )


def case_metadata(config: dict[str, Any]) -> CaseMetadata:
    benchmark = config["benchmark"]
    hardware = config["hardware"]
    ctx = config["worker_config"]["ctx"]
    gen = config["worker_config"]["gen"]
    ctx_workers = int(hardware["num_ctx_servers"])
    gen_workers = int(hardware["num_gen_servers"])
    gpus_per_node = int(hardware["gpus_per_node"])
    total_gpus = ctx_workers * _world_size(ctx) + gen_workers * _world_size(gen)
    return CaseMetadata(
        concurrency=int(str(benchmark["concurrency_list"]).split(",")[0]),
        ctx_num_workers=ctx_workers,
        ctx_tp=int(ctx["tensor_parallel_size"]),
        ctx_ep=int(ctx["moe_expert_parallel_size"]),
        ctx_dp_attn=bool(ctx["enable_attention_dp"]),
        gen_num_workers=gen_workers,
        gen_tp=int(gen["tensor_parallel_size"]),
        gen_ep=int(gen["moe_expert_parallel_size"]),
        gen_dp_attn=bool(gen["enable_attention_dp"]),
        mtp=int(gen["speculative_config"]["max_draft_len"]),
        node_count=math.ceil(total_gpus / gpus_per_node),
    )


def validate_case_against_workflow(
    config: dict[str, Any],
    *,
    concurrency: int,
    prefill_num_workers: int,
    prefill_tp: int,
    prefill_ep: int,
    prefill_dp_attn: bool,
    decode_num_workers: int,
    decode_tp: int,
    decode_ep: int,
    decode_dp_attn: bool,
) -> CaseMetadata:
    """Reject a master-config row that disagrees with its source case YAML."""
    metadata = case_metadata(config)
    expected = {
        "concurrency": concurrency,
        "ctx_num_workers": prefill_num_workers,
        "ctx_tp": prefill_tp,
        "ctx_ep": prefill_ep,
        "ctx_dp_attn": prefill_dp_attn,
        "gen_num_workers": decode_num_workers,
        "gen_tp": decode_tp,
        "gen_ep": decode_ep,
        "gen_dp_attn": decode_dp_attn,
    }
    mismatches = [
        f"{name}: source={getattr(metadata, name)!r}, workflow={value!r}"
        for name, value in expected.items()
        if getattr(metadata, name) != value
    ]
    if mismatches:
        raise ValueError("Workflow/source topology mismatch: " + "; ".join(mismatches))
    if config["benchmark"]["mode"] != "gen_only":
        raise ValueError("Offline disaggregated cases must use benchmark.mode=gen_only")
    if metadata.gen_num_workers != 1:
        raise ValueError("Offline disaggregated cases require exactly one GEN worker")
    return metadata


def render_effective_config(
    source: dict[str, Any],
    *,
    output_path: Path,
    partition: str,
    account: str,
    job_name: str,
    container_image: str,
    container_mount: str,
    model_path: str,
    dataset_root: str,
    log_dir: str,
    decode_steps: int,
    job_time: str = "03:00:00",
) -> tuple[dict[str, Any], CaseMetadata]:
    """Render cluster-local paths without mutating the pinned source case."""
    effective = deepcopy(source)
    metadata = case_metadata(effective)
    output_tokens = timed_max_tokens(decode_steps, metadata.mtp)
    dataset_name = (
        f"DeepSeek-V4-8192-{output_tokens}-16384-ratio-1_for_serve.json"
    )

    effective["slurm"]["partition"] = partition
    effective["slurm"]["account"] = account
    effective["slurm"]["job_time"] = job_time
    effective["slurm"]["job_name"] = job_name
    effective["benchmark"]["input_length"] = 8192
    effective["benchmark"]["output_length"] = output_tokens
    effective["benchmark"]["multi_round"] = 1
    effective["benchmark"]["dataset_file"] = str(
        Path(dataset_root) / dataset_name
    )
    environment = effective["environment"]
    environment["container_image"] = container_image
    environment["container_mount"] = container_mount
    environment["model_path"] = model_path
    environment["log_dir"] = log_dir
    environment["trtllm_repo"] = ""
    environment["trtllm_wheel_path"] = ""
    environment["build_wheel"] = False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(effective, sort_keys=False))
    return effective, metadata


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile without samples")
    ordered = sorted(values)
    index = round((percentile / 100.0) * (len(ordered) - 1))
    return ordered[max(0, min(len(ordered) - 1, index))]


def parse_gen_iterlog(
    path: Path,
    *,
    concurrency: int,
    gen_tp: int,
    mtp: int,
    attention_dp: bool,
) -> IterationSummary:
    """Parse exact-full-batch GEN device iterations from one timed iterlog."""
    if concurrency <= 0 or gen_tp <= 0:
        raise ValueError("concurrency and gen_tp must be positive")
    if attention_dp and concurrency % gen_tp:
        raise ValueError("DEP concurrency must be divisible by GEN TP")

    scheduled = concurrency // gen_tp if attention_dp else concurrency
    generation_tokens = scheduled * (mtp + 1)
    rows: dict[tuple[int, int], float] = {}
    for match in _ITERLOG_RE.finditer(path.read_text(errors="ignore")):
        if int(match["ctx_tokens"]) != 0:
            continue
        if int(match["scheduled"]) != scheduled:
            continue
        if int(match["generation_tokens"]) != generation_tokens:
            continue
        rows[(int(match["iter"]), int(match["global_rank"]))] = float(
            match["device_ms"]
        )

    exact = list(rows.values())
    if not exact:
        raise NoExactFullBatchError(
            f"No exact-full-batch GEN iterations found in {path}"
        )
    median = statistics.median(exact)
    retained = [value for value in exact if median * 0.8 <= value <= median * 1.2]
    if not retained:
        raise ValueError("Median filter removed every exact-full-batch iteration")
    return IterationSummary(
        raw_exact_batch_samples=len(exact),
        retained_samples=len(retained),
        mean_device_step_ms=statistics.fmean(retained),
        median_device_step_ms=statistics.median(retained),
        p90_device_step_ms=_percentile(retained, 90),
        p99_device_step_ms=_percentile(retained, 99),
    )


def build_result(
    *,
    model_id: str,
    source_config: str,
    concurrency: int,
    ctx_gpus: int,
    gen_gpus: int,
    gen_tp: int,
    mtp: int,
    mean_device_step_ms: float,
    median_device_step_ms: float,
    p90_device_step_ms: float,
    p99_device_step_ms: float,
    raw_samples: int,
    retained_samples: int,
    decode_steps: int,
) -> dict[str, Any]:
    """Build the raw JSON consumed by ``utils/process_result.py``."""
    if mean_device_step_ms <= 0:
        raise ValueError("mean_device_step_ms must be positive")
    if gen_gpus <= 0 or ctx_gpus <= 0:
        raise ValueError("CTX and GEN GPU counts must be positive")
    acceptance = DEFAULT_ACCEPTANCE_LENGTHS[mtp]
    step_throughput = concurrency / (mean_device_step_ms / 1000.0)
    token_throughput = step_throughput * acceptance
    return {
        "model_id": model_id,
        "engine_mode": "offline",
        "measurement_boundary": "gen_iteration",
        "tpot_unit": "decode_step",
        "mtp": mtp,
        "max_concurrency": concurrency,
        "decode_steps_target": decode_steps,
        "timed_max_tokens": timed_max_tokens(decode_steps, mtp),
        "assumed_tokens_per_step": acceptance,
        "moe_routing": "perfect",
        "source_config": source_config,
        "num_ctx_gpu": ctx_gpus,
        "num_gen_gpu": gen_gpus,
        "gen_tp": gen_tp,
        "raw_exact_batch_iteration_count": raw_samples,
        "retained_iteration_count": retained_samples,
        "mean_tpot_ms": mean_device_step_ms,
        "median_tpot_ms": median_device_step_ms,
        "p90_tpot_ms": p90_device_step_ms,
        "p99_tpot_ms": p99_device_step_ms,
        "total_token_throughput": step_throughput,
        "output_throughput": step_throughput,
        "decode_step_throughput_per_gen_gpu": step_throughput / gen_gpus,
        "token_equivalent_output_throughput": token_throughput,
        "token_equivalent_output_throughput_per_gen_gpu": (
            token_throughput / gen_gpus
        ),
    }


def build_result_from_logs(
    config: dict[str, Any],
    *,
    log_dir: Path,
    output_path: Path,
    source_config: str,
    model_id: str,
    decode_steps: int,
) -> dict[str, Any]:
    """Parse a completed case and write its workflow-compatible raw JSON."""
    metadata = case_metadata(config)
    candidates = sorted(
        log_dir.glob(f"**/concurrency_{metadata.concurrency}/gen_only*.txt")
    )
    if len(candidates) > 1:
        raise ValueError(
            "Expected at most one timed GEN iterlog for concurrency "
            f"{metadata.concurrency}, found {len(candidates)} under {log_dir}"
        )
    if candidates:
        iteration_log = candidates[0]
        iteration_log_source = "timed_slice"
        try:
            summary = parse_gen_iterlog(
                iteration_log,
                concurrency=metadata.concurrency,
                gen_tp=metadata.gen_tp,
                mtp=metadata.mtp,
                attention_dp=metadata.gen_dp_attn,
            )
        except NoExactFullBatchError as error:
            timed_error = error
        else:
            timed_error = None
    else:
        timed_error = NoExactFullBatchError(
            "No timed GEN iterlog found for concurrency "
            f"{metadata.concurrency} under {log_dir}"
        )

    if timed_error is not None:
        full_log_candidates = sorted(log_dir.glob("**/3_output_GEN_*.log"))
        if len(full_log_candidates) != 1:
            raise ValueError(
                f"{timed_error}; expected exactly one full GEN worker log "
                f"under {log_dir}, found {len(full_log_candidates)}"
            ) from timed_error
        iteration_log = full_log_candidates[0]
        iteration_log_source = "full_gen_worker_fallback"
        try:
            summary = parse_gen_iterlog(
                iteration_log,
                concurrency=metadata.concurrency,
                gen_tp=metadata.gen_tp,
                mtp=metadata.mtp,
                attention_dp=metadata.gen_dp_attn,
            )
        except NoExactFullBatchError as full_log_error:
            raise NoExactFullBatchError(
                f"{timed_error}; fallback also failed: {full_log_error}"
            ) from full_log_error
    ctx = config["worker_config"]["ctx"]
    gen = config["worker_config"]["gen"]
    ctx_gpus = metadata.ctx_num_workers * _world_size(ctx)
    gen_gpus = metadata.gen_num_workers * _world_size(gen)
    result = build_result(
        model_id=model_id,
        source_config=source_config,
        concurrency=metadata.concurrency,
        ctx_gpus=ctx_gpus,
        gen_gpus=gen_gpus,
        gen_tp=metadata.gen_tp,
        mtp=metadata.mtp,
        mean_device_step_ms=summary.mean_device_step_ms,
        median_device_step_ms=summary.median_device_step_ms,
        p90_device_step_ms=summary.p90_device_step_ms,
        p99_device_step_ms=summary.p99_device_step_ms,
        raw_samples=summary.raw_exact_batch_samples,
        retained_samples=summary.retained_samples,
        decode_steps=decode_steps,
    )
    result["iteration_log_source"] = iteration_log_source
    result["iteration_log_file"] = iteration_log.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _bool_arg(value: str) -> bool:
    normalized = value.lower()
    if normalized not in {"true", "false"}:
        raise argparse.ArgumentTypeError("expected true or false")
    return normalized == "true"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source-config", type=Path, required=True)
    prepare.add_argument("--output-config", type=Path, required=True)
    prepare.add_argument("--metadata-output", type=Path, required=True)
    prepare.add_argument("--partition", required=True)
    prepare.add_argument("--account", required=True)
    prepare.add_argument("--job-time", default="03:00:00")
    prepare.add_argument("--job-name", required=True)
    prepare.add_argument("--container-image", required=True)
    prepare.add_argument("--container-mount", required=True)
    prepare.add_argument("--model-path", required=True)
    prepare.add_argument("--dataset-root", required=True)
    prepare.add_argument("--log-dir", required=True)
    prepare.add_argument("--decode-steps", type=int, required=True)
    prepare.add_argument("--concurrency", type=int, required=True)
    prepare.add_argument("--prefill-num-workers", type=int, required=True)
    prepare.add_argument("--prefill-tp", type=int, required=True)
    prepare.add_argument("--prefill-ep", type=int, required=True)
    prepare.add_argument("--prefill-dp-attn", type=_bool_arg, required=True)
    prepare.add_argument("--decode-num-workers", type=int, required=True)
    prepare.add_argument("--decode-tp", type=int, required=True)
    prepare.add_argument("--decode-ep", type=int, required=True)
    prepare.add_argument("--decode-dp-attn", type=_bool_arg, required=True)

    result = subparsers.add_parser("result")
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--log-dir", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--source-config", required=True)
    result.add_argument("--model-id", required=True)
    result.add_argument("--decode-steps", type=int, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        source = _load_yaml(args.source_config)
        effective, metadata = render_effective_config(
            source,
            output_path=args.output_config,
            partition=args.partition,
            account=args.account,
            job_time=args.job_time,
            job_name=args.job_name,
            container_image=args.container_image,
            container_mount=args.container_mount,
            model_path=args.model_path,
            dataset_root=args.dataset_root,
            log_dir=args.log_dir,
            decode_steps=args.decode_steps,
        )
        validate_case_against_workflow(
            effective,
            concurrency=args.concurrency,
            prefill_num_workers=args.prefill_num_workers,
            prefill_tp=args.prefill_tp,
            prefill_ep=args.prefill_ep,
            prefill_dp_attn=args.prefill_dp_attn,
            decode_num_workers=args.decode_num_workers,
            decode_tp=args.decode_tp,
            decode_ep=args.decode_ep,
            decode_dp_attn=args.decode_dp_attn,
        )
        payload = {
            **asdict(metadata),
            "dataset_file": effective["benchmark"]["dataset_file"],
            "timed_max_tokens": effective["benchmark"]["output_length"],
        }
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(payload, sort_keys=True))
        return

    result = build_result_from_logs(
        _load_yaml(args.config),
        log_dir=args.log_dir,
        output_path=args.output,
        source_config=args.source_config,
        model_id=args.model_id,
        decode_steps=args.decode_steps,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
