"""Real GSM8K verification against an srt-owned endpoint, retaining raw samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
from collections import Counter
from importlib.resources import files
from pathlib import Path
from typing import Any

from infx.srt_slurm.contracts import load_mapping

from .common import (
    child_environment,
    child_failed,
    decode_json,
    read_json,
    require_finite,
    run_child,
    validate_endpoint,
    verify_file,
    verify_snapshot_assets,
    write_json,
)
from .identity import LM_EVAL_REVISION, verify_runtime
from .spec import EvalSpec

FILTERS = ("strict-match", "flexible-extract")


def packaged_task_path() -> Path:
    """Installed wheels include the task; this path never depends on the checkout cwd."""
    return Path(str(files("infx.evals").joinpath("gsm8k.yaml")))


def build_argv(spec: EvalSpec, endpoint: str, artifact_root: Path) -> list[str]:
    origin = validate_endpoint(endpoint)
    patch = str(files("infx.evals.patches").joinpath("lm_eval_sitecustomize.py"))
    # Apply the existing compatibility patch explicitly. Python -I must not rely on
    # the legacy PYTHONPATH/sitecustomize injection or on an importable infx in 3.11.
    bootstrap = (
        "import runpy,sys; "
        "runpy.run_path(sys.argv.pop(1)); "
        "runpy.run_module('lm_eval', run_name='__main__')"
    )
    model_args = {
        "model": spec.metadata.model,
        "base_url": f"{origin}/v1/chat/completions",
        "api_key": "EMPTY",
        "eos_string": "</s>",
        "max_retries": 5,
        "num_concurrent": spec.concurrency,
        "timeout": 1800,
        "tokenized_requests": False,
        "max_length": spec.max_length,
    }
    # Harness model_args is a comma-delimited format; reject values that could
    # change its parse instead of shell-quoting an unsafe semantic value.
    if any(
        any(character in str(value) for character in (",", "\n", "\r"))
        for value in model_args.values()
    ):
        raise ValueError("eval model/endpoint cannot contain delimiters in harness model_args")
    return [
        spec.runtime.python,
        "-I",
        "-c",
        bootstrap,
        patch,
        "--model",
        "local-chat-completions",
        "--apply_chat_template",
        "--tasks",
        spec.task.path,
        "--output_path",
        str(artifact_root / "harness"),
        "--log_samples",
        "--model_args",
        ",".join(f"{key}={value}" for key, value in model_args.items()),
        "--gen_kwargs",
        f"max_tokens={spec.max_tokens},temperature=0,top_p=1",
    ]


def eval_metadata(spec: EvalSpec, *, complete: bool) -> dict[str, Any]:
    metadata = spec.metadata
    return {
        "is_multinode": False,
        "disagg": False,
        "framework": metadata.framework,
        "precision": metadata.precision,
        "spec_decoding": metadata.spec_decoding,
        "eval_suite": spec.task_name,
        "recipe_fingerprint": metadata.recipe_fingerprint,
        "tp": metadata.tp,
        "pp": 1,
        "dcp_size": 1,
        "pcp_size": 1,
        "conc": spec.concurrency,
        "ep": 1,
        "dp_attention": False,
        "prefill_tp": metadata.tp,
        "prefill_pp": 1,
        "prefill_dcp_size": 1,
        "prefill_pcp_size": 1,
        "prefill_ep": 1,
        "prefill_dp_attention": False,
        "prefill_num_workers": 0,
        "decode_tp": metadata.tp,
        "decode_pp": 1,
        "decode_dcp_size": 1,
        "decode_pcp_size": 1,
        "decode_ep": 1,
        "decode_dp_attention": False,
        "decode_num_workers": 0,
        "num_gpus": metadata.tp,
        "model": metadata.model,
        "infmax_model_prefix": metadata.model_prefix,
        "hw": metadata.hw,
        "isl": "0",
        "osl": "0",
        "eval_concs": [spec.concurrency],
        "completed_eval_concs": [spec.concurrency] if complete else [],
        "failed_eval_concs": [] if complete else [spec.concurrency],
        "deployment": {
            "kind": "aggregate",
            "nodes": 1,
            "serving_gpus": metadata.tp,
            "tp": metadata.tp,
            "ep": 1,
        },
    }


def stage_outputs(spec: EvalSpec, artifact_root: Path) -> tuple[list[Path], list[Path]]:
    """Keep raw harness files and stage compatibility names without silent overwrite."""
    results: list[Path] = []
    samples: list[Path] = []
    for path in sorted((artifact_root / "harness").rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("results") and path.suffix == ".json":
            group = results
        elif path.name.startswith("sample") and path.suffix == ".jsonl":
            group = samples
        else:
            continue
        target = artifact_root / f"{path.stem}_conc{spec.concurrency}{path.suffix}"
        with target.open("xb") as output, path.open("rb") as source:
            # Raw evidence is byte-for-byte identical; no reserialization.
            import shutil

            shutil.copyfileobj(source, output)
        group.append(target)
    return results, samples


def _expected_task(spec: EvalSpec) -> dict[str, Any]:
    task = load_mapping(verify_file(spec.task))
    expected = {
        "task": "gsm8k",
        "dataset_path": "openai/gsm8k",
        "dataset_name": "main",
        "training_split": "train",
        "test_split": "test",
        "fewshot_split": "train",
        "num_fewshot": 5,
        "repeats": 1,
        "output_type": "generate_until",
    }
    if not isinstance(task, dict) or any(task.get(key) != value for key, value in expected.items()):
        raise ValueError("prepared task does not describe the full canonical GSM8K contract")
    if [item["name"] for item in task.get("filter_list", [])] != list(FILTERS):
        raise ValueError("prepared task must contain strict-match and flexible-extract filters")
    return task


def validate_outputs(
    spec: EvalSpec, endpoint: str, result_files: list[Path], sample_files: list[Path]
) -> list[str]:
    errors: list[str] = []
    if len(result_files) != 1 or len(sample_files) != 1:
        return ["expected exactly one GSM8K result file and one complete sample file"]
    result = read_json(result_files[0])
    task = _expected_task(spec)
    identities = read_json(verify_file(spec.document_identities))
    if not isinstance(identities, dict) or set(identities) != {
        str(index) for index in range(spec.expected_documents)
    }:
        raise ValueError("prepared document identities must cover the entire expected GSM8K split")
    config = result.get("config", {})
    model_args = config.get("model_args", {})
    for key, value in {
        "model": spec.metadata.model,
        "base_url": f"{validate_endpoint(endpoint)}/v1/chat/completions",
        "num_concurrent": spec.concurrency,
        "max_length": spec.max_length,
        "tokenized_requests": False,
    }.items():
        if model_args.get(key) != value:
            errors.append(f"eval model_args.{key} differs from the independently expected contract")
    if config.get("model") != "local-chat-completions" or config.get("limit") is not None:
        errors.append("eval must use the real full-split local-chat-completions adapter")
    if config.get("gen_kwargs") != {"max_tokens": spec.max_tokens, "temperature": 0, "top_p": 1}:
        errors.append("eval generation budget/sampling settings differ from the expected contract")
    if result.get("n-samples") != {
        spec.task_name: {"original": spec.expected_documents, "effective": spec.expected_documents}
    }:
        errors.append("eval n-samples does not contain the full expected task split")
    configs = result.get("configs", {})
    if set(configs) != {spec.task_name}:
        errors.append("eval result contains missing or unexpected tasks")
    emitted_task = configs.get(spec.task_name, {})
    for key, value in task.items():
        if key in {"tag", "metadata", "generation_kwargs"}:
            continue
        if emitted_task.get(key) != value:
            errors.append(f"eval task.{key} differs from the prepared task")
    expected_generation = {**task["generation_kwargs"], **config.get("gen_kwargs", {})}
    if emitted_task.get("generation_kwargs") != expected_generation:
        errors.append("eval task generation settings differ from the prepared task")
    scores = result.get("results", {}).get(spec.task_name, {})
    seen: set[tuple[int, str]] = set()
    sums: Counter[str] = Counter()
    with sample_files[0].open() as stream:
        for line_number, line in enumerate(stream, 1):
            sample = decode_json(line)
            require_finite(sample)
            doc_id, filter_name = sample.get("doc_id"), sample.get("filter")
            if (
                not isinstance(doc_id, int)
                or isinstance(doc_id, bool)
                or str(doc_id) not in identities
                or filter_name not in FILTERS
            ):
                errors.append(f"sample line {line_number} has unexpected document/filter identity")
                continue
            key = (doc_id, filter_name)
            if key in seen:
                errors.append(f"duplicate eval sample: {key}")
                continue
            seen.add(key)
            digest = hashlib.sha256(
                json.dumps(sample.get("doc"), indent=2, ensure_ascii=False).encode()
            ).hexdigest()
            if digest != identities[str(doc_id)] or sample.get("doc_hash") != digest:
                errors.append(f"sample document bytes/hash differ from the prepared split: {key}")
            target = sample.get("target")
            if (
                target != sample.get("doc", {}).get("answer")
                or sample.get("target_hash") != hashlib.sha256(str(target).encode()).hexdigest()
            ):
                errors.append(f"sample target/hash mismatch: {key}")
            value = sample.get("exact_match")
            if isinstance(value, bool) or value not in (0, 1):
                errors.append(f"sample exact_match must be zero or one: {key}")
            else:
                sums[filter_name] += value
    expected = {(index, name) for index in range(spec.expected_documents) for name in FILTERS}
    if seen != expected:
        errors.append(
            f"eval sample coverage incomplete: {len(seen)}/{len(expected)} document/filter pairs"
        )
    for name in FILTERS:
        metric = scores.get(f"exact_match,{name}")
        if (
            not isinstance(metric, int | float)
            or isinstance(metric, bool)
            or not math.isfinite(metric)
        ):
            errors.append(f"missing or non-finite exact_match,{name}")
        elif metric < spec.minimum_score or not math.isclose(
            metric, sums[name] / spec.expected_documents, rel_tol=0, abs_tol=1e-12
        ):
            errors.append(f"exact_match,{name} fails its threshold or disagrees with raw samples")
    return errors


def run(spec: EvalSpec, endpoint: str, artifact_root: Path) -> int:
    endpoint = validate_endpoint(endpoint)
    verify_runtime(spec.runtime, dataset_loader=None, source_pins={"lm-eval": LM_EVAL_REVISION})
    verify_snapshot_assets(
        spec.runtime, "openai/gsm8k", expected_revision=None, only_snapshot=False
    )
    _expected_task(spec)
    verify_file(spec.document_identities)
    artifact_root.mkdir(parents=True, exist_ok=True)
    if (artifact_root / "harness").exists():
        raise ValueError("client artifact root already contains eval output")
    argv = build_argv(spec, endpoint, artifact_root)
    (artifact_root / "eval_command.txt").write_text(shlex.join(argv) + "\n")
    status = run_child(
        argv,
        env=child_environment(spec.runtime),
        cwd=artifact_root,
        log=artifact_root / "eval.log",
        timeout_seconds=spec.runtime.timeout_seconds,
        terminate_grace_seconds=spec.runtime.terminate_grace_seconds,
    )
    errors: list[str] = []
    try:
        result_files, sample_files = stage_outputs(spec, artifact_root)
        errors = validate_outputs(spec, endpoint, result_files, sample_files)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append(f"eval artifact validation failed: {exc}")
    failed = child_failed(status) or bool(errors)
    write_json(artifact_root / "meta_env.json", eval_metadata(spec, complete=not failed))
    write_json(
        artifact_root / "diagnostics" / "client-audit.json",
        {
            "schema_version": 1,
            "client": "lm-eval",
            "status": status,
            "errors": errors,
            "prepared_identity_sha256": spec.runtime.identity.sha256,
            "task_sha256": spec.task.sha256,
            "document_identities_sha256": spec.document_identities.sha256,
            "endpoint": endpoint,
            "expected_documents": spec.expected_documents,
            "max_length": spec.max_length,
            "max_tokens": spec.max_tokens,
        },
    )
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--endpoint", default=os.environ.get("SRT_ENDPOINT"))
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    if not args.endpoint:
        parser.error("--endpoint or runtime-provided SRT_ENDPOINT is required")
    return run(EvalSpec.model_validate(read_json(args.spec)), args.endpoint, args.artifact_root)


if __name__ == "__main__":
    raise SystemExit(main())
