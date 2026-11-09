"""Attest, from the serving engine's own startup arguments and log, that every request did the declared work.

Cross-engine speed numbers compare only when each engine generated the declared clip the
declared way: same resolution, frame count, seed and denoising steps, without quantization,
LoRA or reused conditioning, and with step caching or sparse attention only when the run
declares that optimization. The client cannot observe any of that; the engine's log can. The
document keeps the log's SHA256, selected server arguments, per-request parameters and check
results, never log text, paths, hosts or addresses.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .media import _check

SCHEMA_VERSION = "1.0.0"
BUNDLE_TYPE = "h3_work_attestation"


@dataclass(frozen=True)
class Optimization:
    """A lossy optimization a run can declare: the log line that shows it ran, and the
    attention backends it adds to the dense ones."""

    evidence: re.Pattern[str]
    attention: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Policy:
    """What one runtime's log must show for its measurement to count as the declared work."""

    required_args: Mapping[str, Any]
    recorded_args: tuple[str, ...]
    dense_attention: frozenset[str]
    dit_dtypes: Mapping[str, str]
    optimizations: Mapping[str, Optimization]


SGLANG = Policy(
    # Step caching, quantization, per-component precision overrides and LoRA change the work;
    # disabled conditioning reuse is a declared fairness rule.
    required_args={
        "cache_dit_config": None,
        "quantization": None,
        "component_quantizations": {},
        "component_precisions": {},
        "lora_path": None,
        "disable_conditioning_cache": True,
    },
    recorded_args=(
        "performance_mode",
        "allow_bf16_reduced_precision_reduction",
        "allow_cudnn_tf32",
        "enable_torch_compile",
        "attention_backend",
        "tp_size",
        "ulysses_degree",
        "ring_degree",
        "cfg_parallel_degree",
    ),
    # AttentionBackendEnum at sgl-project/sglang@297e54fd105e974cdf401e5b4d20253a22c16f90
    # (python/sglang/multimodal_gen/runtime/platforms/interface.py) less its sparse,
    # SageAttention, FP8 and no-attention members.
    dense_attention=frozenset(
        {"fa", "fa2", "torch_sdpa", "torch_cudnn_sdpa", "dynamic_cudnn_sdpa", "aiter"}
    ),
    dit_dtypes={"bf16": "torch.bfloat16"},
    # Keys are the App's optimization registry keys. SGLANG_CACHE_DIT_ENABLED leaves
    # server_args.cache_dit_config null, so the mount line, which names the single DiT
    # "transformer", is the only evidence. SubBlock's construction line only marks a layer
    # eligible; the active line is logged once sparse attention runs.
    optimizations={
        "cache-dit": Optimization(re.compile(r"Enabling cache-dit on \S+ with config: ")),
        "subblock-sparse-attn": Optimization(
            re.compile(
                r"SubBlock (?:sparse attention|heterogeneous BSA) active: .*"
                r"keep(?:ing)? \d+/\d+ key blocks"
            ),
            frozenset({"subblock_sparse_attn"}),
        ),
    },
)
POLICIES: dict[str, Policy] = {"sglang": SGLANG}

# SGLang's log grammar. A progress fragment cannot span a line break, so interleaved
# access-log lines split the denoise bar without corrupting any "<step>/<total> [" match.
_SERVER_ARGS = re.compile(r"^\[[^\]\n]*\] server_args: (.*)$", re.M)
_USING_ATTENTION = re.compile(r"Using (\S+) attention backend")
_COMPONENT_ATTENTION = re.compile(r"Attention backends for \S+: (.+)$", re.M)
_DIT_LOAD = re.compile(r"Loading \S+ from .*param_dtype: (\S+)")
_REDUCED_NUMERICS = re.compile(r"Numerics policy: .+ enabled\.")
_IPC_FALLBACK = "IPC all-to-all (multi-rank) unavailable"
_SAMPLING = re.compile(r"^\[[^\]\n]*\] Sampling params:\n((?:[ \t]+\w+: .*\n)*)", re.M)
_FIELD = re.compile(r"^[ \t]+(\w+): (.*)$", re.M)
_PROGRESS = re.compile(r"(\d+)/(\d+) \[")
# Logged sampling field -> plan.generation key; the seed comes from the client record.
_PLAN_FIELDS = {
    "width": "width",
    "height": "height",
    "num_frames": "frame_count",
    "fps": "fps",
    "infer_steps": "num_inference_steps",
    "flow_shift": "flow_shift",
}


def _json_object(text: str) -> dict | None:
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _number(text: str) -> Any:
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    return text


def _attention_backends(text: str) -> list[str]:
    names = _USING_ATTENTION.findall(text)
    # "fa (component constraint), torch_sdpa", or per role "self=fa; cross=torch_sdpa".
    for summary in _COMPONENT_ATTENTION.findall(text):
        for item in re.split(r"[;,]", re.sub(r"\([^)]*\)", "", summary)):
            names.append(item.split("=")[-1].strip())
    return sorted(set(names))


def _server(
    policy: Policy, text: str, precision: str, declared: list[str]
) -> tuple[dict, list[dict]]:
    lines = _SERVER_ARGS.findall(text)
    args = _json_object(lines[0]) if len(lines) == 1 else None
    present = args or {}
    required = policy.required_args
    features_off = args is not None and all(
        key in args and args[key] == value for key, value in required.items()
    )
    backends = _attention_backends(text)
    dtypes = sorted(set(_DIT_LOAD.findall(text)))
    dit_dtype = policy.dit_dtypes.get(precision)
    observed = {key: present[key] for key in policy.recorded_args if key in present}
    observed["reduced_precision_numerics"] = bool(_REDUCED_NUMERICS.search(text))
    ipc_fallbacks = text.count(_IPC_FALLBACK)
    shown = sorted(key for key, item in policy.optimizations.items() if item.evidence.search(text))
    attention = policy.dense_attention.union(
        *(policy.optimizations[key].attention for key in declared if key in policy.optimizations)
    )
    return observed, [
        _check(
            "server.args",
            args is not None,
            len(lines),
            1,
            "exactly one server_args line that parses as a JSON object",
        ),
        _check(
            "server.quality_features_off",
            features_off,
            {key: present[key] for key in required if key in present},
            dict(required),
            "no step cache, quantization, precision override or LoRA; conditioning cache disabled",
        ),
        _check(
            "server.optimizations",
            shown == sorted(declared),
            shown,
            sorted(declared),
            "the log shows exactly the run's declared optimizations running",
        ),
        _check(
            "server.attention_backends",
            bool(backends) and set(backends) <= attention,
            backends,
            sorted(attention),
            "every reported attention backend is dense and full precision, "
            "or belongs to a declared optimization",
        ),
        _check(
            "server.dit_dtype",
            bool(dtypes) and dit_dtype is not None and dtypes == [dit_dtype],
            dtypes,
            dit_dtype,
            f"DiT weights load in the declared {precision} precision",
        ),
        _check(
            "server.no_ipc_fallback",
            ipc_fallbacks == 0,
            ipc_fallbacks,
            0,
            "multi-rank IPC all-to-all did not fall back to NCCL",
        ),
    ]


def _requests(text: str, generation: Mapping[str, Any], records: list[dict]) -> list[dict]:
    blocks = list(_SAMPLING.finditer(text))
    by_job: dict[str, list[tuple[dict, str]]] = {}
    for index, block in enumerate(blocks):
        fields = dict(_FIELD.findall(block.group(1)))
        end = blocks[index + 1].start() if index + 1 < len(blocks) else len(text)
        job = PurePosixPath(fields.get("output_file_path", "")).stem
        by_job.setdefault(job, []).append((fields, text[block.end() : end]))
    steps = generation.get("num_inference_steps")
    full_schedule = {"completed": steps, "totals": [steps]}
    attested = []
    for record in records:
        if not (record.get("attempted") and record.get("job_id")):
            continue
        matches = by_job.get(record["job_id"], [])
        found = len(matches) == 1
        fields, region = matches[0] if found else ({}, "")
        expected = {name: generation.get(key) for name, key in _PLAN_FIELDS.items()}
        expected["seed"] = record.get("seed")
        parameters = {name: _number(fields[name]) for name in expected if name in fields}
        progress = [(int(step), int(total)) for step, total in _PROGRESS.findall(region)]
        denoise = {
            "completed": max((step for step, _ in progress), default=None),
            "totals": sorted({total for _, total in progress}),
        }
        attested.append(
            {
                "slot_id": record.get("slot_id"),
                "job_id": record["job_id"],
                "checks": [
                    _check(
                        "request.server_record",
                        found,
                        len(matches),
                        1,
                        "exactly one engine Sampling params block names this job's output",
                    ),
                    _check(
                        "request.parameters",
                        found and parameters == expected,
                        parameters or None,
                        expected,
                        "the engine ran the declared shape, seed and schedule",
                    ),
                    _check(
                        "request.denoise_steps",
                        found and denoise == full_schedule,
                        denoise,
                        full_schedule,
                        "denoise progress counts to the declared step total",
                    ),
                ],
            }
        )
    return attested


def attest(
    runtime: str,
    server_log: Path,
    plan: dict,
    records: list[dict],
    precision: str,
    optimizations: list[str],
) -> dict:
    """Check the engine's log against the plan, the run's declared optimizations and every
    attempted request; a missing log fails."""
    try:
        data = server_log.read_bytes()
    except OSError:
        data = None
    policy = POLICIES.get(runtime)
    supported = _check(
        "runtime.supported",
        policy is not None,
        runtime,
        sorted(POLICIES),
        "a work-attestation policy exists for this runtime",
    )
    observed, server_checks, requests = {}, [supported], []
    if policy is not None:
        text = (data or b"").decode("utf-8", errors="replace")
        observed, policy_checks = _server(policy, text, precision, optimizations)
        server_checks += policy_checks
        requests = _requests(text, plan["generation"], records)
    every = server_checks + [check for request in requests for check in request["checks"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "bundle_type": BUNDLE_TYPE,
        "runtime": runtime,
        "server_log": {"sha256": None if data is None else hashlib.sha256(data).hexdigest()},
        "server": {"observed": observed, "checks": server_checks},
        "requests": requests,
        "valid": all(check["status"] == "passed" for check in every),
    }
