"""Attest, from the serving engine's own startup arguments and log, that every request did the declared work.

Cross-engine speed numbers compare only when each engine generated the declared clip the
declared way: same resolution, frame count, seed and denoising steps, without step caching,
quantization, LoRA, sparse attention or reused conditioning. The client cannot observe any of
that; the engine's log can. The document keeps the log's SHA256, selected server arguments,
per-request parameters and check results, never log text, paths, hosts or addresses.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from .media import _check

SCHEMA_VERSION = "1.0.0"
BUNDLE_TYPE = "h3_work_attestation"

# Engine log records for one client job id: each record's logged fields and the log text
# that follows it.
JobRecords = dict[str, list[tuple[dict[str, str], str]]]


@dataclass(frozen=True)
class ServerLog:
    """The startup evidence one runtime's log grammar extracts."""

    args: list[dict | None]
    attention: list[str]
    dit_dtypes: list[str]
    observed: dict[str, Any] = field(default_factory=dict)
    checks: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class Policy:
    """What one runtime's log must show for its measurement to count as the declared work."""

    required_args: Mapping[str, Any]
    required_detail: str
    recorded_args: tuple[str, ...]
    dense_attention: frozenset[str]
    dit_dtypes: Mapping[str, str]
    # Logged request field -> plan.generation key; the seed comes from the client record.
    plan_fields: Mapping[str, str]
    read_server: Callable[[str], ServerLog]
    read_jobs: Callable[[str], JobRecords]
    # vLLM echoes only non-default arguments, so an absent key holds its default.
    echo_omits_defaults: bool = False


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

# vLLM-Omni's log grammar, as vllm-project/vllm-omni@5917730a4b02aedb44d374173cdc1b10484ccd52
# (v0.32.0) and vLLM's api_utils.py print it.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_VLLM_ARGS = re.compile(r"\] non-default args: (\{.*\})$", re.M)
_VLLM_ATTENTION = re.compile(
    r"Resolved diffusion attention backend '(\w+)' for role=.* via (.+)$", re.M
)
_VLLM_LORA = re.compile(
    r"Initializing DiffusionLoRAManager: .*dtype=([^,]+), .*static_lora_path=(.*)$", re.M
)
_VLLM_PERSISTED = re.compile(r"Video request (\S+) persisted ")
# Lines vLLM-Omni logs only when a step cache, the prompt-embedding cache or quantization is on.
_VLLM_FEATURE_MARKERS = (
    "Enabling cache-dit",
    "Cache-dit enabled",
    "owns request-scoped Cache-DiT transitions",
    "TeaCache enabled",
    "TeaCache applied",
    "MagCache:",
    "SeaCache enabled",
    "stepcache enabled",
    "Prompt-embed cache installed",
    "Building quantization config",
)


def _json_object(text: str) -> dict | None:
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _literal_dict(text: str) -> dict | None:
    try:
        value = ast.literal_eval(text)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _number(text: str) -> Any:
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    return text


def _sglang_attention(text: str) -> list[str]:
    names = _USING_ATTENTION.findall(text)
    # "fa (component constraint), torch_sdpa", or per role "self=fa; cross=torch_sdpa".
    for summary in _COMPONENT_ATTENTION.findall(text):
        for item in re.split(r"[;,]", re.sub(r"\([^)]*\)", "", summary)):
            names.append(item.split("=")[-1].strip())
    return sorted(set(names))


def _sglang_server(text: str) -> ServerLog:
    ipc_fallbacks = text.count(_IPC_FALLBACK)
    return ServerLog(
        args=[_json_object(line) for line in _SERVER_ARGS.findall(text)],
        attention=_sglang_attention(text),
        dit_dtypes=sorted(set(_DIT_LOAD.findall(text))),
        observed={"reduced_precision_numerics": bool(_REDUCED_NUMERICS.search(text))},
        checks=[
            _check(
                "server.no_ipc_fallback",
                ipc_fallbacks == 0,
                ipc_fallbacks,
                0,
                "multi-rank IPC all-to-all did not fall back to NCCL",
            )
        ],
    )


def _sglang_jobs(text: str) -> JobRecords:
    blocks = list(_SAMPLING.finditer(text))
    jobs: JobRecords = {}
    for index, block in enumerate(blocks):
        fields = dict(_FIELD.findall(block.group(1)))
        end = blocks[index + 1].start() if index + 1 < len(blocks) else len(text)
        job = PurePosixPath(fields.get("output_file_path", "")).stem
        jobs.setdefault(job, []).append((fields, text[block.end() : end]))
    return jobs


def _vllm_omni_server(text: str) -> ServerLog:
    text = _ANSI.sub("", text)
    # Only the platform default carries no attention spec, and so no SAGE quantization or
    # skip-softmax sparsity, which the log line never shows.
    attention = {
        name if source == "platform default" else f"{name} via {source}"
        for name, source in _VLLM_ATTENTION.findall(text)
    }
    managers = _VLLM_LORA.findall(text)
    lora_paths = sorted({path for _, path in managers})
    markers = [marker for marker in _VLLM_FEATURE_MARKERS if marker in text]
    return ServerLog(
        args=[_literal_dict(line) for line in _VLLM_ARGS.findall(text)],
        attention=sorted(attention),
        dit_dtypes=sorted({dtype for dtype, _ in managers}),
        checks=[
            _check(
                "server.no_lora",
                bool(managers) and lora_paths == ["None"],
                lora_paths,
                ["None"],
                "every diffusion worker started its LoRA manager without a startup adapter",
            ),
            _check(
                "server.no_cache_or_quantization",
                not markers,
                markers,
                [],
                "no step cache, prompt-embedding cache or quantization reported enabling",
            ),
        ],
    )


def _vllm_omni_jobs(text: str) -> JobRecords:
    # The only per-request lines that name the job carry no parameters, and the request,
    # frame and denoise-progress lines name no job, so no record ties them to a job.
    jobs: JobRecords = {}
    for job in _VLLM_PERSISTED.findall(_ANSI.sub("", text)):
        jobs.setdefault(job, []).append(({}, ""))
    return jobs


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
    required_detail=(
        "no step cache, quantization, precision override or LoRA; conditioning cache disabled"
    ),
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
    plan_fields={
        "width": "width",
        "height": "height",
        "num_frames": "frame_count",
        "fps": "fps",
        "infer_steps": "num_inference_steps",
        "flow_shift": "flow_shift",
    },
    read_server=_sglang_server,
    read_jobs=_sglang_jobs,
)
VLLM_OMNI = Policy(
    # The vLLM echo omits every vLLM-Omni diffusion argument; these vLLM arguments switch
    # H3 to FP8 weights or FP8 attention.
    required_args={"quantization": None, "kv_cache_dtype": "auto"},
    required_detail="no FP8 weight or attention quantization among the echoed vLLM arguments",
    recorded_args=("enforce_eager", "tensor_parallel_size"),
    # DiffusionAttentionBackendEnum (vllm_omni/diffusion/attention/backends/registry.py) less
    # SAGE_ATTN and SAGE_ATTN_3 (quantized), FASTVIDEO_VSA and RAINFUSION_ATTN (sparse) and
    # VDNH3_ATTN (windowed hybrid).
    dense_attention=frozenset(
        {
            "FLASH_ATTN",
            "TORCH_SDPA",
            "CUDNN_ATTN",
            "FLASHINFER_ATTN",
            "FLASH_ATTN_HUB",
            "FLASH_ATTN_3_HUB",
            "TRTLLM_ATTN",
        }
    ),
    dit_dtypes={"bf16": "torch.bfloat16"},
    plan_fields={
        "width": "width",
        "height": "height",
        "num_frames": "frame_count",
        "fps": "fps",
        "num_inference_steps": "num_inference_steps",
        "flow_shift": "flow_shift",
    },
    read_server=_vllm_omni_server,
    read_jobs=_vllm_omni_jobs,
    echo_omits_defaults=True,
)
POLICIES: dict[str, Policy] = {"sglang": SGLANG, "vllm-omni": VLLM_OMNI}


def _server(policy: Policy, text: str, precision: str) -> tuple[dict, list[dict]]:
    log = policy.read_server(text)
    args = log.args[0] if len(log.args) == 1 else None
    present = args or {}
    required = policy.required_args
    features_off = args is not None and all(
        present.get(key, value) == value
        if policy.echo_omits_defaults
        else key in present and present[key] == value
        for key, value in required.items()
    )
    dit_dtype = policy.dit_dtypes.get(precision)
    observed = {key: present[key] for key in policy.recorded_args if key in present}
    observed.update(log.observed)
    return observed, [
        _check(
            "server.args",
            args is not None,
            len(log.args),
            1,
            "exactly one startup-arguments line that parses as an object",
        ),
        _check(
            "server.quality_features_off",
            features_off,
            {key: present[key] for key in required if key in present},
            dict(required),
            policy.required_detail,
        ),
        _check(
            "server.attention_backends",
            bool(log.attention) and set(log.attention) <= policy.dense_attention,
            log.attention,
            sorted(policy.dense_attention),
            "every reported attention backend is dense and full precision",
        ),
        _check(
            "server.dit_dtype",
            bool(log.dit_dtypes) and dit_dtype is not None and log.dit_dtypes == [dit_dtype],
            log.dit_dtypes,
            dit_dtype,
            f"DiT weights load in the declared {precision} precision",
        ),
        *log.checks,
    ]


def _requests(
    policy: Policy, text: str, generation: Mapping[str, Any], records: list[dict]
) -> list[dict]:
    by_job = policy.read_jobs(text)
    steps = generation.get("num_inference_steps")
    full_schedule = {"completed": steps, "totals": [steps]}
    attested = []
    for record in records:
        if not (record.get("attempted") and record.get("job_id")):
            continue
        matches = by_job.get(record["job_id"], [])
        found = len(matches) == 1
        fields, region = matches[0] if found else ({}, "")
        expected = {name: generation.get(key) for name, key in policy.plan_fields.items()}
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
                        "exactly one engine record names this job",
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


def attest(runtime: str, server_log: Path, plan: dict, records: list[dict], precision: str) -> dict:
    """Check the engine's log against the plan and every attempted request; a missing log fails."""
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
        observed, policy_checks = _server(policy, text, precision)
        server_checks += policy_checks
        requests = _requests(policy, text, plan["generation"], records)
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
