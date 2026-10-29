"""Describe a generation role's drafting and verification, and audit it against its curve.

Each engine spells drafting and verification differently, and some of it is only an engine
default. ``run_collection`` maps a role to the vocabulary of
``infx/golden_al_distribution/identities.yaml``, labelling every fact explicit, documented,
assumed, default or unknown. ``audit`` reports how a run differs from the curve ``curve_name``
selects; it never changes that curve.

python -m infx.srt_slurm.spec_identity snapshot MATRIX_JSON --root INFERENCEX_E2E
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from infx.golden_al_distribution import GOLDEN_DIR, curve_name, golden_length
from infx.golden_al_distribution.identity import (
    Collection,
    Fact,
    compare,
    load_identities,
    normalize_method,
)
from infx.srt_slurm.synthetic_acceptance import ENGINES, spec_parameters, verifier

# Arguments of a generation role that describe speculative decoding, per engine spelling.
SPEC_ARGUMENTS = (
    "speculative", "with-mtp", "num-mtp", "method", "num-speculative-tokens", "draft-model",
    "kv-cache-dtype", "kv_cache_dtype", "kv_cache_config",
)  # fmt: skip
# Spellings of one KV cache dtype; "auto" leaves the engine default.
DTYPE_ALIASES = {"fp8_e4m3": "fp8", "bfloat16": "bf16", "auto": None}
DEFAULT = Fact(None, "default")
UNKNOWN = Fact(None, "unknown")
CHAIN = Fact("chain", "documented")
# Adaptive verification is an opt-in engine flag.
NOT_ADAPTIVE = Fact(False, "documented")


def _set(value: Any) -> Fact:
    return DEFAULT if value is None else Fact(value, "explicit")


def _vllm(spec: Mapping[str, Any]) -> dict[str, Fact]:
    adaptive = spec.get("enable_adaptive_verification")
    return {
        "draft_sample": _set(spec.get("draft_sample_method")),
        "adaptive": (NOT_ADAPTIVE if adaptive is None else Fact(adaptive, "explicit")),
        "topology": CHAIN,
    }


def _sglang(args: Mapping[str, Any], method: str) -> dict[str, Fact]:
    topk = args.get("speculative-eagle-topk")
    if method in ("dspark", "dflash"):
        topology = CHAIN  # block drafters propose one sequence
    elif isinstance(topk, int) and not isinstance(topk, bool):
        topology = Fact("chain" if topk == 1 else f"tree-topk{topk}", "explicit")
    else:
        topology = DEFAULT  # SGLang chooses topk itself when it is unset
    return {
        "draft_sample": DEFAULT,
        "adaptive": NOT_ADAPTIVE,
        "topology": topology,
    }


def _trtllm() -> dict[str, Fact]:
    return {"draft_sample": DEFAULT, "adaptive": NOT_ADAPTIVE, "topology": CHAIN}


def _atom(method: str) -> dict[str, Fact]:
    return {
        # spec_parameters gives ATOM DSpark probabilistic drafting; ATOM has no sampler flag.
        "draft_sample": Fact("probabilistic", "assumed") if method == "dspark" else UNKNOWN,
        "adaptive": DEFAULT,
        "topology": CHAIN,
    }


def _dtype(value: Any) -> Fact:
    if isinstance(value, str):
        value = DTYPE_ALIASES.get(value.lower(), value.lower())
    return _set(value)


def _precision(args: Mapping[str, Any], spec: Mapping[str, Any], engine: str) -> dict[str, Fact]:
    """The draft weights' quantization and the draft's and target's KV cache dtypes."""
    config = args.get("kv_cache_config")
    target = args.get("kv-cache-dtype", args.get("kv_cache_dtype"))
    if target is None and isinstance(config, Mapping):
        target = config.get("dtype")
    if engine == "vllm":
        draft, draft_kv = spec.get("quantization"), spec.get("kv_cache_dtype")
    elif engine == "sglang":
        draft = args.get("speculative-draft-model-quantization")
        draft_kv = args.get("speculative-draft-kv-cache-dtype")
    else:
        draft = draft_kv = None
    unread = engine == "tilert"  # TileRT's draft settings are not readable here
    return {
        "draft_precision": UNKNOWN if unread else _set(draft),
        "draft_kv_cache_dtype": UNKNOWN if unread else _dtype(draft_kv),
        "kv_cache_dtype": _dtype(target),
    }


def _tilert() -> dict[str, Fact]:
    return {"draft_sample": UNKNOWN, "adaptive": DEFAULT, "topology": CHAIN}


def run_collection(model_prefix: str, role: Mapping[str, Any], framework: str) -> Collection | None:
    """How a generation role drafts and verifies, or None when it does not speculate."""
    engine = ENGINES.get(framework)
    if engine is None:
        raise ValueError(f"{framework} is not a speculative engine the connector reads")
    spec = spec_parameters(role, engine)
    if not spec:
        return None
    method = normalize_method(model_prefix, str(spec.get("method", "")))
    args = role.get("args") or {}
    facts = {
        "vllm": lambda: _vllm(spec),
        "sglang": lambda: _sglang(args, method),
        "trtllm": _trtllm,
        "atom": lambda: _atom(method),
        "tilert": _tilert,
    }[engine]()
    facts |= _precision(args, spec, engine)
    facts["verify"] = verifier(role, engine)
    return Collection(engine=engine, draft=str(spec.get("model") or "native"), **facts)


def audit(
    model_prefix: str,
    role: Mapping[str, Any],
    framework: str,
    thinking: str = "thinking_on",
    golden_dir: Path = GOLDEN_DIR,
) -> dict[str, Any] | None:
    """The curve, golden AL and findings of one generation role; None when it does not draft."""
    run = run_collection(model_prefix, role, framework)
    if run is None:
        return None
    spec = spec_parameters(role, ENGINES[framework])
    verify = run.verify.value
    curve = curve_name(model_prefix, spec, golden_dir, verify)
    return {
        "curve": curve,
        "draft_tokens": spec["num_speculative_tokens"],
        "golden": golden_length(model_prefix, spec, thinking, golden_dir, verify),
        "findings": [f.as_list() for f in compare(run, load_identities(golden_dir)[curve])],
    }


def spec_arguments(role: Mapping[str, Any]) -> dict[str, Any]:
    """The speculative arguments of a role, as the snapshot records them."""
    return {k: v for k, v in (role.get("args") or {}).items() if k.startswith(SPEC_ARGUMENTS)}


def snapshot(matrix: list[Mapping[str, Any]], root: Path) -> list[dict[str, Any]]:
    """Audit every distinct speculative generation role of a matrix's agentic rows."""
    from infx.srt_slurm.generate import bound_variant, point_environment
    from infx.srt_slurm.variants import expand_variants

    cases: dict[str, dict[str, Any]] = {}
    for row in matrix:
        if row.get("scenario-type") != "agentic-coding" or not row.get("srt-recipe"):
            continue
        if row.get("spec-decoding", "none") == "none":
            continue
        _, recipe = bound_variant(row, point_environment(row), root, expand=expand_variants)
        roles = recipe.get("roles") or {}
        args = spec_arguments(roles.get("decode", roles.get("agg", {})))
        case = {"model_prefix": row["model-prefix"], "framework": row["framework"], "args": args}
        key = json.dumps(case, sort_keys=True)
        if key not in cases:
            cases[key] = {**case, "recipes": set(), "rows": 0}
        cases[key]["recipes"].add(row["srt-recipe"])
        cases[key]["rows"] += 1
    result = []
    for case in cases.values():
        expected = audit(case["model_prefix"], {"args": case["args"]}, case["framework"])
        result.append({**case, "recipes": sorted(case["recipes"]), "expected": expected})
    return sorted(
        result,
        key=lambda c: (c["model_prefix"], c["framework"], json.dumps(c["args"], sort_keys=True)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    snap = commands.add_parser("snapshot", help="audit every speculative role of a sweep matrix")
    snap.add_argument("matrix", type=Path, help="JSON from infx.matrix.generate full-sweep")
    snap.add_argument("--root", type=Path, required=True, help="the inferencex-e2e directory")
    args = parser.parse_args(argv)
    matrix = json.loads(args.matrix.read_text())
    json.dump(snapshot(matrix, args.root.resolve()), sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
