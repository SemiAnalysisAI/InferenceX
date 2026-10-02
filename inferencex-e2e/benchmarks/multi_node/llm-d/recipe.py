"""Render llm-d role arguments and enforce AgentX benchmark metadata."""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import sys

import yaml

from infx.golden_al_distribution import GOLDEN_DIR, golden_length


def validate_agentic_offload(recipe: dict, env: dict) -> None:
    """An embedded Mooncake store is DRAM offload even with SSD disabled."""
    if env.get("IS_AGENTIC") != "1":
        return
    store = recipe.get("mooncake", {}).get("store_config")
    expected = "dram" if store else "none"
    if env.get("KV_OFFLOADING") != expected:
        raise ValueError(f"Recipe requires KV_OFFLOADING={expected}; fix the master YAML")
    if store and env.get("KV_OFFLOAD_BACKEND") != "mooncake":
        raise ValueError("Mooncake recipe requires KV_OFFLOAD_BACKEND=mooncake")


def role_assignments(recipe: dict, role: str, env: dict, *, golden_dir: Path = GOLDEN_DIR) -> str:
    validate_agentic_offload(recipe, env)
    section = recipe.get(role) or {}
    extra = (section.get("extra-args") or "").strip()
    match = re.search(r"--speculative-config\s+", extra)
    if match:
        config, length = json.JSONDecoder().raw_decode(extra[match.end():])
        if not isinstance(config, dict):
            raise ValueError("speculative-config must be a JSON object")
        agentic = env.get("IS_AGENTIC", "").lower() in ("1", "true")
        if agentic and not env.get("SPEC_DECODING"):
            raise ValueError("Missing SPEC_DECODING for golden AL selection")
        synthetic = (
            agentic
            and env.get("EVAL_ONLY", "").lower() != "true"
            and env.get("SPEC_DECODING") != "none"
        )
        if synthetic:
            if env.get("RUN_EVAL", "").lower() == "true":
                raise ValueError("Run accuracy evals separately with EVAL_ONLY=true, not synthetic AL")
            for key in ("MODEL_PREFIX", "THINKING_MODE"):
                if not env.get(key):
                    raise ValueError(f"Missing {key} for golden AL lookup")
            # Each role's draft depth bounds its acceptance. In particular, a
            # K=1 prefill cannot use the K=3 decode role's golden AL (>2).
            al = golden_length(env["MODEL_PREFIX"], config, env["THINKING_MODE"], golden_dir)
            config.update(
                rejection_sample_method="synthetic",
                synthetic_acceptance_length=al,
            )
            # Adaptive verification can cap the realized AL below the measured
            # target. Golden DSpark curves use a fixed verification depth.
            if config.get("method") == "dspark" or "enable_adaptive_verification" in config:
                config["enable_adaptive_verification"] = False
            print(
                f"{config['method']} {role}: K={config['num_speculative_tokens']}, "
                f"golden AL={al} (thinking={env['THINKING_MODE']})",
                file=sys.stderr,
            )
        else:
            config.pop("synthetic_acceptance_length", None)
            if config.get("rejection_sample_method") == "synthetic" or (
                config.get("method") == "dspark" and "rejection_sample_method" not in config
            ):
                config["rejection_sample_method"] = "block"
        extra = extra[:match.end()] + json.dumps(config, separators=(",", ":")) + extra[match.end() + length:]
    assignments = [f"ROLE_EXTRA_ARGS={shlex.quote(extra)}",
                   f"PREFILL_ENABLE_EP={str(recipe.get('prefill', {}).get('enable-expert-parallel', True)).lower()}"]
    if section.get("tp") is not None:
        assignments.append(f"TP_SIZE={int(section['tp'])}")
    if section.get("enable-expert-parallel") is not None:
        assignments.append(f"ROLE_ENABLE_EP={str(section['enable-expert-parallel']).lower()}")
    for key, value in (section.get("env") or {}).items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid recipe environment variable: {key}")
        assignments.append(f"export {key}={shlex.quote(str(value))}")
    return "\n".join(assignments)


def mooncake_config(recipe: dict, env: dict) -> str:
    validate_agentic_offload(recipe, env)
    config = dict(recipe.get("mooncake", {}).get("store_config") or {})
    if not config:
        return ""
    config["master_server_address"] = f"{env['ALL_IPS'].split(',')[0]}:50051"
    if env.get("IS_AGENTIC") == "1":
        budget_gb = int(env["TOTAL_CPU_DRAM_GB"])
        gpus_per_node = int(env["GPUS_PER_NODE"])
        if budget_gb <= 0 or gpus_per_node <= 0:
            raise ValueError("Mooncake requires a positive per-node DRAM budget and GPU count")
        # The master budget is per node. Each embedded per-GPU store owns a
        # share; transfer buffers are separate from the reusable KV pool.
        config["global_segment_size"] = budget_gb * 10**9 // gpus_per_node
    return json.dumps(config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--role", choices=("prefill", "decode"))
    output.add_argument("--mooncake", action="store_true")
    args = parser.parse_args()
    recipe = yaml.safe_load(args.recipe.read_text())
    print(mooncake_config(recipe, os.environ) if args.mooncake else role_assignments(recipe, args.role, os.environ))


if __name__ == "__main__":
    main()
