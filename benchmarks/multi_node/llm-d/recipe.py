"""Render llm-d role arguments and enforce AgentX benchmark metadata."""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


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


def role_assignments(recipe: dict, role: str, env: dict) -> str:
    validate_agentic_offload(recipe, env)
    section = recipe.get(role) or {}
    extra = (section.get("extra-args") or "").strip()
    if env.get("IS_AGENTIC") == "1":
        match = re.search(r"--speculative-config\s+", extra)
        if match:
            config, length = json.JSONDecoder().raw_decode(extra[match.end():])
            if config.get("method") == "dspark":
                if env.get("SPEC_DECODING") != "mtp":
                    raise ValueError("DSpark requires SPEC_DECODING=mtp in the master YAML")
                if env.get("EVAL_ONLY") == "true":
                    config.pop("synthetic_acceptance_length", None)
                    config.pop("rejection_sample_method", None)
                else:
                    if env.get("RUN_EVAL") == "true":
                        raise ValueError("Run accuracy evals separately with EVAL_ONLY=true, not synthetic AL")
                    if env.get("MODEL_NAME") != "deepseek-ai/DeepSeek-V4-Pro-0813":
                        raise ValueError("No registered DSpark golden AL for this model")
                    golden_path = REPO_ROOT / "golden_al_distribution/dsv4-pro-0813-dspark.yaml"
                    golden = yaml.safe_load(golden_path.read_text())
                    k = config["num_speculative_tokens"]
                    al = golden["deepseek-v4-pro-0813"]["thinking_on"][k]
                    config.update(
                        rejection_sample_method="synthetic",
                        synthetic_acceptance_length=al,
                    )
                    print(f"DSpark {role}: K={k}, golden AL={al} ({golden_path.name})", file=sys.stderr)
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
