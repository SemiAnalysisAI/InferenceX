#!/usr/bin/env python3
"""Configure speculative acceptance in an srt-slurm recipe.

Eval-only runs remove synthetic acceptance so generated text is checked against
the target model. Throughput runs inject a configured synthetic acceptance
length only when ``SYNTHETIC_ACCEPTANCE=true``. Framework-specific rewriting
lives under ``runners/synthetic_injectors/``.

Environment variables:
  EVAL_ONLY                   "true" to restore real target verification
  SYNTHETIC_ACCEPTANCE         "true" to enable (default: "false")
  SYNTHETIC_ACCEPTANCE_LENGTH  target mean acceptance length; if unset, it is
                               auto-resolved from the reference AL YAML using
                               MODEL_PREFIX (+ NUM_SPEC_TOKENS / THINKING_MODE)
  NUM_SPEC_TOKENS              number of speculative tokens (for auto-lookup;
                               falls back to the value parsed from the recipe)
  MODEL_PREFIX                 model prefix key in the reference YAML (e.g. "dsv4")
  THINKING_MODE                "thinking_on" / "thinking_off" — only used when the
                               reference YAML is in the thinking matrix form
                               (default: "thinking_on")
  FRAMEWORK                    framework key selecting the backend (e.g.
                               "dynamo-vllm"); may also be passed as argv[2].

Usage (from a runner; use an absolute path since runners cd into the srt-slurm
clone before invoking this):
  python3 "$GITHUB_WORKSPACE/runners/inject_synthetic_acceptance.py" "${CONFIG_FILE%%:*}" "$FRAMEWORK"
"""

import math
import os
import sys

from synthetic_injectors import get_injector

# MODEL_PREFIX -> top-level key in speedbench-reference-al.yaml.
MODEL_PREFIX_TO_YAML_KEY = {
    "dsv4": "deepseek-v4-pro",
    "dsr1": "deepseek-r1",
    # DSpark ships as its own checkpoint (deepseek-ai/DeepSeek-V4-Pro-DSpark,
    # dated 0813), distinct from the plain MTP checkpoint above.
    "dsv4dspark": "deepseek-v4-pro-0813",
    "dsv4dsparkprob": "deepseek-v4-pro-0813",
}

# Committed golden curves are the authority for AgentX synthetic acceptance.
# Keep this mapping explicit: a model prefix must select one unambiguous curve
# (some models have multiple draft heads / sampling methods).
GOLDEN_AL_REFERENCES = {
    "qwen3.5": (
        "qwen3.5_mtp.yaml",
        "qwen3.5-397b-a17b-nvfp4",
    ),
}


def _log(msg):
    print(f"[Synthetic AR] {msg}")


def _enabled(name: str) -> bool:
    """Return whether an environment flag is explicitly enabled."""
    return os.environ.get(name, "false").strip().lower() == "true"


def _yaml_key(model_prefix):
    return MODEL_PREFIX_TO_YAML_KEY.get(model_prefix, model_prefix)


def _lookup_al(model_block, num_spec_tokens):
    """Resolve AL for num_spec_tokens from either reference-YAML shape.

    Flat list form:   [ {1: 1.90}, {2: 2.60}, ... ]
    Thinking matrix:  { thinking_on: {1: ...}, thinking_off: {1: ...} }
    """
    # Flat list form (each item is a single-key {level: al} mapping).
    if isinstance(model_block, list):
        for item in model_block:
            if num_spec_tokens in item:
                return item[num_spec_tokens]
        return None

    if isinstance(model_block, dict):
        # Thinking matrix form: pick the requested mode, then index by level.
        if any(str(k).startswith("thinking") for k in model_block):
            mode = (
                os.environ.get("THINKING_MODE", "thinking_on").strip() or "thinking_on"
            )
            mode_block = model_block.get(mode)
            if mode_block is None:
                sys.exit(
                    f"ERROR: THINKING_MODE='{mode}' not found in reference YAML "
                    f"(available: {sorted(model_block)})"
                )
            return mode_block.get(num_spec_tokens)
        # Plain {level: al} mapping.
        return model_block.get(num_spec_tokens)

    return None


def _load_reference(ref_yaml, yaml_key, num_spec_tokens):
    if not os.path.isfile(ref_yaml):
        sys.exit(f"ERROR: golden acceptance reference YAML not found: {ref_yaml}")

    import yaml

    with open(ref_yaml) as f:
        data = yaml.safe_load(f)

    model_block = data.get(yaml_key)
    if model_block is None:
        sys.exit(f'ERROR: model key "{yaml_key}" not found in {ref_yaml}')

    al = _lookup_al(model_block, num_spec_tokens)
    if al is None:
        sys.exit(
            f"ERROR: num_spec_tokens={num_spec_tokens} not found for "
            f"{yaml_key} in {ref_yaml}"
        )
    return float(al)


def _resolve_al(config_text, injector, legacy_ref_yaml):
    model_prefix = os.environ.get("MODEL_PREFIX", "").strip()
    explicit = os.environ.get("SYNTHETIC_ACCEPTANCE_LENGTH", "").strip()

    nst_env = os.environ.get("NUM_SPEC_TOKENS", "").strip()
    recipe_spec_tokens = injector.spec_tokens_from_recipe(config_text)

    # Curves listed here are fail closed.  In particular, an explicit AL must
    # not bypass the recipe's actual speculative step count and silently select
    # the next row of the golden table.
    reference = GOLDEN_AL_REFERENCES.get(model_prefix)
    if reference:
        if recipe_spec_tokens is None:
            sys.exit(
                "ERROR: cannot validate golden AL because the recipe does not "
                "declare a speculative-token count"
            )
        validator = getattr(injector, "validate_speculative_shape", None)
        if validator is not None:
            validator(config_text)
        if nst_env and int(nst_env) != recipe_spec_tokens:
            sys.exit(
                f"ERROR: NUM_SPEC_TOKENS={nst_env} disagrees with recipe "
                f"speculative-num-steps={recipe_spec_tokens}"
            )

        filename, key = reference
        ref_yaml = os.path.join(
            os.path.dirname(__file__), "..", "golden_al_distribution", filename
        )
        golden_al = _load_reference(ref_yaml, key, recipe_spec_tokens)
        if explicit and not math.isclose(
            float(explicit), golden_al, rel_tol=0.0, abs_tol=1e-9
        ):
            sys.exit(
                f"ERROR: explicit SYNTHETIC_ACCEPTANCE_LENGTH={explicit} does "
                f"not match golden AL={golden_al:g} for MODEL_PREFIX={model_prefix}, "
                f"num_spec_tokens={recipe_spec_tokens} ({ref_yaml})"
            )
        _log(
            f"Verified golden AL={golden_al:g} from {ref_yaml} "
            f"(model={key}, num_spec_tokens={recipe_spec_tokens})"
        )
        return golden_al

    # Preserve the legacy reference format for model families that have not yet
    # been assigned an unambiguous committed curve above.
    if explicit:
        return float(explicit)

    if not os.path.isfile(legacy_ref_yaml):
        sys.exit(
            "ERROR: SYNTHETIC_ACCEPTANCE_LENGTH not set and reference YAML not "
            f"found: {legacy_ref_yaml}"
        )

    import yaml

    with open(legacy_ref_yaml) as f:
        data = yaml.safe_load(f)

    key = _yaml_key(model_prefix)
    model_block = data.get(key)
    if model_block is None:
        sys.exit(f'ERROR: model key "{key}" not found in {legacy_ref_yaml}')

    num_spec_tokens = (
        int(nst_env) if nst_env else (recipe_spec_tokens if recipe_spec_tokens else 2)
    )

    al = _lookup_al(model_block, num_spec_tokens)
    if al is None:
        sys.exit(
            f"ERROR: num_spec_tokens={num_spec_tokens} not found for {key} in "
            f"{legacy_ref_yaml}"
        )

    _log(
        f"Auto-resolved AL={al} from {legacy_ref_yaml} "
        f"(model={key}, num_spec_tokens={num_spec_tokens})"
    )
    return float(al)


def inject(config_file, framework):
    injector = get_injector(framework)

    if _enabled("EVAL_ONLY"):
        if injector is None or not hasattr(injector, "rewrite_real"):
            print(
                f"[Synthetic AL] EVAL_ONLY=true: no real-acceptance rewriter "
                f"for FRAMEWORK='{framework}'"
            )
            return 0

        with open(config_file) as f:
            content = f.read()
        new_content, count = injector.rewrite_real(content, _log)
        if count:
            with open(config_file, "w") as f:
                f.write(new_content)
            _log(f"Restored real acceptance in {count} speculative-config entries")
        else:
            _log("EVAL_ONLY=true: recipe already uses real acceptance")
        return 0

    if not _enabled("SYNTHETIC_ACCEPTANCE"):
        return 0

    if injector is None:
        sys.exit(
            "ERROR: SYNTHETIC_ACCEPTANCE=true but no synthetic-acceptance "
            f"injector is registered for FRAMEWORK='{framework}'"
        )

    with open(config_file) as f:
        content = f.read()

    al = _resolve_al(
        content,
        injector,
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "benchmarks",
            "speedbench-reference-al.yaml",
        ),
    )

    _log(f"Injecting synthetic acceptance (length={al}) into {config_file}")

    new_content, count = injector.rewrite(content, al, _log)

    if count == 0:
        sys.exit(
            "ERROR: SYNTHETIC_ACCEPTANCE=true but no speculative-config "
            f"entries were found in {config_file}"
        )

    with open(config_file, "w") as f:
        f.write(new_content)
    _log(f"Modified {count} speculative-config entries")
    return 0


def main(argv):
    if len(argv) not in (2, 3):
        sys.exit("Usage: inject_synthetic_acceptance.py CONFIG_FILE [FRAMEWORK]")
    framework = argv[2] if len(argv) == 3 else os.environ.get("FRAMEWORK", "")
    return inject(argv[1], framework)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
