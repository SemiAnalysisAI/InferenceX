#!/usr/bin/env python3
"""Inject synthetic acceptance parameters into an srt-slurm recipe (generic driver).

This is the framework-agnostic half of the synthetic-acceptance mechanism. It
decides *whether* to inject (the ``SYNTHETIC_ACCEPTANCE`` flag) and *what* mean
acceptance length to inject, then delegates the actual recipe rewrite to a
framework-specific backend (see ``runners/synthetic_injectors/``).

The script is a no-op (exit 0, file untouched) when:
  - SYNTHETIC_ACCEPTANCE is unset/false,
so existing callers that do not opt in get exactly the previous behavior. When
enabled it requires a backend registered for the given framework; the vLLM
backend is added in a follow-up framework-support change.

Environment variables:
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

import os
import sys

from synthetic_injectors import get_injector

# MODEL_PREFIX -> top-level key in speedbench-reference-al.yaml.
MODEL_PREFIX_TO_YAML_KEY = {
    "dsv4": "deepseek-v4-pro",
    "dsr1": "deepseek-r1",
}


def _log(msg):
    print(f"[Synthetic AR] {msg}")


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
            mode = os.environ.get("THINKING_MODE", "thinking_on").strip() or "thinking_on"
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


def _resolve_al(config_text, injector, ref_yaml):
    explicit = os.environ.get("SYNTHETIC_ACCEPTANCE_LENGTH", "").strip()
    if explicit:
        return float(explicit)

    if not os.path.isfile(ref_yaml):
        sys.exit(
            "ERROR: SYNTHETIC_ACCEPTANCE_LENGTH not set and reference YAML not "
            f"found: {ref_yaml}"
        )

    import yaml  # local import: only needed on the auto-lookup path

    with open(ref_yaml) as f:
        data = yaml.safe_load(f)

    key = _yaml_key(os.environ.get("MODEL_PREFIX", ""))
    model_block = data.get(key)
    if model_block is None:
        sys.exit(f'ERROR: model key "{key}" not found in {ref_yaml}')

    nst_env = os.environ.get("NUM_SPEC_TOKENS", "").strip()
    if nst_env:
        num_spec_tokens = int(nst_env)
    else:
        num_spec_tokens = injector.spec_tokens_from_recipe(config_text) or 2

    al = _lookup_al(model_block, num_spec_tokens)
    if al is None:
        sys.exit(f"ERROR: num_spec_tokens={num_spec_tokens} not found for {key} in {ref_yaml}")

    _log(
        f"Auto-resolved AL={al} from {ref_yaml} "
        f"(model={key}, num_spec_tokens={num_spec_tokens})"
    )
    return float(al)


def inject(config_file, framework):
    if os.environ.get("SYNTHETIC_ACCEPTANCE", "false") != "true":
        return 0

    injector = get_injector(framework)
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
        os.path.join(os.path.dirname(__file__), "..", "benchmarks", "speedbench-reference-al.yaml"),
    )

    _log(f"Injecting synthetic acceptance (length={al}) into {config_file}")

    new_content, count = injector.rewrite(content, al, _log)

    if count == 0:
        _log("WARNING: No speculative-config entries found to modify; leaving recipe unchanged")
        return 0

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
