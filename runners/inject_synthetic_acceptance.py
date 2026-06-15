#!/usr/bin/env python3
"""Inject synthetic acceptance parameters into an srt-slurm recipe YAML.

When SYNTHETIC_ACCEPTANCE=true, every ``speculative-config`` JSON entry in the
given recipe is rewritten to use synthetic rejection sampling (adds
``rejection_sample_method=synthetic`` and ``synthetic_acceptance_length``).

The script is a no-op (exit 0, file untouched) when:
  - SYNTHETIC_ACCEPTANCE is unset/false, or
  - the recipe contains no ``speculative-config`` entry,
so existing callers that do not opt in get exactly the previous behavior.

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

Usage (from a runner; use an absolute path since runners cd into the srt-slurm
clone before invoking this):
  python3 "$GITHUB_WORKSPACE/runners/inject_synthetic_acceptance.py" "${CONFIG_FILE%%:*}"
"""

import json
import os
import re
import sys

# MODEL_PREFIX -> top-level key in speedbench-reference-al.yaml.
MODEL_PREFIX_TO_YAML_KEY = {
    "dsv4": "deepseek-v4-pro",
    "dsr1": "deepseek-r1",
}

# Matches `speculative-config: '<json>'` (single-quoted JSON, as written in the
# recipe YAML). Capturing the JSON lets us edit it with the json module instead
# of string-munging, so we never produce malformed quoting.
_SPEC_CONFIG_RE = re.compile(r"speculative-config:\s*'([^']+)'")


def _log(msg):
    print(f"[Synthetic AR] {msg}")


def _yaml_key(model_prefix):
    return MODEL_PREFIX_TO_YAML_KEY.get(model_prefix, model_prefix)


def _spec_tokens_from_recipe(text):
    """Best-effort: read num_speculative_tokens from the recipe itself."""
    for m in _SPEC_CONFIG_RE.finditer(text):
        try:
            spec = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        n = spec.get("num_speculative_tokens")
        if n:
            return int(n)
    return 2


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


def _resolve_al(config_text, ref_yaml):
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
    num_spec_tokens = int(nst_env) if nst_env else _spec_tokens_from_recipe(config_text)

    al = _lookup_al(model_block, num_spec_tokens)
    if al is None:
        sys.exit(f"ERROR: num_spec_tokens={num_spec_tokens} not found for {key} in {ref_yaml}")

    _log(
        f"Auto-resolved AL={al} from {ref_yaml} "
        f"(model={key}, num_spec_tokens={num_spec_tokens})"
    )
    return float(al)


def inject(config_file):
    if os.environ.get("SYNTHETIC_ACCEPTANCE", "false") != "true":
        return 0

    with open(config_file) as f:
        content = f.read()

    al = _resolve_al(content, os.path.join(os.path.dirname(__file__), "..", "benchmarks", "speedbench-reference-al.yaml"))

    _log(f"Injecting synthetic acceptance (length={al}) into {config_file}")

    before = [ln.strip() for ln in content.splitlines() if _SPEC_CONFIG_RE.search(ln)]
    if before:
        _log("Before:")
        for ln in before:
            print(f"  {ln}")

    def _replace(match):
        spec = json.loads(match.group(1))
        spec["rejection_sample_method"] = "synthetic"
        spec["synthetic_acceptance_length"] = al
        # Compact separators keep the same style as the hand-written recipes.
        return "speculative-config: '" + json.dumps(spec, separators=(",", ":")) + "'"

    new_content, count = _SPEC_CONFIG_RE.subn(_replace, content)

    if count == 0:
        _log("WARNING: No speculative-config entries found to modify; leaving recipe unchanged")
        return 0

    with open(config_file, "w") as f:
        f.write(new_content)
    _log(f"Modified {count} speculative-config entries")

    after = [ln.strip() for ln in new_content.splitlines() if _SPEC_CONFIG_RE.search(ln)]
    if after:
        _log("After:")
        for ln in after:
            print(f"  {ln.strip()}")
    return 0


def main(argv):
    if len(argv) != 2:
        sys.exit("Usage: inject_synthetic_acceptance.py CONFIG_FILE")
    return inject(argv[1])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
