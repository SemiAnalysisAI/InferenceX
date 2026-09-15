"""vLLM speculative-acceptance recipe rewriting.

Throughput opt-ins can inject synthetic acceptance. Eval-only runs restore real
block verification so model outputs remain valid for accuracy checks. The
backend is registered for both direct vLLM and Dynamo-vLLM recipes.
"""

import json
import re
import sys

from . import register

# Matches `speculative-config: '<json>'` (single-quoted JSON, as written in the
# recipe YAML). Capturing the JSON lets us edit it with the json module instead
# of string-munging, so we never produce malformed quoting.
_SPEC_CONFIG_RE = re.compile(r"speculative-config:\s*'([^']+)'")


def spec_tokens_from_recipe(text):
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


def rewrite(content, al, log):
    """Rewrite every speculative-config entry to synthetic acceptance.

    Returns ``(new_content, count)`` where count is the number of entries
    modified (0 => nothing matched, recipe left unchanged by the driver).
    """
    before = [ln.strip() for ln in content.splitlines() if _SPEC_CONFIG_RE.search(ln)]
    if before:
        log("Before:")
        for ln in before:
            print(f"  {ln}")

    def _replace(match):
        spec = json.loads(match.group(1))
        spec["rejection_sample_method"] = "synthetic"
        spec["synthetic_acceptance_length"] = al
        # Compact separators keep the same style as the hand-written recipes.
        return "speculative-config: '" + json.dumps(spec, separators=(",", ":")) + "'"

    new_content, count = _SPEC_CONFIG_RE.subn(_replace, content)

    if count:
        after = [
            ln.strip() for ln in new_content.splitlines() if _SPEC_CONFIG_RE.search(ln)
        ]
        if after:
            log("After:")
            for ln in after:
                print(f"  {ln}")

    return new_content, count


def rewrite_real(content, log):
    """Restore real block verification in every synthetic config entry."""
    modified = 0

    def _replace(match):
        nonlocal modified
        spec = json.loads(match.group(1))
        if (
            spec.get("rejection_sample_method") != "synthetic"
            and "synthetic_acceptance_length" not in spec
        ):
            return match.group(0)
        spec["rejection_sample_method"] = "block"
        spec.pop("synthetic_acceptance_length", None)
        modified += 1
        return "speculative-config: '" + json.dumps(spec, separators=(",", ":")) + "'"

    new_content = _SPEC_CONFIG_RE.sub(_replace, content)
    if modified:
        log("Restored real block verification for eval-only mode")
    return new_content, modified


register("dynamo-vllm", sys.modules[__name__])
register("vllm", sys.modules[__name__])
