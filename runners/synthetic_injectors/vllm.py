"""vLLM synthetic-acceptance backend (FRAMEWORK=dynamo-vllm).

Rewrites every ``speculative-config: '<json>'`` entry in an srt-slurm recipe to
use synthetic rejection sampling: it adds ``rejection_sample_method=synthetic``
and ``synthetic_acceptance_length=<al>`` to the JSON so the engine emits a
controlled mean acceptance length instead of running the real draft model.

Registered under the "dynamo-vllm" framework key at import time, so importing
the ``synthetic_injectors`` package is enough for the generic driver to resolve
this backend.
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
        after = [ln.strip() for ln in new_content.splitlines() if _SPEC_CONFIG_RE.search(ln)]
        if after:
            log("After:")
            for ln in after:
                print(f"  {ln}")

    return new_content, count


register("dynamo-vllm", sys.modules[__name__])
