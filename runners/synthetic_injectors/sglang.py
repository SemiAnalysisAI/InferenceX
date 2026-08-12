"""SGLang synthetic-acceptance backend for srt-slurm recipes."""

import re
import sys

from . import register

_SPEC_STEPS_RE = re.compile(r"(?m)^\s+speculative-num-steps:\s*([0-9]+)\s*$")
_ENV_BLOCK_RE = re.compile(r"(?m)^(  (?:aggregated|prefill|decode)_environment:\s*)$")


def spec_tokens_from_recipe(text):
    """Read SGLang's speculative step count from the recipe."""
    match = _SPEC_STEPS_RE.search(text)
    return int(match.group(1)) if match else None


def rewrite(content, al, log):
    """Add throughput-only golden-acceptance variables to each worker role."""
    if "SGLANG_SIMULATE_ACC_LEN" in content:
        raise ValueError("recipe already contains SGLANG_SIMULATE_ACC_* variables")

    variables = (
        f'\n    SGLANG_SIMULATE_ACC_LEN: "{al:g}"'
        '\n    SGLANG_SIMULATE_ACC_METHOD: "match-expected"'
        '\n    SGLANG_SIMULATE_ACC_TOKEN_MODE: "real-draft-token"'
    )
    rewritten, count = _ENV_BLOCK_RE.subn(lambda match: match.group(1) + variables, content)
    if count:
        log(f"Added SGLANG_SIMULATE_ACC_* to {count} worker environment block(s)")
    return rewritten, count


register("dynamo-sglang", sys.modules[__name__])
