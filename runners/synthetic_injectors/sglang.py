"""SGLang synthetic-acceptance backend for srt-slurm recipes."""

import re
import sys

from . import register

_SPEC_STEPS_RE = re.compile(r"(?m)^\s+speculative-num-steps:\s*([0-9]+)\s*$")
_SPEC_DRAFT_TOKENS_RE = re.compile(
    r"(?m)^\s+speculative-num-draft-tokens:\s*([0-9]+)\s*$"
)
_ENV_BLOCK_RE = re.compile(
    r"(?m)^([ \t]+)((?:aggregated|prefill|decode)_environment:\s*)$"
)
_SIMULATED_ACCEPTANCE_ENV_RE = re.compile(
    r"(?m)^[ \t]+SGLANG_SIMULATE_ACC_(?:LEN|METHOD|TOKEN_MODE):[^\n]*(?:\n|$)"
)


def spec_tokens_from_recipe(text):
    """Read SGLang's speculative step count from the recipe."""
    values = {int(match) for match in _SPEC_STEPS_RE.findall(text)}
    if len(values) > 1:
        raise ValueError(
            f"recipe declares multiple speculative-num-steps values: {sorted(values)}"
        )
    return next(iter(values)) if values else None


def validate_speculative_shape(text):
    """Fail if SGLang's verification window is not steps + one bonus token."""
    steps = spec_tokens_from_recipe(text)
    drafts = {int(match) for match in _SPEC_DRAFT_TOKENS_RE.findall(text)}
    if steps is None or len(drafts) != 1:
        raise ValueError(
            "recipe must declare one unambiguous speculative-num-steps and "
            "speculative-num-draft-tokens value"
        )
    draft_tokens = next(iter(drafts))
    if draft_tokens != steps + 1:
        raise ValueError(
            f"speculative-num-draft-tokens={draft_tokens} must equal "
            f"speculative-num-steps + 1 ({steps + 1})"
        )


def rewrite(content, al, log):
    """Add throughput-only golden-acceptance variables to each worker role."""
    if "SGLANG_SIMULATE_ACC_LEN" in content:
        raise ValueError("recipe already contains SGLANG_SIMULATE_ACC_* variables")

    def add_variables(match):
        child_indent = match.group(1) + "  "
        variables = (
            f'\n{child_indent}SGLANG_SIMULATE_ACC_LEN: "{al:g}"'
            f'\n{child_indent}SGLANG_SIMULATE_ACC_METHOD: "match-expected"'
            f'\n{child_indent}SGLANG_SIMULATE_ACC_TOKEN_MODE: "real-draft-token"'
        )
        return match.group(0) + variables

    rewritten, count = _ENV_BLOCK_RE.subn(
        add_variables,
        content,
    )
    if count:
        log(f"Added SGLANG_SIMULATE_ACC_* to {count} worker environment block(s)")
    return rewritten, count


def rewrite_real(content, log):
    """Remove throughput-only simulated-acceptance variables for evals."""
    rewritten, count = _SIMULATED_ACCEPTANCE_ENV_RE.subn("", content)
    if count:
        log(f"Removed {count} SGLANG_SIMULATE_ACC_* environment variable(s)")
    return rewritten, count


register("dynamo-sglang", sys.modules[__name__])
