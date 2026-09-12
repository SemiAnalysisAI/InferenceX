"""TensorRT-LLM speculative-acceptance recipe rewriting.

TRT-LLM forces acceptance through the worker environment variable
``TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS``. Its value is the number of
*draft* tokens accepted per step, so an acceptance length AL (target token +
accepted drafts) maps to ``AL - 1``. Throughput opt-ins inject that variable
into every ``*_environment`` block under ``backend:``; eval-only runs remove it
so the verifier's real acceptance drives the generated text. The backend is
registered for both direct trtllm-serve and Dynamo-TRT-LLM recipes.
"""

import re
import sys

from . import register

_ENV_KEY = "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS"
# A `*_environment:` mapping header under backend (aggregated/prefill/decode).
_ENV_BLOCK_RE = re.compile(r"^(?P<indent>[ \t]*)(?:aggregated|prefill|decode)_environment:[ \t]*$")
_ENV_LINE_RE = re.compile(rf"^[ \t]*{_ENV_KEY}:.*\n?", re.MULTILINE)
_DRAFT_LEN_RE = re.compile(r"^\s*(?:max_draft_len|num_nextn_predict_layers):\s*(\d+)", re.MULTILINE)


def spec_tokens_from_recipe(text):
    """Best-effort: read the speculative draft length from the engine config."""
    m = _DRAFT_LEN_RE.search(text)
    return int(m.group(1)) if m else None


def _format_value(al):
    return f"{al - 1:g}"


def rewrite(content, al, log):
    """Force ``AL - 1`` accepted draft tokens in every backend environment block.

    Returns ``(new_content, count)`` where count is the number of environment
    blocks now carrying the variable (0 => no block found, recipe left unchanged).
    """
    value = _format_value(al)
    # Replace existing settings first so a recipe that already carries the
    # variable ends up with exactly one line per block.
    content = _ENV_LINE_RE.sub("", content)

    lines = content.splitlines(keepends=True)
    out, count, i = [], 0, 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        m = _ENV_BLOCK_RE.match(line.rstrip("\n"))
        if m:
            # Child indentation comes from the first non-empty following line;
            # fall back to two spaces past the header.
            child_indent = None
            for nxt in lines[i + 1:]:
                if nxt.strip():
                    child_indent = nxt[: len(nxt) - len(nxt.lstrip())]
                    break
            if child_indent is None or len(child_indent) <= len(m.group("indent")):
                child_indent = m.group("indent") + "  "
            out.append(f"{child_indent}{_ENV_KEY}: '{value}'\n")
            count += 1
        i += 1
    new_content = "".join(out)
    if count:
        log(f"Set {_ENV_KEY}={value} (AL={al}) in {count} environment block(s)")
    return new_content, count


def rewrite_real(content, log):
    """Remove forced acceptance so the verifier's real acceptance is used."""
    new_content, count = _ENV_LINE_RE.subn("", content)
    if count:
        log(f"Removed {_ENV_KEY} from {count} environment block(s) for eval-only mode")
    return new_content, count


register("dynamo-trt", sys.modules[__name__])
register("trt", sys.modules[__name__])
