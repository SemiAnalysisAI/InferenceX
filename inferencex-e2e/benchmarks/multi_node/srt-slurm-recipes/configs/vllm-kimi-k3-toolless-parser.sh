#!/usr/bin/env bash
# Test-only patch (vllm-project/vllm#65872): for Kimi K3 streaming requests without tools, strip XTML content
# markers with the reasoning parser (as the reasoning-only path does) instead of
# gating content on tool framing, which drops all text after the first
# <|close|>response<|sep|> or <|open|>tools<|sep|>. Mirrors
# vllm_kimi_k3_toolless_parser.diff (vllm/parser/kimi_k3.py only).
set -euo pipefail
python3 - <<'EOF'
import importlib.util
import pathlib
import re

tag = "[kimi-k3-toolless-parser]"
root = pathlib.Path(importlib.util.find_spec("vllm").submodule_search_locations[0])
path = root / "parser/kimi_k3.py"
src = path.read_text()
if "if not request.tools and isinstance(" in src:
    print(f"{tag} already applied: {path}")
    raise SystemExit(0)

# The fix calls KimiK3ReasoningParser.strip_content_streaming(previous_text=,
# current_text=); fail loudly if this build lacks it.
rp_src = (root / "reasoning/kimi_k3_reasoning_parser.py").read_text()
assert re.search(
    r"def strip_content_streaming\(\s*self,\s*previous_text: str,\s*current_text: str",
    rp_src,
), "KimiK3ReasoningParser.strip_content_streaming(previous_text, current_text) not found"
assert re.search(
    r"^from vllm\.reasoning\.kimi_k3_reasoning_parser import KimiK3ReasoningParser$",
    src,
    re.M,
), "KimiK3ReasoningParser import not found in parser/kimi_k3.py"
assert src.count("class KimiK3Parser(") == 1, "KimiK3Parser not found exactly once"

fn = src.index("def _extract_tool_calls_streaming(", src.index("class KimiK3Parser("))
sig = re.compile(r"\)\s*->\s*tuple\[DeltaMessage \| None, bool\]:\n").search(src, fn)
assert sig is not None, "_extract_tool_calls_streaming signature end not found"
params = src[fn : sig.start()]
for name in ("previous_text", "current_text", "request"):
    assert re.search(rf"\b{name}\b", params), f"parameter {name} not in signature"
body = src[sig.end() :]
assert re.match(
    r'        if request\.tool_choice != "none":\n', body
), 'first statement is not `if request.tool_choice != "none":`'

block = '''        if not request.tools and isinstance(
            self._reasoning_parser, KimiK3ReasoningParser
        ):
            # No tools means no tools channel to parse: strip the XTML content
            # markers as the reasoning-only path does, instead of gating content
            # on tool framing that would hold back all text after the first
            # <|close|>response<|sep|> or <|open|>tools<|sep|>.
            return self._reasoning_parser.strip_content_streaming(
                previous_text=previous_text,
                current_text=current_text,
            ), False

'''
src = src[: sig.end()] + block + body

compile(src, str(path), "exec")
path.write_text(src)
print(f"{tag} applied: {path}")
EOF
