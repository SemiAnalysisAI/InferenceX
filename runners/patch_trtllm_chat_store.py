#!/usr/bin/env python3
"""Accept OpenAI's non-persistent chat request field in TensorRT-LLM."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


CLASS_HEADER = "class ChatCompletionRequest(OpenAIBaseModel):\n"
NEXT_CLASS = "\nclass "
FIELD_ANCHOR = "    stream: Optional[bool] = False\n"
STORE_FIELD = "    store: Optional[Literal[False]] = False\n"


def installed_protocol_path() -> Path:
    """Return the protocol module path from the installed TensorRT-LLM package."""
    spec = importlib.util.find_spec("tensorrt_llm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("tensorrt_llm package is not installed")
    package_root = Path(next(iter(spec.submodule_search_locations)))
    return package_root / "serve/openai_protocol.py"


def patch_protocol(protocol_path: Path) -> bool:
    """Accept only ``store=false`` and return whether the source changed."""
    source = protocol_path.read_text()
    if source.count(CLASS_HEADER) != 1:
        raise RuntimeError(
            f"unsupported TensorRT-LLM protocol at {protocol_path}: "
            "ChatCompletionRequest not found exactly once"
        )

    class_start = source.index(CLASS_HEADER)
    class_end = source.find(NEXT_CLASS, class_start + len(CLASS_HEADER))
    if class_end == -1:
        raise RuntimeError(
            f"unsupported TensorRT-LLM protocol at {protocol_path}: "
            "ChatCompletionRequest boundary not found"
        )

    class_source = source[class_start:class_end]
    if STORE_FIELD in class_source:
        return False
    if "    store:" in class_source:
        raise RuntimeError(
            f"unsupported TensorRT-LLM store field at {protocol_path}"
        )
    if class_source.count(FIELD_ANCHOR) != 1:
        raise RuntimeError(
            f"unsupported TensorRT-LLM protocol at {protocol_path}: "
            "ChatCompletionRequest stream field not found exactly once"
        )

    patched_class = class_source.replace(
        FIELD_ANCHOR,
        STORE_FIELD + FIELD_ANCHOR,
    )
    protocol_path.write_text(source[:class_start] + patched_class + source[class_end:])
    return True


def main(argv: list[str]) -> int:
    if len(argv) > 2:
        print(f"Usage: {argv[0]} [OPENAI_PROTOCOL_PATH]", file=sys.stderr)
        return 2

    try:
        protocol_path = (
            Path(argv[1]).resolve() if len(argv) == 2 else installed_protocol_path()
        )
        changed = patch_protocol(protocol_path)
    except (OSError, RuntimeError) as error:
        print(f"ERROR: failed to patch TensorRT-LLM chat requests: {error}", file=sys.stderr)
        return 1

    state = "Patched" if changed else "Already patched"
    print(f"{state} TensorRT-LLM ChatCompletionRequest store=false support")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
