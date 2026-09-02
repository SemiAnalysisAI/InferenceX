#!/usr/bin/env bash

set -euo pipefail

python3 - <<'PY'
from importlib.util import find_spec
from pathlib import Path

legacy_spec = find_spec("sglang.srt.server_args_config_parser")
if legacy_spec is not None:
    print("SGLang legacy server-args config parser is already available")
    raise SystemExit(0)

current_spec = find_spec("sglang.srt.utils.server_args_config_parser")
if current_spec is None or current_spec.origin is None:
    raise SystemExit("SGLang server-args config parser is unavailable")

legacy_path = Path(current_spec.origin).parents[1] / "server_args_config_parser.py"
legacy_path.write_text(
    "from sglang.srt.utils.server_args_config_parser import *  # noqa: F403\n"
)
print(f"Installed SGLang server-args compatibility module at {legacy_path}")
PY
