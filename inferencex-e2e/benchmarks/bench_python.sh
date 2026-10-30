#!/usr/bin/env bash

bench_python() {
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return
    fi
    # Serving images without an interpreter (LayerScale) run the stdlib client on uv's standalone
    # CPython, the version infx/bench/agentic/venv.py builds the AgentX venv from.
    local uv_dir
    uv_dir="$(mktemp -d)" || return
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$uv_dir" UV_NO_MODIFY_PATH=1 sh >&2 || return
    "$uv_dir/uv" python install 3.11 >&2 && "$uv_dir/uv" python find 3.11
}
