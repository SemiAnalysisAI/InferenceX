"""Resolve CLI spellings from the installed SGLang parser."""

from __future__ import annotations

import argparse


def cuda_graph_flags(parser: argparse.ArgumentParser) -> tuple[str, str]:
    """Preserve legacy graph settings, or use the explicit phase options."""
    options = parser._option_string_actions
    if "--cuda-graph-bs" in options:
        return "--cuda-graph-bs", "--cuda-graph-bs"
    split = "--cuda-graph-bs-prefill", "--cuda-graph-bs-decode"
    if all(option in options for option in split):
        return split
    raise ValueError("Installed SGLang has no supported CUDA graph batch-size flags")


def main() -> None:
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    print(*cuda_graph_flags(parser))


if __name__ == "__main__":
    main()
