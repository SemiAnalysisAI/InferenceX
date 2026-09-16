from __future__ import annotations

import argparse

import pytest

from benchmarks.multi_node.amd_utils.sglang_cli import cuda_graph_flags


@pytest.mark.parametrize("split_available", [False, True])
def test_legacy_option_preserves_batch_sizes(split_available: bool) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-graph-bs", nargs="+", type=int)
    if split_available:
        parser.add_argument("--cuda-graph-bs-prefill", nargs="+", type=int)
        parser.add_argument("--cuda-graph-bs-decode", nargs="+", type=int)

    prefill, decode = cuda_graph_flags(parser)

    for flag in (prefill, decode):
        args = parser.parse_args([flag, "1", "4", "8"])
        assert args.cuda_graph_bs == [1, 4, 8]


def test_split_options_select_the_correct_phase() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-graph-bs-prefill", nargs="+", type=int)
    parser.add_argument("--cuda-graph-bs-decode", nargs="+", type=int)

    prefill, decode = cuda_graph_flags(parser)
    args = parser.parse_args([prefill, "2", "4", decode, "1", "8"])

    assert args.cuda_graph_bs_prefill == [2, 4]
    assert args.cuda_graph_bs_decode == [1, 8]


@pytest.mark.parametrize("available", [None, "--cuda-graph-bs-decode"])
def test_missing_supported_options_fail_closed(available: str | None) -> None:
    parser = argparse.ArgumentParser()
    if available:
        parser.add_argument(available, nargs="+", type=int)

    with pytest.raises(ValueError, match="no supported CUDA graph"):
        cuda_graph_flags(parser)
