"""The planner's srtctl-free variant expansion selects what srtctl selects at launch."""

import sys
from pathlib import Path

import pytest

from infx.srt_slurm.synthetic_acceptance import selected_recipes
from infx.srt_slurm.variants import expand_variants

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "utils/srt-slurm/src"))

BUNDLE = {
    "schema": 2,
    "base": {
        "name": "bundle", "engine": "sglang",
        # srtctl apply drops this null from zip variants only.
        "frontend": {"type": "sglang", "router": None},
        "roles": {"agg": {"gpus": 4, "args": {
            "tensor-parallel-size": 4, "chunked-prefill-size": 8192, "cuda-graph-bs": [1, 2, 4],
        }}},
        "benchmark": {"env": {"KEEP": "1"}},
    },
    # Null deletes a key and lists replace.
    "override_wide": {"roles": {"agg": {"gpus": 8, "args": {
        "tensor-parallel-size": 8, "chunked-prefill-size": None, "cuda-graph-bs": [8],
    }}}},
    "override_named": {"name": "custom", "benchmark": None},
    # Length-1 lists broadcast across the group.
    "zip_override_conc": {
        "roles": {"agg": {"args": {"max-running-requests": [2, 4, 8]}}},
        "benchmark": {"env": {"CONC": ["2", "4", "8"], "KEEP": ["0"]}},
    },
    "zip_override_named": {"name": ["first", "second"], "roles": {"agg": {"gpus": [1, 2]}}},
}  # fmt: skip


@pytest.mark.parametrize("selector", [
    None, "base", "override_wide", "override_named", "zip_override_conc[2]", "zip_override_named[1]",
])  # fmt: skip
def test_expansion_matches_srtctl(selector):
    assert expand_variants(BUNDLE, selector) == selected_recipes(BUNDLE, selector)


@pytest.mark.parametrize(("raw", "selector"), [
    (BUNDLE, "override_missing"),
    (BUNDLE, "zip_override_conc[3]"),
    ({"schema": 2, "engine": "sglang"}, "base"),
    ({"base": {}, "zip_override_uneven": {"a": [1, 2], "b": [1, 2, 3]}}, None),
])  # fmt: skip
def test_selections_srtctl_rejects_are_rejected(raw, selector):
    with pytest.raises(ValueError):
        selected_recipes(raw, selector)
    with pytest.raises(ValueError):
        expand_variants(raw, selector)
