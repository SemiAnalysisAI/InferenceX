import matplotlib.pyplot as plt
import pytest

from generate_aiperf_plots import has_atom_metrics, panel_prefix_cache_hit_rate


def _counter(value: float) -> dict:
    return {
        "series": [
            {
                "timeslices": [
                    {"start_ns": 1_000_000_000, "total": value},
                ]
            }
        ]
    }


@pytest.mark.parametrize("separator", [":", "_"])
def test_atom_prefix_cache_panel_includes_external_hits(separator: str) -> None:
    server_metrics = {
        "metrics": {
            f"atom{separator}prefix_cache_cached_tokens": _counter(40),
            f"atom{separator}prefix_cache_full_tokens": _counter(100),
            f"atom{separator}lmcache_loaded_tokens": _counter(20),
        }
    }

    assert has_atom_metrics(server_metrics)

    figure, axis = plt.subplots()
    try:
        panel_prefix_cache_hit_rate(axis, server_metrics, 0)
        series = {collection.get_label(): collection for collection in axis.collections}

        assert series["GPU (HBM)"].get_offsets()[0, 1] == pytest.approx(40)
        assert series["External"].get_offsets()[0, 1] == pytest.approx(20)
    finally:
        plt.close(figure)
