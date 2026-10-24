from __future__ import annotations

from types import ModuleType

import pytest
from matplotlib.figure import Figure

from infx.datasets import plot_subagent_distributions, plot_weka_distributions


@pytest.mark.parametrize("module", [plot_weka_distributions, plot_subagent_distributions])
@pytest.mark.parametrize("values", [[0.0002, 0.0005], [64.0, 64.0, 64.0]])
def test_log_histogram_counts_every_positive_value(
    module: ModuleType, values: list[float]
) -> None:
    ax = Figure().subplots()

    module._draw_histogram(
        ax=ax,
        values=values,
        title="metric",
        xlabel="x",
        bins=40,
        log_x=True,
        value_fmt="{:,.4f}",
        linear_clip_pct=99.0,
    )

    assert sum(bar.get_height() for bar in ax.patches) == len(values)
