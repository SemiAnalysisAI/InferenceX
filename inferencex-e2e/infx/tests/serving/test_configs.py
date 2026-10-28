from pathlib import Path

import pytest

from infx.serving.to_matrix import build

CONFIGS = sorted((Path(__file__).resolve().parents[3] / "configs/serving").glob("*.yaml"))


def test_eleven_examples_one_per_pattern():
    assert len(CONFIGS) == 11


@pytest.mark.parametrize("path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_every_serving_config_validates(path):
    assert build(path.read_text())["points"]
