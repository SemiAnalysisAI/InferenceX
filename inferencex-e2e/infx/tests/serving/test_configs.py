from pathlib import Path

import pytest

from infx.serving.to_matrix import build

CONFIGS = sorted((Path(__file__).resolve().parents[3] / "configs/serving").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_every_serving_config_validates(path):
    assert build(path.read_text())["points"]
