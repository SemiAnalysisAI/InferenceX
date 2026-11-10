import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from infx.launch.drivers.srt import checkout


def test_setup_supplies_the_scraper_release_to_make(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(checkout, "TACHOMETER_RELEASE", "v9.8.7")
    (tmp_path / "Makefile").write_text(
        "TACHOMETER_RELEASE ?= latest\n"
        "setup:\n"
        "\t@test '$(TACHOMETER_RELEASE)' != latest\n"
        "\t@printf '%s/$(ARCH)' '$(TACHOMETER_RELEASE)' > installed-release\n"
    )
    run = SimpleNamespace(
        workspace=tmp_path,
        env=dict(os.environ),
        cluster=SimpleNamespace(arch="aarch64"),
        life=SimpleNamespace(event=Mock()),
    )

    assert checkout.run_setup(run, checkout.Checkout(tmp_path, "test")) == 0
    assert (tmp_path / "installed-release").read_text() == "v9.8.7/aarch64"
    run.life.event.fail.assert_not_called()
