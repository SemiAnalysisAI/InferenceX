# SPDX-License-Identifier: GPL-3.0-only
import shutil

import pytest

from power_model.tables import TABLES_DIR, load_tables


def test_unresolved_references_fail_at_load(tmp_path):
    root = tmp_path / "tables"
    shutil.copytree(TABLES_DIR, root, ignore=shutil.ignore_patterns("*.py", "__pycache__"))
    parts = root / "parts.yaml"
    parts.write_text(
        parts.read_text()
        .replace("provenance: hgx-idle-nvme", "provenance: hgx-idle-nvme-v2")
        .replace("per_drive_power_w: $power_w", "per_drive_power_w: $drive_w")
    )

    with pytest.raises(ValueError, match="provenance hgx-idle-nvme-v2.*detail variable drive_w"):
        load_tables(root)
