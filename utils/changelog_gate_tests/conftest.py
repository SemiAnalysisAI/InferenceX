"""Make the colocated utils/ modules importable from this test subfolder.

These tests live in their own directory but exercise modules that remain in
utils/ (validate_perf_changelog, prepare_perf_changelog_merge,
recover_failed_ingest, and matrix_logic). Under pytest's default prepend import
mode only this directory is added to sys.path, so prepend utils/ as well to
resolve the top-level imports.
"""

from __future__ import annotations

import sys
from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parent.parent
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))
