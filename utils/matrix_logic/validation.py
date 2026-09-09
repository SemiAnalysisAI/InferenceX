"""Compatibility import for :mod:`infx.matrix.validation`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from infx.matrix import validation

sys.modules[__name__] = validation
