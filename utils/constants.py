"""Compatibility imports for the sweep planner's default input paths."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infx.matrix import GENERATE_SWEEPS_PY_SCRIPT, MASTER_CONFIGS, RUNNER_CONFIG
