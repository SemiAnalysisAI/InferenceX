"""Compatibility entrypoint for :mod:`infx.workflows.calc_success_rate`."""

import importlib
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    runpy.run_module("infx.workflows.calc_success_rate", run_name="__main__", alter_sys=True)
else:
    sys.modules[__name__] = importlib.import_module("infx.workflows.calc_success_rate")
