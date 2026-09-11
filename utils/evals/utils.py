"""Compatibility entrypoint for :mod:`infx.evals.utils`."""

import importlib
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __name__ == "__main__":
    runpy.run_module("infx.evals.utils", run_name="__main__", alter_sys=True)
else:
    sys.modules[__name__] = importlib.import_module("infx.evals.utils")
