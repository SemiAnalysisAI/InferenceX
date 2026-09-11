"""Compatibility entrypoint for :mod:`infx.results.compare_results`."""

import importlib
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    runpy.run_module("infx.results.compare_results", run_name="__main__", alter_sys=True)
else:
    sys.modules[__name__] = importlib.import_module("infx.results.compare_results")
