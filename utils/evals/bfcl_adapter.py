"""Compatibility entrypoint for :mod:`infx.evals.bfcl_adapter`."""

import importlib
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __name__ == "__main__":
    runpy.run_module("infx.evals.bfcl_adapter", run_name="__main__", alter_sys=True)
else:
    sys.modules[__name__] = importlib.import_module("infx.evals.bfcl_adapter")
