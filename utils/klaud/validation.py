"""Compatibility entrypoint for :mod:`infx.klaud.validation`."""

import importlib
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __name__ == "__main__":
    runpy.run_module("infx.klaud.validation", run_name="__main__", alter_sys=True)
else:
    sys.modules[__name__] = importlib.import_module("infx.klaud.validation")
