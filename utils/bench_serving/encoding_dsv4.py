"""Compatibility entrypoint for :mod:`infx.bench_serving.encoding_dsv4`."""

import importlib
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __name__ == "__main__":
    runpy.run_module("infx.bench_serving.encoding_dsv4", run_name="__main__", alter_sys=True)
else:
    sys.modules[__name__] = importlib.import_module("infx.bench_serving.encoding_dsv4")
