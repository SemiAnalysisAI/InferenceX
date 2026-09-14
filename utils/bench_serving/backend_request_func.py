"""Compatibility entrypoint for :mod:`infx.bench_serving.backend_request_func`."""

import importlib
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __name__ == "__main__":
    runpy.run_module("infx.bench_serving.backend_request_func", run_name="__main__", alter_sys=True)
else:
    sys.modules[__name__] = importlib.import_module("infx.bench_serving.backend_request_func")
