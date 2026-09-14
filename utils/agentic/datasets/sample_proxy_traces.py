"""Compatibility entrypoint for :mod:`infx.datasets.sample_proxy_traces`."""

import importlib
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

if __name__ == "__main__":
    runpy.run_module("infx.datasets.sample_proxy_traces", run_name="__main__", alter_sys=True)
else:
    sys.modules[__name__] = importlib.import_module("infx.datasets.sample_proxy_traces")
