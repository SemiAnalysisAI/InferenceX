"""Compatibility entrypoint for :mod:`infx.evals.patches.lm_eval_sitecustomize`."""

import importlib
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

if __name__ == "__main__":
    runpy.run_module("infx.evals.patches.lm_eval_sitecustomize", run_name="__main__", alter_sys=True)
else:
    sys.modules[__name__] = importlib.import_module("infx.evals.patches.lm_eval_sitecustomize")
