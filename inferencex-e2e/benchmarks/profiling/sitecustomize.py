"""Load the op-attribution profiler (infx_prof) in every engine process; see infx_prof/__init__.py."""

import importlib.machinery
import importlib.util
import os
import sys


def _chain_next_sitecustomize():
    here = os.path.dirname(os.path.abspath(__file__))
    for entry in sys.path:
        if not entry or os.path.abspath(entry) == here:
            continue
        spec = importlib.machinery.PathFinder.find_spec("sitecustomize", [entry])
        if spec is not None and spec.origin and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return


if os.environ.get("INFX_PROF_DIR"):
    import infx_prof

    infx_prof.boot()

try:
    _chain_next_sitecustomize()
except Exception:
    if os.environ.get("INFX_PROF_DIR"):
        from infx_prof import runtime

        runtime.write_error("chain sitecustomize")
