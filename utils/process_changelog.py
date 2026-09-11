"""Compatibility entrypoint for :mod:`infx.matrix.plan`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infx.matrix import plan

if __name__ == "__main__":
    plan.main()
else:
    sys.modules[__name__] = plan
