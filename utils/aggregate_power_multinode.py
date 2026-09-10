"""Compatibility entrypoint for :mod:`infx.results.power.multinode`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infx.results.power import multinode

if __name__ == "__main__":
    sys.exit(multinode.main())
else:
    # Keep legacy imports and monkeypatches attached to the canonical module.
    sys.modules[__name__] = multinode
