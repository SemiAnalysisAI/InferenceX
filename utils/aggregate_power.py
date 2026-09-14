"""Compatibility entrypoint for :mod:`infx.results.power.single_node`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infx.results.power import single_node

if __name__ == "__main__":
    sys.exit(single_node.main())
else:
    # Keep legacy imports and monkeypatches attached to the canonical module.
    sys.modules[__name__] = single_node
