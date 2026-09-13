"""Compatibility entrypoint for :mod:`infx.workflows.reuse`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infx.workflows import reuse

if __name__ == "__main__":
    reuse.cli()
else:
    # Preserve legacy imports and monkeypatches on the canonical module.
    sys.modules[__name__] = reuse
