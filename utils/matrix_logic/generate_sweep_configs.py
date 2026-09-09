"""Compatibility entrypoint for :mod:`infx.matrix.generate`."""

import sys
from pathlib import Path

# Direct script execution adds this folder, not the repository root, to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from infx.matrix import generate

if __name__ == "__main__":
    generate.main()
else:
    # Preserve legacy imports and monkeypatches without loading a second module.
    sys.modules[__name__] = generate
