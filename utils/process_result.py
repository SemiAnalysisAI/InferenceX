"""Compatibility entrypoint for :mod:`infx.results.fixed_sequence`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infx.results.fixed_sequence import main


if __name__ == "__main__":
    sys.exit(main())
