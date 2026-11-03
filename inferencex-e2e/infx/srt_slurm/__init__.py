from __future__ import annotations

import sys

from infx.config import repository_root


def load_srt() -> None:
    """Make the repository's pinned SRT checkout available to planning and launch adapters."""
    source = repository_root() / "utils/srt-slurm/src"
    if not (source / "srtctl").is_dir():
        raise ValueError(f"No srt-slurm checkout at {source}; initialize the submodule")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
