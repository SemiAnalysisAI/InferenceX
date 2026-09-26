"""Git operation helpers for InferenceX.

Subprocess-based helpers work everywhere (including ``--no-project`` CI runs).
The ``GitRepo`` class requires GitPython and is available under the ``workflows``
extra.
"""

from infx.git.repo import (
    diff_added_lines,
    ls_tree_batch_read,
    rev_parse,
    run_git,
    show_file,
    show_stage,
)

__all__ = [
    "GitRepo",
    "diff_added_lines",
    "ls_tree_batch_read",
    "rev_parse",
    "run_git",
    "show_file",
    "show_stage",
]


def __getattr__(name: str) -> object:
    if name == "GitRepo":
        from infx.git.repo import GitRepo

        return GitRepo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
