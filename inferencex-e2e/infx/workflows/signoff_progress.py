"""Show an eyes reaction on a CODEOWNER sign-off while its verification runs.

Reactions notify nobody. GitHub has no reactions for review summaries, so those
sign-offs get no progress marker; the verdict comment is unchanged either way.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Literal

from infx import github

IN_PROGRESS = "eyes"
SIGNOFF_COMMENT = re.compile(r"(issuecomment-|discussion_r)([1-9][0-9]*)")
COLLECTIONS: dict[str, Literal["issues", "pulls"]] = {
    "issuecomment-": "issues",
    "discussion_r": "pulls",
}


def reaction_target(signoff_key: str) -> tuple[Literal["issues", "pulls"], int] | None:
    """Comment collection and id for a sign-off key, or None for a review summary."""
    match = SIGNOFF_COMMENT.fullmatch(signoff_key)
    return (COLLECTIONS[match[1]], int(match[2])) if match else None


def mark(repo: str, token: str, signoff_key: str, *, running: bool) -> str:
    """Add or clear the bot's in-progress reaction. Never raises on GitHub errors."""
    target = reaction_target(signoff_key)
    if target is None:
        return f"{signoff_key}: no reaction (review summaries do not support reactions)"
    collection, comment_id = target
    try:
        github.set_comment_reaction(
            repo,
            comment_id,
            token,
            IN_PROGRESS if running else None,
            collection=collection,
            replace=() if running else (IN_PROGRESS,),
        )
    except (github.APIError, github.ListingError) as error:
        return f"::warning::Could not update the progress reaction on {signoff_key}: {error}"
    return f"{signoff_key}: {'added' if running else 'cleared'} {IN_PROGRESS} reaction"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", choices=("start", "finish"))
    args = parser.parse_args(argv)
    print(
        mark(
            os.environ["GITHUB_REPOSITORY"],
            os.environ["GH_TOKEN"],
            os.environ["SIGNOFF_KEY"],
            running=args.state == "start",
        )
    )


if __name__ == "__main__":
    main()
