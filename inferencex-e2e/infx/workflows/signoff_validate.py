"""Check a CODEOWNER verdict file against the publisher's rules without publishing it.

The sign-off verifier runs this on its own verdict before finishing, so it can fix a
formatting slip instead of having the whole verdict discarded:

    python -m infx.workflows.signoff_validate /tmp/codeowner-signoff-verdict.md
"""

from __future__ import annotations

import sys
from pathlib import Path

from infx.workflows.signoff_publish import verdict_errors

VALID = "Verdict format is valid."


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m infx.workflows.signoff_validate <verdict.md>", file=sys.stderr)
        return 2
    try:
        verdict = Path(args[0]).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        print(f"Cannot read {args[0]}: {exc}", file=sys.stderr)
        return 2
    errors = verdict_errors(verdict)
    print("\n".join(errors) if errors else VALID)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
