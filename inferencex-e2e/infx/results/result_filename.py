"""Bound artifact basenames without losing recipe or point identity."""

from __future__ import annotations

import argparse
import hashlib
import os


def compact(value: str, budget: int) -> str:
    """Return ``value`` if it fits ``budget`` UTF-8 bytes, else a truncated prefix plus a hash suffix.

    Raises ValueError for empty values, path separators or a budget too small for the suffix.
    """
    if not value or any(char in value for char in "/\\\r\n\x00"):
        raise ValueError("Expected a nonempty filename component")
    encoded = value.encode("utf-8")
    if len(encoded) <= budget:
        return value
    suffix = "-" + hashlib.sha256(encoded).hexdigest()[:20]
    if budget < len(suffix):
        raise ValueError("Result suffix leaves no filename identity budget")
    return encoded[: budget - len(suffix)].decode("utf-8", errors="ignore") + suffix


def result_stem(base: str, fingerprint: str = "") -> str:
    """Return the result basename for ``base`` and an optional recipe ``fingerprint``, capped at 120 bytes."""
    value = base + (f"_recipe-{fingerprint}" if fingerprint else "")
    return compact(value, 120)


def point_filename(
    stem: str, config: str, conc: str, gpus: str, ctx: str = "", gen: str = ""
) -> str:
    """Return the point result filename for ``stem``, ``config`` and numeric point fields within 255 bytes.

    Raises ValueError when a point field is not decimal.
    """
    numbers = [conc, gpus, *([ctx, gen] if ctx or gen else [])]
    if any(not number.isascii() or not number.isdecimal() for number in numbers):
        raise ValueError("Expected numeric point identity")
    tail = f"_conc{conc}_gpus_{gpus}" + (f"_ctx_{ctx}_gen_{gen}" if ctx else "")
    budget = 255 - len(("power_validation_" + stem + "_" + tail + ".json.tmp").encode())
    return f"{stem}_{compact(config, budget)}{tail}.json"


def main() -> None:
    """Print a point filename for ``--point``, else the result stem from RESULT_FILENAME_BASE."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--point", nargs=6, metavar=("STEM", "CONFIG", "CONC", "GPUS", "CTX", "GEN")
    )
    args = parser.parse_args()
    print(
        point_filename(*args.point)
        if args.point
        else result_stem(
            os.environ["RESULT_FILENAME_BASE"], os.environ.get("RECIPE_FINGERPRINT", "")
        )
    )


if __name__ == "__main__":
    main()
