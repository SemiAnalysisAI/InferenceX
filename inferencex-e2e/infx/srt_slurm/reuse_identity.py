"""Write the launch identities an AgentX server-reuse group checks, from bound recipes.

``python -m infx.srt_slurm.reuse_identity --point 8=c8/recipe.yaml --point 16=c16/recipe.yaml
--output identities.json`` reads each point's bound single-node recipe (the ``recipe.yaml``
``infx.srt_slurm.single_node prepare`` writes) and fails unless every point would launch the
identical server.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from infx.bench.agentic.reuse import SCHEMA, check_identities
from infx.bench.env import InputError

CLIENT_KEYS = frozenset({"benchmark", "name"})
"""Bound-recipe keys that hold the client's per-point inputs or a variant label, not the server."""


def server_identity(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """The ``{"image", "model", "server"}`` identity of one bound single-node recipe."""
    block = recipe["model"]
    return {
        "image": block["container"],
        "model": str(block["path"]).removeprefix("hf:"),
        "server": {key: deepcopy(value) for key, value in recipe.items() if key not in CLIENT_KEYS},
    }


def identities(recipes: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """The identities document for ``recipes`` (concurrency to bound recipe, in run order).

    Raises ``ValueError`` when a recipe was bound to another concurrency, and
    ``InputError`` when the points' servers differ or fall outside reuse's scope.
    """
    for conc, recipe in recipes.items():
        bound = str(recipe["benchmark"]["env"]["CONC"])
        if bound != str(conc):
            raise ValueError(f"the recipe given for point {conc} was bound to CONC={bound}")
    document = {
        "schema": SCHEMA,
        "points": {str(conc): server_identity(recipe) for conc, recipe in recipes.items()},
    }
    first = document["points"][str(next(iter(recipes)))]
    check_identities(document, list(recipes), first["model"], None)
    return document


def _point(text: str) -> tuple[int, Path]:
    """Parse ``CONC=PATH``."""
    conc, separator, path = text.partition("=")
    if not separator or not conc.isdecimal() or int(conc) <= 0 or str(int(conc)) != conc:
        raise argparse.ArgumentTypeError(f"expected CONC=PATH with a positive CONC, got {text!r}")
    return int(conc), Path(path)


def main(argv: list[str] | None = None) -> None:
    """Write the identities of the ``--point`` recipes to ``--output``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--point", type=_point, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(argv)
    if len({conc for conc, _ in parsed.point}) != len(parsed.point):
        parser.error("each --point concurrency must be unique")
    try:
        recipes = {conc: yaml.safe_load(path.read_text()) for conc, path in parsed.point}
        document = identities(recipes)
    except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError, InputError) as error:
        parser.error(str(error))
    parsed.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
