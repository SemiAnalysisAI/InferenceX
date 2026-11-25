import importlib.util
import sys
from pathlib import Path

SPECS = "video-generation/specs"


def released_plans(tree: str | Path) -> dict[tuple[str, str], str]:
    """Each ``release-v*`` set's plan hash under ``tree``, computed by ``tree``'s own loader."""
    specs = Path(tree) / SPECS
    module_spec = importlib.util.spec_from_file_location("video_spec", specs / "video_spec.py")
    loader = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(loader)
    plans = {}
    for path in sorted(specs.iterdir()):
        if path.suffix in {".yaml", ".yml"}:
            spec = loader.load_spec(path)
            for set_id in spec["sets"]:
                if set_id.startswith("release-v"):
                    plans[path.name, set_id] = loader.plan_sha256(loader.plan_for(spec, set_id))
    return plans


def main(argv: list[str]) -> None:
    """Exit nonzero when the second tree changes or drops a released set of the first."""
    before, after = (released_plans(tree) for tree in argv[1:])
    violations = [
        f"{spec} {set_id}: " + ("plan changed" if (spec, set_id) in after else "removed")
        for (spec, set_id), plan in before.items()
        if after.get((spec, set_id)) != plan
    ]
    if violations:
        rule = "a released set never changes; add the next release-v<N> (STANDARD.md rule 1):"
        sys.exit("\n".join([rule, *violations]))


if __name__ == "__main__":
    main(sys.argv)
