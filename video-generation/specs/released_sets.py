import importlib.util
import io
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

SPECS = "video-generation/specs"
# The check starts at this commit on main, the last to touch the specs before the rule.
RULE_START = "283fb864d08613b04c51ab581816089b0d3954c3"


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


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, stdout=subprocess.PIPE).stdout


def _released_plans_at(commit: str, *, must_load: bool) -> dict[tuple[str, str], str]:
    """``commit``'s released plans, or none when its specs do not load and it may be skipped."""
    with tempfile.TemporaryDirectory() as tree:
        with tarfile.open(fileobj=io.BytesIO(_git("archive", commit, SPECS))) as archive:
            archive.extractall(tree, filter="data")
        try:
            return released_plans(tree)
        except Exception as error:
            if must_load:
                raise
            print(f"skipped {commit[:9]}, whose specs do not load: {error!r}", file=sys.stderr)
            return {}


def main(argv: list[str]) -> None:
    """Exit nonzero unless HEAD holds every released set as its history first committed it.

    Reads ``argv[1]`` (default ``RULE_START``) and each later first-parent commit of HEAD that
    touches the specs, so a set that was added and then edited fails until the edit is undone.
    An earlier commit whose specs do not load is skipped: this check fails there, so it
    published nothing.
    """
    start = argv[1] if len(argv) > 1 else RULE_START
    if subprocess.run(["git", "merge-base", "--is-ancestor", start, "HEAD"]).returncode:
        sys.exit(f"HEAD does not descend from {start}; the check needs main's full history")
    log = _git("rev-list", "--first-parent", "--reverse", f"{start}..HEAD", "--", SPECS)
    commits = [start, *log.decode().split()]
    plans = {
        commit: _released_plans_at(commit, must_load=commit == commits[-1]) for commit in commits
    }
    head = plans[commits[-1]]  # No later commit touches the specs, so these are HEAD's plans.
    first: dict[tuple[str, str], tuple[str, str]] = {}
    for commit in commits:
        for key, plan in plans[commit].items():
            first.setdefault(key, (plan, commit))
    violations = [
        f"{spec} {set_id}: {'plan changed' if (spec, set_id) in head else 'removed'}"
        f" since {commit[:9]}"
        for (spec, set_id), (plan, commit) in first.items()
        if head.get((spec, set_id)) != plan
    ]
    if violations:
        rule = (
            "a released set never changes; add the next release-v<N>, or a new spec file when"
            " the model, generation or fairness changes (video-generation/specs/README.md):"
        )
        sys.exit("\n".join([rule, *violations]))


if __name__ == "__main__":
    main(sys.argv)
