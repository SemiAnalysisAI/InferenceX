# infx

Repo-wide rules: [root `AGENTS.md`](../../AGENTS.md).

## Style

- Python uses typed signatures and strict Pydantic schemas.
- All stable Ruff rules apply (exclusions in `ruff.toml`), line length 100, and the Ruff formatter; CI's Lint job fails on any finding. Run the [local checks](#local-checks) before pushing Python changes and fix findings where practical; justified exceptions use inline `# noqa: CODE`, never file-wide ignores.
- Shared benchmark behavior that runs inside serving containers belongs in `bench/` as stdlib-only, Python 3.10 compatible commands (`python3 -m infx.bench <command>`) configured by env vars or flags. Bash entrypoints stay thin shims.
- Never hard-code synthetic acceptance lengths; the srt driver picks measured values from `golden_al_distribution/` ([details](golden_al_distribution/README.md#using-golden-curves-in-srt-slurm-runs)).

## Tests

**A test must run the real implementation on concrete inputs and assert on what it computes, returns, writes, or raises.** Tests that read source text, parse code structure, grep the repo, pin checked-in config, are tautological, reimplement the code, test fixtures, are smoke-only, or duplicate a covered branch must be deleted. Before adding or approving a test, answer:

1. What line does it run, and what bug would make it fail?
2. Would it still pass if the code were rewritten with identical behavior?
3. Would it fail because someone added a recipe, bumped an image tag, or reworded a comment?
4. Does an existing test already reach this branch?

Full rules: [test quality](../../CONTRIBUTING.md#test-quality).

## Local checks

Run from `inferencex-e2e/`:

```bash
uv sync --locked --all-extras --group test
uvx --exclude-newer PT12H ruff@latest check --fix infx
uvx --exclude-newer PT12H ruff@latest format infx
uv run --locked --all-extras --group test python -m pytest -c pyproject.toml infx/tests/ -n 4
```

Run the focused test path first. Use `--extra workflows` for GitHub integrations and `--extra results` for result tooling. Dependency changes update `pyproject.toml` and `uv.lock` via `uv add` or `uv lock --upgrade-package NAME`. Schema and generator checks do not prove serving behavior.
