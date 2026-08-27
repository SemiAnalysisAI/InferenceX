# AGENTS.md

Guidance for AI agents working with InferenceX.

## Start here

1. **Start every task with [`docs/index.md`](docs/index.md).** Choose the one focused guide that matches the task. Do not load every documentation page.
2. Repository source, schemas, workflows, launchers, and collectors are authoritative. If documentation disagrees with implementation, follow the implementation and update the nearest English guide plus its Chinese counterpart.
3. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening or reviewing a PR or changing review, sweep, or merge policy.
4. Read [`KLAUD_DEBUG.md`](KLAUD_DEBUG.md) before debugging a Klaud-Cold or `claude/*` image-bump PR.

## Agent-specific policy

- Repository skills are canonical under `.agents/skills/`. Add or update skills there. `.claude/skills/` contains compatibility symlinks for Claude discovery.
- PR and issue titles, descriptions, and human-authored PR comments must include English and natural Simplified Chinese. Keep code, commands, logs, stack traces, model names, hardware SKUs, framework names, flags, and identifiers unchanged. The exact CODEOWNER sign-off template is English-only. See [`docs/documentation-procedures.md`](docs/documentation-procedures.md) and [`.github/AGENT_OPERATIONS.md`](.github/AGENT_OPERATIONS.md#translation-terminology).
- Commit subjects use conventional English style, while commit bodies include the Chinese translation. Contributor-facing docs use English as the source version and ship with a synchronized `_zh.md` page and language switcher.
- Follow the nearest existing pattern. Python uses typed signatures and strict Pydantic schemas. YAML uses kebab-case fields. Shared benchmark Bash behavior belongs in `benchmark_lib.sh`, with parameters passed through environment variables.

## Non-negotiable benchmark invariants

- Every priority-scheduled benchmark job on a self-hosted cluster must request exactly one `nodes:N` label, where `N` is the positive integer number of physical Slurm nodes required. Single-node jobs use `nodes:1`; generated multi-node jobs must forward their computed `node-count`. A queued job missing this label is ineligible for priority scheduling, and labels cannot be added retroactively, so fix the source branch and dispatch a new run.
- Every change that can affect benchmark performance and every recipe addition or modification requires a new `perf-changelog.yaml` entry. The file is append-only and byte-sensitive. Preserve all existing bytes and separator whitespace, and append only at the tail.
- Multi-node srt-slurm changes update the recipe YAML and matching master config together. For image bumps, `model.container` must equal `image`.
- Every `*_mtp.sh` passes `--use-chat-template` to `run_benchmark_serving`.
- Benchmarks create no new directories under `/workspace`. Root containers must not leave root-owned files in shared AMD runner workspaces.
- Generated configuration is not runtime proof. Run the narrowest local check, then the applicable smoke, sweep, or eval procedure from [`docs/procedures.md`](docs/procedures.md).

All repository maps, task routes, commands, schemas, sweep semantics, artifact contracts, recovery steps, and detailed conventions live behind [`docs/index.md`](docs/index.md).
