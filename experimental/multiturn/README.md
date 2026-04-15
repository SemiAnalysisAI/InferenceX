# Experimental multiturn notes

This directory contains working notes, investigations, and planning material for multiturn and long-context benchmarking.

## Official ISB1 replay status lives elsewhere

Do **not** treat this directory as the source of truth for the currently supported InferenceX ISB1 surface.

For the official, reviewable statement of what is landed now, use:
- `datasets/isb1/SUPPORT_MATRIX.md`
- `datasets/isb1/README.md`
- `.github/configs/isb1-master.yaml`

## Relevant roadmap docs

- `ISB1_MULTITURN_LONG_CONTEXT_CANONICAL_SYNTHESIS_2026-04-09.md` — canonical synthesis for next implementation phases; use this first for planning context.
- `ISB1_INFERENCEX_PHASED_PR_ROADMAP_2026-04-09.md` — phased landing plan used to split schema/workflow/data/extension/polish work into mergeable stages.

## Scope warning

Files in this directory may discuss future or experimental directions such as:
- KV offload investigations
- synthetic multiturn ideas
- broader long-context expansion
- experiments outside the currently merged official replay lane

Those notes are useful for planning, but they are **not** themselves an official support claim.
