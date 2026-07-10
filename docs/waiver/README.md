# Engine Patch Waivers

<div align="center">

**English** | [中文](./README_zh.md)

</div>

InferenceX benchmarks what the community can actually pull and run. **PRs must not patch the inference engine or serving stack.** The pinned image must run as shipped — no modifications of any kind, including (non-exhaustively):

- `.patch` files, `git apply`, or `patch` invocations
- **inline patches embedded in benchmark scripts** — e.g. a `python3 - <<EOF` / `sed` / `cat > file` heredoc in a `benchmarks/**` `.sh` script that rewrites installed engine sources before `vllm serve` / SGLang launch
- `sed`/`awk`/in-place edits of installed engine sources (e.g. anything under `site-packages`)
- Python monkey-patching injected via env hooks, sitecustomize, or copied-in files
- overwriting or shadowing files inside the container image
- `pip install` of forked or rebuilt engine wheels on top of the pinned image
- rebuilding the engine image from a modified source tree

Installing the benchmark harness and its client-side dependencies (e.g. aiperf, eval tooling) is fine — this rule covers the **serving** stack that produces the numbers.

**The only exception is a patch covered by a filled-out `WAIVER.md` in this folder.**

## When is a waiver acceptable?

A waiver is a temporary bridge, not a loophole. Legitimate cases look like:

- an upstream fix is already merged but not yet in any released image, and the benchmark cannot run at all without it
- new hardware or new model architecture bring-up where upstream support is actively landing

A waiver is **not** acceptable for performance optimizations that upstream has not accepted — if a patch makes the benchmark faster, the community cannot reproduce that number from the released image, which defeats the point of InferenceX.

## How to file a waiver

1. In the **same PR** that introduces the patch, create `docs/waiver/<slug>/WAIVER.md`, where `<slug>` identifies the patch (e.g. `2026-07-glm5-gb200-sglang-moe-kernel`).
2. Copy the template below and fill out **every** field, in English.
3. Link the waiver in the "Additional detail section" of the CODEOWNER sign-off — the sign-off verifier ([codeowner-signoff-verify.yml](../../.github/workflows/codeowner-signoff-verify.yml)) independently checks that any patching in a PR is covered by a waiver.
4. A core maintainer must explicitly approve the waiver in a PR comment; link that comment in the waiver.
5. When the upstream fix ships, remove the patch and delete the waiver folder in the same PR.

## Template (keep in English verbatim)

```markdown
# WAIVER: <one-line description of the patch>

- **PR:** <link to the InferenceX PR that introduces the patch>
- **Date filed:** <YYYY-MM-DD>
- **Filed by:** @<github-username>
- **Configs affected:** <model / precision / SKU / framework, and the master-config entries touched>
- **What is patched:** <exact files and where the patch is applied (script line, Dockerfile step); include the patch contents or a link to it>
- **Why the patch is required:** <why the unmodified upstream image cannot run this benchmark>
- **Upstream status:** <link to the upstream PR/issue that removes the need for this patch>
- **Removal plan:** <which upstream release or condition lets us drop the patch, and the expected date>
- **Performance impact:** <does the patch change performance vs. the unpatched upstream image? link the evals run>
- **Core maintainer approval:** @<maintainer> — <link to the approving PR comment>
```

## Active waivers

None.
