# power_model

Repo-wide rules: [root `AGENTS.md`](../AGENTS.md).

Changes here must also update the InferenceX-app TypeScript port, which mirrors `src/power_model` file for file under `packages/app/src/lib/power/power_model/` and pins one upstream merge commit. A port fix that upstream also needs starts here. Every `power_model/` PR description lists all four steps:

1. Land this InferenceX PR first, with matching tests.
2. Open the app PR, cross-linked both ways, with matching tests.
3. In the app PR, pin this PR's merge commit.
4. In the app PR, regenerate the port manifest and the parity fixture.

Procedure: [InferenceX-app `AGENTS.md`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/master/AGENTS.md#power-model--ts-port-and-upstream-python-must-stay-in-sync).
