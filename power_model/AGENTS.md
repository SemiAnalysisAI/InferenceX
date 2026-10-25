# power_model

Repo-wide rules: [root `AGENTS.md`](../AGENTS.md).

- Changes here update the InferenceX-app TypeScript port with matching tests and cross-linked PRs. The app mirrors `src/power_model` file for file under `packages/app/src/lib/power/power_model/` and pins one upstream merge commit: land the InferenceX PR first, then the app PR pins that commit and regenerates its manifest and parity fixture. A port fix that upstream also needs starts with the InferenceX PR. Procedure: [InferenceX-app `AGENTS.md`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/master/AGENTS.md#power-model--ts-port-and-upstream-python-must-stay-in-sync).
