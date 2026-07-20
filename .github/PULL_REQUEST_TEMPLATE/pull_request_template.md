## Description

<!-- Provide a brief description of your changes -->

## Related Issue

<!-- Link to related issue(s) if applicable -->
Fixes #

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Configuration change
- [ ] Documentation update
- [ ] Other (please describe)

## Checklist

- [ ] I have tested my changes locally
- [ ] I have updated documentation if necessary
- [ ] **If I changed a container image or config, I have already updated `perf-changelog.yaml`**
  - [ ] New `perf-changelog.yaml` entries are **appended to the end** of the file (the file is chronological: oldest at top, newest at bottom)
- [ ] **Before merging via reuse, an authorized maintainer (`OWNER`/`MEMBER`/`COLLABORATOR`) has commented `/reuse-sweep-run` on this PR** — do this **only once there is a final full sweep that is all green with evals passing**, since after this comment the sweep label will no longer automatically kick off new sweeps (remove and re-add the label to force one)
