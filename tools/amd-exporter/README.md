# AMD power exporter image

**English** | [中文](README_zh.md)

The CPU-only `AMD exporter image` workflow builds a fresh-sample exporter and publishes a unique tag to `ghcr.io/semianalysisai/amd-device-metrics-exporter`. It does not start a GPU benchmark or change production data.

The base is the official ROCm nightly artifact from run `36867107435`, source `9d0eb8c88af99f1dbe2914d803382092671456d9`. Its archive checksum and image config digest are verified before use. The runtime includes the upstream 255 W sentinel correction. `cache.patch` applies the configurable GPUGet cache change from [functionstackx/device-metrics-exporter#1](https://github.com/functionstackx/device-metrics-exporter/pull/1), with a default-cache regression and concurrent zero-cache tests.

Only `/home/amd/bin/server` is replaced. The original ROCm and GPUAgent layers and entrypoint are retained. `AMD_GPU_GET_CACHE_TTL=0s` forces each exporter request to issue a new GPUGet RPC. This removes exporter response reuse; actual hardware freshness, overhead and one-second benchmark collection still require live validation.

The workflow verifies the patch checksum, runs focused cache race tests, builds Linux/amd64, checks the image's version and invalid-setting rejection, and verifies the unchanged base layers and runtime configuration. It then publishes a commit-specific tag and saves an Enroot squash with `SHA256SUMS` and `provenance.json` in `amd-dme-cache-squash`. Consumers must record the published registry digest and the squash checksum. The source commit, patch hash, GPUAgent commit and workflow revision are separate provenance fields; the embedded ROCm commit is explicitly unknown.

Dispatch this workflow manually on the reviewed branch, or push the dedicated `chore/powerx-amd-exporter-image` branch. The job has a 30-minute CPU limit and needs `packages: write`. It does not move a `latest` tag. A successful image build is preparation, not AMD measurement acceptance.
