# SRT streaming smoke

**English** | [中文](README_zh.md)

Tests NVIDIA/srt-slurm#539 on B300 without loading a model. The services-only
SRT job emits stdout/stderr for 90 seconds while Tachometer captures real host
process metrics. It uses one node, with a ten-minute Slurm limit.

The workflow builds Tachometer from the exact SRT commit so the uploader uses
that commit's atomic Arrow writer. It does not download the released Tachometer
binary. Binary checksums and the source commit are saved with the run artifacts.

After deploying the Dash collector, add `srt-streaming-test` to this PR. The build
runs on a GitHub-hosted machine and the smoke uses the B300 scheduler. Remove the
label before unrelated pushes; labeled pushes run the smoke again. The existing
`SRT_STATUS_ENDPOINT` and `SRTCTL_STATUS_TOKEN` repository secrets provide the
shared collector endpoint and bearer token.

Open `/srt` on InferenceX Dash and select the `b300-dsxe` job ID printed by the
workflow. Check the live log markers during the run and captured Tachometer data
afterward. The artifact checks local output; successful delivery must also be
confirmed in Dash. No benchmark results are published.
