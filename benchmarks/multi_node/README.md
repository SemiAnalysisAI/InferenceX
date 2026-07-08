# Multi-node benchmark contract

<div align="center">

**English** | [中文](./README_zh.md)

</div>

Files under benchmarks/multi_node describe workloads. They must not select a
cluster, submit a scheduler job, map host storage, choose container privileges,
or infer fabric policy from hostnames. Those decisions belong under runners.

The workflow and master config provide workload intent:

- model and image: MODEL, IMAGE
- shape: ISL, OSL, CONC_LIST, RANDOM_RANGE_RATIO
- engine: FRAMEWORK, SPEC_DECODING
- topology: PREFILL_* and DECODE_*

The cluster runner resolves that intent to local resources and injects one
entrypoint:

    export MULTINODE_LAUNCHER="$GITHUB_WORKSPACE/runners/<cluster>/submit.sh"
    bash "benchmarks/multi_node/<recipe>.sh"

run_disaggregated.sh validates the workload contract, normalizes EP/DP and
node-list values, and calls the injected launcher. The launcher receives the
workload through exported environment variables; it must add cluster values
such as model paths, scheduler account/partition, device names, host mounts,
container runtime settings, and log locations.

## Adding a recipe

Most recipes are intentionally only a tail-call to disaggregated_recipe.sh.
Add exports before that call only for model or engine behavior that is valid on
every cluster. A host path, node name, Slurm flag, or Docker setting is runner
configuration, not recipe configuration.

## Adding a cluster

Create a launcher under runners/launch_<cluster>.sh and, when needed, a cluster
adapter under runners/<cluster>/. Declare public-model-to-local-storage aliases
and cache roots in `configs/runners.yaml`; consume them through
`runners/lib/runner_config.sh`. The launcher sets the remaining cluster policy.
The adapter launches the containers or scheduler job and arranges for the
portable runtime scripts to execute inside them.

The MI355X AMDS implementation in runners/mi355x-amds is the reference for a
Slurm-plus-Docker adapter. NVIDIA srt-slurm runners follow the same ownership
boundary using generated srtctl configuration.

## srt-slurm

All NVIDIA launchers use the repository, branch, and immutable main snapshot in
`configs/runners.yaml` through `runners/lib/srt_slurm.sh`. Launchers must not
clone or select their own srt-slurm branches. The helper overlays checked-in
InferenceX recipes, retrieves recipes that only exist in the locked legacy
snapshot as data, resolves recipe selectors, and converts `sa-bench` entries to
srt-slurm's custom benchmark interface. That custom command runs
`srt_benchmark.sh`, which calls InferenceX's in-tree benchmark client. Existing
custom benchmarks, such as agentic workloads, pass through unchanged.

Small compatibility patches that have not landed upstream live in
`runners/srt-slurm-patches/` and are applied to that exact snapshot with
`git apply --check`. They must not select another branch or revision.
