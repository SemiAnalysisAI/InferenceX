# PySlurm 25.11.2 Patches for Slurm 25.05.x

These patches port pyslurm 25.11.2 (designed for Slurm 25.11) to build and run
against Slurm 25.05.x (any 25.05 point release: 25.05.1 through 25.05.7+).

## Patch series

### 0001-slurm-2505-pxi-declarations.patch

**Target**: Slurm 25.05.x (`SLURM_VERSION_NUMBER` 0x1905xx)

Modifies the Cython extern declarations in `pyslurm/slurm/*.pxi` to match the
Slurm 25.05 public C API. Key changes:

- **struct field renames**: Many structs gained a `slurm_step_id_t step_id`
  member in 25.11 that replaces the `uint32_t job_id` in 25.05 (`job_desc_msg_t`,
  `slurm_job_info_t`, `submit_response_msg_t`, etc.). This patch reverts those.
- **select_type_flags_t enum -> CR_\* defines**: 25.11 turned the old
  `CR_CPU`/`CR_CORE`/... preprocessor constants into a typed enum. Cython name
  aliases (`SELECT_CPU "CR_CPU"`) keep the `.pyx` code using `SELECT_*` while
  the C compiler resolves the `CR_*` defines.
- **Removed fields/constants**: `MAX_VAL*`, `JOB_EXPEDITING`,
  `SPREAD_SEGMENTS`, `DEBUG_FLAG_METRICS`, config fields added in 25.11
  (`namespace_plugin` -> `job_container_plugin`), etc.
- **Function signatures**: `slurm_get_job_steps`, `slurm_kill_job_step`,
  `slurm_signal_job_step`, `slurm_complete_job`, `slurm_terminate_job_step`
  changed from `slurm_step_id_t*` to separate `(job_id, step_id)` args.
- **Error codes**: 25.11-only error enums removed;
  `ESLURM_INVALID_NAMESPACE_CHANGE` reverted to `ESLURM_INVALID_JOB_CONTAINER_CHANGE`.
- **Internal struct** `job_id_msg_t` (from `extra.pxi`): `step_id` -> `job_id`.

### 0002-slurm-2505-pyx-code.patch

**Target**: Slurm 25.05.x (`SLURM_VERSION_NUMBER` 0x1905xx)

Updates the Cython `.pyx` implementation files to use the 25.05 struct layouts
and function signatures established by patch 0001:

- `submission.pyx`: `resp.step_id.job_id` -> `resp.job_id`
- `job.pyx`: `changes.ptr.step_id.job_id` -> `changes.ptr.job_id` (modify),
  `msg.step_id.job_id` -> `msg.job_id` (batch script RPC)
- `step.pyx`: `slurm_get_job_steps(&step_id, ...)` -> `slurm_get_job_steps(0, job_id, step_id, ...)`;
  `slurm_signal_job_step` / `slurm_kill_job_step` adjusted for separate args
- `config.pyx`: `self.ptr.namespace_plugin` -> `self.ptr.job_container_plugin`

## Applying

The build helper (`infx/runners/pyslurm_build.py`) applies these patches
automatically when it detects `SLURM_VERSION_NUMBER` in the 0x1905xx range.
To apply manually:

```bash
cd third_party/pyslurm
git apply patches/0001-slurm-2505-pxi-declarations.patch
git apply patches/0002-slurm-2505-pyx-code.patch
```

## What is unsupported on 25.05

- `pyslurm.Job.submit_line` — the `submit_line` field was added to
  `slurm_job_info_t` in 25.11. Accessing it on 25.05 would return `None`.
- `node_info_t.parameters` — not in 25.05.
- `reserve_info.allowed_parts` / `reserve_info.qos` — not in 25.05.
- `slurm_load_job_sluid` / `slurm_get_resource_layout` functions — 25.11 only.

## Porting notes for other Slurm versions

- **25.11.x**: No patches needed — pyslurm 25.11.2 targets this natively.
- **26.05.x**: Will need a new patch series; the `slurm_step_id_t` migration
  is complete there and additional struct changes are expected.
