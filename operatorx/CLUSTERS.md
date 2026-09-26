# Clusters

Reference for the clusters operatorx targets: hardware, how the runner is
launched per platform, and the per-platform quirks. Access details
(hostnames, credentials, checkout paths) are deployment-specific and are not
included here. Fill them in for your own environment.

Platform → cluster routing lives in `operatorx/clusters.py` (`CLUSTER_PLATFORMS`).
SLURM/env defaults live in `scripts/submit_run.py` (`DEFAULT_CLUSTER`).

## Conventions

- `submit_run.py` is **SLURM-only**. It submits `sbatch`/`srun` jobs and is
  used on the NVIDIA and AMD clusters.
- Point `OPERATORX_SQUASH_DIR` at your container squash directory, and set
  `OPERATORX_PARTITION` / `OPERATORX_ACCOUNT` / `OPERATORX_QOS` to your
  cluster's SLURM values (see the env-var table at the bottom).
- Set `OPERATORX_JOB_NAME` to a recognizable SLURM job name for your runs.

## At a glance

| Cluster id    | Platform | Hardware                        | Scheduler  |
|---------------|----------|---------------------------------|------------|
| `b200_dgx_8x` | nvidia   | 8× B200 SXM (DGX-style)         | SLURM      |
| `b300_hgx_8x` | nvidia   | 8× B300 (HGX-style)             | SLURM      |
| `b200_nvl72`  | nvidia   | B200 NVL72 (routing only)       | SLURM      |
| `mi355x_8x`   | amd      | 8× MI355 OAM per node           | SLURM      |

## NVIDIA

### b200 — DGX-style, 8× B200 SXM (`b200_dgx_8x`)

| Setting  | Value                                                    |
|----------|----------------------------------------------------------|
| Hardware | 8× B200 SXM per node                                     |
| GRES     | e.g. `gpu:nvidia_b200:8`                                 |
| Defaults | `submit_run.py` defaults target this cluster (`OPERATORX_CLUSTER=b200_dgx_8x`, partition/squash dir from the script/env) |

```bash
OPERATORX_JOB_NAME=<job-name> python3 scripts/submit_run.py nvidia
```

### b300 — HGX-style, 8× B300 (`b300_hgx_8x`)

| Setting           | Value                                                        |
|-------------------|--------------------------------------------------------------|
| Hardware          | 8× B300 per node                                             |
| OS note           | If the login node runs Python 3.10, `submit_run.py` falls back to `tomli` for TOML parsing |

Override the SLURM knobs for your cluster:

```bash
OPERATORX_CLUSTER=b300_hgx_8x \
OPERATORX_PARTITION=<partition> \
OPERATORX_ACCOUNT=<account> \
OPERATORX_QOS=<qos> \
OPERATORX_SQUASH_DIR=<squash-dir> \
OPERATORX_BACKENDS=vllm \
OPERATORX_JOB_NAME=<job-name> \
python3 scripts/submit_run.py nvidia
```

### `b200_nvl72`

Present in `CLUSTER_PLATFORMS` (routes to nvidia) as a placeholder. There is no
hardware-specific guidance yet.

## AMD — `mi355x_8x`

| Setting   | Value                                                        |
|-----------|--------------------------------------------------------------|
| Hardware  | 8× MI355 OAM per node                                        |
| GRES      | e.g. `gpu:amd_instinct_mi355_oam:8`                          |
| Backends  | `containers.toml` `amd.torch` / `amd.vllm` → `vllm-openai-rocm` |

```bash
OPERATORX_CLUSTER=mi355x_8x \
OPERATORX_PARTITION=<partition> \
OPERATORX_SQUASH_DIR=<squash-dir> \
OPERATORX_JOB_NAME=<job-name> \
python3 scripts/submit_run.py amd
```

## Cluster id → platform (`operatorx/clusters.py`)

```python
CLUSTER_PLATFORMS = {
    "b200_dgx_8x": "nvidia",
    "b300_hgx_8x": "nvidia",
    "b200_nvl72":  "nvidia",
    "mi355x_8x":   "amd",
}
```

## `submit_run.py` env vars

| Var                    | Default                       | Notes                                                               |
|------------------------|-------------------------------|---------------------------------------------------------------------|
| `OPERATORX_CLUSTER`    | per-platform (see script)     | Routes the runner via `CLUSTER_PLATFORMS`                           |
| `OPERATORX_PARTITION`  | site default                  | SLURM partition                                                     |
| `OPERATORX_ACCOUNT`    | (omitted)                     | SLURM `--account`                                                   |
| `OPERATORX_QOS`        | (omitted)                     | SLURM `--qos`                                                       |
| `OPERATORX_SQUASH_DIR` | site default                  | Where `<safe_image>.sqsh` lives (used by `srun --container-image=`) |
| `OPERATORX_BACKENDS`   | all backends for the platform | CSV allowlist                                                       |
| `OPERATORX_JOB_NAME`   | `benchmark`                   | SLURM job name                                                      |
| `OPERATORX_TESTLISTS`  | all under `testlists/`        | CSV of testlist stems to run                                        |
| `OPERATORX_TIME_MIN`   | `30`                          | `--time` (minutes)                                                  |

`WORLD_SIZES = [1, 2, 4, 8]` supports single-node runs only. Multi-node NCCL IB bring-up
currently hangs on the B200/B300 fabrics. `MASTER_ADDR` is derived by parsing
`SLURM_NODELIST`.
