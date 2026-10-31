---
name: serving-runner-setup
description: Bring a VNG Slurm GPU node into the InferenceX runner fleet the standard way (on-node listeners as stackops under systemd, one cluster record per node in configs/runners.yaml), or check, move or retire existing VNG runners. Use when the user says "setup runner", "thêm node vào InferenceX", "đăng ký runner", "runner offline", "chuyển runner sang stackops", or a serving job lands on the wrong node or cannot start a container.
---

# serving-runner-setup

A thin layer over upstream `inferencex-e2e/utils/runner_setup/RUNNER_SETUP.md`: read that first. VNG differs in one way: there is no shared filesystem big enough, so each runner lives **on its GPU node** and each node is **one cluster** pinned by `salloc-args --nodelist`.

Every step that registers, removes or restarts a runner, or changes a node, needs the user's explicit OK first. Nodes are shared with the team: never `--exclusive`, never cancel someone else's job.

## Convention (all pools)

1. User `stackops`, uid/gid 1001 on every node (same as the existing hgx/han nodes; Slurm shows it as `1001`). Its home holds `~/.config/enroot/.credentials` (VCR pull).
2. Cluster id `<site>-<node>`: site `hcm` or `han` (`hcm-hgx02`, `hcm-rtx01`, `han-h200-01`). Runner name `<cluster-id>_<NN>` (`hcm-hgx02_00`), exactly one `cluster:<cluster-id>` label. A new site needs its prefix in `VNG_CLUSTER_PREFIXES` (`infx/serving/schema.py`).
3. Directory `<large-local-disk>/gharunners/<cluster-id>/gharunnerNN/` (`/mnt/gharunners/hcm-hgx02/...` on hgx, `/data/gharunners/hcm-rtx01/...` on rtx, `/mnt/gharunners/han-h200-01/...` on HAN). The per-cluster folder keeps new slots away from any old `gharunnerNN` directory on the same disk.
4. GitHub labels `slurm,<sku>,<pool>,cluster:<cluster-id>` (pool = `h200-hcm`, `rtx6000pro-hcm` or `h200-han`). `benchmark-tmpl.yml` puts the serving YAML's `runner:` straight into `runs-on`, so every value a YAML may use (pool, cluster label, slot name) must be a real label on the runner.
5. systemd: `sudo ./svc.sh install stackops && sudo ./svc.sh start` in each runner directory. No tmux, no hand-run `run.sh`.
6. Slots per node = maximum concurrent jobs (hgx 4, rtx 2, HAN 1).

## 1. Preflight the node

Run from `slurm-client` as yourself; each probe is a 1-CPU job of a few seconds:

```bash
NODE=hgx-h200-02; PART=hardware-hcm
srun -p $PART -w $NODE -n1 -c1 --mem=512M -t 2 bash -c '
  hostname; getent passwd stackops || echo "MISSING passwd entry for stackops"
  grep -v "^#" /etc/enroot/enroot.conf | grep -v "^$"
  df -h / /mnt /data 2>/dev/null; nvidia-smi -L | wc -l'
srun -p $PART -w $NODE -n1 -c1 --mem=512M -t 5 --container-image=ubuntu:24.04 true && echo "pyxis OK"
scontrol show node $NODE | grep -E "NodeHostName|Gres|RealMemory|CPUTot"
```

Pass criteria:
- `stackops` exists.
- Pyxis starts a container.
- The enroot paths are on a disk with > 100 GB free.
- The GPU count in `Gres` equals `nvidia-smi -L`.
- `NodeHostName` matches the node name.

If `stackops` is missing, create it as root on the node (step 1b). Fix anything else on the infra side before registering. Known state (2026-10-25):
- HAN `test` partition, 1xH200 / 16 CPU / ~120 GB RAM per node, no `/data` disk on han-1/2 (caches on `/mnt`, which is on `/`):
  - `han-1`, `han-2`: healthy (pyxis OK, ~300 GB free), no `stackops` yet. Clusters `han-h200-01`, `han-h200-02`.
  - `han-4`: `/data` cleaned to 73% (267G free, 2026-10-25), `stackops` exists. Cluster `han-h200-04`, runner in `/data/gharunners/han-h200-04`, squash `/data/containers`, HF cache the existing `/data/hf-cache`. `/` (enroot runtime) has only 48G free.
  - `han-3-1`: `/` at 99% with the enroot paths on it. Out.
  - `han-3`: slurmd name mismatch (`srun`: Invalid node name). Out.
- `hgx-h200-01`: gres declares 4 of 8 GPUs.
- `/shared` NFS at 96%.

## 1b. Create `stackops` (root, only if preflight says MISSING)

```bash
groupadd -g 1001 stackops && useradd -u 1001 -g 1001 -m -s /bin/bash stackops
install -d -o stackops -g stackops -m 700 ~stackops/.config/enroot
# copy ~stackops/.config/enroot/.credentials from a working node (VCR pull), then:
chown stackops:stackops ~stackops/.config/enroot/.credentials && chmod 600 ~stackops/.config/enroot/.credentials
install -d -o stackops -g stackops /mnt/gharunners /mnt/hf_hub_cache /mnt/containers   # HAN/hgx; rtx uses /data
```

## 2. Register the listeners

As `stackops` on the node, with a registration token from https://github.com/vngcloud/InferenceX/settings/actions/runners/new:

```bash
ID=hcm-hgx02; BASE=/mnt/gharunners/$ID          # rtx: ID=hcm-rtx01; BASE=/data/gharunners/$ID   HAN: ID=han-h200-01
mkdir -p $BASE && cd $BASE
curl -fsSLO https://raw.githubusercontent.com/vngcloud/InferenceX/refactor/vng-benchmark/inferencex-e2e/utils/runner_setup/setup.sh
bash setup.sh <TOKEN> https://github.com/actions/runner/releases/download/v2.338.1/actions-runner-linux-x64-2.338.1.tar.gz \
  0 3 $BASE $ID slurm,h200,h200-hcm,cluster:$ID https://github.com/vngcloud/InferenceX   # rtx: 0 1 ... slurm,rtx6000pro,rtx6000pro-hcm,cluster:$ID   HAN: 0 0 ... slurm,h200,h200-han,cluster:$ID
for d in $BASE/gharunner0*/actions-runner; do (cd "$d" && sudo ./svc.sh install stackops && sudo ./svc.sh start); done
```

The registration token comes from the settings page or `gh api -X POST repos/vngcloud/InferenceX/actions/runners/registration-token --jq .token` (valid exactly 1 hour, not configurable; mint it right before running `setup.sh`). `setup.sh` prints `All runners configured!` even when registration failed: check that every slot printed its own `[<name>] Done.` line.

Every `datasets/lock.yaml` entry marked `private: true` (GreenSpec, SPEED-Bench) needs `GREENLAB_HF_TOKEN`; the other repos are public. `infx/serving/datasets.py` downloads anonymously when `GREENLAB_HF_TOKEN` is unset, so a runner that is to run `scenario: greenspec` or `scenario: speedbench` needs `GREENLAB_HF_TOKEN=…` in `actions-runner/.env`. Other scenarios need no `.env`.

## 3. Add the cluster record

In `inferencex-e2e/configs/runners.yaml`, add the `cluster:<id>` label with the slot names, add the slots to the pool label (`h200-hcm` / `rtx6000pro-hcm` / `h200-han`), and copy the nearest VNG record under `clusters:`. Change:
- the `--nodelist` in `salloc-args`;
- `gpus-per-node`, from `Gres`;
- `available-cpu-dram-mib`, from `RealMemory`;
- `--mem-per-gpu`: about 95% of `RealMemory` / GPUs, in M;
- the volume and squash paths;
- `gres`: copy the form `scontrol show node` prints (`gpu:nvidia_h200:{gpus}` on hgx, plain `gpu:{gpus}` on HAN).

Keep `exclusive: false` and `--cpus-per-gpu=16`. Run `PYTEST infx/tests/serving/test_clusters.py` (extend its `NODES` table) and commit.

## 4. Smoke

Dispatch an `agentx` + `overrides: {duration: 180}` serving YAML on `cluster:<id>` with the `serving-bench` skill. Watch it get scheduled with `squeue -w <node>`.

## Move or retire a runner

- **Move to `stackops`** (for example the RTX runners that ran as `thanglq5` in `/data/thanglq5/runner-rtx0N`):
  1. Stop the process.
  2. `./config.sh remove --token <removal-token>`.
  3. Register again under the convention.
  4. Move the HF cache and squash files: `mv /data/thanglq5/hf_hub_cache /data/hf_hub_cache`, `mv /data/thanglq5/containers /data/containers`, then `chown -R stackops`.
- **Retire**:
  1. `sudo ./svc.sh stop && sudo ./svc.sh uninstall`.
  2. `./config.sh remove --token <removal-token>`.
  3. Delete the slot from `configs/runners.yaml`.
