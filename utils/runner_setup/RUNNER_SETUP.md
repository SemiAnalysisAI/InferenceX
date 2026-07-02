# Self-hosted runner setup

Scripts for bulk-provisioning the self-hosted GitHub Actions runners that execute
InferenceX benchmark jobs on the GPU clusters.

- [`setup.sh`](setup.sh) — downloads the actions-runner tarball once and configures N runner
  instances in parallel under a base directory.
- [`start_runners.sh`](start_runners.sh) — starts the configured runners inside a tmux session,
  one tiled pane per runner.

> **This is the standard way the SemiAnalysis team sets up runners — it is not the only
> way.** This guide is written primarily for **Slurm clusters**, where the GHA runner
> listener processes all run on the **login/controller node** and each benchmark job is
> dispatched onto the **compute nodes** via `srun`. The same runners can also be brought
> up on a **bare-metal single node** (with Docker) or on **Kubernetes** — those setups
> differ enough that they aren't covered here yet (more docs to be added in future).

> **Runner setup is node-specific and varies a lot from cluster to cluster** —
> storage mounts, the runner user, weight-staging paths, container runtime, and Slurm
> defaults all differ. If you're an agent (or new team member) following this doc,
> **ask the user for clarification** whenever a step's specifics aren't obvious for the
> target cluster rather than guessing. The per-cluster choices are ultimately encoded in
> that cluster's `runners/launch_<cluster>.sh`.

## Viewing the current runners

The live list of registered runners (name, status, labels) is at:

> https://github.com/SemiAnalysisAI/InferenceX/settings/actions/runners

Or via the `gh` CLI / REST API:

```bash
gh api repos/SemiAnalysisAI/InferenceX/actions/runners \
  --jq '.runners[] | "\(.name)\t\(.status)\t\([.labels[].name] | join(","))"'
```

## GitHub access (token / permissions)

Provisioning runners hits authenticated GitHub endpoints (listing runners, minting the
registration token, removing runners), so you need a GitHub credential with sufficient
permissions. Provide it one of two ways:

- **`gh` CLI logged in** as a user with **admin access to the repo** (`gh auth login`),
  which `gh api` uses automatically; or
- a **personal access token (PAT)** exported as `GH_TOKEN` / `GITHUB_TOKEN` (or pasted
  into the CLI when prompted).

Required permissions (all of these endpoints require **admin access to the repository**):

| Token type | What it needs |
|------------|---------------|
| **Classic PAT** | `repo` scope (covers list runners, create registration/remove tokens, delete runners — all require repo admin) |
| **Fine-grained PAT** | Repository **Administration** permission — **Read** to list runners, **Read & write** to create the registration/remove token and to add/delete runners |

> If you're an agent, you will need the user to supply this credential (pasted in the CLI
> or via an env var) — you cannot create the runner registration token without it. Ask
> for it explicitly. The registration token produced from it expires after ~1 hour.

## Prerequisites

1. **Decide which user the GitHub Actions processes will run under.** This user's home
   directory must be on shared storage that is mounted on **all compute nodes** of the
   cluster — the runner work directories (`_work`) hold checkouts, logs, and artifacts
   that Slurm jobs on compute nodes read and write.
2. The host needs `curl`, `tar`, and `tmux`. Run the setup from a normal login shell so
   that the runner captures a sane `PATH` (including the Slurm binaries — `sinfo`,
   `srun`, `sbatch`); the runner snapshots `PATH` into `.path` at configuration time.

## Setup

1. Under the chosen user's home directory, create the runner base directory:

   ```bash
   mkdir -p ~/gharunners && cd ~/gharunners
   ```

2. Clone InferenceX (or just copy the two scripts in this directory onto the host):

   ```bash
   git clone https://github.com/SemiAnalysisAI/InferenceX.git
   ```

3. Navigate to
   [github.com/SemiAnalysisAI/InferenceX/settings/actions/runners/new?arch=x64&os=linux](https://github.com/SemiAnalysisAI/InferenceX/settings/actions/runners/new?arch=x64&os=linux)
   to fetch the **registration token** and **runner tarball URL**, which are inputs to
   `setup.sh`:

   ![Where to find the runner URL and token](assets/new-runner-page.png)

   > Note: the registration token expires after ~1 hour. If `config.sh` starts failing with
   > authentication errors partway through, refresh the page and re-run with a new token.

4. Configure the runners:

   ```bash
   ./InferenceX/utils/runner_setup/setup.sh \
     <TOKEN> \
     <RUNNER_URL> \
     <START_INDEX> <END_INDEX> \
     ~/gharunners \
     <BASE_RUNNER_NAME> \
     <ADDITIONAL_RUNNER_TAGS>
   ```

   Example — 14 runners (`b300-nv_00` … `b300-nv_13`) on the B300 cluster:

   ```bash
   ./InferenceX/utils/runner_setup/setup.sh \
     AOPHAHI... \
     https://github.com/actions/runner/releases/download/v2.335.1/actions-runner-linux-x64-2.335.1.tar.gz \
     0 13 \
     ~/gharunners \
     b300-nv \
     slurm,b300
   ```

   This creates `gharunner00/actions-runner` … `gharunner13/actions-runner` under the
   base directory, all sharing one downloaded tarball.

5. Start the runners:

   ```bash
   ./InferenceX/utils/runner_setup/start_runners.sh 0 13 ~/gharunners
   ```

   This (re)creates a tmux session (default name: `github-actions`) with one tiled pane
   per runner running `./run.sh`. Reattach later with `tmux attach -t github-actions`.

6. Verify the runners show up as **Idle** on the
   [runners settings page](https://github.com/SemiAnalysisAI/InferenceX/settings/actions/runners),
   then register them in the repo config (see below).

## Naming convention — read this before picking `BASE_RUNNER_NAME`

Runner names are **load-bearing**. Each runner is named `<BASE_RUNNER_NAME>_<NN>`
(zero-padded two-digit index), e.g. `b300-nv_07`, and two pieces of CI infrastructure
key off that name:

1. **The launch script is selected from the name prefix.** The benchmark workflows run

   ```bash
   bash ./runners/launch_${RUNNER_NAME%%_*}.sh
   ```

   so everything before the first `_` must match an existing script in
   [`runners/`](../../runners) — e.g. runner `b300-nv_07` -> `runners/launch_b300-nv.sh`.
   For a brand-new cluster, add a `runners/launch_<BASE_RUNNER_NAME>.sh` first.
   Corollary: `BASE_RUNNER_NAME` itself must not contain `_` (use hyphens).

2. **Sweep scheduling looks runner nodes up by label.** Jobs are distributed across the
   runner names listed under each `labels` entry in
   [`configs/runners.yaml`](../../configs/runners.yaml). New runners do
   **not** receive sweep jobs until they are added there, and the entries must match the
   registered names exactly — including zero-padding. (Some older fleets predate the
   padded convention, e.g. `h200-dgxc-slurm_0`; `setup.sh` always zero-pads, so new
   entries should use the padded form.)

## Labels / `ADDITIONAL_RUNNER_TAGS`

`setup.sh` registers each runner with labels `<ADDITIONAL_RUNNER_TAGS>,<RUNNER_NAME>`
(on top of the implicit `self-hosted`, `Linux`, `X64`). Conventions in use:

- `slurm` — the runner submits work through Slurm.
- The SKU name (`b200`, `b300`, `h200`, `gb300`, …) — coarse hardware targeting.
- Exactly one `cluster:<name>` label, e.g. `cluster:b200-dgxc` — required exact
  hardware/fleet identity for success-rate reporting and hardware-specific config.
  Every runner in the same physical cluster with identical hardware should use the same
  cluster label.
- Optional capacity tags, e.g. `b200-dsv4`, `b300-p1` — used to carve out dedicated
  benchmark subsets.

The per-runner name label (`b300-nv_07`) is what `runs-on` resolves for sweep jobs, so
always keep it (the script appends it automatically). A typical registered runner ends
up with labels like:

```
self-hosted, Linux, X64, slurm, b200, b200-dsv4, cluster:b200-dgxc, b200-dgxc_00
```

Labels can be edited later on the runners settings page without re-registering.

## Storage layout

The login node (where the runners live) and the Slurm compute nodes (where benchmarks
run) exchange everything through the filesystem, so every path the CI touches must be
visible from the compute node that the job lands on. That means each path must either
live on **shared storage**, or **exist identically on every compute node** (e.g. local
NVMe at the same mount point). Four classes of paths to set up per cluster — the host
side of each is defined in that cluster's `runners/launch_<cluster>.sh`:

1. **Runner home / `_work` directories** — must be shared storage (see Prerequisites).
   The job checkout, scripts, and result artifacts live here and are bind-mounted into
   the benchmark container (`$GITHUB_WORKSPACE`).
2. **HF hub cache** — the workflows set the *container-side* path globally
   (`HF_HUB_CACHE=/mnt/hf_hub_cache/` in `benchmark-tmpl.yml`); each launch script
   bind-mounts a per-cluster *host* path `HF_HUB_CACHE_MOUNT` over it. Examples in use:
   `/mnt/nfs/sa-shared/gharunners/hf-hub-cache/` (h100, shared NFS),
   `/mnt/vast/gharunner/hf-hub-cache` (CoreWeave, shared VAST),
   `/tmp/gharunner/hf-hub-cache` (b200-cw, node-local — same path on every node, but
   each node downloads its own copy, so prefer shared storage where available).
3. **Pre-staged model weights** — large models are not downloaded from HF in CI. The
   launch scripts override `MODEL_PATH` to per-cluster staging directories
   (e.g. `/lustre/fsw/models/...` on b200-dgxc, `/data/models/...` on b300,
   read-only `/scratch/models/` on b300 multinode). Bringing up a new model on a
   cluster means staging the weights there first.
4. **Squash images** — launch scripts `enroot import` each Docker image once into a
   `.sqsh` file under a shared `SQUASH_DIR` (e.g. `/home/sa-shared/containers` on
   b200-dgxc, `/mnt/lustre01/users-public/sa-shared`
   on gb200), then launch with `--container-image=<file>.sqsh`. This must be on shared
   storage because pyxis reads the file on the **compute** node, and it lets concurrent
   jobs reuse one import instead of each pulling the registry image. Note
   `ENROOT_CACHE_PATH` (import scratch space) defaults under `$HOME/.cache/enroot`.

Size accordingly: weights run hundreds of GB to TB per model, `.sqsh` files are
~20–40 GB each and accumulate one per image tag (clean old tags periodically), and the
HF cache grows with datasets/tokenizers.

When provisioning a **new cluster**, decide these locations up front and encode them in
the new `runners/launch_<cluster>.sh`.

## Gotchas

- **Runners do not survive reboots.** They run via `./run.sh` in tmux, not as a systemd
  service — after node maintenance, re-run `start_runners.sh` (it kills and recreates
  the session, which is safe for idle runners).
- **Large fleets vs. tmux panes:** `start_runners.sh` puts every runner in one tiled
  window; with ~15+ runners panes can get too small and `split-window` may fail with
  `no space for new pane`. Split into ranges across multiple sessions via the optional
  `SESSION_NAME` argument.
- **Removing runners:** from the runner directory, stop the process and run
  `./config.sh remove --token <removal-token>` (token from the runners settings page).
  Remember to also delete the name from the matching `labels` entry in `configs/runners.yaml`.

## Record the cluster in the team canvas (SemiAnalysis only)

There is an internal **InferenceX Clusters** Slack canvas that tracks every cluster's
hardware, node count, login address, runner user, runner directory, and per-node host RAM.
The link is intentionally not stored in this repo — **if you are a SemiAnalysis employee,
ask the user for the Slack link to the InferenceX Clusters canvas.**

**If you have authenticated access to that canvas, add the corresponding cluster
information to it** after provisioning —
a new row in the Clusters table (and the Host-RAM table), plus any access notes (jumpbox,
non-Slurm/bare-metal, Tailscale, etc.). Keep the canvas consistent with
[`configs/runners.yaml`](../../configs/runners.yaml) and the live
[runners settings page](https://github.com/SemiAnalysisAI/InferenceX/settings/actions/runners),
which remain the sources of truth.

- If you **do not** have access to that canvas, ignore this step.
- If you **don't have a Slack integration available**, or you're otherwise **unsure
  whether you have access**, **confirm with the user** before attempting it — don't guess.

> Note for agents: editing this canvas via the Slack `update_canvas` tool has a data-loss
> footgun — replacing a table section leaves a stray empty table, and replacing a
> non-header section can swallow trailing content. Prefer a full-document replace
> (reconstructed from a fresh read, omitting the leading `# InferenceX Clusters` H1) and
> re-read the canvas afterward to verify.
