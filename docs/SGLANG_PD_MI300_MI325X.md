# SGLang prefill–decode (PD) disaggregation on AMD MI300X / MI325X

This guide applies to **AMD MI300X** and **MI325X** (and similar CDNA3-class ROCm deployments) running **SGLang disaggregated inference** (prefill and decode on separate GPU nodes) via the **InferenceX** **SSH + Docker** launchers—no Slurm required.

The launcher scripts are `run_1p1d_sglang_mi300_mi325x.sh` and `run_1p2d_sglang_mi300_mi325x.sh` (optional wrapper `start_sglang_pd_mi300_mi325x.sh`). You choose the **container image** (`IMAGE`), **GPU count**, and **host drivers** (e.g. Broadcom `bnxt_re` / libbnxt tarball) to match your hardware.

For the broader InferenceX project (benchmarking across NVIDIA/AMD, CI, and the public dashboard), see the [repository README](../README.md).

---

## 1. What you are running

| Topology | Script | Nodes | Roles |
|----------|--------|-------|--------|
| **1 prefill + 1 decode** | `run_1p1d_sglang_mi300_mi325x.sh` | 2 | Rank 0: prefill, router, benchmark client. Rank 1: decode. |
| **1 prefill + 2 decode** | `run_1p2d_sglang_mi300_mi325x.sh` | 3 | Rank 0: prefill + router + benchmark. Ranks 1–2: decode workers. |

**IP order matters.** The cluster addresses passed to the engine are:

- **1P+1D:** `IPADDRS = prefill_ip,decode_ip` (prefill must be first).
- **1P+2D:** `IPADDRS = prefill_ip,decode1_ip,decode2_ip` (decode indices must align with `benchmarks/multi_node/amd_utils/server.sh`).

The launchers resolve data-plane IPs by SSH’ing each host and running `ip route get 1.1.1.1` (unless you set the IPs explicitly—see below).

---

## 2. Prerequisites checklist

Use this before the first production run.

### 2.1 Control machine (launcher)

- **InferenceX** cloned at a path you will use as `REMOTE_REPO` on **every** GPU node (the same path is bind-mounted into the container as `/workspace`).
- **Passwordless SSH** from the launcher to all GPU nodes, with a key the scripts can use (default: `${HOME}/.ssh/id_rsa`; override with `SSH_OPTS` if needed).
- Optional: **local** check that model directories exist, or set `SKIP_LOCAL_MODEL_CHECK=1` to validate only on the cluster.

### 2.2 Every GPU node

- **Docker** installed; the scripts use `sudo docker` by default (`USE_SUDO_FOR_DOCKER=1`). If your user is in the `docker` group, set `USE_SUDO_FOR_DOCKER=0`.
- **Container image** pulled. Default in the scripts is `lmsysorg/sglang:v0.5.9-rocm700-mi30x` (`IMAGE`). Pick a tag that matches **your** GPU generation and ROCm line; if a build targets one SKU explicitly, align with vendor or SGLang release notes.
- **Model weights** on local disk (often `/dev/shm` or `/mnt`): each node needs `HOST_MODEL_DIR/MODEL_NAME` as a directory (e.g. `DeepSeek-R1-0528`). Prefill and decode can use different host parents via `PREFILL_MODEL_HOST_DIR` and `DECODE_MODEL_HOST_DIR` (and `DECODE_MODEL_HOST_DIR_2` for the second decode in 1P+2D).
- **RDMA / InfiniBand device list** for MoRI: set `IBDEVICES` to a comma-separated list of HCAs, or install host tools so `benchmarks/multi_node/amd_utils/detect_ibdevices_bnxt.sh` can populate it automatically.

### 2.3 Network between nodes

Open **TCP** between all participating nodes for the PD and routing stack (defaults used in `server.sh`):

- **5000** — barrier / sync
- **8000** — SGLang PD HTTP endpoints
- **30000** — router / head traffic

Example (host firewall): allow the peer node IPs on these ports (e.g. `ufw` rules from each node to the others).

Do **not** set `DOCKER="sudo docker"` as a single env var for remote runs; use `DOCKER_BIN=docker` and `USE_SUDO_FOR_DOCKER=1` instead—multi-word `DOCKER` breaks SSH `env` passing.

---

## 3. Quick start

### 3.1 Environment file pattern

A minimal pattern (from `start_sglang_pd_mi300_mi325x.sh`) is:

```bash
export PREFILL_MODEL_HOST_DIR="${PREFILL_MODEL_HOST_DIR:-/dev/shm}"
export DECODE_MODEL_HOST_DIR="${DECODE_MODEL_HOST_DIR:-/dev/shm}"
export IMAGE="${IMAGE:-lmsysorg/sglang:v0.5.9-rocm700-mi30x}"

# Optional: rebuild MoRI inside the container (e.g. topology / latest fixes)
export INSTALL_MORI_IN_CONTAINER=1
export INSTALL_MORI_MODE=git
export MORI_GIT_REF=main

# Optional: rebuild libbnxt in-container from Broadcom tarball (match host bnxt_re)
# Use the tarball that matches the bnxt_re version on *this* host (see /usr/src/bnxt_re-* on the node).
# Examples:
#   MI325X-class:  libbnxt_re-231.0.162.0.tar.gz
#   MI300X-class:  libbnxt_re-230.2.52.0.tar.gz
export REBUILD_LIBBNXT_IN_CONTAINER=1
export PATH_TO_BNXT_TAR_PACKAGE=/workspace/driver/libbnxt_re-231.0.162.0.tar.gz
# export PATH_TO_BNXT_TAR_PACKAGE=/workspace/driver/libbnxt_re-230.2.52.0.tar.gz

export PREFILL_NODE="<prefill-host-or-ip>"
export DECODE_NODE="<decode-host>"          # 1P+1D
# or for 1P+2D:
export DECODE_NODE_1="<decode1>"
export DECODE_NODE_2="<decode2>"

cd /path/to/InferenceX
bash run_1p1d_sglang_mi300_mi325x.sh
# or: bash run_1p2d_sglang_mi300_mi325x.sh
```

Driver tarballs can be downloaded from https://www.broadcom.com/support/download-search. Please download the tarball that matches the bnxt_re version on **this** host (see /usr/src/bnxt_re-* on the node). We also provide the tarball under `InferenceX/driver/` on **each** node at the same `REMOTE_REPO` layout so the in-container path `/workspace/driver/...` exists after bind-mount.

### 3.2 Validate without running

```bash
DRY_RUN=1 bash run_1p1d_sglang_mi300_mi325x.sh
```

### 3.3 Common benchmark overrides

```bash
export MODEL_NAME=DeepSeek-R1-0528
export ISL=1024 OSL=1024
export CONC_LIST="8 16"
bash run_1p1d_sglang_mi300_mi325x.sh
```

Parallelism defaults (override as needed): `PREFILL_TP`, `DECODE_TP`, `PREFILL_EP`, `DECODE_EP`, `PREFILL_DP_ATTN`, `DECODE_DP_ATTN`, `DECODE_MTP_SIZE`, `GPUS_PER_NODE`, `xP`, `yD`.

---

## 4. How the launch works (mental model)

1. **Launcher** SSHs to each node with a large `env ... bash /path/to/InferenceX/scripts/_disagg_ssh_remote_inner.sh`.
2. **`_disagg_ssh_remote_inner.sh`** starts one **Docker** container per node: GPUs, `/dev/infiniband`, host network, shared memory, and volumes:
   - `HOST_MODEL_DIR` → `/models`
   - `HOST_REPO` (InferenceX root) → `/workspace`
   - Log directories for benchmarks and `/run_logs`
3. **`_disagg_container_entry.sh`** (inside the image) optionally runs **libbnxt** rebuild, then **MoRI** install, then execs **`benchmarks/multi_node/amd_utils/server.sh`**, which implements the SGLang PD topology and benchmark client for your `MODEL_NAME` (via `models.yaml` in that directory).

Startup order in the 1P+1D script: **decode first** (background), short sleep, then **prefill** (foreground with tee to log). Adjust if you customize scripts.

---

## 5. Configuration reference (environment)

| Variable | Purpose |
|----------|---------|
| `PREFILL_NODE`, `DECODE_NODE` | SSH targets for 1P+1D (`user@host`). |
| `DECODE_NODE_1`, `DECODE_NODE_2` | Second decode host for 1P+2D. |
| `SSH_USER` | Defaults to `whoami` on launcher if not set in script defaults. |
| `REMOTE_REPO` / `INFERENCEX_DIR` | InferenceX root; must match on all nodes for `/workspace`. |
| `PREFILL_MODEL_HOST_DIR`, `DECODE_MODEL_HOST_DIR` | Host paths containing `MODEL_NAME`. |
| `DECODE_MODEL_HOST_DIR_2` | Optional second decode model root (1P+2D). |
| `IMAGE` | SGLang ROCm container tag; **must match your AMD SKU / ROCm** (MI300X vs MI325X, etc.). |
| `MODEL_NAME` | Key in `benchmarks/multi_node/amd_utils/models.yaml`. |
| `IBDEVICES` | RDMA devices for MoRI (required unless auto-detected). |
| `PREFILL_IP`, `DECODE_IP` | Optional explicit data-plane IPs (1P+1D). |
| `PREFILL_IP`, `DECODE1_IP`, `DECODE2_IP` | Optional explicit IPs (1P+2D). |
| `BARRIER_SYNC_PORT`, `SGLANG_PD_PORT`, `ROUTER_PORT` | Override ports if defaults conflict. |
| `MORI_RDMA_TC` | Optional RDMA traffic class. |
| `REBUILD_LIBBNXT_IN_CONTAINER`, `PATH_TO_BNXT_TAR_PACKAGE` | In-container libbnxt build from tarball (**match host NIC driver version**). |
| `INSTALL_MORI_IN_CONTAINER`, `INSTALL_MORI_MODE`, `MORI_GIT_REF`, … | Build MoRI from git or a mounted path. |
| `SKIP_LOCAL_MODEL_CHECK` | `1` to skip launcher-side weight directory check. |
| `USE_SUDO_FOR_DOCKER`, `DOCKER_BIN`, `EXTRA_DOCKER_ARGS`, `DOCKER_SHM_SIZE` | Docker invocation tuning. |

---

## 6. Logs and artifacts

- **Launcher:** `${HOME}/logs/sglang_disagg/benchmark_logs_<timestamp>/`
  - `ssh_prefill_<node>.log`, `ssh_decode_<node>.log` (naming varies for 1P+2D).
- **On each node:** `/tmp/inferencex_disagg_logs_${JOB_ID}` and `/tmp/run_logs_${JOB_ID}` (mapped into the container as documented in the scripts).

Use these logs for MoRI/SGLang startup errors, RDMA (`ibv_devinfo`), and benchmark throughput lines.

---

## 7. Related InferenceX entry points

- **Slurm / CI-style** multi-node benchmarks: `benchmarks/multi_node/dsr1_*_sglang-disagg.sh` and `amd_utils/submit.sh` (different orchestration, same `server.sh` family).
- **Matrix / GitHub Actions**: see `AGENTS.md` and `.github/configs/` for `sglang-disagg` framework definitions.

---

## 8. Support expectations

InferenceX is an open benchmarking and research platform. This SSH-based PD path is provided to reproduce disaggregated SGLang setups with explicit scripts; production hardening (secrets, monitoring, upgrades) remains your responsibility.

Please be aware that only special/private Thor2 driver supports IBGDA, current configuration (bxnt NIC + public driver) has not been tested for MoRI-EP enablement, but coming later.

For upstream engine behavior and APIs, refer to **SGLang** and **AMD ROCm / MoRI** documentation.
