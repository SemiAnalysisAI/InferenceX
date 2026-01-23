# InferenceMAX Runners

This document aims to provide a brief overview of some of the runners used for InferenceMAX. Each script in this `runners/` directory
contains the logic for launching the inference runtime container on a specific system. Therefore, these scripts may contain hardcoded
references to directories, users, etc. on the specified system. On the other hand, the scripts in `benchmarks/` are agnostic of the
host system and are run primarily in containers (vLLM, SGLang, etc.).

# MI355X AMDS

Provided by AMD (in part by TensorWave)

`runners/launch_mi355x-amds.sh`

## Overview

| Metric | Value |
|--------|-------|
| Compute Nodes | 12 |
| GPUs per Node | 8× AMD Instinct MI355X |
| Total GPUs | 96 |
| Total GPU VRAM | 27.6 TB |
| Network Fabric | 400 Gbps RoCEv2 |


## GPU

| Specification | Value |
|---------------|-------|
| Model | AMD Instinct MI355X |
| Architecture | gfx950 (CDNA 4) |
| Form Factor | OAM |
| VRAM | 288 GB HBM3e per GPU |
| Intra-Node Interconnect | XGMI (fully connected, 1 hop) |
| Firmware (SMC) | 04.86.11.02 |

```
~$ rocm-smi --showtopo

=============================== Link Type between two GPUs ===============================
       GPU0         GPU1         GPU2         GPU3         GPU4         GPU5         GPU6         GPU7
GPU0   0            XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         XGMI
GPU1   XGMI         0            XGMI         XGMI         XGMI         XGMI         XGMI         XGMI
GPU2   XGMI         XGMI         0            XGMI         XGMI         XGMI         XGMI         XGMI
GPU3   XGMI         XGMI         XGMI         0            XGMI         XGMI         XGMI         XGMI
GPU4   XGMI         XGMI         XGMI         XGMI         0            XGMI         XGMI         XGMI
GPU5   XGMI         XGMI         XGMI         XGMI         XGMI         0            XGMI         XGMI
GPU6   XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         0            XGMI
GPU7   XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         0
```

## CPU

| Specification | Value |
|---------------|-------|
| Model | AMD EPYC 9575F |
| Architecture | Zen 5 |
| Sockets per Node | 2 |
| Cores per Node | 128 (SMT disabled) |
| Base / Boost Frequency | 1.5 GHz / 3.3 GHz |

## Memory & Storage

| Specification | Value |
|---------------|-------|
| System RAM | ~3 TB DDR5 per node |
| NVMe Storage | 28 TB (8× Micron 7450 3.5 TB in RAID) |

## Network

### RDMA NICs (GPU Scale-Out)

| Specification | Value |
|---------------|-------|
| Hardware | Pensando DSC |
| Ports per Node | 8 |
| Speed | 400 Gbps per port (3.2 Tbps total) |
| Protocol | RoCEv2 |
| GPU Direct RDMA | Enabled |
| Driver | ionic 25.11.1.001 |
| Firmware | 1.117.5-a-45 |

### Data NICs (Storage + Management)

| Specification | Value |
|---------------|-------|
| Hardware | Broadcom BCM57508 (Thor 2) |
| Ports per Node | 2 |
| Speed | 400 Gbps per port |
| Driver | bnxt_en 1.10.3-233.0.152.2 |
| Firmware | 231.2.63.0 |

### Switch Fabric

| Specification | Value |
|---------------|-------|
| Leaf Switches | Arista DCS-7060X6-64PE |
| Topology | Rail-optimized leaf-spine |


## Software

### OS & Drivers

| Component | Version |
|-----------|---------|
| OS | Ubuntu 22.04.5 LTS |
| Kernel | 6.8.0-84-generic |
| ROCm | 7.1.1 |
| amdgpu Driver | 6.16.6 |
| RCCL | 2.27.7.70101 |

### System

| Component | Version |
|-----------|---------|
| SLURM | 25.05.3 |
| Docker | 28.2.2 |
| Python | 3.10.12 |

--- 

