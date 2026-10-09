# InferenceX Power Model (Beta Experimental)

On most publicly accessible clusters, only GPU-level power telemetry is available. This model takes a pragmatic approach to estimating total power draw by combining that telemetry with estimates for the remaining components and overheads.

## Install

Requires Python 3.12 or newer. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m power_model --help
```

## Run

```bash
python -m power_model --gpu-level-power-per-gpu=400 --system=h100 \
  --workload=agentic-cpu-offloading --scale-out-enabled --power-breakdown-per-chassis
```

## CLI

| Option | Meaning |
| --- | --- |
| `--gpu-level-power-per-gpu` | Required actual GPU electrical watts per GPU; underscore spelling also accepted |
| `--cpu-socket-measured-power` | GB200/GB300 NVL72 only: ACPI `Grace Power Socket` watts per socket; see below |
| `--system` | Required system from the table below |
| `--model` | `oss` (default) or `example` |
| `--workload` | `fixed-seq-len` (default), `agentic`, or `agentic-cpu-offloading` |
| `--scale-out-enabled` | Activate NICs and external switches; alias `--using-scale-out`; default off |
| `--systems` | OSS model quantity of the selected chassis or rack; default 1 |
| `--power-breakdown-per-chassis` | OSS model nested BoM for one chassis or rack, followed by cluster totals |
| `--help` | List options, systems, and models |

### Measured Grace socket

`--cpu-socket-measured-power` takes the per-socket average of the ACPI `Grace Power Socket` sensor. A PowerX `avg_cpu_socket_power_w` qualifies only when `power_audit.cpu.sensor_kind` is `grace_socket`. Never pass DCGM field 1130 (`CPU<n>:cpuPowerUsageW`): it is the CPU rail alone and misses SysIO, LPDDR5X and regulator loss, about 50 W per socket on GB300.

For Grace Hopper/Blackwell superchips, the [NVIDIA Grace power guide](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/power-thermals.html) defines this sensor only as "Power of Grace socket." The model treats it as CPU and SysIO rails, LPDDR5X and socket regulator loss. This follows from the guide's statement that the socket total includes DRAM power and regulator loss, and from a measured GB300 remainder of about 42 W above CPU plus SysIO. GB200 firmware that reported the extra regulator rail only in module power under-reports the socket. The model also assumes that the sensor sits downstream of the compute-tray DC/DC converter.

On every Grace socket (36 per rack), the value replaces the modeled `GraceCPU` and `LPDDR5XMemory` lines with one `Grace socket (measured)` line; no Grace-side loss is added. Tray DC/DC loss, fans, power shelves and facility power through PUE respond to the new load. NICs, optics, NVMe drives and NVSwitch trays do not depend on it. Tray fans take only the workload's modeled LPDDR5X share as air heat and treat the rest of the socket, including regulator loss, as cold-plate heat; routing all ~42 W to air would add up to about 0.5 kW of rack IT power. A cluster without Grace sockets, such as any HGX `--system`, rejects the flag, as do `--model=example` and a negative or non-finite value.

```bash
python -m power_model --system=gb200 --gpu-level-power-per-gpu=594.191 \
  --cpu-socket-measured-power=98.066 --power-breakdown-per-chassis
```
