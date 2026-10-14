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
| `--cpu-and-dram-measured-power-per-socket` | Measured CPU and DRAM watts per socket; currently available from ACPI `Grace Power Socket` on rack-scale GB200/GB300 NVL72 |
| `--cpu-measured-power-per-socket` | Measured CPU-only watts per socket, excluding DRAM; currently available from RAPL package power on HGX |
| `--system` | Required system from the table below |
| `--model` | `oss` (default) or `example` |
| `--workload` | `fixed-seq-len` (default), `agentic`, or `agentic-cpu-offloading` |
| `--scale-out-enabled` | Activate NICs and external switches; alias `--using-scale-out`; default off |
| `--systems` | OSS model quantity of the selected chassis or rack; default 1 |
| `--power-breakdown-per-chassis` | OSS model nested BoM for one chassis or rack, followed by cluster totals |
| `--help` | List options, systems, and models |

### Measured Grace socket

`--cpu-and-dram-measured-power-per-socket` takes measured CPU-plus-DRAM watts per socket. The currently available rack-scale source is the per-socket average of the ACPI `Grace Power Socket` sensor. A PowerX `avg_cpu_socket_power_w` qualifies for that use only when `power_audit.cpu.sensor_kind` is `grace_socket`. Never pass DCGM field 1130 (`CPU<n>:cpuPowerUsageW`) as a CPU-plus-DRAM measurement: it is the CPU rail alone and misses SysIO, LPDDR5X and regulator loss, about 50 W per socket on GB300.

For Grace Hopper/Blackwell superchips, the [NVIDIA Grace power guide](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/power-thermals.html) defines this sensor only as "Power of Grace socket." The model treats it as CPU and SysIO rails, LPDDR5X and socket regulator loss. This follows from the guide's statement that the socket total includes DRAM power and regulator loss, and from a measured GB300 remainder of about 42 W above CPU plus SysIO. GB200 firmware that reported the extra regulator rail only in module power under-reports the socket. The model also assumes that the sensor sits downstream of the compute-tray DC/DC converter.

On every Grace socket (36 per rack), the value replaces the modeled `GraceCPU` and `LPDDR5XMemory` lines with one `Grace socket (measured)` line; no Grace-side loss is added. Tray DC/DC loss, fans, power shelves and facility power through PUE respond to the new load. NICs, optics, NVMe drives and NVSwitch trays do not depend on it. Tray fans take only the workload's modeled LPDDR5X share as air heat and treat the rest of the socket, including regulator loss, as cold-plate heat; routing all ~42 W to air would add up to about 0.5 kW of rack IT power.

The input is also supported on HGX: it replaces both modeled `X86CPU` and `DDR5DIMM` power for the two CPU sockets, with the DIMMs apportioned across them. Fan draw and PSU loss respond, while fan sizing stays modeled. No qualified CPU-and-DRAM socket measurement is currently available for HGX. `--model=example` rejects this input because it has no CPU or DRAM inventory; negative and non-finite values are invalid.

```bash
python -m power_model --system=gb200 --gpu-level-power-per-gpu=594.191 \
  --cpu-and-dram-measured-power-per-socket=98.066 --power-breakdown-per-chassis
```

### Measured x86 CPU

`--cpu-measured-power-per-socket` takes measured CPU-only watts per socket. The currently available HGX source is RAPL package power: compute each socket's `package-N` energy delta over the measurement window, divide it by the window length, and average across sockets. Pass package power only. Do not add a RAPL DRAM domain on Intel hosts, because the model keeps DIMMs modeled. AMD EPYC exposes no RAPL DRAM domain.

On each HGX chassis, the value replaces the modeled `X86CPU` line for both sockets with a `CPU` line of measured provenance. The `DDR5DIMM` line stays modeled, as do NICs, optics, NVMe drives and PCIe switches. The measured CPU is part of the chassis DC load, so fan draw, PSU loss and facility power through PUE respond to it. Fan sizing still uses the modeled design-state CPU.

The input is also supported on rack-scale Grace systems: it replaces modeled `GraceCPU` power for all 36 sockets while LPDDR5X stays modeled. The input must exclude DRAM. No qualified CPU-only socket measurement is currently available for these racks. When both inputs are provided, HGX uses the CPU-only input and rack-scale systems use the CPU-and-DRAM input. `--model=example` rejects the input; negative and non-finite values are invalid.

AMD Zen RAPL has not been validated against wall power, and Schöne et al. (TU Dresden, 2021) found Zen 2 RAPL inaccurate. The model labels the value as measured with that caveat.

```bash
python -m power_model --system=mi355 --gpu-level-power-per-gpu=612.5 \
  --cpu-measured-power-per-socket=180 --power-breakdown-per-chassis
```
