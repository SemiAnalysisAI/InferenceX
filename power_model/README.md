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
| `--system` | Required system from the table below |
| `--model` | `advanced` (default) or `basic-example` |
| `--workload` | `fixed-seq-len` (default), `agentic`, or `agentic-cpu-offloading` |
| `--scale-out-enabled` | Activate NICs and external switches; alias `--using-scale-out`; default off |
| `--systems` | Advanced model quantity of the selected chassis or rack; default 1 |
| `--power-breakdown-per-chassis` | Advanced model nested BoM for one chassis or rack, followed by cluster totals |
| `--help` | List options, systems, and models |

| System | Additional aliases | Cooling/PUE |
| --- | --- | --- |
| `hopper` | `h100`, `h200` | Air / 1.3 |
| `mi300`, `mi325`, `mi355`, `b200`, `b300` | — | Air / 1.3 |
| `gb200-nvl72` | `gb200` | DLC / 1.1 |
| `gb300-nvl72` | `gb300` | DLC / 1.1 |
