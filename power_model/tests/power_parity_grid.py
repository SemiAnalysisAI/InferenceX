# SPDX-License-Identifier: GPL-3.0-only
"""The power-model input grid the InferenceX-app TypeScript port is checked against.

Ported from InferenceX-app packages/app/scripts/generate-power-model-parity.py: every
catalog system, workload and scale-out state, measured Grace-socket or x86 CPU inputs,
a sweep of GPU watts including out-of-domain inputs, seeded random inputs, and the
one-GPU example model. Systems keep the catalog order so the seeded draws match.
"""

import itertools
import random
from typing import NamedTuple

SYSTEMS = (
    "mi300",
    "mi325",
    "mi355",
    "hopper",
    "b200",
    "b300",
    "gb200-nvl72",
    "gb300-nvl72",
)
RACKS = frozenset({"gb200-nvl72", "gb300-nvl72"})
WORKLOADS = ("idle", "fixed-seq-len", "agentic", "agentic-cpu-offloading")
SCALE_OUT = (False, True)
# Spans idle, typical, TDP, and both PSU-capacity and rack-shelf limits; -1 is rejected.
# 594.191 W/GPU is the measured GB300 NVL72 Qwen3.5 8k/1k replay mean (app NVL72 fixtures).
GPU_WATTS = (
    -1.0, 0.0, 150.5, 350.0, 594.191, 612.375, 700.0, 1000.0,
    1200.0, 1400.0, 1430.0, 1440.0, 1600.0, 2400.0,
)  # fmt: skip
# Other socket inputs use this subset of GPU watts.
SPARSE_WATTS = (0.0, 700.0, 1400.0)
# Four-system clusters scale the shared network allocation.
CLUSTER_WATTS = (700.0, 1440.0)
# None models Grace; 98.066 W is a measured GB300 NVL72 replay mean.
RACK_SOCKET_WATTS = (None, 0.0, 98.066, 250.0, 500.0)
# None models the x86 CPU; 180 W per package is the CLI test's input.
CHASSIS_CPU_WATTS = (None, 0.0, 180.0, 400.0)
FUZZ_SEED = 20261012
FUZZ_CASES = 240


class Case(NamedTuple):
    system: str
    workload_state: str
    using_scale_out: bool
    systems: int
    gpu_level_power_per_gpu: float
    cpu_and_dram_measured_power_per_socket: float | None
    cpu_measured_power_per_socket: float | None
    model: str = "oss"


def grid():
    for system in SYSTEMS:
        rack = system in RACKS
        # A rack measures its Grace sockets, a chassis its x86 CPU packages; the dense
        # value is swept across every GPU watt.
        measured, dense = (RACK_SOCKET_WATTS, 98.066) if rack else (CHASSIS_CPU_WATTS, 180.0)

        def inputs(watts, rack=rack):
            return (watts, None) if rack else (None, watts)

        for workload, scale_out in itertools.product(WORKLOADS, SCALE_OUT):
            if rack and workload == "idle":
                # Grace has no idle profile: one modeled and one measured rejection suffice.
                for socket_watts in (None, 98.066):
                    yield Case(system, workload, scale_out, 1, 700.0, socket_watts, None)
                continue
            for gpu_watts in GPU_WATTS:
                for watts in measured:
                    if watts not in (None, dense) and gpu_watts not in SPARSE_WATTS:
                        continue
                    yield Case(system, workload, scale_out, 1, gpu_watts, *inputs(watts))
            for gpu_watts in CLUSTER_WATTS:
                for watts in (None, dense):
                    yield Case(system, workload, scale_out, 4, gpu_watts, *inputs(watts))
            # Exercise the alternate input and both-input precedence on each family.
            if rack:
                yield Case(system, workload, scale_out, 1, 700.0, None, 180.0)
            else:
                yield Case(system, workload, scale_out, 1, 700.0, 100.0, None)
                yield Case(system, workload, scale_out, 1, 700.0, None, -1.0)
            yield Case(system, workload, scale_out, 1, 700.0, 98.066, 180.0)


def fuzz():
    """Random grid-shaped inputs; a measured input matches the system's CPU family."""
    rng = random.Random(FUZZ_SEED)
    for _ in range(FUZZ_CASES):
        system = rng.choice(SYSTEMS)
        gpu_watts = rng.uniform(0.0, 2000.0 if rng.random() < 0.8 else 40000.0)
        watts = rng.uniform(0.0, 1500.0) if rng.random() < 0.6 else None
        socket_watts, cpu_watts = (watts, None) if system in RACKS else (None, watts)
        workload, scale_out, systems = rng.choice(WORKLOADS), rng.random() < 0.5, rng.randint(1, 8)
        yield Case(system, workload, scale_out, systems, gpu_watts, socket_watts, cpu_watts)


def example_cases():
    """The one-GPU example model: every workload, without and with each measured input."""
    for system, workload, (socket_watts, cpu_watts) in itertools.product(
        ("hopper", "gb200-nvl72"), WORKLOADS, ((None, None), (100.0, None), (None, 180.0))
    ):
        yield Case(system, workload, False, 1, 700.0, socket_watts, cpu_watts, "example")


def cases():
    return itertools.chain(grid(), fuzz(), example_cases())
