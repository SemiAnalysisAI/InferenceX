"""CU-to-CU L2 / Infinity Cache latency distance via pointer chasing (AMD CDNA3/CDNA4).

Each CU chases a random linked list through the per-XCD L2 cache (default) or,
with ``--chain-mb`` larger than the L2, through the Infinity Cache. Comparing
the per-address latency profiles of every CU pair reveals the GPU's internal
structure: CUs in the same XCD (and shader engine) see near-identical profiles,
while XCDs attached to different IODs diverge measurably once the chain misses
L2.

Outputs an N*N pairwise CSV (cu_a, cu_b, mean_abs_diff, xcd_a, xcd_b) to
stdout, plus results/cu_info.csv (per-CU XCD/SE mapping and mean latency).
Paste the CSV into ``results/`` by hand.

Requirements: ROCm with ``hipcc`` and ``rocminfo`` (gfx942 or gfx950).

Usage:
    python ubenchX/l2_distance/amd/bench.py                    # auto-detect arch, L2-sized chain
    python ubenchX/l2_distance/amd/bench.py --chain-mb 64      # Infinity-Cache-resident chain
    python ubenchX/l2_distance/amd/bench.py --arch gfx950 --num-cus 256

Ported from ubenchX/l2_distance/nvidia (NVIDIA Blackwell).
"""

import argparse
import os
import re
import subprocess
import sys


def rocminfo_gpu_block(device: int) -> str:
    """Return the rocminfo text of the device-th GPU (gfx*) agent."""
    out = subprocess.run(["rocminfo"], capture_output=True, text=True, check=True).stdout
    agents = list(re.finditer(r"^\s*Name:\s+(gfx\S*)", out, flags=re.MULTILINE))
    if not agents:
        raise RuntimeError("rocminfo lists no gfx agent")
    if device >= len(agents):
        raise RuntimeError(f"rocminfo lists {len(agents)} gfx agents; --device {device} is out of range")
    start = agents[device].start()
    end = agents[device + 1].start() if device + 1 < len(agents) else len(out)
    return out[start:end]


def detect_gpu_arch(device: int) -> str:
    """Return the gfx architecture string (e.g. 'gfx950') for the device-th GPU."""
    m = re.search(r"^\s*Name:\s+(gfx[0-9a-f]+)", rocminfo_gpu_block(device), flags=re.MULTILINE)
    return m.group(1)


def detect_num_cus(device: int) -> int:
    """Return the compute-unit count for the device-th GPU."""
    m = re.search(r"^\s*Compute Unit:\s+(\d+)", rocminfo_gpu_block(device), flags=re.MULTILINE)
    if not m:
        raise RuntimeError("rocminfo lists no Compute Unit count for the gfx agent")
    return int(m.group(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arch", help="GPU arch (e.g. gfx950); auto-detected if omitted")
    parser.add_argument("--num-cus", type=int, help="CU count; auto-detected if omitted")
    parser.add_argument("--chain-mb", type=float, default=0,
                        help="Chain size in MiB (0 = per-XCD L2 size, the default)")
    parser.add_argument("--num-hops", type=int, default=0, help="Override hop count (0 = full chain)")
    parser.add_argument("--num-passes", type=int, default=5, help="Averaging passes (default: 5)")
    parser.add_argument("--warm-hops", type=int, default=-1,
                        help="Untimed hops before timing (-1 = full chain, 0 = none)")
    parser.add_argument("--device", type=int, default=0, help="HIP device index (default: 0)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    arch = args.arch or detect_gpu_arch(args.device)
    num_cus = args.num_cus or detect_num_cus(args.device)

    print(f"Detected arch={arch}, num_cus={num_cus}", file=sys.stderr)

    # Compile
    src = os.path.join(script_dir, "l2_pointer_chase.hip")
    binary = os.path.join(script_dir, "l2_pointer_chase")
    compile_cmd = [
        "hipcc", "-O2", "-std=c++17",
        f"--offload-arch={arch}",
        f"-DNUM_CUS={num_cus}",
        "-o", binary, src,
    ]
    print(f"Compiling: {' '.join(compile_cmd)}", file=sys.stderr)
    subprocess.run(compile_cmd, check=True)

    # Run
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    run_cmd = [binary, "--device", str(args.device), "--passes", str(args.num_passes)]
    if args.chain_mb > 0:
        run_cmd += ["--chain-mb", str(args.chain_mb)]
    if args.num_hops > 0:
        run_cmd += ["--hops", str(args.num_hops)]
    if args.warm_hops >= 0:
        run_cmd += ["--warm-hops", str(args.warm_hops)]

    print(f"Running: {' '.join(run_cmd)}", file=sys.stderr)
    # Binary writes the pairwise CSV to stdout, diagnostics to stderr
    result = subprocess.run(run_cmd, cwd=script_dir, check=True, capture_output=True, text=True)

    # Print the CSV to stdout (caller can redirect)
    sys.stdout.write(result.stdout)
    # Forward stderr diagnostics
    sys.stderr.write(result.stderr)


if __name__ == "__main__":
    main()
