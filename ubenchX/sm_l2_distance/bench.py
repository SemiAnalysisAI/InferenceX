"""SM-to-SM L2 cache latency distance via pointer chasing.

Each SM chases a random linked list through the L2 cache. Comparing the
per-address latency profiles of every SM pair reveals the GPU's internal
structure: SMs in the same GPC (and die half) see near-identical profiles,
while SMs on different dies diverge measurably.

Outputs an N*N pairwise CSV (sm_a, sm_b, mean_abs_diff, gpc_a, gpc_b) to
stdout, plus results/sm_info.csv (per-SM GPC mapping and mean latency).
Paste the CSV into ``results/`` by hand.

Requirements: CUDA toolkit (nvcc) with cooperative-groups support.

Usage:
    python ubenchX/sm_l2_distance/bench.py          # auto-detect arch
    python ubenchX/sm_l2_distance/bench.py --arch sm_100a --num-sms 148

Ported from SemiAnalysisAI/microbench-blackwell (sm_l2_distance/).
"""

import argparse
import os
import subprocess
import sys
import tempfile


def detect_gpu_arch() -> str:
    """Return the SM architecture string (e.g. 'sm_100a') for GPU 0."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader", "--id=0"],
        capture_output=True,
        text=True,
        check=True,
    )
    cap = result.stdout.strip().split("\n")[0].strip()
    major, minor = cap.split(".")
    return f"sm_{major}{minor}"


def detect_num_sms() -> int:
    """Return the multiprocessor count for GPU 0."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=count",
            "--format=csv,noheader",
            "--id=0",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    # nvidia-smi --query-gpu=count returns device count, not SM count.
    # Use CUDA deviceQuery-style approach via nvcc + a tiny program.
    prog = r'''
#include <cstdio>
#include <cuda_runtime.h>
int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("%d\n", p.multiProcessorCount);
    return 0;
}
'''
    with tempfile.NamedTemporaryFile(suffix=".cu", mode="w", delete=False) as f:
        f.write(prog)
        src = f.name
    out = src.replace(".cu", "")
    try:
        subprocess.run(["nvcc", "-o", out, src], check=True, capture_output=True)
        r = subprocess.run([out], capture_output=True, text=True, check=True)
        return int(r.stdout.strip())
    finally:
        for path in (src, out):
            if os.path.exists(path):
                os.remove(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arch", help="SM architecture (e.g. sm_100a); auto-detected if omitted")
    parser.add_argument("--num-sms", type=int, help="SM count; auto-detected if omitted")
    parser.add_argument("--num-hops", type=int, default=0, help="Override hop count (0 = full chain)")
    parser.add_argument("--num-passes", type=int, default=5, help="Averaging passes (default: 5)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    arch = args.arch or detect_gpu_arch()
    num_sms = args.num_sms or detect_num_sms()

    print(f"Detected arch={arch}, num_sms={num_sms}", file=sys.stderr)

    # Compile
    src = os.path.join(script_dir, "l2_pointer_chase.cu")
    binary = os.path.join(script_dir, "l2_pointer_chase")
    gencode = f"-gencode=arch=compute_{arch.replace('sm_', '')},code=\"{arch},compute_{arch.replace('sm_', '')}\""
    compile_cmd = [
        "nvcc", "-O2", "-std=c++17",
        gencode,
        f"-DNUM_SMS={num_sms}",
        "-o", binary, src,
    ]
    print(f"Compiling: {' '.join(compile_cmd)}", file=sys.stderr)
    subprocess.run(compile_cmd, check=True)

    # Run
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    run_cmd = [binary]
    if args.num_hops > 0:
        run_cmd.append(str(args.num_hops))
        run_cmd.append(str(args.num_passes))
    elif args.num_passes != 5:
        run_cmd.extend(["0", str(args.num_passes)])

    print(f"Running: {' '.join(run_cmd)}", file=sys.stderr)
    # Binary writes the pairwise CSV to stdout, diagnostics to stderr
    result = subprocess.run(run_cmd, cwd=script_dir, check=True, capture_output=True, text=True)

    # Print the CSV to stdout (caller can redirect)
    sys.stdout.write(result.stdout)
    # Forward stderr diagnostics
    sys.stderr.write(result.stderr)


if __name__ == "__main__":
    main()
