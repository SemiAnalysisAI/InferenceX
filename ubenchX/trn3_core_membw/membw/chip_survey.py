"""Chip-to-chip and stack-to-stack HBM copy bandwidth: one large copy on every HBM stack of every chip.

The first mem-bw sweeps sharded sizes across chips, and the plateau differed by chip (~1,075 vs ~990 GB/s
per stack), which drew a sawtooth. This survey measures that spread directly so the curves can be taken
on one chip and the spread reported separately.

  python membw/chip_survey.py --bytes 1G --rounds 3 --out /w/runs/survey

One NeuronCore per stack (LNC=1, cores 8c, 8c+2, 8c+4, 8c+6 of chip c: one per HBM stack, the even core of
each pair), copy_h2h_static. All chips run at once, each chip's four stacks one after another, rounds
repeated so time drift and chip identity can be told apart. Result per run: <out>/r<k>/core<n>/result.json.
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sweep import parse_bytes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", default="1G")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--chips", default="0-15")
    ap.add_argument("--num-exec", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    nb = parse_bytes(a.bytes)
    lo, hi = (int(x) for x in a.chips.split("-"))
    chips = list(range(lo, hi + 1))
    neff = f"{a.out}/neff"
    if not os.path.exists(neff + "/kernel.neff"):
        subprocess.run([sys.executable, f"{HERE}/run_point.py", "--kernel", "copy_h2h_static", "--bytes", str(nb),
                        "--out", neff, "--compile-only"], check=True)

    def chip_round(job):
        k, c = job
        lines = []
        for s in range(4):
            core = 8 * c + 2 * s
            d = f"{a.out}/r{k}/core{core}"
            if os.path.exists(d + "/result.json"):
                continue
            r = subprocess.run([sys.executable, f"{HERE}/run_point.py", "--kernel", "copy_h2h_static",
                                "--bytes", str(nb), "--neff", neff + "/kernel.neff", "--input", neff + "/input.npy",
                                "--core", str(core), "--num-exec", str(a.num_exec), "--out", d],
                               capture_output=True, text=True)
            lines.append(f"r{k} chip {c} stack {s} core {core}: " + (r.stdout.strip().splitlines() or [r.stderr[-300:]])[-1])
        return lines

    for k in range(a.rounds):
        with ThreadPoolExecutor(len(chips)) as ex:
            for lines in ex.map(chip_round, [(k, c) for c in chips]):
                print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
