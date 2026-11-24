"""Print each idle Neuron device's TensorE clock, measured from the driver's pe_cntrs counters.

  python3 ubenchX/full_chip_matmul/te_clock.py --seconds 2

On an idle core the idle counter of tensor_engine/pe_cntrs advances once per TensorE clock cycle, so
its rate shows which performance profile a device runs at (the profile tool cannot read it back on
Trainium3). Prints one line per device: the clock in GHz averaged over its cores, and the spread.
"""

import argparse
import glob
import os
import time

SYSFS = "/sys/devices/virtual/neuron_device"


def idle_counts():
    out = {}
    for path in glob.glob(os.path.join(SYSFS, "neuron*", "neuron_core*", "stats", "hardware", "tensor_engine", "pe_cntrs")):
        parts = path.split(os.sep)
        rows = [x.split(",") for x in open(path).read().strip().split("\n")[1:]]
        out[int(parts[-6][6:]), int(parts[-5][11:])] = (time.time(), int(rows[0][3]), int(rows[0][1]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, required=True, help="measurement interval")
    a = ap.parse_args()
    first = idle_counts()
    time.sleep(a.seconds)
    second = idle_counts()
    for dev in sorted({d for d, _ in first}):
        ghz = []
        for (d, c), (t1, idle1, mm1) in sorted(second.items()):
            t0, idle0, mm0 = first[d, c]
            # Skip cores that ran a matmul or whose counters were reset in the interval.
            if d == dev and mm1 == mm0 and idle1 > idle0:
                ghz.append((idle1 - idle0) / (t1 - t0) / 1e9)
        if ghz:
            print(f"neuron{dev}: {sum(ghz) / len(ghz):.3f} GHz (cores {len(ghz)}, min {min(ghz):.3f}, max {max(ghz):.3f})")
        else:
            print(f"neuron{dev}: no idle core")


if __name__ == "__main__":
    main()
