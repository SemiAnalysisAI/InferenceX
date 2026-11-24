"""Sample the Neuron driver's per-core throttle and TensorE counters from sysfs until killed.

  python ubenchX/full_chip_matmul/sample_throttle.py --device 0 --interval 1 --out throttle.jsonl

Each sample is one JSON line per core: host epoch time, the raw text of
neuron<device>/neuron_core<c>/stats/hardware/ham_throttle/throttle and of .../tensor_engine/pe_cntrs.
"""

import argparse
import glob
import json
import os
import time

SYSFS = "/sys/devices/virtual/neuron_device"


def read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError as e:
        return f"ERROR {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=int, required=True, help="Neuron device (chip) index")
    ap.add_argument("--interval", type=float, required=True, help="seconds between samples")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = os.path.join(SYSFS, f"neuron{a.device}")
    cores = sorted(int(p.rsplit("neuron_core", 1)[1]) for p in glob.glob(os.path.join(dev, "neuron_core*")))
    with open(a.out, "a", buffering=1) as out:
        while True:
            t0 = time.time()
            for c in cores:
                hw = os.path.join(dev, f"neuron_core{c}", "stats", "hardware")
                out.write(json.dumps({"t": round(time.time(), 3), "device": a.device, "core": c,
                                      "throttle": read(os.path.join(hw, "ham_throttle", "throttle")),
                                      "pe_cntrs": read(os.path.join(hw, "tensor_engine", "pe_cntrs"))}) + "\n")
            time.sleep(max(0.0, a.interval - (time.time() - t0)))


if __name__ == "__main__":
    main()
