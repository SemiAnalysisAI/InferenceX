"""Low-frequency, read-only GPU/host memory samples in the job's service log."""

import argparse
import json
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--interval-seconds", type=float, required=True)
parser.add_argument("--duration-seconds", type=float, required=True)
args = parser.parse_args()
if args.interval_seconds <= 0 or args.duration_seconds <= 0:
    parser.error("interval and duration must be positive")
deadline = time.monotonic() + args.duration_seconds
while time.monotonic() < deadline:
    sample = {"time": datetime.now(timezone.utc).isoformat(), "host": socket.gethostname(), "gpus": []}
    for device in sorted(Path("/sys/class/drm").glob("card[0-9]*/device")):
        if not (device / "mem_info_vram_total").exists():
            continue
        gpu = {"card": device.parent.name, "pci": device.resolve().name}
        for field in ("mem_info_vram_used", "mem_info_vram_total", "mem_info_gtt_used", "gpu_busy_percent"):
            try:
                gpu[field] = int((device / field).read_text().strip())
            except (OSError, ValueError) as error:
                gpu[field] = {"error": str(error)}
        sample["gpus"].append(gpu)
    try:
        sample["host_memory"] = {
            key: value.strip()
            for line in Path("/proc/meminfo").read_text().splitlines()
            for key, value in [line.split(":", 1)]
            if key in {"MemAvailable", "HugePages_Total", "HugePages_Free", "HugePages_Rsvd", "Hugepagesize"}
        }
    except OSError as error:
        sample["host_memory"] = {"error": str(error)}
    print(json.dumps(sample), flush=True)
    time.sleep(min(args.interval_seconds, max(0, deadline - time.monotonic())))
