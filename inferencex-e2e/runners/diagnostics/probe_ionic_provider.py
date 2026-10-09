"""Inspect and open verbs devices inside an existing container; no MR, QP or traffic.

Run next to latest_rdma_provider_probe_0928.py. Use the same immutable image
first without, then with the provider bind. No scheduler/container is launched.
"""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys


def loaded_libraries():
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            path = Path(fields[5])
            if "libionic" in path.name or "libibverbs" in path.name:
                paths.add(path)
    result = {}
    for path in sorted(paths):
        try:
            result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            result[str(path)] = str(error)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, choices=("image-only", "host-provider"))
    args = parser.parse_args()
    report = {"case": args.label, "traffic_tested": False, "gpu_tested": False}
    report["library_mounts"] = [
        line for line in Path("/proc/self/mountinfo").read_text().splitlines()
        if any(part in line for part in ("/usr/lib", "/usr/local/lib", "/etc/libibverbs.d"))
    ]
    report["kernel_abi"] = {
        str(path): path.read_text().strip()
        for path in Path("/sys/class/infiniband_verbs").glob("*/abi_version")
    }
    try:
        packages = subprocess.run(
            ["dpkg-query", "-W", "libionic1", "libibverbs1", "ibverbs-providers"],
            capture_output=True, text=True, check=False, timeout=10,
        )
        report["packages"] = packages.stdout.strip()
        report["package_query_errors"] = packages.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        report["package_query_errors"] = str(error)
    passed = False
    try:
        runpy.run_path(str(Path(__file__).with_name("latest_rdma_provider_probe_0928.py")))
        passed = True
    except (AssertionError, OSError) as error:
        report["error"] = str(error)
    finally:
        report["loaded_libraries_sha256"] = loaded_libraries()
        report["eight_devices_opened_and_closed"] = passed
        print("K3_IONIC_REPORT=" + json.dumps(report), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
