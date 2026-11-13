#!/usr/bin/env python3
"""Trainium3 chip inventory, the trn3 analogue of the tpc-skyline page.

Read-only: neuron-ls JSON plus the driver's sysfs tree (/sys/class/neuron_device). Every value
here comes from the driver or the device firmware, not from a static table in a Python package
(unlike JAX's get_tpu_info on TPU). Run on every node; analyze_inventory.py checks that every
device reports the same values, which is how harvesting or asymmetry would show up.

  python3 collect_inventory.py > inventory-<host>.json
"""
import json
import os
import socket
import subprocess
import sys

ROOT = "/sys/class/neuron_device"
SKIP_DIRS = {"power", "subsystem", "ctrl", "stats"}  # counters and controls, not inventory


def read_tree(path, depth=0):
    out = {}
    for name in sorted(os.listdir(path)):
        p = os.path.join(path, name)
        if name in SKIP_DIRS or name == "uevent" or os.path.islink(p) and depth > 0:
            continue
        if os.path.isdir(p):
            if depth < 4:
                out[name] = read_tree(p, depth + 1)
        else:
            try:
                with open(p) as f:
                    out[name] = f.read().strip()
            except OSError as e:
                out[name] = f"<unreadable: {e.strerror}>"
    return out


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else f"<rc={r.returncode}> {r.stderr[-500:]}"


def main():
    ls_text = run(["/opt/aws/neuron/bin/neuron-ls", "--json-output"])
    try:
        ls = json.loads(ls_text)
    except ValueError:
        ls = ls_text
    for d in ls if isinstance(ls, list) else []:
        d.pop("neuron_processes", None)  # who is running is not inventory
    top = {}
    for name in sorted(os.listdir(ROOT)):
        p = os.path.join(ROOT, name)
        if not name.startswith("neuron"):
            if os.path.isdir(p):
                top[name] = read_tree(p, 1)
            else:
                with open(p) as f:
                    top[name] = f.read().strip()
    devices = {name: read_tree(os.path.join(ROOT, name))
               for name in sorted(os.listdir(ROOT), key=lambda n: (len(n), n))
               if name.startswith("neuron") and os.path.isdir(os.path.join(ROOT, name))}
    header = run(["/opt/aws/neuron/bin/neuron-ls"]).split("+--", 1)[0]
    json.dump({"host": socket.gethostname(), "neuron_ls": ls, "neuron_ls_header": header,
               "sysfs_top": top, "sysfs_devices": devices,
               "driver": run(["modinfo", "-F", "version", "neuron"]).strip()},
              sys.stdout, indent=1)


if __name__ == "__main__":
    main()
