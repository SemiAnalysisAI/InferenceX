#!/opt/k31-dynamo/bin/python
"""Record process limits and inherit the available memlock allowance into Dynamo."""

import json
import os
from pathlib import Path
import resource
import sys


def main():
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    print(json.dumps({"event": "worker_memlock_before", "pid": os.getpid(),
                      "soft_bytes": soft, "hard_bytes": hard}), flush=True)
    if soft != hard:
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (hard, hard))
    print(Path("/proc/self/limits").read_text(), flush=True)
    executable = "/opt/k31-dynamo/bin/python"
    os.execv(executable, [executable, *sys.argv[1:]])


if __name__ == "__main__":
    main()
