#!/opt/k31-dynamo/bin/python
"""Record process limits and inherit the available memlock allowance into Dynamo."""

import json
import os
from pathlib import Path
import resource
import sys


def main():
    args = sys.argv[1:]
    if "--model" in args:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from infx.bench.model_metadata import stage_model_metadata

        index = args.index("--model") + 1
        args[index] = str(
            stage_model_metadata(
                Path(args[index]),
                Path(os.environ["K31_MODEL_METADATA_DIR"]),
                Path(os.environ["K31_MODEL_METADATA_MANIFEST"]),
            )
        )
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    print(
        json.dumps(
            {
                "event": "worker_memlock_before",
                "pid": os.getpid(),
                "soft_bytes": soft,
                "hard_bytes": hard,
            }
        ),
        flush=True,
    )
    if soft != hard:
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (hard, hard))
    print(Path("/proc/self/limits").read_text(), flush=True)
    executable = "/opt/k31-dynamo/bin/python"
    os.execv(executable, [executable, *args])


if __name__ == "__main__":
    main()
