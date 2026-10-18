#!/usr/bin/env bash
set -eo pipefail
package=$(python3 -P -c 'import importlib.util,os;print(os.path.dirname(importlib.util.find_spec("sglang").origin))')
echo "Patching SGLang at $package"
python3 - "$package" <<'PY'
import sys
from pathlib import Path

package = Path(sys.argv[1])
edits = [
    (
        "kernels/ops/muse_spark_v12/gemm_ag.py",
        "and torch.cuda.get_device_capability(x.device) == (10, 3)",
        "and torch.cuda.get_device_capability(x.device) >= (9, 0)",
        1,
    ),
    (
        "srt/models/muse_spark.py",
        "and torch.cuda.get_device_capability() == (10, 3)\n        ):\n            ca_comm = get_tp_group().ca_comm",
        "and torch.cuda.get_device_capability() >= (9, 0)\n        ):\n            ca_comm = get_tp_group().ca_comm",
        1,
    ),
    (
        "srt/models/muse_spark.py",
        "and torch.cuda.get_device_capability(hidden_states.device) == (10, 3)",
        "and torch.cuda.get_device_capability(hidden_states.device) >= (9, 0)",
        1,
    ),
]
for name, old, new, count in edits:
    path = package / name
    text = path.read_text()
    if text.count(old) == count:
        path.write_text(text.replace(old, new))
        print(f"patched {path}")
    elif old not in text and text.count(new) == count:
        print(f"already patched {path}")
    else:
        sys.exit(f"{path}: anchor {old!r} found {text.count(old)} times, expected {count}")
PY
