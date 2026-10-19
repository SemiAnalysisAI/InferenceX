#!/usr/bin/env bash
set -eo pipefail
package=$(python3 -P -c 'import importlib.util,os;print(os.path.dirname(importlib.util.find_spec("sglang").origin))')
echo "Patching SGLang at $package"
python3 - "$package/srt" <<'PY'
import sys
from pathlib import Path

package = Path(sys.argv[1])
edits = [
    (
        "server_args.py",
        """    LANGUAGE_MODEL_ONLY_ARCHITECTURES = (
        "MuseGlimmerForConditionalGeneration",
""",
        """    LANGUAGE_MODEL_ONLY_ARCHITECTURES = (
        "MuseGlimmerForConditionalGeneration",
        "MuseSparkForConditionalGeneration",
""",
    ),
    (
        "models/muse_spark_omni.py",
        """        self.language_only = get_disagg().language_only
""",
        """        self.language_only = get_disagg().language_only or bool(
            getattr(config, "language_model_only", False)
        )
""",
    ),
]
for name, old, new in edits:
    path = package / name
    text = path.read_text()
    if text.count(new) == 1:
        print(f"already patched {path}")
    elif text.count(old) == 1:
        path.write_text(text.replace(old, new))
        print(f"patched {path}")
    else:
        sys.exit(f"{path}: anchor found {text.count(old)} times, expected 1:\n{old}")
PY
