#!/usr/bin/env bash

# DeepSWE through Pier against an OpenAI-compatible endpoint: run_deepswe.sh, inputs from the
# environment. See docs/deepswe-standalone.md for the configuration this pins.
set -eo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
source "$root/benchmarks/check_env.sh"
check_env_vars KIMI_MODEL_BASE_URL KIMI_MODEL_API_KEY KIMI_MODEL_NAME KIMI_MODEL_MAX_CONTEXT_SIZE \
    DEEPSWE_TASKS_DIR DEEPSWE_TASKS DEEPSWE_N_ATTEMPTS DEEPSWE_CONCURRENCY DEEPSWE_OUTPUT_DIR
for tool in pier docker python3; do
    if ! command -v "$tool" >/dev/null; then
        echo "ERROR: $tool is not on PATH" >&2
        exit 1
    fi
done
if [[ ! -f "$DEEPSWE_TASKS_DIR/abs-module-cache-flags/task.toml" ]]; then
    echo "ERROR: DEEPSWE_TASKS_DIR is not the deep-swe tasks directory: $DEEPSWE_TASKS_DIR" >&2
    exit 1
fi

run="deepswe-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$DEEPSWE_OUTPUT_DIR/jobs"
config="$DEEPSWE_OUTPUT_DIR/pier.yaml"
# The agent environment carries the API key, so the job file is private to the caller.
(umask 077 && python3 - "$config" <<'PYEOF'
import os
import sys
from pathlib import Path

config = Path(sys.argv[1])
tasks_dir = Path(os.environ["DEEPSWE_TASKS_DIR"])
wanted = os.environ["DEEPSWE_TASKS"]
if wanted == "all":
    names = sorted(path.parent.name for path in tasks_dir.glob("*/task.toml"))
else:
    names = [name.strip() for name in wanted.split(",") if name.strip()]
missing = [name for name in names if not (tasks_dir / name / "task.toml").is_file()]
if missing or not names:
    sys.exit(f"tasks not found under {tasks_dir}: {missing or wanted}")
lines = [
    "datasets:",
    f"  - path: {tasks_dir}",
    "    task_names:",
    *[f"      - {name}" for name in names],
    "agents:",
    "  - import_path: kimi_code_pier:KimiCode",
    f"    model_name: {os.environ['KIMI_MODEL_NAME']}",
    "    env:",
    *[
        f"      {key}: {os.environ[key]!r}"
        for key in (
            "KIMI_MODEL_NAME",
            "KIMI_MODEL_API_KEY",
            "KIMI_MODEL_BASE_URL",
            "KIMI_MODEL_MAX_CONTEXT_SIZE",
        )
    ],
    "      KIMI_MODEL_PROVIDER_TYPE: openai",
    "environment:",
    "  type: docker",
    f"n_attempts: {int(os.environ['DEEPSWE_N_ATTEMPTS'])}",
    f"n_concurrent_trials: {int(os.environ['DEEPSWE_CONCURRENCY'])}",
    "agent_timeout_multiplier: 50",
    "retry:",
    "  max_retries: 3",
]
config.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"{len(names)} tasks, n_attempts {os.environ['DEEPSWE_N_ATTEMPTS']}, "
      f"concurrency {os.environ['DEEPSWE_CONCURRENCY']}", file=sys.stderr)
PYEOF
)

echo "pier job $run -> $DEEPSWE_OUTPUT_DIR/jobs/$run" >&2
PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}" \
    pier run -c "$config" --job-name "$run" --jobs-dir "$DEEPSWE_OUTPUT_DIR/jobs" \
    || echo "pier exited $? (reducing anyway: errored trials are counted)" >&2
python3 "$here/reduce_rewards.py" --job-dir "$DEEPSWE_OUTPUT_DIR/jobs/$run" \
    --output-dir "$DEEPSWE_OUTPUT_DIR" --model "$KIMI_MODEL_NAME" --n-attempts "$DEEPSWE_N_ATTEMPTS"
