"""Runtime fixes for SWE-bench 4.1.0 Modal sandbox resource handling."""

import math
import os
import sys
from pathlib import Path

CPU_ANCHOR = "cpu=4,"
CPU_MARKER = "reduce idle Modal CPU billing"
LIFECYCLE_ANCHOR = (
    "            log_dir=log_dir,\n"
    "            errored=True,\n"
    "        )\n"
    "\n"
    "\n"
    "def run_instances_modal("
)
LIFECYCLE_REPLACEMENT = (
    "            log_dir=log_dir,\n"
    "            errored=True,\n"
    "        )\n"
    "    finally:  # stop billing after evaluation\n"
    "        try:\n"
    "            runner.sandbox.terminate()\n"
    "        except Exception:\n"
    "            pass\n"
    "\n"
    "\n"
    "def run_instances_modal("
)
LIFECYCLE_MARKER = "stop billing after evaluation"


def _cpu_value() -> str:
    value = os.environ.get("SWEBENCH_EVAL_SANDBOX_CPU", "2")
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(
            f"SWEBENCH_EVAL_SANDBOX_CPU={value!r} must be numeric"
        ) from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(
            f"SWEBENCH_EVAL_SANDBOX_CPU={value!r} must be positive and finite"
        )
    return value


def patch(path: str, cpu: str) -> bool:
    source_path = Path(path)
    source = source_path.read_text()
    cpu_applied = CPU_MARKER in source
    lifecycle_applied = LIFECYCLE_MARKER in source
    failures = []

    if not cpu_applied and source.count(CPU_ANCHOR) != 1:
        failures.append(f"{CPU_ANCHOR!r} found {source.count(CPU_ANCHOR)} times")
    if not lifecycle_applied and source.count(LIFECYCLE_ANCHOR) != 1:
        failures.append(
            f"lifecycle anchor found {source.count(LIFECYCLE_ANCHOR)} times"
        )
    if failures:
        for failure in failures:
            print(
                f"WARN: [swebench] {source_path}: {failure}; no changes written",
                file=sys.stderr,
            )
        return False

    if not cpu_applied:
        source = source.replace(
            CPU_ANCHOR,
            f"cpu={cpu},  # {CPU_MARKER}",
        )
    if not lifecycle_applied:
        source = source.replace(LIFECYCLE_ANCHOR, LIFECYCLE_REPLACEMENT)

    if cpu_applied and lifecycle_applied:
        print(f"[swebench] {source_path}: patch already applied")
    else:
        source_path.write_text(source)
        print(f"[swebench] {source_path}: patch applied with cpu={cpu}")
    return True


def main() -> int:
    try:
        cpu = _cpu_value()
    except ValueError as error:
        print(f"WARN: {error}; leaving SWE-bench unmodified", file=sys.stderr)
        return 1

    import swebench.harness.modal_eval.run_evaluation_modal as modal_evaluation

    return 0 if patch(modal_evaluation.__file__, cpu) else 1


if __name__ == "__main__":
    sys.exit(main())
