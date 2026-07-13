"""Patch swebench's Modal scorer (run_evaluation_modal.py), idempotent and
anchor-checked like patch_swebench_agent.py. Run by _patch_swebench_scoring
in benchmarks/benchmark_lib.sh.

1. cpu=4 hardcoded per eval sandbox: Modal bills RESERVED cores and the test
   runs are overwhelmingly single-threaded pytest, so 300 instances reserve
   ~4x what they use. Patched to SWEBENCH_EVAL_SANDBOX_CPU (default 2 ->
   measured $80.83 -> $41.57 per full-300 with identical results).

2. run_instance_modal never finalizes its ModalSandboxRuntime (the __exit__
   that terminates the sandbox exists but nothing calls it), so every eval
   sandbox idle-bills after its tests finish until the 30-min sandbox
   timeout or app teardown. Invisible on the fast 50-slice (the whole app
   ends in ~2 min); on full-300 the slow tail keeps the app alive ~40 min
   and all 300 sandboxes bill ~30 min for ~3 min of work. A finally: on the
   function's main try/except chain terminates the sandbox on every exit
   path the moment the instance's evaluation ends.
"""
import os, sys

cpu = os.environ.get("SWEBENCH_EVAL_SANDBOX_CPU", "2")
try:
    float(cpu)
except ValueError:
    print(f"WARN: SWEBENCH_EVAL_SANDBOX_CPU={cpu!r} is not numeric; leaving cpu=4", file=sys.stderr)
    sys.exit(1)

import swebench.harness.modal_eval.run_evaluation_modal as rem
path = rem.__file__
src = open(path).read()
ok = True
changed = False

# hunk 1: reserved-cpu reduction
if "inferencex scoring-cpu patch" in src:
    print(f"[swebench] {path}: scoring-cpu patch already applied")
elif src.count("cpu=4,") != 1:
    print(f"WARN: [swebench] {path}: cpu=4 anchor not found exactly once "
          f"(count={src.count('cpu=4,')}); skipping", file=sys.stderr)
    ok = False
else:
    src = src.replace("cpu=4,", f"cpu={cpu},  # inferencex scoring-cpu patch")
    changed = True
    print(f"[swebench] {path}: eval sandbox cpu=4 -> cpu={cpu}")

# hunk 2: terminate the sandbox when the instance's evaluation ends
LIFECYCLE_ANCHOR = (
    "            log_dir=log_dir,\n"
    "            errored=True,\n"
    "        )\n"
    "\n"
    "\n"
    "def run_instances_modal("
)
LIFECYCLE_NEW = (
    "            log_dir=log_dir,\n"
    "            errored=True,\n"
    "        )\n"
    "    finally:  # inferencex sandbox lifecycle patch\n"
    "        try:\n"
    "            runner.sandbox.terminate()\n"
    "        except Exception:\n"
    "            pass\n"
    "\n"
    "\n"
    "def run_instances_modal("
)
if "inferencex sandbox lifecycle patch" in src:
    print(f"[swebench] {path}: sandbox lifecycle patch already applied")
elif src.count(LIFECYCLE_ANCHOR) != 1:
    print(f"WARN: [swebench] {path}: lifecycle anchor not found exactly once "
          f"(count={src.count(LIFECYCLE_ANCHOR)}); skipping", file=sys.stderr)
    ok = False
else:
    src = src.replace(LIFECYCLE_ANCHOR, LIFECYCLE_NEW)
    changed = True
    print(f"[swebench] {path}: eval sandboxes now terminate on instance completion")

if changed:
    open(path, "w").write(src)
sys.exit(0 if ok else 1)
