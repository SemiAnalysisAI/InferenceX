"""One bounded baseline/candidate run against the installed pinned vLLM package."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

bundle = Path("/workspace/diagnostics/pr3088-runtime-cpu/runtime-tests")
out = Path("/evidence")
out.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("candidate_patch", "/workspace/runners/patch_kimik3_mooncake_recovery.py")
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)
import torch
import vllm
package = Path(vllm.__file__).resolve().parent
if str(package).startswith(("/workspace/", str(bundle))):
    raise RuntimeError(f"vLLM source-shadowing: {package}")
state = {"installed_vllm": str(package), "vllm_version": vllm.__version__,
         "torch_version": torch.__version__, "cuda_build": torch.version.cuda,
         "cpu_affinity": sorted(os.sched_getaffinity(0)),
         "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
         "slurm_job_id": os.getenv("SLURM_JOB_ID"),
         "slurm_job_gpus": os.getenv("SLURM_JOB_GPUS"),
         "slurm_step_gpus": os.getenv("SLURM_STEP_GPUS"), "phases": {}}
# NVML identity only: enumerate devices visible to this step, no tensor/model allocation.
state["visible_gpu_identity"] = subprocess.check_output([
    "nvidia-smi", "--query-gpu=uuid,pci.bus_id,name", "--format=csv,noheader"], text=True)
for relative, original_sha, fixed_sha, edits in patch.PATCHES:
    actual = hashlib.sha256((package / relative).read_bytes()).hexdigest()
    if actual != original_sha:
        raise RuntimeError(f"Expected exact pristine image source: {relative}: {actual}")
state["original_sources_verified"] = True

def save():
    (out / "runtime-result.json").write_text(json.dumps(state, indent=2) + "\n")

def run(name, selection, deadline):
    command = [sys.executable, "-m", "pytest", "-c", str(bundle / "pytest.ini"),
               "--confcutdir=" + str(bundle), "--import-mode=importlib", "-q", "-rA",
               "--junitxml=" + str(out / (name + ".xml")), *selection]
    with (out / (name + ".log")).open("w") as log:
        try:
            result = subprocess.run(command, cwd=bundle, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=deadline)
        except subprocess.TimeoutExpired:
            state["phases"][name] = {"timeout_seconds": deadline}
            save()
            raise
    document = ET.parse(out / (name + ".xml")).getroot()
    cases = list(document.iter("testcase"))
    failures = list(document.iter("failure"))
    errors = list(document.iter("error"))
    skips = list(document.iter("skipped"))
    state["phases"][name] = {"exit_code": result.returncode, "tests": len(cases),
        "failures": len(failures), "errors": len(errors), "skipped": len(skips)}
    save()
    return result.returncode, cases, failures, errors, skips

save()
unit = "tests/v1/kv_connector/unit/"
rc, cases, failures, errors, skips = run("original", [unit + "test_kv_load_failure_recovery.py::test_async_load_failure"], 90)
# Upstream's three real Scheduler.schedule/update cases: original fails only the
# new first-block callback assertion; missing imports/configs are NOT regression proof.
if not (rc == 1 and len(cases) == 3 and len(failures) == 1 and not errors and not skips
        and "on_load_failure" in (failures[0].text or "")
        and "Called 0 times" in (failures[0].text or "")):
    raise RuntimeError("Original source did not reproduce the expected upstream callback assertion")
# Separate interpreter on next phase ensures no original classes survive in pytest.
patch.patch_mooncake(package)
state["candidate_sources_verified"] = all(
    hashlib.sha256((package / relative).read_bytes()).hexdigest() == fixed_sha
    for relative, original_sha, fixed_sha, edits in patch.PATCHES)
save()
if not state["candidate_sources_verified"]:
    raise RuntimeError("Candidate source hashes differ")
selection = [unit + "test_kv_load_failure_recovery.py",
             unit + "test_error_propagation.py",
             unit + "test_mooncake_store_scheduler.py",
             unit + "test_multi_connector.py::test_load_failure_notifies_every_connector"]
rc, cases, failures, errors, skips = run("candidate", selection, 240)
if rc or not cases or failures or errors or skips:
    raise RuntimeError("Candidate runtime regression check failed or was incomplete")
state["status"] = "runtime_cpu_regression_passed"
save()
print(json.dumps(state, indent=2))
