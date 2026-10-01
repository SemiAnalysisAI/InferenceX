"""Fake Slurm/enroot/git/uv/srtctl binaries for driving ``python -m infx.launch`` in tests.

Only external executables are faked; the real launcher, cluster records, image
staging, recipe binder, golden-acceptance planner, and artifact code run.
Cluster host paths are remapped under a sandbox so every cluster record can run
on a developer machine.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
JOB_ID = "42"

_BASH_FAKES = {
    "git": r"""
printf '%s\n' "$*" >> "$FAKE_LOG_DIR/git.log"
case " $* " in
  *" clone "*) dest="${@: -1}"; mkdir -p "$dest/configs" "$dest/.git" ;;
  *" init "*) mkdir -p "${@: -1}" ;;
  *" rev-parse HEAD"*) echo "$FAKE_SRT_COMMIT" ;;
esac
""",
    "uv": r"""
printf '%s\n' "$*" >> "$FAKE_LOG_DIR/uv.log"
if [[ "$1" == venv ]]; then
  dest="${@: -1}"; mkdir -p "$dest/bin"
  printf '#!/bin/bash\nexec "%s" "$@"\n' "$FAKE_PYTHON" > "$dest/bin/python"; chmod +x "$dest/bin/python"
fi
""",
    "make": r"""
printf '%s\n' "$*" >> "$FAKE_LOG_DIR/make.log"
if [[ -n "${FAKE_MAKE_RC:-}" ]]; then echo "make setup broke"; exit "$FAKE_MAKE_RC"; fi
# First run leaves a truncated NATS download behind, as flaky GitHub releases do.
if [[ -n "${FAKE_TRUNCATED_NATS:-}" && ! -e configs/.downloaded ]]; then
  touch configs/.downloaded; echo truncated > configs/nats-server-v2.10.28-amd64.deb
  echo "dpkg: error processing archive"; exit 2
fi
mkdir -p bin && touch bin/uv
""",
    "squeue": r"""
if [[ " $* " == *" -j 42 "* && -n "${FAKE_ACTIVE:-}" && -e "$FAKE_ACTIVE" ]]; then echo "42|RUNNING"; fi
exit 0
""",
    "sacct": r"""
state="${FAKE_STATE:-COMPLETED|0:0}"
if [[ " $* " == *" 4242 "* && "$(cat "$FAKE_LOG_DIR/batch-rc" 2>/dev/null)" != 0 ]]; then state="FAILED|1:0"; fi
echo "$state"
""",
    "scontrol": "exit 1",
    "sacctmgr": "exit 0",
    "scancel": r"""
printf '%s\n' "$*" >> "$FAKE_LOG_DIR/scancel.log"
[[ -n "${FAKE_ACTIVE:-}" ]] && rm -f "$FAKE_ACTIVE"
exit 0
""",
    "srun": r"""
printf '%s\n' "$*" >> "$FAKE_LOG_DIR/srun.log"
while [[ "$1" == -* ]]; do shift; done
exec "$@"
""",
    "tail": r"""
pid=""
for arg; do
  case "$arg" in --pid=*) pid="${arg#--pid=}" ;; -*|+*) ;; *) [[ -f "$arg" ]] && cat "$arg" ;; esac
done
if [[ -n "${FAKE_TAIL_MARKER:-}" ]]; then
  touch "$FAKE_TAIL_MARKER"
  while kill -0 "$pid" 2>/dev/null; do sleep 0.1; done
fi
exit 0
""",
    "enroot": r"""
printf '%s\n' "$*" >> "$FAKE_LOG_DIR/enroot.log"
[[ "$1" == import && "$2" == -o ]] && printf 'squash %s\n' "$4" > "$3"
""",
    "unsquashfs": r"""
printf '%s\n' "$*" >> "$FAKE_LOG_DIR/unsquashfs.log"
[[ -s "${@: -1}" ]]
""",
    "flock": "exit 0",
    "findmnt": r"""echo "${FAKE_FSTYPE:-lustre}" """,
    "rsync": r"""printf '%s\n' "$*" >> "$FAKE_LOG_DIR/rsync.log" """,
}

_SRTCTL = r"""
import json, os, pathlib, sys
import yaml
argv = sys.argv[1:]
keep = ("RUNNER_NAME", "INFMAX_WORKSPACE", "MODEL_PATH", "SERVED_MODEL_NAME", "VIRTUAL_ENV",
        "UCX_NET_DEVICES", "ENROOT_ROOTFS_WRITABLE", "SRT_SRUN_OPTIONS")
record = {"argv": argv, "cwd": os.getcwd(), "env": {name: os.environ.get(name) for name in keep}}
with open(os.path.join(os.environ["FAKE_LOG_DIR"], "srtctl.jsonl"), "a") as handle:
    handle.write(json.dumps(record) + "\n")
assert pathlib.Path("bin/uv").is_file(), "make setup did not run in the srtctl root"
config = yaml.safe_load(pathlib.Path("srtslurm.yaml").read_text())
if "--output" in argv:
    base = pathlib.Path(argv[argv.index("--output") + 1])
elif config.get("output_dir"):
    base = pathlib.Path(config["output_dir"])
else:
    base = pathlib.Path(config["srtctl_root"]) / "outputs"
output = base / "42"
logs = output / "logs"
logs.mkdir(parents=True, exist_ok=True)
(logs / "sweep_42.log").write_text("benchmark complete\n")
result = os.environ["RESULT_FILENAME"]
mode = os.environ["FAKE_RESULTS"]
if mode == "single":
    (logs / f"{result}.json").write_text('{"completed": 2}')
    (logs / "gpu_metrics.csv").write_text("gpu,power\n0,300\n")
    (logs / "gpu_metrics_context.json").write_text('{"device_count": 4}')
elif mode == "fixed":
    point = logs / "sweep_isl_1024_osl_1024"
    point.mkdir()
    (point / "results_concurrency_4_gpus_16_ctx_8_gen_8.json").write_text('{"conc": 4}')
elif mode == "agentic":
    workspace = pathlib.Path(os.environ["INFMAX_WORKSPACE"])
    for conc in os.environ["CONC_LIST"].split():
        (workspace / f"{result}_conc{conc}.json").write_text(json.dumps({"conc": int(conc)}))
        (logs / "agentic" / f"conc_{conc}").mkdir(parents=True)
if os.environ.get("FAKE_PROFILE"):
    (logs / "infx_profile" / "steps").mkdir(parents=True)
    (logs / "infx_profile" / "steps" / "dp0_tp0.jsonl").write_text('{"step": 0}\n')
if os.environ.get("RUN_EVAL") == "true" or os.environ.get("EVAL_ONLY") == "true":
    (logs / "eval_results").mkdir()
    (logs / "eval_results" / "results_gsm8k.json").write_text("{}")
    (logs / "infx-eval-exit-code").write_text("0\n")
if os.environ.get("FAKE_ACTIVE"):
    pathlib.Path(os.environ["FAKE_ACTIVE"]).touch()
if "--json" in argv:
    print(json.dumps({"status": "submitted", "slurm_job_id": "42", "output_dir": str(output)}))
else:
    print("✅ Job 42 submitted!")
sys.exit(int(os.environ.get("FAKE_SRTCTL_RC", "0")))
"""

_SBATCH = r"""
import os, subprocess, sys
args = sys.argv[1:]
with open(os.path.join(os.environ["FAKE_LOG_DIR"], "sbatch.log"), "a") as handle:
    handle.write(" ".join(args) + "\n")
output = next(arg.split("=", 1)[1] for arg in args if arg.startswith("--output="))
chdir = next(arg.split("=", 1)[1] for arg in args if arg.startswith("--chdir="))
env = {**os.environ, "SLURM_JOB_ID": "4242"}
with open(output, "w") as log:
    rc = subprocess.run(["bash", args[-1]], cwd=chdir, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
with open(os.path.join(os.environ["FAKE_LOG_DIR"], "batch-rc"), "w") as handle:
    handle.write(str(rc))
print("4242")
"""


def install_fakes(directory: Path) -> Path:
    """Write the fake binaries into ``directory`` and return it."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, body in _BASH_FAKES.items():
        binary = directory / name
        binary.write_text(f"#!/bin/bash\n{body.strip()}\n")
        binary.chmod(0o755)
    for name, body in {"srtctl": _SRTCTL, "sbatch": _SBATCH}.items():
        binary = directory / name
        binary.write_text(f"#!{sys.executable}\n{body.lstrip()}")
        binary.chmod(0o755)
    return directory


def _sandboxed(value: str, sandbox: Path) -> str:
    """Map an absolute (or ``~``) cluster host path under ``sandbox``."""
    if value.startswith("~/"):
        return str(sandbox / "home" / value[2:])
    return str(sandbox / value.lstrip("/"))


def sandbox_runner_config(sandbox: Path) -> Path:
    """Write configs/runners.yaml with every cluster host path moved under ``sandbox``."""
    data: dict[str, Any] = yaml.safe_load((ROOT / "configs/runners.yaml").read_text())
    for cluster in data["clusters"].values():
        slurm = cluster["slurm"]
        squash = slurm.get("squash") or {}
        locations = [squash, *squash.get("framework-dirs", {}).values(), *squash.get("helper-dirs", {}).values()]
        for framework in squash.get("framework-dirs", {}).values():
            locations += framework.get("model-prefixes", {}).values()
        for location in locations:
            if "dir" in location:
                location["dir"] = _sandboxed(location["dir"], sandbox)
        for volume in slurm.get("volumes", {}).values():
            volume["path"] = _sandboxed(volume["path"], sandbox)
        srt = slurm.get("srt-slurm") or {}
        for key in ("outputs", "shared-run-root", "uv-cache-root"):
            if key in srt:
                srt[key] = _sandboxed(srt[key], sandbox)
        if "mounts" in srt:
            srt["mounts"] = {
                host if host.startswith("/dev/") else _sandboxed(host, sandbox): target
                for host, target in srt["mounts"].items()
            }
        for record in (cluster, srt):
            record["env"] = {
                name: _sandboxed(value, sandbox) if value.startswith("/") else value
                for name, value in record.get("env", {}).items()
            }
    path = sandbox / "runners.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def runner_for(cluster_id: str) -> str:
    """The first physical runner of ``cluster:<cluster_id>``."""
    labels = yaml.safe_load((ROOT / "configs/runners.yaml").read_text())["labels"]
    return labels[f"cluster:{cluster_id}"][0]


def make_workspace(workspace: Path) -> Path:
    """A GITHUB_WORKSPACE with the recipe mirror, one patch beside the patches README, and a stub benchmark_lib."""
    recipes = workspace / "benchmarks/multi_node/srt-slurm-recipes"
    (recipes / "configs").mkdir(parents=True)
    (recipes / "configs/setup.sh").write_text("true\n")
    patches = workspace / "runners/srt-slurm/patches"
    patches.mkdir(parents=True)
    (patches / "README.md").write_text("# srt-slurm patches\n")
    (patches / "001-fixture.patch").write_text("fixture\n")
    (workspace / "benchmarks/benchmark_lib.sh").write_text(
        '_write_lm_eval_meta_json() { printf \'{"conc": "%s"}\\n\' "$3" > "$1"; }\n'
    )
    return workspace


def base_env(*, fakes: Path, logs: Path, workspace: Path, sandbox: Path) -> dict[str, str]:
    """Environment of a workflow launch step, pointed at the fakes."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("SLURM_", "FAKE_", "SRT_", "INFERENCEX_", "INFX_", "UV_"))
        and key not in {"VIRTUAL_ENV", "MODEL_PATH", "CONFIG_FILE", "EVAL_CONFIG_FILE", "BENCH_SCRIPT_OVERRIDE"}
    }
    env.update(
        PATH=f"{fakes}{os.pathsep}{Path(sys.executable).parent}{os.pathsep}/usr/bin{os.pathsep}/bin",
        PYTHONPATH=f"{ROOT}{os.pathsep}{ROOT / 'utils/srt-slurm/src'}",
        HOME=str(sandbox / "home"),
        USER="runner",
        GITHUB_WORKSPACE=str(workspace),
        FAKE_LOG_DIR=str(logs),
        FAKE_PYTHON=sys.executable,
        FAKE_SRT_COMMIT="0123456789abcdef0123456789abcdef01234567",
        ENROOT_IMPORT_TIME_LIMIT="10",
        SALLOC_TIME_LIMIT="10",
        HF_HUB_CACHE="/hf",
        THINKING_MODE="thinking_on",
        GPU_MONITOR_INTERVAL="3",
        EVAL_ONLY="false",
        RUN_EVAL="false",
        REQUIRE_POWER="0",
        RESULT_FILENAME="point-identity",
        GITHUB_RUN_ID="9001",
        GITHUB_RUN_ATTEMPT="1",
    )
    logs.mkdir(parents=True, exist_ok=True)
    return env


def launch(
    env: dict[str, str], config: Path, cwd: Path, command: str = "run", timeout: float = 120
) -> subprocess.CompletedProcess[str]:
    """Run ``python -m infx.launch --runner-config config <command>``."""
    return subprocess.run(
        [sys.executable, "-m", "infx.launch", "--runner-config", str(config), command],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout, check=False,
    )  # fmt: skip


def srtctl_calls(logs: Path) -> list[dict[str, Any]]:
    """Recorded ``srtctl`` invocations."""
    path = logs / "srtctl.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def lines(logs: Path, name: str) -> list[str]:
    """Recorded argument lines of a bash fake."""
    path = logs / f"{name}.log"
    return path.read_text().splitlines() if path.exists() else []
