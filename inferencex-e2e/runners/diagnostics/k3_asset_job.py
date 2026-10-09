"""Disposable CPU-only platform diagnostic, not a benchmark launcher."""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from string import Template
from types import SimpleNamespace


def command(argv, **kwargs):
    print("+ " + shlex.join(map(str, argv)), flush=True)
    return subprocess.run(list(map(str, argv)), text=True, check=True, **kwargs)


def plan():
    import yaml
    from infx.clusters import load_inventory
    from infx.clusters.slurm import slurm_settings
    from infx.launch.backends.slurm.squash import registry_reference, squash_path
    from infx.launch.drivers.srt.lanes import srt_lane
    from infx.launch.policy import LaunchPath, runtime_env

    root = Path(__file__).resolve().parents[2]
    cluster = load_inventory(yaml.safe_load((root / "configs/runners.yaml").read_text())).clusters["mi355x-amds"]
    settings = slurm_settings(cluster)
    request = SimpleNamespace(env=dict(os.environ), model_prefix="kimik3", framework="vllm-disagg")
    env = runtime_env(cluster, request, settings.srt_slurm.env)
    lane = srt_lane(cluster.id, LaunchPath.SRT_MULTI)
    [draft] = [mount for mount in lane.mounts if mount.volume == "k3-draft" and mount.when(request)]
    recipe = yaml.safe_load((root / "benchmarks/multi_node/srt-slurm-recipes/kimik3/vllm/mi355x-fp4/agentx/disagg-variants.yaml").read_text())["base"]
    [provider_mount] = [Template(spec).substitute(env) for spec in recipe["extra_mount"]]
    image = recipe["model"]["container"]
    assert "@sha256:" in image, "diagnostic requires an immutable image"
    return {
        "image": image, "registry_reference": registry_reference(image),
        "squash": str(squash_path(image, settings.squash.policy("vllm-disagg", "kimik3"))),
        "partition": settings.partition, "account": settings.account,
        "exclude": list(settings.exclude), "outputs": str(settings.srt_slurm.outputs),
        "draft_source": str(settings.path(draft.volume)), "draft_target": draft.target,
        "provider_source": env["IONIC_PROVIDER_PATH"], "provider_mount": provider_mount,
        "node-count": 1, "gpus": 0, "cpus": 8, "memory": "64G", "walltime": "00:30:00",
    }


def report(stdout, marker):
    values = [json.loads(line.removeprefix(marker)) for line in stdout.splitlines() if line.startswith(marker)]
    if len(values) != 1:
        raise RuntimeError(f"Expected one {marker} report, got {len(values)}")
    return values[0]


def node(plan_file):
    data = json.loads(plan_file.read_text())
    work = plan_file.parent
    job_id = os.environ["SLURM_JOB_ID"]
    print(json.dumps({"job_id": job_id, "node": socket.gethostname(), "plan": data}), flush=True)
    provider = Path(data["provider_source"])
    draft = Path(data["draft_source"])
    provider_hash = hashlib.sha256(provider.read_bytes()).hexdigest()
    config_hash = hashlib.sha256((draft / "config.json").read_bytes()).hexdigest()
    def metadata():
        return {str(p): (p.stat().st_mode, p.stat().st_mtime_ns, p.stat().st_ctime_ns)
                for p in (provider, draft, draft / "config.json")}
    before = metadata()
    print(json.dumps({"host_provider_sha256": provider_hash, "host_draft_config_sha256": config_hash}), flush=True)
    command(["scontrol", "show", "job", job_id])
    # Reuse only the exact existing squash, without changing the shared cache.
    image = data["registry_reference"]
    squash = Path(data["squash"])
    if squash.is_file() and subprocess.run(["unsquashfs", "-s", str(squash)], capture_output=True).returncode == 0:
        image = str(squash)
    temporary = Path(tempfile.mkdtemp(prefix=f"k3-assets-{job_id}-", dir="/var/tmp"))
    env = dict(os.environ)
    for name in ("CACHE", "DATA", "RUNTIME", "TEMP"):
        path = temporary / name.lower()
        path.mkdir()
        env[f"ENROOT_{name}_PATH"] = str(path)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", ROCR_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="")
    # Image conversion also runs within the CPU job's memory/processor budget.
    env.update(ENROOT_MAX_PROCESSORS=str(data["cpus"]),
               ENROOT_SQUASH_OPTIONS="-comp lzo -noD -exit-on-error -mem 4G")
    outputs = {}
    try:
        for label, extra, payload in (
            ("image-only", [], ["probe_ionic_provider.py", "--label", "image-only"]),
            ("host-provider", [data["provider_mount"]], ["probe_ionic_provider.py", "--label", "host-provider"]),
            ("draft", [], ["probe_draft_assets.py", data["draft_target"]]),
        ):
            mounts = [f"{work}:/probe:ro", "/dev/infiniband:/dev/infiniband",
                      "/dev/kfd:/dev/kfd", "/dev/dri:/dev/dri",
                      f"{draft}:{data['draft_target']}", *extra]
            argv = ["srun", "--jobid", job_id, "--nodes=1", "--ntasks=1", f"--cpus-per-task={data['cpus']}",
                    "--gres=gpu:0", f"--mem={data['memory']}", "--container-image", image,
                    "--container-writable", "--container-remap-root", "--no-container-mount-home",
                    "--no-container-entrypoint", "--container-workdir=/tmp", "--container-mounts",
                    ",".join(mounts), "timeout", "--signal=TERM", "--kill-after=10s", "120s",
                    "python3", f"/probe/{payload[0]}", *payload[1:]]
            print(f"BEGIN_ASSET_PROBE {label}", flush=True)
            print("+ " + shlex.join(argv), flush=True)
            result = subprocess.run(argv, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900)
            (work / f"{label}.log").write_text(result.stdout)
            print(result.stdout, flush=True)
            outputs[label] = {"returncode": result.returncode}
            try:
                outputs[label]["report"] = report(result.stdout, "K3_DRAFT_REPORT=" if label == "draft" else "K3_IONIC_REPORT=")
            except (ValueError, RuntimeError) as error:
                outputs[label]["report_error"] = str(error)
            print(f"END_ASSET_PROBE {label} rc={result.returncode}", flush=True)
            if result.returncode and "pyxis: failed to import docker image" in result.stdout:
                outputs[label]["setup_error"] = "image_import"
                print("Image import failed before the probe; skipping remaining containers.", flush=True)
                break
        bound = outputs.get("host-provider", {}).get("report", {})
        plain = outputs["image-only"].get("report", {})
        loaded = bound.get("loaded_libraries_sha256", {})
        config = outputs.get("draft", {}).get("report", {})
        core_before = {k: v for k, v in plain.get("loaded_libraries_sha256", {}).items() if "libibverbs.so" in k}
        core_after = {k: v for k, v in loaded.items() if "libibverbs.so" in k}
        checks = {
            "bound_devices_open": outputs.get("host-provider", {}).get("returncode") == 0 and bound.get("eight_devices_opened_and_closed") is True,
            "host_provider_loaded": any("libionic" in path and digest == provider_hash for path, digest in loaded.items()),
            "verbs_core_unchanged": bool(core_before) and core_before == core_after,
            "draft_config_matches": outputs.get("draft", {}).get("returncode") == 0 and config.get("config_loaded") is True and config.get("config_sha256") == config_hash,
            "host_assets_unchanged": before == metadata() and hashlib.sha256(provider.read_bytes()).hexdigest() == provider_hash,
        }
        summary = {"checks": checks, "cases": outputs,
                   "not_run": [label for label in ("image-only", "host-provider", "draft") if label not in outputs],
                   "traffic_tested": False, "gpu_model_loaded": False}
        (work / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("K3_ASSET_RESULT=" + json.dumps(summary), flush=True)
        return 0 if all(checks.values()) else 1
    finally:
        # Only the exact mkdtemp directory above, never a shared cache.
        try:
            shutil.rmtree(temporary)
        except OSError as error:
            print(f"Job-local Enroot data retained at {temporary}: {error}", flush=True)


def submit():
    data = plan()
    work = Path(tempfile.mkdtemp(prefix=f"k3-assets-{os.environ['GITHUB_RUN_ID']}-", dir=data["outputs"]))
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(source, work / source.name)
    plan_file = work / "plan.json"
    plan_file.write_text(json.dumps(data, indent=2) + "\n")
    log = work / "slurm.log"
    argv = ["sbatch", "--parsable", "--partition", data["partition"], "--nodes=1", "--ntasks=1",
            f"--cpus-per-task={data['cpus']}", "--gres=gpu:0", f"--mem={data['memory']}", "--time", data["walltime"],
            "--job-name", f"k3-assets-{os.environ['GITHUB_RUN_ID']}", "--output", str(log),
            "--chdir", str(work), "--export=ALL"]
    if data["account"]:
        argv += ["--account", data["account"]]
    if data["exclude"]:
        argv += ["--exclude", ",".join(data["exclude"])]
    argv += ["--wrap", shlex.join(["python3", "-u", str(work / Path(__file__).name), "node", str(plan_file)])]
    # Caller defaults must not accidentally turn this into a GPU/exclusive job.
    env = {k: v for k, v in os.environ.items() if not k.startswith("SBATCH_")}
    result = command(argv, env=env, capture_output=True)
    job_id = result.stdout.strip().split(";")[0]
    if not job_id.isdecimal():
        raise RuntimeError(f"Invalid sbatch receipt: {result.stdout!r}")
    print(f"ASSET_JOB_ID={job_id} ASSET_WORKDIR={work}", flush=True)

    def cancel(_signum, _frame):
        raise InterruptedError("diagnostic cancelled")

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    position = 0
    try:
        while True:
            if log.exists():
                with log.open() as stream:
                    stream.seek(position)
                    print(stream.read(), end="", flush=True)
                    position = stream.tell()
            queued = subprocess.run(["squeue", "-h", "-j", job_id, "-o", "%T"], capture_output=True, text=True, check=True, timeout=15)
            if not queued.stdout.strip():
                break
            time.sleep(5)
        if log.exists():
            with log.open() as stream:
                stream.seek(position)
                print(stream.read(), end="", flush=True)
        command(["sacct", "-j", job_id, "--format=JobID,State,ExitCode,AllocTRES,Elapsed", "-P"], timeout=20)
        terminal = ""
        for _ in range(12):
            receipt = subprocess.run(["sacct", "-X", "-n", "-P", "-j", job_id,
                                      "--format=State,ExitCode"], capture_output=True,
                                     text=True, check=True, timeout=15)
            terminal = receipt.stdout.strip()
            if terminal:
                break
            time.sleep(1)
        if terminal != "COMPLETED|0:0":
            raise RuntimeError(f"Slurm did not complete cleanly: {terminal!r}")
        outcome = json.loads((work / "result.json").read_text())
        return 0 if all(outcome["checks"].values()) else 1
    except BaseException:
        command(["scancel", job_id], timeout=15)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "submit", "node"))
    parser.add_argument("path", nargs="?", type=Path)
    args = parser.parse_args()
    if args.mode == "plan":
        print(json.dumps(plan(), indent=2))
        return 0
    if args.mode == "submit":
        return submit()
    if args.path is None:
        parser.error("node requires the plan file")
    return node(args.path)


if __name__ == "__main__":
    raise SystemExit(main())
