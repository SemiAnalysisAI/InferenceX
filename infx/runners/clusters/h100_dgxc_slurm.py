"""Python launcher for the h100-dgxc-slurm cluster.

Ports all three execution paths from ``runners/launch_h100-dgxc-slurm.sh``:

1. **native-single-node** — SRT-slurm single-node launch via
   :func:`infx.runners.srt_launch.launch_srt_single_node`.
2. **multinode** — srt-slurm multi-node (dynamo-sglang / dynamo-trt) with
   srtctl submission, log tailing, result collection, and NFS cleanup.
3. **agentic (legacy)** — ``salloc`` holder job, squash import under flock,
   ``srun --container-image`` Pyxis step, then ``scancel``.

Cluster facts (Ubuntu 24.04.4 x86_64, Slurm 25.05.7, Python 3.12.3):
- ``/tmp/enroot`` is not writable by the runner user (mkdir permission denied);
  the bash launcher does not set ``ENROOT_CACHE_PATH`` or ``ENROOT_RUNTIME_PATH``
  explicitly — enroot falls back to its own defaults (``$HOME/.local/share/enroot``
  or ``$XDG_RUNTIME_DIR``).  We preserve that behavior (no custom enroot paths).
- No Slurm headers installed; pyslurm is built by a companion agent that vendors
  headers into ``third_party/pyslurm/``.  See ``infx/runners/pyslurm_build.py``.
"""

from __future__ import annotations

import contextlib
import datetime
import glob
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from infx.runners.slurm import SlurmClient
from infx.runners.srt_launch import (
    _build_eval_passthrough,
    _env,
    _env_opt,
    _image_to_squash_key,
    apply_srt_recipe,
    bundle_server_logs,
    check_staged_srt_assets,
    copy_fixed_sequence_results,
    copy_to_workspace,
    resolve_h100_srt_container,
    run_srt_setup,
    setup_srt_slurm,
)

log = logging.getLogger(__name__)

SLURM_PARTITION = "hpc-gpu-1"
SLURM_ACCOUNT = "customer"


def _spec_suffix() -> str:
    return "_mtp" if os.environ.get("SPEC_DECODING") == "mtp" else ""


# ---------------------------------------------------------------------------
# Execution path: native-single-node
# ---------------------------------------------------------------------------


def _launch_native_single_node() -> int:
    """Port of the native-single-node path."""
    image = _env("IMAGE")
    hf_hub_cache_mount = "/mnt/nfs/sa-shared/gharunners/hf-hub-cache"
    srt_model_path = f"hf:{_env('MODEL')}"
    srt_squash_file = f"/mnt/nfs/lustre/containers/{_image_to_squash_key(image)}.sqsh"

    # Per-runner uv cache — the host home is not writable on compute nodes.
    runner_name = _env("RUNNER_NAME")
    os.environ["UV_CACHE_DIR"] = f"/mnt/nfs/sa-shared/.uv/cache-{runner_name}"
    os.environ["UV_PYTHON_INSTALL_DIR"] = "/mnt/nfs/sa-shared/.uv/python"

    os.environ["HF_HUB_CACHE_MOUNT"] = hf_hub_cache_mount
    os.environ["SRT_MODEL_PATH"] = srt_model_path
    os.environ["SRT_SQUASH_FILE"] = srt_squash_file

    from infx.runners.srt_launch import launch_srt_single_node

    return launch_srt_single_node(
        "h100-dgxc-slurm",
        extra_cluster_config_args=[
            "--var",
            "SLURM_ACCOUNT",
            SLURM_ACCOUNT,
            "--var",
            "SLURM_PARTITION",
            SLURM_PARTITION,
            "--var",
            "CONTAINER_KEY",
            image,
        ],
    )


# ---------------------------------------------------------------------------
# Execution path: multinode
# ---------------------------------------------------------------------------


def _launch_multinode() -> int:
    """Port of the multinode path."""
    github_workspace = _env("GITHUB_WORKSPACE")
    inferencex_root = github_workspace
    inferencex_slurm_utils_dir = os.path.join(github_workspace, "runners")
    framework = _env("FRAMEWORK")
    model_prefix = _env("MODEL_PREFIX")
    precision = _env("PRECISION")
    image = _env("IMAGE")
    isl = _env("ISL")
    osl = _env("OSL")
    eval_only = _env("EVAL_ONLY")
    run_eval = _env_opt("RUN_EVAL", "false")
    result_filename = _env("RESULT_FILENAME")
    runner_name = _env("RUNNER_NAME")

    # Model path resolution.
    model_path: str
    srt_slurm_model_prefix: str
    served_model_name: str | None = None

    if framework == "dynamo-sglang":
        if model_prefix == "dsr1" and precision == "fp8":
            model_path = "/mnt/nfs/lustre/models/dsr1-fp8"
            srt_slurm_model_prefix = "dsr1-fp8"
        else:
            msg = (
                f"Unsupported model prefix/precision for dynamo-sglang: {model_prefix}/{precision}"
            )
            raise ValueError(msg)
    elif framework == "dynamo-trt":
        if model_prefix == "dsr1" and precision == "fp8":
            model_path = "/mnt/nfs/lustre/models/dsr1-fp8"
            served_model_name = "DeepSeek-R1-0528"
            srt_slurm_model_prefix = "DeepSeek-R1-0528"
        else:
            msg = f"Unsupported model prefix/precision for dynamo-trt: {model_prefix}/{precision}"
            raise ValueError(msg)
    else:
        msg = f"Unsupported framework: {framework}. Supported frameworks are: dynamo-trt, dynamo-sglang"
        raise ValueError(msg)

    os.environ["MODEL_PATH"] = model_path
    if served_model_name:
        os.environ["SERVED_MODEL_NAME"] = served_model_name
    os.environ["SRT_SLURM_MODEL_PREFIX"] = srt_slurm_model_prefix

    # Prepare srt-slurm checkout.
    log.info("Preparing job-local srt-slurm checkout...")
    srt_repo_dir = "srt-slurm"
    abs_srt_repo = os.path.join(github_workspace, srt_repo_dir)
    if Path(abs_srt_repo).is_dir():
        log.info("Removing existing %s...", srt_repo_dir)
        shutil.rmtree(abs_srt_repo)

    setup_srt_slurm(
        abs_srt_repo,
        framework,
        False,
        github_workspace=github_workspace,
        inferencex_slurm_utils_dir=inferencex_slurm_utils_dir,
    )

    # Install uv + srtctl.
    log.info("Installing srtctl...")
    uv_install_dir = "/mnt/nfs/sa-shared/.uv/bin"
    uv_cache_dir = "/mnt/nfs/sa-shared/.uv/cache"
    uv_python_dir = "/mnt/nfs/sa-shared/.uv/python"
    os.environ["UV_INSTALL_DIR"] = uv_install_dir
    os.environ["UV_CACHE_DIR"] = uv_cache_dir
    os.environ["UV_PYTHON_INSTALL_DIR"] = uv_python_dir
    Path(uv_install_dir).mkdir(parents=True, exist_ok=True)
    Path(uv_cache_dir).mkdir(parents=True, exist_ok=True)
    Path(uv_python_dir).mkdir(parents=True, exist_ok=True)

    uv_path = os.path.join(uv_install_dir, "uv")
    if not Path(uv_path).is_file() or not os.access(uv_path, os.X_OK):
        subprocess.run(
            ["bash", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"],
            check=True,
        )
    os.environ["PATH"] = f"{uv_install_dir}:{os.environ['PATH']}"
    env_script = os.path.join(uv_install_dir, "env")
    if Path(env_script).is_file():
        # Source the env file to set any additional vars.
        subprocess.run(["bash", "-c", f"source {env_script}"], check=False)

    old_cwd = str(Path.cwd())
    os.chdir(abs_srt_repo)
    try:
        subprocess.run(["uv", "venv", "--quiet"], check=True)
        venv_bin = os.path.join(abs_srt_repo, ".venv", "bin")
        os.environ["VIRTUAL_ENV"] = os.path.join(abs_srt_repo, ".venv")
        os.environ["PATH"] = f"{venv_bin}:{os.environ['PATH']}"
        subprocess.run(["uv", "pip", "install", "--quiet", "-e", "."], check=True)

        if not shutil.which("srtctl"):
            msg = "Failed to install srtctl"
            raise RuntimeError(msg)

        log.info("Configs available at: %s/", srt_repo_dir)

        nginx_squash = "/mnt/nfs/lustre/containers/nginx_1.27.4.sqsh"
        squash_file, container_key = resolve_h100_srt_container(image, framework)
        os.environ["SQUASH_FILE"] = squash_file
        os.environ["CONTAINER_KEY"] = container_key
        check_staged_srt_assets(model_path, squash_file)

        os.environ["ISL"] = isl
        os.environ["OSL"] = osl

        srtctl_root = os.path.join(github_workspace, srt_repo_dir)
        os.environ["SRTCTL_ROOT"] = srtctl_root
        os.environ["SLURM_ACCOUNT"] = SLURM_ACCOUNT
        os.environ["SLURM_PARTITION"] = SLURM_PARTITION
        os.environ["NGINX_SQUASH_FILE"] = nginx_squash
        os.environ["IMAGE"] = image

        log.info("Creating srtslurm.yaml configuration...")

        # write_srt_cluster_config — call cluster_config.py directly.
        profile_yaml = os.path.join(
            inferencex_slurm_utils_dir,
            "srt-slurm",
            "h100-dgxc-slurm.yaml",
        )
        subprocess.run(
            [
                "python3",
                "-m",
                "infx.srt_slurm.cluster_config",
                profile_yaml,
                "srtslurm.yaml",
                "--var",
                "SLURM_ACCOUNT",
                SLURM_ACCOUNT,
                "--var",
                "SLURM_PARTITION",
                SLURM_PARTITION,
                "--var",
                "SRTCTL_ROOT",
                srtctl_root,
                "--var",
                "SQUASH_FILE",
                squash_file,
                "--var",
                "NGINX_SQUASH_FILE",
                nginx_squash,
                "--var",
                "IMAGE",
                image,
                "--var",
                "CONTAINER_KEY",
                container_key,
                "--model",
                srt_slurm_model_prefix,
                model_path,
            ],
            check=True,
            env={
                **os.environ,
                "PYTHONPATH": f"{inferencex_root}:{os.environ.get('PYTHONPATH', '')}",
            },
        )

        log.info("Generated srtslurm.yaml:")
        print(Path("srtslurm.yaml").read_text())

        run_srt_setup(srtctl_root, arch="x86_64", github_workspace=github_workspace)

        os.environ["INFMAX_WORKSPACE"] = github_workspace

        # CONFIG_FILE check.
        config_file = os.environ.get("CONFIG_FILE", "")
        if not config_file:
            log.error(
                "CONFIG_FILE is not set. The srt-slurm path requires a CONFIG_FILE "
                "in additional-settings. Config: MODEL_PREFIX=%s PRECISION=%s FRAMEWORK=%s",
                model_prefix,
                precision,
                framework,
            )
            return 1

        # Patch config file name.
        config_path = Path(config_file.split(":")[0])
        content = config_path.read_text()
        content = re.sub(r"^name:.*", f'name: "{runner_name}"', content, flags=re.MULTILINE)
        config_path.write_text(content)

        # Add dist-timeout after watchdog-timeout.
        raw = config_path.read_text()
        if "dist-timeout:" not in raw:
            raw = raw.replace(
                "      watchdog-timeout:",
                "      watchdog-timeout:\n      dist-timeout: 1800",
            )
            config_path.write_text(raw)

        # Submit via apply_srt_recipe.
        srt_eval_passthrough = _build_eval_passthrough()
        srtctl_eval_args = [
            "--set",
            (
                'post_eval.command=["bash", "{infmax_workspace}/benchmarks/multi_node/srt_eval.sh",'
                ' "{endpoint}", "{infmax_workspace}"]'
            ),
            "--set",
            f"post_eval.passthrough_env={srt_eval_passthrough}",
            "--set",
            "benchmark.stream_output=true",
        ]

        tags = f"h100,{model_prefix},{precision},{isl}x{osl},infmax-{datetime.datetime.now(tz=datetime.UTC).strftime('%Y%m%d')}"
        srtctl_output = apply_srt_recipe(
            config_file,
            framework,
            [*srtctl_eval_args, "-f", config_file, "--tags", tags],
            inferencex_root=inferencex_root,
        )
        print(srtctl_output)

        # Extract JOB_ID.
        match = re.search(r"Job (\d+)", srtctl_output)
        if not match:
            log.error("Failed to extract JOB_ID from srtctl output")
            return 1
        job_id = int(match.group(1))
        log.info("Extracted JOB_ID: %d", job_id)

        # Stream logs.
        logs_dir = f"outputs/{job_id}/logs"
        log_file = f"{logs_dir}/sweep_{job_id}.log"
        SlurmClient.stream_job_log(job_id, log_file)

        log.info("Job %d completed!", job_id)
        log.info("Collecting results...")

        if not Path(logs_dir).is_dir():
            log.warning("Logs directory not found at %s", logs_dir)
            return 1

        log.info("Found logs directory: %s", logs_dir)

        # Copy logs.
        workspace_logs = os.path.join(github_workspace, "LOGS")
        if os.path.exists(workspace_logs):
            shutil.rmtree(workspace_logs)
        shutil.copytree(logs_dir, workspace_logs)
        bundle_server_logs(logs_dir, os.path.join(github_workspace, "multinode_server_logs.tar.gz"))

        # Collect results.
        if eval_only != "true":
            copy_fixed_sequence_results(
                logs_dir,
                github_workspace,
                result_filename,
                inferencex_root=inferencex_root,
            )
        else:
            log.info("EVAL_ONLY=true: Skipping benchmark result collection")

        if run_eval == "true" or eval_only == "true":
            eval_dir = os.path.join(logs_dir, "eval_results")
            if Path(eval_dir).is_dir():
                log.info("Extracting eval results from %s", eval_dir)
                for eval_file in Path(eval_dir).iterdir():
                    if eval_file.is_file():
                        copy_to_workspace(
                            str(eval_file), os.path.join(github_workspace, eval_file.name)
                        )
            else:
                log.warning("RUN_EVAL=true but no eval results found at %s", eval_dir)

    finally:
        os.chdir(old_cwd)

        # NFS cleanup — retry to handle silly-rename lock files.
        log.info("Cleaning up srt-slurm outputs...")
        outputs_dir = os.path.join(abs_srt_repo, "outputs")
        for attempt in range(1, 6):
            try:
                if Path(outputs_dir).is_dir():
                    shutil.rmtree(outputs_dir)
                break
            except OSError:
                log.info("Retry %d/5: Waiting for NFS locks to release...", attempt)
                time.sleep(10)
        # Remove stale NFS files.
        for nfs_file in glob.glob(os.path.join(abs_srt_repo, ".nfs*")):
            with contextlib.suppress(OSError):
                Path(nfs_file).unlink()

    return 0


# ---------------------------------------------------------------------------
# Execution path: agentic (legacy salloc + srun --container-image)
# ---------------------------------------------------------------------------


def _launch_agentic() -> int:
    """Port of the legacy agentic path (salloc --no-shell + srun container)."""
    github_workspace = _env("GITHUB_WORKSPACE")
    image = _env("IMAGE")
    gpu_count = _env("GPU_COUNT")
    runner_name = _env("RUNNER_NAME")
    salloc_time_limit = _env("SALLOC_TIME_LIMIT")
    model_prefix = _env("MODEL_PREFIX")
    exp_name = _env("EXP_NAME")
    precision = _env("PRECISION")
    scenario_subdir = _env_opt("SCENARIO_SUBDIR", "")
    spec_suffix = _spec_suffix()
    framework = _env("FRAMEWORK")
    hf_hub_cache = _env("HF_HUB_CACHE")

    hf_hub_cache_mount = "/mnt/nfs/sa-shared/gharunners/hf-hub-cache/"
    aiperf_mmap_cache_host = "/mnt/nfs/sa-shared/gharunners/ai-perf-cache"
    squash_file = f"/mnt/nfs/lustre/containers/{_image_to_squash_key(image)}.sqsh"
    lock_file = f"{squash_file}.lock"

    # Submit holder job via pyslurm.
    job_id = SlurmClient.submit_holder_job(
        partition=SLURM_PARTITION,
        account=SLURM_ACCOUNT,
        gres=f"gpu:{gpu_count}",
        time_limit=salloc_time_limit,
        job_name=runner_name,
        exclusive=True,
    )

    try:
        SlurmClient.wait_for_running(job_id)

        # Squash import under flock — mirrors the bash logic.
        # Check the shared cache before opening the lock. A valid squash file is
        # immutable, so readers do not need to touch a lock owned by another user.
        import_script = f"""\
if unsquashfs -l "{squash_file}" > /dev/null 2>&1; then
    echo 'Squash file already exists and is valid, skipping import'
else
    if ! {{ exec 9>"{lock_file}"; }} 2>/dev/null; then
        exec 9<"{lock_file}" || {{ echo 'Failed to open lock for {squash_file}'; exit 1; }}
    fi
    flock -w 600 9 || {{ echo 'Failed to acquire lock for {squash_file}'; exit 1; }}
    if unsquashfs -l "{squash_file}" > /dev/null 2>&1; then
        echo 'Squash file was imported by another job'
    else
        rm -f "{squash_file}"
        enroot import -o "{squash_file}" docker://{image}
    fi
fi
"""
        SlurmClient.run_step(job_id, ["bash", "-c", import_script])

        # Prefer framework-tagged benchmark script.
        bench_base = (
            f"benchmarks/single_node/{scenario_subdir}{exp_name.split('_')[0]}_{precision}_h100"
        )
        bench_script = f"{bench_base}_{framework}{spec_suffix}.sh"
        if not Path(bench_script).is_file():
            bench_script = f"{bench_base}{spec_suffix}.sh"

        # DeepSeek-V4.1-Flash uses /ix instead of /workspace.
        if model_prefix == "dsv41flash":
            container_mount_dir = "/ix"
            os.environ["INFMAX_CONTAINER_WORKSPACE"] = "/ix"
            os.environ["RESULT_DIR"] = "/ix/results"
        else:
            container_mount_dir = "/workspace"

        mounts = (
            f"{github_workspace}:{container_mount_dir}/,"
            f"{hf_hub_cache_mount}:{hf_hub_cache},"
            f"{aiperf_mmap_cache_host}:/aiperf_mmap_cache"
        )

        SlurmClient.run_step(
            job_id,
            ["bash", bench_script],
            container=squash_file,
            container_mounts=mounts,
            container_workdir=f"{container_mount_dir}/",
            no_container_mount_home=True,
            no_container_entrypoint=True,
            export="ALL,PORT=8888,AIPERF_DATASET_MMAP_CACHE_DIR=/aiperf_mmap_cache",
        )
    finally:
        SlurmClient.cancel(job_id)

    return 0


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------


def launch() -> int:
    """Dispatch to the correct execution path.  Returns process exit code."""
    # Validate early — mirrors the bash check_env_vars at the top of the launcher.
    for name in ("EVAL_ONLY", "IS_MULTINODE", "RUN_EVAL", "SALLOC_TIME_LIMIT", "IS_AGENTIC"):
        _env(name)

    is_multinode = os.environ["IS_MULTINODE"]
    is_agentic = os.environ["IS_AGENTIC"]
    srt_recipe = os.environ.get("SRT_RECIPE", "")

    if is_multinode == "true":
        execution_path = "multinode"
    elif is_agentic == "0" or srt_recipe:
        if not srt_recipe:
            msg = "SRT_RECIPE is required for the native-single-node path"
            raise OSError(msg)
        execution_path = "native-single-node"
    else:
        execution_path = "agentic"

    log.info("h100-dgxc-slurm: execution path = %s", execution_path)

    if execution_path == "native-single-node":
        return _launch_native_single_node()
    if execution_path == "multinode":
        return _launch_multinode()
    return _launch_agentic()
