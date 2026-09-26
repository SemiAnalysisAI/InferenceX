"""Python port of the native srt-slurm single-node launch from slurm_utils.sh.

Faithfully replicates ``launch_srt_single_node`` and its helpers for the
native-single-node execution path.  Same files written, same env vars read,
same srtctl/synthetic_acceptance invocation, same result copying and exit codes.
Where the bash calls sacct/squeue/scancel this uses :class:`SlurmClient`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from infx.runners.slurm import SlurmClient

log = logging.getLogger(__name__)


def _env(name: str) -> str:
    """Read a required environment variable, fail clearly if missing."""
    value = os.environ.get(name, "")
    if not value:
        msg = f"Required environment variable {name} is not set or empty"
        raise OSError(msg)
    return value


def _env_opt(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _image_to_squash_key(image: str) -> str:
    """Replicate ``printf '%s' "$IMAGE" | sed 's/[\\/:@#]/_/g'``."""
    out = image
    for ch in "/\\:@#":
        out = out.replace(ch, "_")
    return out


# ---------------------------------------------------------------------------
# setup_srt_slurm — prepare a job-local srt-slurm checkout
# ---------------------------------------------------------------------------


def setup_srt_slurm(
    destination: str,
    framework: str,
    uses_power: bool,
    *,
    github_workspace: str,
    inferencex_slurm_utils_dir: str,
) -> str:
    """Prepare a job-local srt-slurm checkout and return the srt-slurm commit SHA.

    Mirrors ``setup_srt_slurm`` from slurm_utils.sh.
    """
    eval_passthrough = _build_eval_passthrough()
    source = os.path.join(inferencex_slurm_utils_dir, "..", "utils", "srt-slurm")

    srt_slurm_commit: str
    if framework == "tilert":
        srt_slurm_commit = "6bc3f306bdafa1edfb5dded2fcda8f1ccede1bde"
        subprocess.run(["git", "init", "--quiet", destination], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                destination,
                "remote",
                "add",
                "origin",
                "https://github.com/SemiAnalysisAI/srt-slurm.git",
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", destination, "fetch", "--quiet", "--depth=1", "origin", srt_slurm_commit],
            check=True,
        )
        subprocess.run(
            ["git", "-C", destination, "checkout", "--quiet", "--detach", srt_slurm_commit],
            check=True,
        )
    else:
        git_dir = os.path.join(source, ".git")
        if not os.path.exists(git_dir):
            msg = "Missing srt-slurm submodule; run git submodule update --init before launching."
            raise FileNotFoundError(msg)
        result = subprocess.run(
            ["git", "-C", source, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        srt_slurm_commit = result.stdout.strip()
        subprocess.run(
            [
                "git",
                "-c",
                "advice.detachedHead=false",
                "clone",
                "--quiet",
                "--no-hardlinks",
                source,
                destination,
            ],
            check=True,
        )
        # Apply patches.
        patches_dir = os.path.join(github_workspace, "runners", "srt-slurm", "patches")
        if Path(patches_dir).is_dir():
            for patch in sorted(Path(patches_dir).glob("*.patch")):
                subprocess.run(
                    ["git", "-C", destination, "apply", str(patch)],
                    check=True,
                )

    # Verify HEAD matches.
    result = subprocess.run(
        ["git", "-C", destination, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip() != srt_slurm_commit:
        msg = f"srt-slurm HEAD mismatch: {result.stdout.strip()} != {srt_slurm_commit}"
        raise RuntimeError(msg)

    log.info("Using srt-slurm revision %s", srt_slurm_commit)

    # Write SHA files.
    sha_path = os.path.join(github_workspace, "srt-slurm-sha.txt")
    Path(sha_path).write_text(srt_slurm_commit + "\n")
    if uses_power:
        shutil.copy2(sha_path, os.path.join(github_workspace, "power-producer-sha.txt"))

    # Set up recipe and config directories.
    recipes_dir = os.path.join(destination, "recipes")
    benchmarks_mn = os.path.join(destination, "benchmarks", "multi_node")
    Path(recipes_dir).mkdir(parents=True, exist_ok=True)
    Path(benchmarks_mn).mkdir(parents=True, exist_ok=True)

    src_recipes = os.path.join(github_workspace, "benchmarks", "multi_node", "srt-slurm-recipes")
    if Path(src_recipes).is_dir():
        shutil.copytree(src_recipes, recipes_dir, dirs_exist_ok=True)

    # Symlink recipes into benchmarks/multi_node/.
    link = Path(benchmarks_mn) / "srt-slurm-recipes"
    if not link.exists():
        link.symlink_to(Path("..", "..", "recipes"))

    # Copy configs.
    src_configs = os.path.join(src_recipes, "configs")
    if Path(src_configs).is_dir():
        configs_dir = os.path.join(destination, "configs")
        Path(configs_dir).mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_configs, configs_dir, dirs_exist_ok=True)

    return eval_passthrough


def _build_eval_passthrough() -> str:
    """Build the eval passthrough JSON — mirrors the inline Python in setup_srt_slurm."""
    names = [
        "EVAL_FRAMEWORK",
        "EVAL_CONC",
        "EVAL_LIMIT",
        "EVAL_SUITE",
        "SWEBENCH_GEN_MODE",
        "SWEBENCH_USE_MODAL",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "IS_AGENTIC",
        "SCENARIO_TYPE",
        "TP",
        "EP_SIZE",
        "DP_ATTENTION",
        "PP_SIZE",
        "DCP_SIZE",
        "PCP_SIZE",
        "CONC",
    ]
    runtime_vars = os.environ.get("INFERENCEX_RUNTIME_ENV_VARS", "")
    return json.dumps(names + runtime_vars.split())


# ---------------------------------------------------------------------------
# run_srt_setup — ``make setup``
# ---------------------------------------------------------------------------


def run_srt_setup(srtctl_root: str, *, arch: str = "x86_64", github_workspace: str) -> None:
    """Run ``make setup`` in the srt-slurm checkout.  Mirrors ``run_srt_setup``."""
    setup_log = os.path.join(github_workspace, "srt-setup.log")
    log.info("Setting up srt-slurm (details: srt-setup.log)")
    with open(setup_log, "a") as logfp:
        result = subprocess.run(
            ["make", "setup", f"ARCH={arch}"],
            cwd=srtctl_root,
            stdout=logfp,
            stderr=logfp,
            check=False,
        )
    if result.returncode != 0:
        sys.stderr.write(Path(setup_log).read_text())
        msg = f"srt-slurm setup failed with exit code {result.returncode}"
        raise RuntimeError(msg)
    log.info("srt-slurm setup complete")


# ---------------------------------------------------------------------------
# apply_srt_recipe — invoke synthetic_acceptance
# ---------------------------------------------------------------------------


def apply_srt_recipe(
    config: str,
    framework: str,
    extra_args: list[str],
    *,
    inferencex_root: str,
) -> str:
    """Run ``python3 -m infx.srt_slurm.synthetic_acceptance`` and return stdout.

    Mirrors ``apply_srt_recipe`` from slurm_utils.sh.
    """
    env = {**os.environ}
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{inferencex_root}:{pp}" if pp else inferencex_root
    # Slurm creates a separate compute venv; do not inherit the login marker.
    env.pop("VIRTUAL_ENV", None)

    result = subprocess.run(
        [
            "python3",
            "-m",
            "infx.srt_slurm.synthetic_acceptance",
            config,
            framework,
            "--",
            *extra_args,
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        msg = f"apply_srt_recipe failed with exit code {result.returncode}"
        raise RuntimeError(msg)
    return result.stdout


# ---------------------------------------------------------------------------
# copy helpers — faithful to slurm_utils.sh
# ---------------------------------------------------------------------------


def copy_to_workspace(source_file: str, destination_file: str) -> None:
    """Copy a file unless source and destination are the same inode."""
    if os.path.exists(destination_file) and Path(source_file).samefile(destination_file):
        log.info("Result already present at %s", destination_file)
        return
    shutil.copy2(source_file, destination_file)
    log.info("Copied %s to %s", Path(source_file).name, destination_file)


def bundle_server_logs(logs_dir: str, archive: str) -> None:
    """Create a tar.gz of logs_dir contents.  Mirrors ``bundle_server_logs``."""
    if not Path(logs_dir).is_dir():
        return
    if not any(Path(logs_dir).iterdir()):
        return
    try:
        subprocess.run(
            ["tar", "czf", archive, "-C", logs_dir, "."],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        log.warning("Failed to bundle %s", archive)


def copy_fixed_sequence_results(
    logs_dir: str,
    workspace: str,
    result_filename: str,
    *,
    inferencex_root: str,
) -> None:
    """Copy fixed-sequence results.  Mirrors ``copy_fixed_sequence_results``."""
    result_subdirs = [
        str(p)
        for p in Path(logs_dir).iterdir()
        if p.is_dir() and "isl" in p.name and "osl" in p.name
    ]
    if not result_subdirs:
        log.warning("No result subdirectories found in %s", logs_dir)
        return

    for result_subdir in result_subdirs:
        config_name = Path(result_subdir).name
        log.info("Processing result subdirectory: %s", result_subdir)
        result_files = list(Path(result_subdir).glob("results_concurrency_*.json"))

        for result_file in result_files:
            filename = result_file.name
            # Parse concurrency, gpus, ctx, gen from filename.
            conc_match = re.search(r"results_concurrency_(\d+)_gpus_", filename)
            gpus_match = re.search(r"_gpus_(\d+)", filename)
            ctx_match = re.search(r"_ctx_(\d+)_gen_", filename)
            gen_match = re.search(r"_gen_(\d+)\.json", filename)

            concurrency = conc_match.group(1) if conc_match else ""
            gpus = gpus_match.group(1) if gpus_match else ""
            ctx = ctx_match.group(1) if ctx_match else ""
            gen = gen_match.group(1) if gen_match else ""

            log.info(
                "Processing concurrency %s with %s GPUs (ctx: %s, gen: %s): %s",
                concurrency,
                gpus,
                ctx,
                gen,
                result_file,
            )

            env = {**os.environ}
            pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{inferencex_root}:{pp}" if pp else inferencex_root
            proc = subprocess.run(
                [
                    "python3",
                    "-m",
                    "infx.results.result_filename",
                    "--point",
                    result_filename,
                    config_name,
                    concurrency,
                    gpus,
                    ctx,
                    gen,
                ],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )
            workspace_result = os.path.join(workspace, proc.stdout.strip())
            copy_to_workspace(str(result_file), workspace_result)


def copy_eval_artifacts(eval_dir: str, workspace: str) -> None:
    """Copy eval result files.  Mirrors ``copy_eval_artifacts``."""
    if not Path(eval_dir).is_dir():
        log.warning("Eval results not found at %s", eval_dir)
        return
    for entry in Path(eval_dir).iterdir():
        if entry.is_file():
            copy_to_workspace(str(entry), os.path.join(workspace, entry.name))


# ---------------------------------------------------------------------------
# launch_srt_single_node — the main orchestrator
# ---------------------------------------------------------------------------


def launch_srt_single_node(
    profile: str,
    *,
    extra_cluster_config_args: list[str] | None = None,
) -> int:
    """Python port of ``launch_srt_single_node`` from slurm_utils.sh.

    Returns 0 on success, non-zero on failure.
    """
    github_workspace = _env("GITHUB_WORKSPACE")
    inferencex_root = github_workspace
    inferencex_slurm_utils_dir = os.path.join(github_workspace, "runners")

    # Validate all required env vars upfront.
    for name in (
        "GITHUB_WORKSPACE",
        "SRT_RECIPE",
        "FRAMEWORK",
        "MODEL",
        "MODEL_PREFIX",
        "IMAGE",
        "PRECISION",
        "TP",
        "PP_SIZE",
        "DCP_SIZE",
        "PCP_SIZE",
        "EP_SIZE",
        "DP_ATTENTION",
        "GPU_COUNT",
        "IS_AGENTIC",
        "SPEC_DECODING",
        "CONC",
        "ISL",
        "OSL",
        "RANDOM_RANGE_RATIO",
        "RESULT_FILENAME",
        "GPU_MONITOR_INTERVAL",
        "SRT_MODEL_PATH",
        "HF_HUB_CACHE_MOUNT",
        "HF_HUB_CACHE",
        "SALLOC_TIME_LIMIT",
    ):
        _env(name)

    srt_single_node_root = tempfile.mkdtemp(prefix="srt-single.", dir=github_workspace)
    srtctl_root = os.path.join(srt_single_node_root, "checkout")
    os.environ["INFMAX_WORKSPACE"] = github_workspace
    framework = _env("FRAMEWORK")

    # Setup srt-slurm checkout.
    srt_eval_passthrough = setup_srt_slurm(
        srtctl_root,
        framework,
        False,
        github_workspace=github_workspace,
        inferencex_slurm_utils_dir=inferencex_slurm_utils_dir,
    )

    # Install uv + venv.
    _ensure_uv()
    old_cwd = str(Path.cwd())
    os.chdir(srtctl_root)
    try:
        subprocess.run(["uv", "venv", "--quiet", ".venv"], check=True)
        # Activate the venv by adjusting PATH and VIRTUAL_ENV.
        venv_bin = os.path.join(srtctl_root, ".venv", "bin")
        os.environ["VIRTUAL_ENV"] = os.path.join(srtctl_root, ".venv")
        os.environ["PATH"] = f"{venv_bin}:{os.environ['PATH']}"
        subprocess.run(["uv", "pip", "install", "--quiet", "-e", "."], check=True)
    finally:
        os.chdir(old_cwd)

    os.environ["PYTHONPATH"] = f"{github_workspace}:{os.environ.get('PYTHONPATH', '')}"

    # Prepare recipe arguments.
    args_file = os.path.join(srt_single_node_root, "arguments")
    subprocess.run(
        [
            "python3",
            "-m",
            "infx.srt_slurm.single_node",
            "prepare",
            os.path.join(github_workspace, _env("SRT_RECIPE")),
            args_file,
        ],
        check=True,
    )
    raw_args = Path(args_file).read_bytes().split(b"\0")
    raw_args = [a.decode() for a in raw_args if a]
    srt_selected_recipe = raw_args[0]
    srt_runtime_args = raw_args[1:]
    srt_runtime_args.extend(
        [
            "--set",
            (
                'post_eval.command=["bash", "{infmax_workspace}/benchmarks/single_node/srt_eval.sh",'
                ' "{endpoint}", "/logs/infx-eval-exit-code"]'
            ),
            "--set",
            f"post_eval.passthrough_env={srt_eval_passthrough}",
        ]
    )

    # Resolve container.
    image = _env("IMAGE")
    srt_squash_file = _env_opt("SRT_SQUASH_FILE")
    srt_container = image
    if srt_squash_file and Path(srt_squash_file).is_file():
        check = subprocess.run(
            ["unsquashfs", "-s", srt_squash_file],
            capture_output=True,
            check=False,
        )
        if check.returncode == 0:
            srt_container = srt_squash_file

    # Generate cluster config.
    model = _env("MODEL")
    hf_hub_cache_mount = _env("HF_HUB_CACHE_MOUNT")
    hf_hub_cache = _env("HF_HUB_CACHE")
    salloc_time_limit = _env("SALLOC_TIME_LIMIT")
    srt_model_path = _env("SRT_MODEL_PATH")

    cluster_config_cmd = [
        "python3",
        "-m",
        "infx.srt_slurm.cluster_config",
        os.path.join(inferencex_slurm_utils_dir, "srt-slurm", f"{profile}.yaml"),
        "srtslurm.yaml",
        "--var",
        "SRTCTL_ROOT",
        srtctl_root,
        "--var",
        "SQUASH_FILE",
        srt_container,
        "--var",
        "IMAGE",
        image,
        "--var",
        "NGINX_SQUASH_FILE",
        "nginx:1.27.4",
        "--var",
        "SRT_DEFAULT_TIME_LIMIT",
        salloc_time_limit,
        "--model",
        f"hf:{model}",
        srt_model_path,
        "--container",
        image,
        srt_container,
        "--mount",
        hf_hub_cache_mount,
        hf_hub_cache,
        "--exclusive",
    ]
    if extra_cluster_config_args:
        cluster_config_cmd.extend(extra_cluster_config_args)
    subprocess.run(cluster_config_cmd, check=True, cwd=srtctl_root)

    run_srt_setup(
        srtctl_root,
        arch=_env_opt("SRT_SETUP_ARCH", "x86_64"),
        github_workspace=github_workspace,
    )

    # Track submission for cleanup.
    srt_job_id: str = ""
    srt_job_output: str = ""
    rc = 0

    def _finish(exit_rc: int) -> int:
        """Cleanup handler — mirrors finish_native_single_node."""
        nonlocal srt_job_id, srt_job_output
        final_rc = exit_rc

        # Try to read submission if not yet known.
        if not srt_job_id:
            submission_json = os.path.join(github_workspace, "srt-single-node-submission.json")
            fields_file = os.path.join(srt_single_node_root, "submission-fields")
            try:
                subprocess.run(
                    [
                        "python3",
                        "-m",
                        "infx.srt_slurm.single_node",
                        "submission",
                        submission_json,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    stdout=open(fields_file, "w"),  # noqa: SIM115
                )
                lines = Path(fields_file).read_text().strip().splitlines()
                srt_job_id = lines[0]
                srt_job_output = lines[1] if len(lines) > 1 else ""
            except Exception:  # noqa: BLE001
                log.debug("Could not read submission fields during cleanup")

        # Cancel active job.
        if srt_job_id:
            try:
                if SlurmClient.job_is_active(int(srt_job_id)):
                    SlurmClient.cancel(int(srt_job_id))
            except Exception:  # noqa: BLE001
                log.debug("Could not cancel job during cleanup")

        # Collect artifacts.
        result_filename = _env("RESULT_FILENAME")
        if srt_job_output and Path(srt_job_output).is_dir():
            bundle_server_logs(
                srt_job_output,
                os.path.join(github_workspace, "srt-single-node-logs.tar.gz"),
            )
            result_json = os.path.join(srt_job_output, "logs", f"{result_filename}.json")
            if Path(result_json).is_file():
                try:
                    copy_to_workspace(
                        result_json,
                        os.path.join(
                            github_workspace,
                            Path(result_json).name,
                        ),
                    )
                except Exception:  # noqa: BLE001
                    final_rc = 1

            for gpu_metric in Path(srt_job_output).glob("logs/gpu_metrics*"):
                try:
                    copy_to_workspace(
                        str(gpu_metric),
                        os.path.join(
                            github_workspace,
                            gpu_metric.name,
                        ),
                    )
                except Exception:  # noqa: BLE001
                    final_rc = 1

            # AgentX artifacts.
            agentic_dir = os.path.join(srt_job_output, "logs", "agentic")
            if Path(agentic_dir).is_dir():
                try:
                    dest = os.path.join(github_workspace, "results")
                    if os.path.exists(dest):
                        shutil.rmtree(dest)
                    shutil.copytree(agentic_dir, dest)
                except Exception:  # noqa: BLE001
                    final_rc = 1

        return final_rc

    try:
        # Submit recipe.
        submission_json = os.path.join(github_workspace, "srt-single-node-submission.json")
        stdout = apply_srt_recipe(
            srt_selected_recipe,
            framework,
            [
                "--json",
                "--yes",
                "--output",
                os.path.join(srt_single_node_root, "outputs"),
                *srt_runtime_args,
            ],
            inferencex_root=inferencex_root,
        )
        Path(submission_json).write_text(stdout)

        # Read submission fields.
        fields_file = os.path.join(srt_single_node_root, "submission-fields")
        result = subprocess.run(
            ["python3", "-m", "infx.srt_slurm.single_node", "submission", submission_json],
            check=True,
            capture_output=True,
            text=True,
        )
        Path(fields_file).write_text(result.stdout)
        lines = result.stdout.strip().splitlines()
        srt_job_id = lines[0]
        srt_job_output = lines[1] if len(lines) > 1 else ""

        # Stream logs.
        log_file = os.path.join(srt_job_output, "logs", f"sweep_{srt_job_id}.log")
        SlurmClient.stream_job_log(int(srt_job_id), log_file)

        # Verify job completed successfully.
        SlurmClient.verify_job_status(int(srt_job_id))

        # Check eval exit code.
        run_eval = _env_opt("RUN_EVAL", "false")
        eval_only = _env("EVAL_ONLY")
        if run_eval == "true" or eval_only == "true":
            exit_code_file = os.path.join(srt_job_output, "logs", "infx-eval-exit-code")
            if not Path(exit_code_file).is_file():
                msg = f"Eval exit code file not found: {exit_code_file}"
                raise FileNotFoundError(msg)
            exit_val = Path(exit_code_file).read_text().strip()
            if exit_val != "0":
                msg = f"Eval failed with exit code {exit_val}"
                raise RuntimeError(msg)

        # Check result file exists.
        result_filename = _env("RESULT_FILENAME")
        if eval_only != "true":
            result_path = os.path.join(srt_job_output, "logs", f"{result_filename}.json")
            if not Path(result_path).is_file() or Path(result_path).stat().st_size == 0:
                msg = f"Result file not found or empty: {result_path}"
                raise FileNotFoundError(msg)

    except Exception:
        rc = 1
        raise
    finally:
        rc = _finish(rc)

    return rc


def _ensure_uv() -> None:
    """Install uv if not already on PATH."""
    if shutil.which("uv"):
        return
    subprocess.run(
        ["bash", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"],
        check=True,
    )
    home_bin = os.path.join(str(Path.home()), ".local", "bin")
    if home_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{home_bin}:{os.environ['PATH']}"


# ---------------------------------------------------------------------------
# Multinode helpers
# ---------------------------------------------------------------------------


def resolve_h100_srt_container(image: str, framework: str) -> tuple[str, str]:
    """Resolve container paths for multinode.  Mirrors ``resolve_h100_srt_container``.

    Returns ``(squash_file, container_key)``.
    """
    if not image or " " in image or "\t" in image:
        msg = f"Invalid image: {image!r}"
        raise ValueError(msg)
    container_key = image.replace("nvcr.io/", "nvcr.io#")
    if framework == "dynamo-sglang":
        squash = f"/mnt/nfs/lustre/containers/{_image_to_squash_key(image)}.sqsh"
    elif framework == "dynamo-trt":
        stripped = image.removeprefix("nvcr.io/")
        safe = stripped
        for ch in "/\\:@#":
            safe = safe.replace(ch, "+")
        squash = f"/mnt/nfs/sa-shared/containers/{safe}.sqsh"
    else:
        msg = f"Unsupported framework for resolve_h100_srt_container: {framework!r}"
        raise ValueError(msg)
    return squash, container_key


def check_staged_srt_assets(model_path: str, squash_file: str) -> None:
    """Check model and container readiness.  Mirrors ``check_staged_srt_assets``."""
    config_json = os.path.join(model_path, "config.json")
    if not Path(config_json).is_file() or not os.access(config_json, os.R_OK):
        msg = "readiness-blocked: staged model/config or requested container is unavailable"
        raise RuntimeError(msg)
    result = subprocess.run(
        ["unsquashfs", "-s", squash_file],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        msg = "readiness-blocked: staged model/config or requested container is unavailable"
        raise RuntimeError(msg)
