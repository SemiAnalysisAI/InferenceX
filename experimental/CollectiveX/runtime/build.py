"""Pinned source staging and per-node backend environments, with shared cache lifecycles."""

from __future__ import annotations

import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile

from .config import RANK_ENV_VARS, DEEPEP_UNSETS
from .scheduler import locked, log, log_path, log_tail, run, write_json


# Upstream DeepEP main carries #630 (single-node V2 init), #642 (the Blackwell LL combine
# fence, issue #700), #715 (release before GIN barrier), #688 (NCCL Device API), and
# #640/#627 (pip-wheel SO-name resolution). Cache identity includes the exact commit.
DEEPEP_REPO = "https://github.com/deepseek-ai/DeepEP"
DEEPEP_COMMIT = "01dc3aaac82068020353dce2c302e38153c0bfaa"

# The cu12 NVSHMEM wheel on cu130 images poisons sm103's MNNVL heap initialization. Torch
# 2.10.0+cu130 also poisoned that context on driver 580.159.03; 2.11.0 matches the image.
# Both pins belong in the venv cache key, not just in its installation command.
DEEPEP_NVSHMEM = "nvidia-nvshmem-cu13==3.4.5"
DEEPEP_TORCH = "torch==2.11.0"
# Bump when build flags change without a pin change, preventing stale .ready cache reuse.
DEEPEP_BUILD_GEN = "dlarch1"

UCCL_REPO = "https://github.com/uccl-project/uccl"
UCCL_COMMIT = "fc1b582031221645ea9fce58aeb57187713145e3"

# nccl-extensions owns nccl.ep since nccl4py 0.4 stopped bundling it. Its combine-recv
# fence releases the LL ladder clamp. nccl4py is pinned alongside it; both key the cache.
NCCL_EP_SPECS = ("nccl-extensions[cu13]==0.1.0", "nccl4py[cu13]==0.5.0")


SOURCES = {
    "deepep-v2": ("deepep-v2", DEEPEP_REPO, DEEPEP_COMMIT, ("third-party/fmt",)),
    "uccl-ep": ("uccl", UCCL_REPO, UCCL_COMMIT, ()),
}
NODE = Path(__file__).with_name("node.py")


def remove(path: Path) -> None:
    """Match rm -rf on one generated file, symlink, or directory."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def stage_source(
    destination: Path,
    name: str,
    repository: str,
    revision: str,
    submodules: tuple[str, ...],
    output: Path,
) -> Path:
    """Fetch an exact pin on the submit host, which has network access before allocation."""
    source = destination / f"{name}-{revision}"
    if source.is_dir():
        return source
    destination.mkdir(parents=True, exist_ok=True)
    destination.chmod(0o700)
    temporary = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=destination))
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    # B300's NFS UID mapping can make a new stage root-owned. The job's private HOME scopes
    # this exemption and passes it to submodule child git, as in the original launcher.
    run(
        ["git", "config", "--global", "--add", "safe.directory", "*"],
        path=output,
        env=env,
        check=False,
    )
    try:
        run(["git", "init", "-q", str(temporary)], path=output, env=env)
        run(
            ["git", "-C", str(temporary), "remote", "add", "origin", repository],
            path=output,
            env=env,
        )
        run(
            [
                "git",
                "-C",
                str(temporary),
                "fetch",
                "-q",
                "--no-tags",
                "--depth",
                "1",
                "origin",
                revision,
            ],
            path=output,
            env=env,
        )
        run(
            [
                "git",
                "-C",
                str(temporary),
                "-c",
                "advice.detachedHead=false",
                "checkout",
                "-q",
                "--detach",
                "FETCH_HEAD",
            ],
            path=output,
            env=env,
        )
        if (
            run(["git", "-C", str(temporary), "rev-parse", "HEAD"], env=env).stdout.strip()
            != revision
        ):
            raise RuntimeError("staged source does not match its pinned commit")
        if submodules:
            run(
                [
                    "git",
                    "-C",
                    str(temporary),
                    "submodule",
                    "update",
                    "-q",
                    "--init",
                    "--depth",
                    "1",
                    *submodules,
                ],
                path=output,
                env=env,
            )
        temporary.rename(source)
        return source
    except BaseException:
        shutil.rmtree(temporary)
        log_tail(output)
        raise


def stage_backend_source(stage: Path, backend: str, root: Path) -> None:
    """UCCL keeps the whole tree for common_hip.hpp -> util/gpu_rt.h, without thirdparty submodules."""
    if backend in SOURCES:
        name, repository, revision, submodules = SOURCES[backend]
        stage_source(
            stage / "experimental/CollectiveX/.collx_sources",
            name,
            repository,
            revision,
            submodules,
            log_path(root, f"backend-source-{name}"),
        )


def materialize_source(destination: Path, backend: str, env: dict[str, str]) -> None:
    """Give each build a writable copy of its staged, immutable upstream source."""
    name, _, revision, _ = SOURCES[backend]
    base = env.get("COLLX_BACKEND_SOURCE_ROOT", "")
    source = Path(base) / f"{name}-{revision}"
    if not base or not source.is_dir():
        raise RuntimeError(f"{backend} staged source is invalid")
    remove(destination)
    shutil.copytree(source, destination, symlinks=True)


def slug(value: str) -> str:
    """The existing tr -cs cache spelling, including squeezing literal '-' runs."""
    return re.sub("-+", "-", re.sub(r"[^A-Za-z0-9_.-]", "-", value)).removeprefix("-")


def cache_root(backend: str, arch: str, env: dict[str, str]) -> Path | None:
    """Keep existing cache keys byte-for-byte so a refactor does not rebuild the fleet."""
    cpu, base = platform.machine(), env.get("COLLX_BACKEND_CACHE_ROOT", "")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", cpu) or not Path(base).is_absolute():
        return None
    image = slug(env.get("COLLECTIVEX_IMAGE") or "manual")
    if backend == "deepep-v2":
        nvshmem = DEEPEP_NVSHMEM.removeprefix("nvidia-").replace("==", "-")
        recipe = (
            f"{DEEPEP_COMMIT[:12]}-{DEEPEP_TORCH.replace('==', '-')}-{nvshmem}-{DEEPEP_BUILD_GEN}"
        )
        target = "sm" + arch.replace(".", "", 1)
    else:
        recipe = UCCL_COMMIT[:12] if backend == "uccl-ep" else slug(" ".join(NCCL_EP_SPECS))
        target = slug(arch)
    return Path(base) / f"{backend}-{cpu}-{target}-{image}-{recipe}"


def cache_ready(root: Path, kind: str) -> bool:
    """A ready marker alone cannot authorize a partial cache."""
    if not (root / ".ready").is_file():
        return False
    if kind == "venv":
        return os.access(root / "venv/bin/python", os.X_OK) and (root / "source").is_dir()
    return kind == "site" and (root / "site").is_dir()


def install_cached(root: Path, kind: str, installer, *args) -> None:
    """Install under one private lock; activation and probing happen after releasing it."""
    with locked(Path(f"{root}.lock")):
        if not cache_ready(root, kind):
            installer(root, *args)


def python_command(
    env: dict[str, str],
    command: str,
    *arguments: str,
    interpreter: str = "python3",
    capture: bool = True,
) -> str:
    """Run capability checks in the interpreter whose packages are being validated."""
    result = run([interpreter, str(NODE), command, *map(str, arguments)], env=env, capture=capture)
    return result.stdout.strip() if capture else ""


def cuda_arch(runner: str, env: dict[str, str]) -> str:
    """Prove the registered CUDA target against the allocated GPU before building."""
    from .config import _platforms

    arch = _platforms()[runner]["arch"]
    digits = arch.removeprefix("sm")
    expected = f"{digits[:-1]}.{digits[-1]}" if arch.startswith("sm") and digits.isdigit() else ""
    if not expected:
        raise RuntimeError(f"no CUDA target registered for {runner}")
    detected = python_command(env, "cuda-arch")
    if detected != expected:
        raise RuntimeError(f"{runner} expected CUDA target {expected}, detected {detected}")
    return detected


def nvshmem_overlay(root: Path, packaged: Path) -> Path:
    """Expose the wheel's host SONAME without changing the installed wheel."""
    overlay = root / "nvshmem-overlay"
    with locked(root / "nvshmem-overlay.lock"):
        if not overlay.is_dir():
            temporary = root / f".nvshmem-overlay.{os.getpid()}"
            remove(temporary)
            (temporary / "lib").mkdir(parents=True)
            (temporary / "include").symlink_to(packaged / "include")
            for path in sorted((packaged / "lib").iterdir()):
                (temporary / "lib" / path.name).symlink_to(path)
            if (packaged / "lib/libnvshmem_host.so.3").exists():
                host = temporary / "lib/libnvshmem_host.so"
                host.unlink(missing_ok=True)
                host.symlink_to(packaged / "lib/libnvshmem_host.so.3")
            temporary.rename(overlay)
        if (
            overlay.is_symlink()
            or (overlay / "include").resolve() != (packaged / "include").resolve()
            or not (overlay / "lib/libnvshmem_host.so").exists()
            or not (overlay / "lib/libnvshmem_device.a").exists()
        ):
            raise RuntimeError("DeepEP V2 NVSHMEM overlay is invalid")
    return overlay


def deepep_environment(root: Path, env: dict[str, str]) -> dict[str, str]:
    """Activate the pinned venv and toolchain without sourcing a shell activation script."""
    interpreter = root / "venv/bin/python"
    sites = sorted((root / "venv/lib").glob("python*/site-packages"))
    if not os.access(interpreter, os.X_OK) or not sites or not sites[0].is_dir():
        raise RuntimeError("DeepEP V2 venv interpreter or site-packages is unavailable")
    nccl = Path(
        python_command(
            env, "package-root", "nvidia-nccl-cu13", "nccl", interpreter=str(interpreter)
        )
    )
    nvshmem = Path(
        python_command(
            env,
            "package-root",
            DEEPEP_NVSHMEM.split("==")[0],
            "nvshmem",
            interpreter=str(interpreter),
        )
    )
    overlay = nvshmem_overlay(root, nvshmem)
    compiler = shutil.which("nvcc", path=env.get("PATH"))
    if not compiler:
        raise RuntimeError("CUDA nvcc is unavailable")
    nvcc = Path(compiler).resolve()
    cuda = nvcc.parent.parent
    cccl = sorted(cuda.glob("targets/*/include/cccl"))
    if (
        nvcc.parent.name != "bin"
        or not os.access(cuda / "bin/nvcc", os.X_OK)
        or not (cuda / "include").is_dir()
        or not (cuda / "lib64").is_dir()
        or not cccl
        or not cccl[0].is_dir()
    ):
        raise RuntimeError("CUDA toolkit root or CCCL headers are unavailable")
    execution = env.get("COLLECTIVEX_EXECUTION_ID") or "manual"
    if not re.fullmatch(r"[A-Za-z0-9._-]+", execution):
        raise RuntimeError("DeepEP V2 execution identity is invalid")
    jit = Path(f"/tmp/collectivex-deepep-v2-jit-{execution}")
    if jit.is_symlink():
        raise RuntimeError("DeepEP V2 JIT cache path is unsafe")
    jit.mkdir(parents=True, exist_ok=True)
    jit.chmod(0o700)
    result = {
        **env,
        "VIRTUAL_ENV": str(root / "venv"),
        "PATH": f"{root}/venv/bin:" + env.get("PATH", "").removeprefix(f"{root}/venv/bin:"),
        "PYTHONPATH": str(sites[0]) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        "CUDA_HOME": str(cuda),
        "CPATH": f"{cccl[0]}:{env.get('CPATH', '')}",
        "NVCC_PREPEND_FLAGS": f"-I{cccl[0]} {env.get('NVCC_PREPEND_FLAGS', '')}",
        "NVSHMEM_DIR": str(overlay),
        "EP_NCCL_ROOT_DIR": str(nccl),
        "EP_NVSHMEM_ROOT_DIR": str(overlay),
        "EP_JIT_CACHE_DIR": str(jit),
        "EP_REUSE_NCCL_COMM": "1",
        "NCCL_CUMEM_ENABLE": "1",
        "LD_LIBRARY_PATH": f"{overlay}/lib:{nccl}/lib:{nvshmem}/lib:{env.get('LD_LIBRARY_PATH', '')}",
    }
    for name in DEEPEP_UNSETS:
        result.pop(name, None)
    return result


def reset_cache(root: Path) -> None:
    """Remove an incomplete cache before rebuilding; readiness is written last."""
    remove(root)
    root.mkdir(mode=0o700)


def install_deepep(root: Path, arch: str, env: dict[str, str]) -> None:
    reset_cache(root)
    run(["python3", "-m", "venv", str(root / "venv")], env=env, capture=False)
    interpreter = str(root / "venv/bin/python")
    pip = [interpreter, "-m", "pip", "install", "-q", "--disable-pip-version-check", "--no-input"]
    run(
        [
            *pip,
            "pip==26.1.2",
            "setuptools==82.0.1",
            "wheel==0.47.0",
            "ninja==1.13.0",
            "numpy==2.2.6",
            DEEPEP_NVSHMEM,
        ],
        env=env,
        capture=False,
    )
    run(
        [
            *pip,
            "--index-url",
            "https://download.pytorch.org/whl/cu130",
            "--extra-index-url",
            "https://pypi.org/simple",
            DEEPEP_TORCH,
        ],
        env=env,
        capture=False,
    )
    # Torch pins NCCL 2.28.9; ElasticBuffer needs the Device API in 2.30.4.
    run(
        [*pip, "--force-reinstall", "--no-deps", "nvidia-nccl-cu13==2.30.4"], env=env, capture=False
    )
    active = deepep_environment(root, env)
    source = root / "source"
    materialize_source(source, "deepep-v2", active)
    # nvcc's RDC device link gets no -gencode from the extension build and otherwise defaults
    # to sm75 on CUDA13. NVCC_PREPEND_FLAGS reaches dlink too, preventing sm103 context faults.
    target = arch.replace(".", "", 1)
    build_env = {
        **active,
        "TORCH_CUDA_ARCH_LIST": arch,
        "MAX_JOBS": "16",
        "NVCC_PREPEND_FLAGS": f"-gencode=arch=compute_{target},code=sm_{target} {active['NVCC_PREPEND_FLAGS']}",
    }
    run(
        [
            interpreter,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-build-isolation",
            "--no-deps",
            "--force-reinstall",
            ".",
        ],
        cwd=source,
        env=build_env,
        capture=False,
    )
    python_command(active, "check-backend", "deepep-v2", interpreter=interpreter, capture=False)
    (root / ".ready").touch()


def pip_fallback(
    arguments: list[str], env: dict[str, str], cwd: Path | None = None, *, docker: bool = False
) -> None:
    """Keep the image's PEP668 fallback and its original interpreter selection."""
    base = (
        ["pip", "install", "-q"]
        if docker
        else ["python3", "-m", "pip", "install", "-q", "--disable-pip-version-check", "--no-input"]
    )
    try:
        run([*base, *arguments], env=env, cwd=cwd, capture=False)
    except subprocess.CalledProcessError:
        run([*base, "--break-system-packages", *arguments], env=env, cwd=cwd, capture=False)


def install_uccl(root: Path, arch: str, env: dict[str, str], *, docker: bool = False) -> None:
    if docker:
        # /uccl_pfx is a bind mount. Clear its contents, never remove the mount point itself.
        for child in root.iterdir():
            remove(child)
    else:
        reset_cache(root)
    source = Path("/tmp/ub" if docker else f"/tmp/collectivex-uccl-{UCCL_COMMIT}")
    log(f"UCCL-EP: building {UCCL_COMMIT} from source (USE_DMABUF, PER_EXPERT_BATCHING)")
    pip_fallback(["nanobind"], env, docker=docker)
    materialize_source(source, "uccl-ep", env)
    amd = env.get("COLLX_VENDOR", "nvidia") == "amd"
    if amd:
        # hipMallocManaged fails even for 4KiB on CDNA. Pinned host memory is coherent and
        # device-accessible for the CPU-proxy handles and atomic buffers on gfx942/gfx950.
        for name in ("uccl_ep.cc", "uccl_proxy.cpp"):
            path = source / "ep/src" / name
            path.write_text(path.read_text().replace("cudaMallocManaged", "cudaMallocHost"))
    active = {
        **env,
        "USE_DMABUF": "1",
        "PER_EXPERT_BATCHING": "1",
        "PYTORCH_ROCM_ARCH" if amd else "TORCH_CUDA_ARCH_LIST": arch,
    }
    run(["python3", "setup.py", "install"], cwd=source / "ep", env=active, capture=False)
    # The wrapper's dependency resolves to an unsuitable PyPI uccl-cu12 wheel. The source
    # build above already supplies uccl.ep on both CUDA and ROCm, so retain --no-deps.
    pip_fallback(["--no-deps", "."], env, cwd=source / "ep/deep_ep_wrapper", docker=docker)
    site = Path(
        run(
            ["python3", "-c", "import site; print(site.getsitepackages()[0])"], env=env
        ).stdout.strip()
    )
    destination = root if docker else root / "site"
    destination.mkdir(exist_ok=True)
    for pattern in ("deep_ep*", "uccl*"):
        matches = sorted(site.glob(pattern))
        if not matches:
            raise RuntimeError(f"UCCL cache population failed: {pattern}")
        for path in matches:
            target = destination / path.name
            if path.is_dir():
                shutil.copytree(path, target, symlinks=True)
            else:
                shutil.copy2(path, target, follow_symlinks=False)
    if docker:
        python_command(
            {**env, "PYTHONPATH": str(root)}, "check-backend", "uccl-docker", capture=False
        )
    (root / ".ready").touch()


def site_environment(root: Path, backend: str, env: dict[str, str]) -> dict[str, str]:
    site = root / "site"
    if not site.is_dir():
        raise RuntimeError(f"{backend} cache site is unavailable")
    result = {
        **env,
        "PYTHONPATH": str(site) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
    }
    if backend == "uccl-ep" and env.get("COLLX_VENDOR", "nvidia") == "amd":
        result["UCCL_EP_ENABLE_AGGRESSIVE_ATOMIC"] = "1"
    if backend == "nccl-ep":
        # The wheel's newer NCCL must precede torch's loader path, and Device API symmetric
        # allocations require cuMem. Persist both settings for every subsequent rank.
        libraries = sorted((site / "nvidia").glob("nccl*/lib"))
        if libraries:
            result["LD_LIBRARY_PATH"] = str(libraries[0]) + (
                ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
            )
        result["NCCL_CUMEM_ENABLE"] = "1"
    return result


def install_nccl(root: Path, env: dict[str, str]) -> None:
    reset_cache(root)
    (root / "site").mkdir()
    # --target avoids the image's externally-managed system environment (PEP668).
    run(
        [
            "python3",
            "-m",
            "pip",
            "install",
            "-q",
            "--disable-pip-version-check",
            "--no-input",
            "--target",
            str(root / "site"),
            *NCCL_EP_SPECS,
        ],
        env=env,
        capture=False,
    )
    python_command(
        site_environment(root, "nccl-ep", env), "check-backend", "nccl-ep", capture=False
    )
    (root / ".ready").touch()


def prepare_backend(backend: str, runner: str, env: dict[str, str]) -> dict[str, str]:
    """Prepare exactly one backend per node, then return its rank environment."""
    if backend in ("mori", "flashinfer-ep"):
        python_command(env, "check-backend", backend, capture=False)
        return dict(env)
    if backend not in ("deepep-v2", "uccl-ep", "nccl-ep"):
        raise ValueError("unknown backend preparation request")
    if backend == "uccl-ep" and env.get("COLLX_VENDOR", "nvidia") == "amd":
        from .config import _platforms

        arch = _platforms()[runner]["arch"]
    else:
        arch = cuda_arch(runner, env)
    root = cache_root(backend, arch, env)
    if backend == "deepep-v2":
        if root is None:
            raise RuntimeError("DeepEP V2 shared cache root is unavailable")
        install_cached(root, "venv", install_deepep, arch, env)
        active = deepep_environment(root, env)
        interpreter = str(root / "venv/bin/python")
    else:
        installer, arguments = (
            (install_uccl, (arch, env)) if backend == "uccl-ep" else (install_nccl, (env,))
        )
        if root is None:
            suffix = UCCL_COMMIT if backend == "uccl-ep" else slug(" ".join(NCCL_EP_SPECS))
            label = "uccl" if backend == "uccl-ep" else "nccl-ep"
            root = Path(f"/tmp/collectivex-{label}-cache-{suffix}")
            if not cache_ready(root, "site"):
                installer(root, *arguments)
        else:
            install_cached(root, "site", installer, *arguments)
        active = site_environment(root, backend, env)
        interpreter = "python3"
    python_command(active, "check-backend", backend, interpreter=interpreter, capture=False)
    log(f"{backend} ready")
    return active


def write_rank_environment(root: Path, node_id: str, backend: str, env: dict[str, str]) -> None:
    """Persist only the allowlisted settings; ranks consume data instead of sourcing code."""
    if not re.fullmatch(r"[0-9]+", node_id):
        raise ValueError("invalid Slurm node identity")
    write_json(
        root / ".collx_backend/env" / f"node-{node_id}.json",
        {
            "set": {name: env[name] for name in RANK_ENV_VARS if name in env},
            "unset": list(DEEPEP_UNSETS) if backend == "deepep-v2" else [],
        },
    )
