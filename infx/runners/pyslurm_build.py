"""Build pyslurm 25.11.2 against the system (or vendored) Slurm headers.

Usage:
    python3 -m infx.runners.pyslurm_build --slurm-include /usr/include --out /tmp/pyslurm_wheel
    python3 -m infx.runners.pyslurm_build --cache-root ~/.cache/inferencex/pyslurm

``--cache-root`` builds once per content key (vendored tree, patches, headers,
Slurm version, Python version, arch) and prints ``PYSLURM_PATH=<dir>``, an
importable directory reused by later runs on the same host.

The builder:
1. Copies the vendored pyslurm source to a temp directory.
2. Detects the Slurm version from the headers.
3. Applies the matching patch series (e.g. 25.05.x patches).
4. Runs Cython + setuptools to build an in-tree extension or wheel.
5. Prints the importable path on success.

If system Slurm headers are not found, falls back to vendored headers
under third_party/slurm-headers/<version>/ that match the runtime Slurm
version (detected from ``sinfo -V`` or ``scontrol show config``).
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_DIR = REPO_ROOT / "third_party" / "pyslurm"
PATCHES_DIR = VENDOR_DIR / "patches"
HEADERS_DIR = REPO_ROOT / "third_party" / "slurm-headers"


def _detect_runtime_slurm_version() -> str | None:
    """Try to get the Slurm version string from sinfo -V."""
    try:
        out = subprocess.check_output(["sinfo", "-V"], text=True, timeout=5).strip()
        # "slurm 25.05.7" -> "25.05.7"
        m = re.search(r"(\d+\.\d+\.\d+)", out)
        return m.group(1) if m else None
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def _parse_version_number_from_header(include_dir: Path) -> int | None:
    """Read SLURM_VERSION_NUMBER from slurm_version.h."""
    vh = include_dir / "slurm" / "slurm_version.h"
    if not vh.exists():
        return None
    for line in vh.read_text().splitlines():
        if "SLURM_VERSION_NUMBER" in line and "#define" in line:
            m = re.search(r"0x([0-9a-fA-F]+)", line)
            if m:
                return int(m.group(1), 16)
    return None


def _version_number_to_major_minor(vnum: int) -> tuple[int, int]:
    """Extract (major, minor) from SLURM_VERSION_NUMBER."""
    major = (vnum >> 16) & 0xFF
    minor = (vnum >> 8) & 0xFF
    return major, minor


def _find_slurm_include(args_include: str | None) -> Path:
    """Resolve the Slurm include directory, falling back to vendored headers."""
    # 1. Explicit --slurm-include
    if args_include:
        p = Path(args_include)
        if (p / "slurm" / "slurm.h").exists():
            return p
        # Maybe they pointed at the slurm/ subdir
        if (p / "slurm.h").exists():
            return p.parent

    # 2. System locations
    for candidate in [
        Path("/usr/include"),
        Path("/usr/local/include"),
    ]:
        if (candidate / "slurm" / "slurm.h").exists():
            print(f"[pyslurm-build] Found system Slurm headers at {candidate}")
            return candidate

    # 3. Vendored fallback — match runtime version
    runtime_ver = _detect_runtime_slurm_version()
    if runtime_ver:
        # Try exact match first, then major.minor match
        for pattern in [runtime_ver, ".".join(runtime_ver.split(".")[:2])]:
            for d in sorted(HEADERS_DIR.iterdir()) if HEADERS_DIR.exists() else []:
                if d.name.startswith(pattern) and (d / "slurm" / "slurm.h").exists():
                    print(
                        f"[pyslurm-build] Using vendored Slurm {d.name} headers "
                        f"(runtime Slurm {runtime_ver})"
                    )
                    return d

    # 4. Fall back to any vendored 25.05.x headers
    if HEADERS_DIR.exists():
        for d in sorted(HEADERS_DIR.iterdir()):
            if d.name.startswith("25.05") and (d / "slurm" / "slurm.h").exists():
                print(f"[pyslurm-build] Falling back to vendored Slurm {d.name} headers")
                return d

    raise FileNotFoundError("Cannot find Slurm headers. Pass --slurm-include or install slurm-dev.")


def _slurm_plugin_dir() -> Path | None:
    """Return Slurm's PluginDir from ``scontrol show config``, if available."""
    try:
        out = subprocess.check_output(["scontrol", "show", "config"], text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "PluginDir" and value.strip():
            return Path(value.strip().split(":")[0])
    return None


def _find_slurm_lib() -> Path:
    """Find the directory holding libslurmfull.so.

    pyslurm uses Slurm-internal symbols (e.g. assoc_mgr_tres_list) that only
    libslurmfull exports; linking the public libslurm builds a wheel that
    fails at import. Distributions install libslurmfull beside the plugins.
    """
    roots = [
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/lib64"),
        Path("/usr/lib/aarch64-linux-gnu"),
        Path("/usr/local/lib"),
        Path("/usr/lib"),
    ]
    plugin_dir = _slurm_plugin_dir()
    candidates = ([plugin_dir] if plugin_dir else []) + [d / "slurm" for d in roots] + roots
    for d in candidates:
        if (d / "libslurmfull.so").exists():
            return d
    searched = ", ".join(str(d) for d in candidates)
    raise FileNotFoundError(f"Cannot find libslurmfull.so (searched {searched})")


def _apply_patches(build_dir: Path, slurm_major_minor: tuple[int, int]) -> None:
    """Apply the patch series matching the detected Slurm version."""
    major, minor = slurm_major_minor
    if major == 25 and minor == 5:
        patch_prefix = "slurm-2505"
    elif major == 25 and minor == 11:
        print("[pyslurm-build] Slurm 25.11 detected — no patches needed")
        return
    else:
        print(
            f"[pyslurm-build] WARNING: Slurm {major}.{minor:02d} — "
            f"no patches available, trying unpatched build"
        )
        return

    patches = sorted(PATCHES_DIR.glob(f"*-{patch_prefix}-*.patch"))
    if not patches:
        raise FileNotFoundError(f"No patches found for Slurm {major}.{minor:02d} in {PATCHES_DIR}")

    for p in patches:
        print(f"[pyslurm-build] Applying {p.name}")
        subprocess.check_call(
            # Patches use a/ b/ prefixes relative to the pyslurm source root.
            ["git", "apply", "-p1", str(p)],
            cwd=build_dir,
        )


def _build_wheel(build_dir: Path, include_dir: Path, lib_dir: Path, out_dir: Path) -> Path:
    """Cythonize and build pyslurm, return the wheel path."""
    env = os.environ.copy()
    env["SLURM_INCLUDE_DIR"] = str(include_dir)
    env["SLURM_LIB_DIR"] = str(lib_dir)

    # Determine which library name to link against
    lib_name = "slurmfull" if (lib_dir / "libslurmfull.so").exists() else "slurm"

    # Patch setup.py to use the correct library name if needed
    setup_py = build_dir / "setup.py"
    content = setup_py.read_text()
    if lib_name != "slurmfull":
        content = content.replace('SLURM_LIB = "libslurmfull"', f'SLURM_LIB = "lib{lib_name}"')

    # Disable the version check — we've patched the declarations
    content = content.replace(
        "if Version(self.version) != Version(SLURM_VERSION):",
        "if False:  # Version check disabled by pyslurm_build patch",
    )
    setup_py.write_text(content)

    # Build wheel
    subprocess.check_call(
        [sys.executable, "setup.py", "bdist_wheel", "-d", str(out_dir)],
        cwd=build_dir,
        env=env,
    )

    wheels = list(out_dir.glob("pyslurm-*.whl"))
    if not wheels:
        raise RuntimeError("No wheel produced")
    return wheels[0]


def _build_inplace(build_dir: Path, include_dir: Path, lib_dir: Path) -> Path:
    """Build pyslurm extensions in-place, return the pyslurm package dir."""
    env = os.environ.copy()
    env["SLURM_INCLUDE_DIR"] = str(include_dir)
    env["SLURM_LIB_DIR"] = str(lib_dir)

    lib_name = "slurmfull" if (lib_dir / "libslurmfull.so").exists() else "slurm"

    setup_py = build_dir / "setup.py"
    content = setup_py.read_text()
    if lib_name != "slurmfull":
        content = content.replace('SLURM_LIB = "libslurmfull"', f'SLURM_LIB = "lib{lib_name}"')
    content = content.replace(
        "if Version(self.version) != Version(SLURM_VERSION):",
        "if False:  # Version check disabled by pyslurm_build patch",
    )
    setup_py.write_text(content)

    subprocess.check_call(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=build_dir,
        env=env,
    )
    return build_dir / "pyslurm"


def _content_key(include_dir: Path, lib_dir: Path, vnum: int) -> str:
    """Hash every input that changes the built extension."""
    digest = hashlib.sha256()
    for root in (VENDOR_DIR, include_dir / "slurm"):
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    digest.update(f"{vnum}|{lib_dir}|{sys.version_info[:2]}|{platform.machine()}".encode())
    return digest.hexdigest()[:16]


def _cached_build(cache_root: Path, build: callable, key: str, label: str) -> Path:
    """Return an importable pyslurm directory, building it at most once per key."""
    dest = cache_root / f"pyslurm-{label}-{key}"
    marker = dest / ".complete"
    if marker.exists():
        print(f"[pyslurm-build] Cache hit: {dest}")
        return dest
    cache_root.mkdir(parents=True, exist_ok=True)
    with (cache_root / ".lock").open("w") as lock:
        # Runners on one login host share the cache; build under an exclusive lock.
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker.exists():
            print(f"[pyslurm-build] Cache hit after wait: {dest}")
            return dest
        staging = Path(tempfile.mkdtemp(prefix=f"{dest.name}.", dir=cache_root))
        wheel = build(staging / "dist")
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(staging / "site")
        (staging / "site" / ".complete").write_text(wheel.name + "\n")
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(dest)
        (staging / "site").rename(dest)
        shutil.rmtree(staging, ignore_errors=True)
    print(f"[pyslurm-build] Cached build: {dest}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pyslurm against system Slurm")
    parser.add_argument("--slurm-include", help="Path to Slurm include dir")
    parser.add_argument("--out", help="Output directory for the wheel")
    parser.add_argument(
        "--cache-root",
        help="Build once per content key under this directory and print PYSLURM_PATH",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Build in-place instead of a wheel; print the importable pyslurm path",
    )
    args = parser.parse_args()

    include_dir = _find_slurm_include(args.slurm_include)
    vnum = _parse_version_number_from_header(include_dir)
    if vnum is None:
        raise RuntimeError(f"Cannot read SLURM_VERSION_NUMBER from {include_dir}")

    major, minor = _version_number_to_major_minor(vnum)
    print(
        f"[pyslurm-build] Slurm version: {major}.{minor:02d} "
        f"(0x{vnum:06x}), headers at {include_dir}"
    )

    lib_dir = _find_slurm_lib()
    print(f"[pyslurm-build] Slurm library dir: {lib_dir}")

    if args.cache_root:
        key = _content_key(include_dir, lib_dir, vnum)
        label = f"{major}.{minor:02d}-cp{sys.version_info[0]}{sys.version_info[1]}"

        def build(out_dir: Path) -> Path:
            tree = _prepare_tree(major, minor)
            out_dir.mkdir(parents=True, exist_ok=True)
            return _build_wheel(tree, include_dir, lib_dir, out_dir)

        path = _cached_build(Path(args.cache_root).expanduser(), build, key, label)
        print(f"PYSLURM_PATH={path}")
        return

    build_dir = _prepare_tree(major, minor)
    _finish(args, build_dir, include_dir, lib_dir)


def _prepare_tree(major: int, minor: int) -> Path:
    """Copy the vendored source to a temp dir and apply the matching patches."""
    build_dir = Path(tempfile.mkdtemp(prefix="pyslurm-build-"))
    print(f"[pyslurm-build] Build directory: {build_dir}")

    shutil.copytree(
        VENDOR_DIR,
        build_dir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("patches", "VENDORED.md"),
    )

    _apply_patches(build_dir, (major, minor))
    return build_dir


def _finish(args: argparse.Namespace, build_dir: Path, include_dir: Path, lib_dir: Path) -> None:
    """Build in place or as a wheel, per the CLI flags."""
    if args.inplace:
        pkg_dir = _build_inplace(build_dir, include_dir, lib_dir)
        print(f"\n[pyslurm-build] SUCCESS — importable from: {pkg_dir}")
        print(f"PYSLURM_PATH={build_dir}")
    else:
        out_dir = Path(args.out) if args.out else build_dir / "dist"
        out_dir.mkdir(parents=True, exist_ok=True)
        whl = _build_wheel(build_dir, include_dir, lib_dir, out_dir)
        print(f"\n[pyslurm-build] SUCCESS — wheel at: {whl}")
        print(f"PYSLURM_WHEEL={whl}")


if __name__ == "__main__":
    main()
