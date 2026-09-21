"""Preserve native V4.1 DSpark FP8 WO_A weights in the official SGLang nightly.

Applies only to the exact upstream source shipped at 0f6761b54facebb47f2068f87ecccd8f14da3a0e.
The dense FP8 linear path supports the checkpoint's 32x32 scales; the upstream
grouped projection instead dequantizes these draft weights permanently to BF16.
"""

import hashlib
import importlib.util
import subprocess
from pathlib import Path

BASE_SHA256 = "61dc79f075c9e1e5a68a466de5eb95a85ff1fa2e5a9f4d66c2fa69e956fd7b61"
PATCHED_SHA256 = "f33b370d9e49f87c5909e23109c1f22968166690765a7df169144d149f0f3d87"
HELPER_SHA256 = "9e0d5e2cb5afe0d729ffffa160f1b44ddcca163e44a675a393154a6148a610de"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_native_wo_a_patch(package_root: Path) -> None:
    model = package_root / "srt/models/deepseek_v4_dspark.py"
    helper = package_root / "srt/models/dspark_projection.py"
    actual = digest(model)
    if (
        actual == PATCHED_SHA256
        and helper.is_file()
        and digest(helper) == HELPER_SHA256
    ):
        print(f"Native DSpark WO_A patch already applied: sha256:{actual}")
        return
    if actual != BASE_SHA256 or helper.exists():
        raise RuntimeError(
            "Refusing to patch unexpected SGLang DSpark source: "
            f"model sha256:{actual}; expected sha256:{BASE_SHA256}. "
            "Revalidate the native FP8 fix when changing the serving image."
        )
    patch = Path(__file__).parent / "patches/sglang_dsv41_native_wo_a.patch"
    command = [
        "patch",
        "--batch",
        "--forward",
        "--fuzz=0",
        "-p3",
        "-i",
        str(patch.resolve()),
    ]
    subprocess.run([*command, "--dry-run"], cwd=package_root, check=True)
    subprocess.run(command, cwd=package_root, check=True)
    if digest(model) != PATCHED_SHA256 or digest(helper) != HELPER_SHA256:
        raise RuntimeError(
            "Patched DSpark source does not match the validated native FP8 fix"
        )
    print(f"Applied native DSpark WO_A patch: sha256:{PATCHED_SHA256}")


def main() -> None:
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        raise RuntimeError("SGLang package is unavailable in the serving container")
    apply_native_wo_a_patch(Path(spec.origin).parent)


if __name__ == "__main__":
    main()
