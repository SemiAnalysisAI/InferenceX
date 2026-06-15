#!/usr/bin/env python3
"""Apply the pinned vLLM eager-path patch for the M3 AITER AR+RMS experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import os
import stat
import tempfile
from pathlib import Path


PATCH_MARKER = "InferenceX M3 AITER AR+RMS experiment"
EXPECTED_SOURCE_SHA256 = {
    "vllm/model_executor/layers/fused_allreduce_gemma_rms_norm.py": (
        "0169300358b731605508407ff39e9376497866b102b105f1e96113b80a658eac"
    ),
}
EXPECTED_PATCHED_SHA256 = {
    "vllm/model_executor/layers/fused_allreduce_gemma_rms_norm.py": (
        "fc018fb9e89af9f734e7f7c0f5618f3d280bd83fd004781fd9dfd2665559eec6"
    ),
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one source anchor, found {count}")
    return text.replace(old, new, 1)


def patch_helper_source(source: str) -> str:
    """Use AITER directly from M3's eager allreduce+Gemma RMSNorm helper."""
    if PATCH_MARKER in source:
        return source

    source = _replace_once(
        source,
        "import torch\n\nfrom vllm.distributed.communication_op",
        "import os\n\nimport torch\n\nfrom vllm.distributed.communication_op",
        "fused_allreduce_gemma_rms_norm import",
    )
    return _replace_once(
        source,
        """    # Fallback: explicit all-reduce + GemmaRMSNorm (matches the unfused model).
    reduced = tensor_model_parallel_all_reduce(hidden_states)
    return norm(reduced, residual)
""",
        """    is_m3_rocm_norm = (
        type(norm).__module__ == "vllm.models.minimax_m3.amd.model"
        and type(norm).__name__ == "MiniMAXGemmaRMSNorm"
    )
    use_m3_aiter = (
        os.getenv("M3_AITER_AR_RMS_MODE") == "fused"
        and is_m3_rocm_norm
        and hidden_states.dim() == 2
        and hidden_states.is_contiguous()
        and hidden_states.dtype in (torch.bfloat16, torch.float16)
        and hidden_states.shape[0] <= 512
    )
    if use_m3_aiter:
        # InferenceX M3 AITER AR+RMS experiment: M3 runs eager, so invoke
        # AITER directly instead of relying on a torch.compile fusion pass.
        from vllm._aiter_ops import rocm_aiter_ops

        aiter_ar = rocm_aiter_ops.get_aiter_allreduce()
        if aiter_ar is None:
            rocm_aiter_ops.initialize_aiter_allreduce(
                get_tp_group().cpu_group, hidden_states.device
            )
            aiter_ar = rocm_aiter_ops.get_aiter_allreduce()
        if aiter_ar is None:
            raise RuntimeError("AITER custom allreduce initialization failed")
        if not hasattr(aiter_ar, "_pool") and hidden_states.shape[-1] not in (
            512,
            1024,
            2048,
            4096,
        ):
            raise RuntimeError(
                "AITER <0.1.12 does not support M3's allreduce hidden size"
            )

        gamma = getattr(norm, "_inferencex_aiter_gamma", None)
        if (
            gamma is None
            or gamma.device != hidden_states.device
            or gamma.dtype != hidden_states.dtype
        ):
            gamma = (
                norm.weight.detach().to(
                    device=hidden_states.device, dtype=hidden_states.dtype
                )
                + 1.0
            ).contiguous()
            norm._inferencex_aiter_gamma = gamma

        # amd-aiter 0.1.13.post1's one-stage kernel lacks the exit barrier
        # fixed by ROCm/aiter#3514 and is unsafe under HIP graph replay.
        result = aiter_ar.fused_ar_rms(
            hidden_states,
            residual,
            w=gamma,
            eps=norm.variance_epsilon,
            registered=torch.cuda.is_current_stream_capturing(),
            use_1stage=False,
        )
        if result is None:
            raise RuntimeError("AITER fused allreduce RMSNorm returned no result")
        return result[0], result[1]

    # Fallback: explicit all-reduce + GemmaRMSNorm (matches the unfused model).
    reduced = tensor_model_parallel_all_reduce(hidden_states)
    return norm(reduced, residual)
""",
        "fused_allreduce_gemma_rms_norm fallback",
    )


def find_vllm_root() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("Unable to locate the installed vLLM package")
    return Path(spec.origin).resolve().parent.parent


def _write_atomic(path: Path, text: str) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temp_file:
        temp_file.write(text)
        temp_path = Path(temp_file.name)
    temp_path.chmod(mode)
    os.replace(temp_path, path)


def apply_runtime_patch(vllm_root: Path, check_only: bool = False) -> None:
    transforms = {
        "vllm/model_executor/layers/fused_allreduce_gemma_rms_norm.py": (
            patch_helper_source
        ),
    }
    originals: dict[Path, str] = {}
    patched: dict[Path, str] = {}

    for relative_path, transform in transforms.items():
        path = vllm_root / relative_path
        source = path.read_text(encoding="utf-8")
        originals[path] = source
        if PATCH_MARKER in source:
            actual_sha = _sha256(source)
            expected_sha = EXPECTED_PATCHED_SHA256[relative_path]
            if actual_sha != expected_sha:
                raise RuntimeError(
                    f"{relative_path}: patched source fingerprint mismatch; "
                    f"expected {expected_sha}, got {actual_sha}"
                )
            compile(source, str(path), "exec")
            patched[path] = source
            continue

        actual_sha = _sha256(source)
        expected_sha = EXPECTED_SOURCE_SHA256[relative_path]
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"{relative_path}: source fingerprint mismatch; "
                f"expected {expected_sha}, got {actual_sha}"
            )
        patched_source = transform(source)
        actual_patched_sha = _sha256(patched_source)
        expected_patched_sha = EXPECTED_PATCHED_SHA256[relative_path]
        if actual_patched_sha != expected_patched_sha:
            raise RuntimeError(
                f"{relative_path}: generated patch fingerprint mismatch; "
                f"expected {expected_patched_sha}, got {actual_patched_sha}"
            )
        patched[path] = patched_source
        compile(patched_source, str(path), "exec")

    marker_count = sum(PATCH_MARKER in source for source in originals.values())
    if marker_count not in (0, len(originals)):
        raise RuntimeError("vLLM runtime patch is only partially applied")

    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    action = "validated" if check_only else "applied"
    if not check_only:
        for path, source in patched.items():
            _write_atomic(path, source)

    print(f"M3 AITER AR+RMS runtime patch {action} for vLLM {version}")
    for path, source in patched.items():
        print(f"  {path.relative_to(vllm_root)} sha256={_sha256(source)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-root", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    root = args.vllm_root.resolve() if args.vllm_root else find_vllm_root()
    apply_runtime_patch(root, check_only=args.check)


if __name__ == "__main__":
    main()
