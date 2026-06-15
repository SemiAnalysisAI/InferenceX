#!/usr/bin/env python3
"""Apply the pinned vLLM runtime patch for the M3 AITER AR+RMS experiment."""

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
    "vllm/compilation/passes/fusion/allreduce_rms_fusion.py": (
        "4c34948c820c514d20a85f8d06fdff62ffa4601e62feef61e80c1fab57c2ed39"
    ),
}
EXPECTED_PATCHED_SHA256 = {
    "vllm/model_executor/layers/fused_allreduce_gemma_rms_norm.py": (
        "bd40e3470a0b07dacc630eec788d84cf14ca4b761e6f4832a16c719a2f8fe7a1"
    ),
    "vllm/compilation/passes/fusion/allreduce_rms_fusion.py": (
        "70b86423139ca96c498b807086ff142197300c7ae240a6172fc4e92daf62c43f"
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
    """Expose the M3 post-attention Gemma norm as vLLM IR."""
    if PATCH_MARKER in source:
        return source

    source = _replace_once(
        source,
        "import torch\n\nfrom vllm.distributed.communication_op",
        "import torch\n\nimport vllm.ir.ops\nfrom vllm.distributed.communication_op",
        "fused_allreduce_gemma_rms_norm import",
    )
    return _replace_once(
        source,
        """    # Fallback: explicit all-reduce + GemmaRMSNorm (matches the unfused model).
    reduced = tensor_model_parallel_all_reduce(hidden_states)
    return norm(reduced, residual)
""",
        """    # Fallback: explicit all-reduce + GemmaRMSNorm (matches the unfused model).
    reduced = tensor_model_parallel_all_reduce(hidden_states)
    if (
        type(norm).__module__ == "vllm.models.minimax_m3.amd.model"
        and type(norm).__name__ == "MiniMAXGemmaRMSNorm"
    ):
        # InferenceX M3 AITER AR+RMS experiment: expose the post-attention
        # operation to the compiler while preserving Gemma's (1 + weight).
        weight = norm.weight.data.to(reduced.dtype) + 1.0
        return vllm.ir.ops.fused_add_rms_norm(
            reduced, residual, weight, norm.variance_epsilon
        )
    return norm(reduced, residual)
""",
        "fused_allreduce_gemma_rms_norm fallback",
    )


def patch_fusion_source(source: str) -> str:
    """Add PR #43953's copy-aware AITER pattern and decode-only guard."""
    if PATCH_MARKER in source:
        return source

    copy_aware_pattern = '''

class AiterAllreduceFusedAddRMSNormWithCopyPattern(
    BasePattern, VllmPatternReplacement
):
    """Non-quant AR+RMS fusion for all_reduce with an external copy_ user."""

    def __init__(self, epsilon, dtype, device):
        super().__init__(dtype, device)
        self.epsilon = epsilon
        self.FUSED_OP = rocm_aiter_ops.get_fused_allreduce_rmsnorm_op()

    def get_inputs(self):
        return [self.empty(5, 16), self.empty(5, 16), self.empty(16)]

    @property
    def pattern(self):
        eps = self.epsilon

        def _pattern(residual, input_, weight):
            ar_out = tensor_model_parallel_all_reduce(input_)
            rms, res_out = vllm.ir.ops.fused_add_rms_norm(
                ar_out, residual, weight, eps
            )
            return rms, res_out, ar_out

        return _pattern

    @property
    def replacement(self):
        eps = self.epsilon

        def _replacement(residual, input_, weight):
            fused = self.FUSED_OP(
                input_=input_,
                residual=residual,
                weight=weight.to(input_.dtype),
                epsilon=eps,
            )
            return fused[0], fused[1], fused[1]

        return _replacement
'''
    source = _replace_once(
        source,
        "\n\nclass RocmAiterAllReduceFusionPass(VllmFusionPatternMatcherPass):",
        copy_aware_pattern
        + "\n\nclass RocmAiterAllReduceFusionPass(VllmFusionPatternMatcherPass):",
        "copy-aware AITER pattern insertion",
    )
    source = _replace_once(
        source,
        """        self.max_token_num = min(
            max_token_num,
            config.scheduler_config.max_num_batched_tokens,
        )
""",
        """        # InferenceX M3 AITER AR+RMS experiment: keep the fusion on
        # cudagraph decode ranges; prefill retains quick-reduce + Triton norm.
        max_cg = config.compilation_config.max_cudagraph_capture_size or 512
        self.max_token_num = min(
            max_token_num,
            config.scheduler_config.max_num_batched_tokens,
            max_cg,
        )
""",
        "AITER allreduce decode range guard",
    )
    return _replace_once(
        source,
        """        for epsilon in [1e-5, 1e-6]:
            # Quant-fused variants must register first so the pattern matcher
""",
        """        for epsilon in [1e-5, 1e-6]:
            self.register(
                AiterAllreduceFusedAddRMSNormWithCopyPattern(
                    epsilon, self.model_dtype, self.device
                )
            )
            torch._inductor.pattern_matcher._seen_patterns.clear()

        for epsilon in [1e-5, 1e-6]:
            # Quant-fused variants must register first so the pattern matcher
""",
        "copy-aware AITER pattern registration",
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
        "vllm/compilation/passes/fusion/allreduce_rms_fusion.py": (
            patch_fusion_source
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
