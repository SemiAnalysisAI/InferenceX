"""Install the digest-pinned, V4.1-only ROCm attention source backport."""

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

ANCHOR = """    elif _is_hip:
        from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
"""
REPLACEMENT = """    elif _is_hip:
        if runner.model_config.hf_text_config.model_type == "deepseek_v41":
            from sglang.srt.layers.attention.dsv41_rocm.backend import (
                DeepseekV4HipRadixBackend as DeepseekV41BackportBackend,
            )

            logger.info("Using digest-pinned V4.1 ROCm source backport on latest nightly.")
            return DeepseekV41BackportBackend(runner)
        from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
"""
FP8_ANCHOR = (
    "    # Only Triton reads the block size at launch; DeepGEMM, the FlashInfer\n"
)
FP8_REPLACEMENT = (
    """    if _is_hip and _is_gfx95_supported and weight_block_size == [32, 32] and act_scale_ue8m0:
        from sglang.srt.runtime_context import process_model_config

        config = process_model_config().hf_text_config
        if getattr(config, "model_type", None) == "deepseek_v41":
            from sglang.srt.layers.attention.dsv41_rocm.fp8_linear import (
                rocm_v41_block_fp8_linear,
            )

            return rocm_v41_block_fp8_linear
"""
    + FP8_ANCHOR
)
HOST_REGISTER_ANCHOR = """        if int(err) != 0:
            raise RuntimeError(f"cudaHostRegister({nbytes} bytes) failed: {err}")
"""
HOST_REGISTER_REPLACEMENT = (
    HOST_REGISTER_ANCHOR
    + """        self.device_ptr = self.bytes.data_ptr()
        if torch.version.hip is not None:
            from sglang.srt.layers.attention.dsv41_rocm.host_table import (
                hip_host_device_pointer,
            )

            self.device_ptr = hip_host_device_pointer(self.device_ptr)
"""
)
HOST_METHOD_ANCHOR = (
    "    def _load_rows(self, param: nn.Parameter, loaded_weight: torch.Tensor):\n"
)
HOST_METHOD_REPLACEMENT = (
    """    def _table_pointer(self, tensor: torch.Tensor) -> int:
        if self.host_table is None or torch.version.hip is None:
            return tensor.data_ptr()
        return (
            self.host_table.device_ptr
            + tensor.data_ptr()
            - self.host_table.bytes.data_ptr()
        )

"""
    + HOST_METHOD_ANCHOR
)
ENGRAM_REWRITES = (
    (HOST_REGISTER_ANCHOR, HOST_REGISTER_REPLACEMENT, 1),
    (HOST_METHOD_ANCHOR, HOST_METHOD_REPLACEMENT, 1),
    ("self.weight.data_ptr()", "self._table_pointer(self.weight)", 2),
    ("self.scale.data_ptr()", "self._table_pointer(self.scale)", 2),
)
MODEL_ANCHOR = "    return x_quant, x_bf16\n"
MODEL_REPLACEMENT = (
    """    if _is_hip and _is_gfx95_supported:
        from sglang.srt.runtime_context import process_model_config

        if process_model_config().hf_text_config.model_type == "deepseek_v41":
            # The V4 fused quantizer emits 128-wide groups. Keep its existing
            # normalized BF16 row; the V4.1 linear applies native 32-wide UE8M0.
            return x_bf16, x_bf16
"""
    + MODEL_ANCHOR
)

MLP_ANCHOR = "        self.use_fused_clamp_act_mul = _is_hip\n"
MLP_REPLACEMENT = (
    MLP_ANCHOR
    + '        if _is_hip:\n            from sglang.srt.runtime_context import process_model_config\n\n            if process_model_config().hf_text_config.model_type == "deepseek_v41":\n                # AITER\'s fused path requires 128-wide groups/alignment. V4.1\n                # uses 32-wide groups and shared-expert widths such as 576.\n                self.use_fused_clamp_act_mul = False\n                self._infx_v41_hip = True\n'
)
MLP_CALL_ANCHOR = (
    "                silu_and_mul_clamp(gate_up, x, float(self.swiglu_limit))\n"
)
MLP_CALL_REPLACEMENT = '                if getattr(self, "_infx_v41_hip", False):\n                    from sglang.srt.layers.attention.dsv41_rocm.activation import (\n                        rocm_v41_silu_and_mul_clamp,\n                    )\n\n                    rocm_v41_silu_and_mul_clamp(gate_up, x, float(self.swiglu_limit))\n                else:\n                    silu_and_mul_clamp(gate_up, x, float(self.swiglu_limit))\n'

POOL_REWRITES = (
    (
        "        self.uses_aiter_fp4_layout = _is_hip and self.use_fp4_indexer\n",
        '        self.uses_aiter_fp4_layout = _is_hip and self.use_fp4_indexer\n        self._infx_v41_hip = False\n        if self.uses_aiter_fp4_layout:\n            from sglang.srt.runtime_context import process_model_config\n\n            self._infx_v41_hip = (\n                process_model_config().hf_text_config.model_type == "deepseek_v41"\n            )\n',
    ),
    (
        "        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (\n            store_fp4_index_k_cache,\n        )\n",
        "        if self._infx_v41_hip:\n            from sglang.srt.layers.attention.dsv41_rocm.fp4_indexer import (\n                store_fp4_index_k_cache_split,\n            )\n\n            return store_fp4_index_k_cache_split(\n                cache_k,\n                self.index_k_payload_buffer[layer_id - self.start_layer],\n                self.index_k_scale_buffer[layer_id - self.start_layer],\n                loc,\n                page_size=self.page_size,\n                rne=self.index_k_rne,\n            )\n        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (\n            store_fp4_index_k_cache,\n        )\n",
    ),
    (
        '        assert self.use_fp4_indexer, "packed readback only applies to the fp4 layout"\n',
        '        assert self.use_fp4_indexer, "packed readback only applies to the fp4 layout"\n        if self._infx_v41_hip:\n            from sglang.srt.layers.attention.dsv41_rocm.fp4_indexer import (\n                read_fp4_index_k_split,\n            )\n\n            return read_fp4_index_k_split(\n                self.index_k_payload_buffer[layer_id - self.start_layer],\n                self.index_k_scale_buffer[layer_id - self.start_layer],\n                slots,\n                page_size=self.page_size,\n            )\n',
    ),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def install(package: Path, evidence: Path) -> None:
    source = Path(__file__).resolve().parent
    manifest = json.loads((source / "provenance.json").read_text())
    registry = package / "srt/layers/attention/attention_registry.py"
    original = registry.read_text()
    if REPLACEMENT in original:
        original = original.replace(REPLACEMENT, ANCHOR, 1)
    if sha256(original.encode()) != manifest["registry_sha256"]:
        raise RuntimeError(
            "Unexpected SGLang attention registry; refusing to patch another revision"
        )
    if original.count(ANCHOR) != 1:
        raise RuntimeError("Expected exactly one HIP dsv4 registry anchor")
    fp8_utils = package / "srt/layers/quantization/fp8_utils.py"
    fp8_original = fp8_utils.read_text()
    if FP8_REPLACEMENT in fp8_original:
        fp8_original = fp8_original.replace(FP8_REPLACEMENT, FP8_ANCHOR, 1)
    if sha256(fp8_original.encode()) != manifest["fp8_utils_sha256"]:
        raise RuntimeError(
            "Unexpected SGLang FP8 dispatch source; refusing to patch another revision"
        )
    if fp8_original.count(FP8_ANCHOR) != 1:
        raise RuntimeError("Expected exactly one block-FP8 dispatch anchor")
    engram = package / "srt/layers/engram.py"
    engram_original = engram.read_text()
    for before, after, count in ENGRAM_REWRITES:
        if after in engram_original:
            engram_original = engram_original.replace(after, before, count)
    if sha256(engram_original.encode()) != manifest["engram_sha256"]:
        raise RuntimeError(
            "Unexpected Engram source; refusing to patch another revision"
        )
    for before, _, count in ENGRAM_REWRITES:
        if engram_original.count(before) != count:
            raise RuntimeError("Unexpected Engram host-pointer patch anchor count")
    model = package / "srt/models/deepseek_v4.py"
    model_original = model.read_text()
    if MODEL_REPLACEMENT in model_original:
        model_original = model_original.replace(MODEL_REPLACEMENT, MODEL_ANCHOR, 1)
    if sha256(model_original.encode()) != manifest["model_sha256"]:
        raise RuntimeError(
            "Unexpected V4 model source; refusing to patch another revision"
        )
    if model_original.count(MODEL_ANCHOR) != 1:
        raise RuntimeError("Unexpected V4 fused-normalization patch anchor count")
    mlp = package / "srt/models/deepseek_v2.py"
    mlp_original = mlp.read_text()
    if MLP_REPLACEMENT in mlp_original:
        mlp_original = mlp_original.replace(MLP_REPLACEMENT, MLP_ANCHOR, 1)
    if MLP_CALL_REPLACEMENT in mlp_original:
        mlp_original = mlp_original.replace(MLP_CALL_REPLACEMENT, MLP_CALL_ANCHOR, 1)
    if sha256(mlp_original.encode()) != manifest["mlp_sha256"]:
        raise RuntimeError("Unexpected MLP source; refusing to patch another revision")
    if mlp_original.count(MLP_ANCHOR) != 1 or mlp_original.count(MLP_CALL_ANCHOR) != 1:
        raise RuntimeError("Unexpected MLP fused-clamp patch anchor count")
    pool = package / "srt/mem_cache/deepseek_v4_memory_pool.py"
    pool_original = pool.read_text()
    for before, after in POOL_REWRITES:
        if after in pool_original:
            pool_original = pool_original.replace(after, before, 1)
    if sha256(pool_original.encode()) != manifest["pool_sha256"]:
        raise RuntimeError(
            "Unexpected KV-pool source; refusing to patch another revision"
        )
    for before, _ in POOL_REWRITES:
        if pool_original.count(before) != 1:
            raise RuntimeError("Unexpected indexer-pool access patch anchor count")
    for item in manifest["files"]:
        data = (source / "dsv41_rocm" / item["installed_name"]).read_bytes()
        if sha256(data) != item["adapted_sha256"]:
            raise RuntimeError(
                f"Backport source hash mismatch: {item['installed_name']}"
            )
    destination = registry.parent / "dsv41_rocm"
    destination.mkdir(exist_ok=True)
    for path in (source / "dsv41_rocm").glob("*.py"):
        shutil.copy2(path, destination / path.name)
    patched = original.replace(ANCHOR, REPLACEMENT, 1)
    registry.write_text(patched)
    fp8_patched = fp8_original.replace(FP8_ANCHOR, FP8_REPLACEMENT, 1)
    fp8_utils.write_text(fp8_patched)
    engram_patched = engram_original
    for before, after, count in ENGRAM_REWRITES:
        engram_patched = engram_patched.replace(before, after, count)
    engram.write_text(engram_patched)
    model_patched = model_original.replace(MODEL_ANCHOR, MODEL_REPLACEMENT, 1)
    model.write_text(model_patched)
    mlp_patched = mlp_original.replace(MLP_ANCHOR, MLP_REPLACEMENT, 1)
    mlp_patched = mlp_patched.replace(MLP_CALL_ANCHOR, MLP_CALL_REPLACEMENT, 1)
    mlp.write_text(mlp_patched)
    pool_patched = pool_original
    for before, after in POOL_REWRITES:
        pool_patched = pool_patched.replace(before, after, 1)
    pool.write_text(pool_patched)
    manifest["installed_registry_sha256"] = sha256(patched.encode())
    manifest["installed_fp8_utils_sha256"] = sha256(fp8_patched.encode())
    manifest["installed_engram_sha256"] = sha256(engram_patched.encode())
    manifest["installed_model_sha256"] = sha256(model_patched.encode())
    manifest["installed_mlp_sha256"] = sha256(mlp_patched.encode())
    manifest["installed_pool_sha256"] = sha256(pool_patched.encode())
    evidence.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"V4.1 ROCm backport installed; exact source evidence: {evidence}", flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    args = parser.parse_args()
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Cannot locate installed SGLang package")
    install(Path(next(iter(spec.submodule_search_locations))), args.evidence)
