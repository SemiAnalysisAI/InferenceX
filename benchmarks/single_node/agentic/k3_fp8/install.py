"""Install the isolated K3 dense-FP8 experiment into the writable container."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

BASE_SHA256 = "53823d8ce73750d59638daccdcfba44619eeea10bd1cc5fd4534a038fbcd375f"


def main():
    package = Path(
        next(iter(importlib.util.find_spec("vllm").submodule_search_locations))
    )
    quant = package / "model_executor/layers/quantization"
    source = quant / "mxfp4.py"
    original = source.read_bytes()
    assert hashlib.sha256(original).hexdigest() == BASE_SHA256, source
    old = """        if isinstance(layer, LinearBase):
            if self.ignored_layers and is_layer_skipped("""
    new = """        if isinstance(layer, LinearBase):
            from .k3_dense_fp8 import K3DenseFp8LinearMethod, eligible_prefix

            if eligible_prefix(prefix):
                return K3DenseFp8LinearMethod()
            if self.ignored_layers and is_layer_skipped("""
    text = original.decode()
    assert text.count(old) == 1
    patched = text.replace(old, new)
    compile(patched, str(source), "exec")
    module = Path(__file__).with_name("dense_fp8.py").read_bytes()
    compile(module, "k3_dense_fp8.py", "exec")
    (quant / "k3_dense_fp8.py").write_bytes(module)
    source.write_text(patched)
    report = {
        "base_sha256": BASE_SHA256,
        "patched_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "module_sha256": hashlib.sha256(module).hexdigest(),
        "scope": "eligible K3 dense LinearBase; original RoutedExperts dispatch",
    }
    Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
