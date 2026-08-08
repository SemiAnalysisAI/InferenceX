"""Build-time proof that the DSv4 EPLB patch is really in the installed vLLM.

Static (ast) rather than an import: importing the model module initializes
CUDA via vllm.platforms.rocm, which has no GPU during a docker build.
"""

import ast
import pathlib
import sys

ROOT = pathlib.Path("/src/vllm/vllm/models/deepseek_v4/amd")

WANT = {
    "model.py": [
        ("class", "DeepseekV4MixtureOfExperts"),
        ("method", "DeepseekV4MixtureOfExperts.extract_moe_parameters"),
        ("method", "DeepseekV4MixtureOfExperts.update_physical_experts_metadata"),
        ("method", "DeepseekV4MoE._init_mega_moe_experts"),
    ],
    "mega_moe_experts.py": [
        ("func", "_ensure_symmetric_heap_size"),
        ("method", "DeepseekV4MegaMoEExperts.get_expert_weights"),
        ("method", "DeepseekV4MegaMoEExperts.set_eplb_state"),
        ("method", "DeepseekV4MegaMoEExperts.update_expert_map"),
        ("method", "DeepseekV4MegaMoEExperts._map_global_expert_id"),
    ],
}


def _defs(node):
    return {
        b.name
        for b in node.body
        if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def main() -> int:
    failures = []
    trees = {}

    for fname, checks in WANT.items():
        trees[fname] = tree = ast.parse((ROOT / fname).read_text())  # catches truncation
        classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
        module_funcs = _defs(tree)

        for kind, name in checks:
            if kind == "class":
                ok = name in classes
            elif kind == "func":
                ok = name in module_funcs
            else:
                cls, meth = name.split(".")
                ok = cls in classes and meth in _defs(classes[cls])
            if not ok:
                failures.append(f"{fname}: missing {kind} {name}")
        print(f"checked {fname}: {len(checks)} symbols")

    # The MTP hazard, asserted at build time. If _init_mega_moe_experts gated
    # the redundancy read on self.enable_eplb, MTP layers would size themselves
    # 384/48 while target layers sat at 392/49 -- two MegaMoEShape keys, so two
    # ~7.7 GiB sets of mori symmetric buffers instead of one shared runtime.
    init = next(
        b
        for c in trees["model.py"].body
        if isinstance(c, ast.ClassDef) and c.name == "DeepseekV4MoE"
        for b in c.body
        if isinstance(b, ast.FunctionDef) and b.name == "_init_mega_moe_experts"
    )
    src = ast.unparse(init)
    if "eplb_config.num_redundant_experts" not in src:
        failures.append("model.py: mega sizing lost its redundancy read")
    if "self.enable_eplb" in src:
        failures.append("model.py: mega sizing is gated on enable_eplb (MTP shape split)")

    if failures:
        print("EPLB patch verification FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print("EPLB patch baked in and verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
