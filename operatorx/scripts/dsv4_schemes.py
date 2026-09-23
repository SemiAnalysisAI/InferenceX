"""Rewrite DeepSeek-V4-architecture serving GEMM testlists to checkpoint schemes.

The serving testlists were captured with coarse weight dtypes (bf16 / fp8 /
mxfp4) on bf16 activations. The DeepSeek-V4-Pro checkpoint quantizes every
dense Linear to FP8 128x128 blocks with ue8m0 scales and 1x128 dynamic
activation groups - including the shared experts the capture labelled mxfp4 -
except the compressor and the indexer's kv/gate/weights projections, which
stay bf16. The router, mHC projection and lm_head
were already bf16. Shapes are unchanged; an (n, k) that is both an FP8
projection and a bf16 compressor (n = head_dim) gets one entry of each.

    python -m scripts.dsv4_schemes testlists/gemm_serving_8k1k_min.json [...]
"""
from __future__ import annotations

import argparse
import json

HIDDEN = {7168, 10240, 12288}  # DSv4-Pro and the DSv4-architecture proxies
HEAD_DIM = 512
BF16_ONLY_N = {64, 256, 2 * HEAD_DIM}  # indexer weights_proj, indexer kv/gate, CSA compressor
FP8_AND_BF16_N = {HEAD_DIM}  # kv_proj (fp8) and HCA compressor (bf16)

BF16 = {"a": {"dtype": "bf16"}, "b": {"dtype": "bf16"}, "out": "bf16"}
FP8_BLOCK = {  # FP8 128x128 weight blocks, 1x128 dynamic activation groups, ue8m0 scales
    "a": {"dtype": "e4m3", "scale": {"dtype": "ue8m0", "static": False, "group": [1, 128]}},
    "b": {"dtype": "e4m3", "scale": {"dtype": "ue8m0", "static": True, "group": [128, 128]}},
    "out": "bf16",
}


def schemes(args: dict) -> list[dict]:
    w, n, k = args["dtype_b"], args["n"], args["k"]
    if w == "bf16":
        return [BF16]
    if w == "mxfp4":  # shared experts
        return [FP8_BLOCK]
    if w != "fp8":
        raise ValueError(f"unexpected captured weight dtype {w!r}")
    if k in HIDDEN and n in BF16_ONLY_N:
        return [BF16]
    if k in HIDDEN and n in FP8_AND_BF16_N:
        return [FP8_BLOCK, BF16]
    return [FP8_BLOCK]


def rewrite(entries: list[dict]) -> list[dict]:
    out, seen = [], set()
    for e in entries:
        if e["type"] != "gemm":
            raise ValueError(f"not a gemm entry: {e['type']}")
        for s in schemes(e["args"]):
            args = {"m": e["args"]["m"], "n": e["args"]["n"], "k": e["args"]["k"], **s}
            key = json.dumps(args, sort_keys=True)
            if key not in seen:
                seen.add(key)
                out.append({"type": "gemm", "args": args})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--out-dir", help="write here instead of rewriting in place")
    a = p.parse_args()
    for path in a.paths:
        entries = json.load(open(path))
        new = rewrite(entries)
        shapes = lambda es: {(x["args"]["m"], x["args"]["n"], x["args"]["k"]) for x in es}
        assert shapes(new) == shapes(entries)
        dst = path if a.out_dir is None else f"{a.out_dir}/{path.rsplit('/', 1)[-1]}"
        with open(dst, "w") as f:
            json.dump(new, f, indent=2)
            f.write("\n")
        print(f"{path}: {len(entries)} -> {len(new)} entries, {len(shapes(new))} shapes -> {dst}")


if __name__ == "__main__":
    main()
