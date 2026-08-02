#!/usr/bin/env python3
"""Apply vllm-project/vllm#50578 to the vLLM installed in this container.

PR: "[ROCm][MLA] Use asm decode for non-divisor small head counts"
    https://github.com/vllm-project/vllm/pull/50578
    head c5d4f2f6c2704f3c2ae9973d45b0c391745288ec (vanshbhatia-amd/vllm,
    branch rocm-aiter-mla-asm-pad), base e2fa28594f7baad142a426b0b6a2cfe2c79201c7.

Why this matters for Kimi-K3
----------------------------
On ROCM_AITER_MLA, MLA decode with fewer than 16 query heads per rank runs the
Gluon decode kernel. Gluon parallelizes only over heads, so one workgroup walks
the whole KV cache and per-token latency scales linearly with context length.
The asm *persistent* decode needs num_heads >= 16; smaller counts are padded up,
but the stock get_mla_padded_q / get_mla_unpadded_o only handle *divisors* of 16
(repeat_interleave / strided slice), so non-divisor counts fall back to Gluon.

Kimi-K3 has 96 attention heads => **12 heads/rank at TP8**, a non-divisor, so its
long-context decode is stuck on Gluon. The PR pads non-divisor counts to 16 by
tiling the query heads and slicing, and routes those decodes through asm.

Scope of this script
--------------------
The PR touches two files:

  1. vllm/v1/attention/backends/mla/rocm_aiter_mla.py -- the whole diff is
     confined to `class AiterMLAHelper`. This script replaces that class
     wholesale with the PR's version, verbatim. Replacing the entire class
     (rather than applying a context diff) is deliberate: the kimi-k3 preview
     image reports vLLM 0.1.dev19253+g5f76ae224, and that sha is not resolvable
     in vllm-project/vllm, so the installed file cannot be assumed byte-equal to
     the PR base. Whole-class replacement is exact regardless of drift elsewhere
     in the file.

  2. vllm/model_executor/layers/quantization/mxfp4.py -- forces
     AITER_BF16_FP8_MOE_BOUND=0 when AITER_SITUV2_A8W4 is on. **Deliberately not
     applied**: kimik3_fp4_mi355x.sh already does `export
     AITER_BF16_FP8_MOE_BOUND=0` unconditionally (reference env block), which is
     strictly broader than the PR's in-process os.environ write. Nothing to do.

This script FAILS LOUDLY. A silently-unpatched run would produce a "no perf lift"
number that means nothing, which is the one outcome worth spending a node on
avoiding, so every failure path exits non-zero.
"""

import argparse
import difflib
import importlib.util
import os
import re
import sys

# Verbatim from the PR head. Do not reflow: this is the artifact under test.
PR_CLASS = '''class AiterMLAHelper:
    """
    AITER MLA persistent (asm) decode requires num_heads >= 16. Head counts
    < 16 are padded up to exactly 16: divisors of 16 by repeat_interleave,
    other counts (e.g. 12 heads/rank at TP8, 6 at TP16) by tiling the query
    heads and slicing to 16. Non-divisor padded decodes take the asm path;
    divisors and max_qo_len > 1 small-head verify still use Gluon.
    """

    _AITER_MIN_MLA_HEADS: Final = 16
    _AITER_UNSUPPORTED_HEADS: ClassVar[tuple[int, ...]] = ()

    @staticmethod
    def check_num_heads_validity(num_heads: int):
        assert AiterMLAHelper.is_valid_num_heads(num_heads), (
            "ROCM AITER MLA requires 1-15 heads (padded to 16 for asm "
            "persistent decode; exact divisors of 16 may keep Gluon) or a "
            f"multiple of 16 heads, but got {num_heads}.\\n"
            f"Try adjusting tensor_parallel_size value."
        )

    @staticmethod
    def is_valid_num_heads(num_heads: int) -> bool:
        return num_heads > 0 and (
            num_heads < AiterMLAHelper._AITER_MIN_MLA_HEADS
            or num_heads % AiterMLAHelper._AITER_MIN_MLA_HEADS == 0
        )

    @staticmethod
    def get_actual_mla_num_heads(num_heads: int) -> int:
        return max(num_heads, AiterMLAHelper._AITER_MIN_MLA_HEADS)

    @staticmethod
    def get_mla_padded_q(num_heads: int, q: torch.Tensor) -> torch.Tensor:
        m = AiterMLAHelper._AITER_MIN_MLA_HEADS
        if num_heads >= m:
            return q
        if m % num_heads == 0:
            return q.repeat_interleave(m // num_heads, dim=1)
        # Non-divisor head counts (e.g. 12 heads/rank at TP8, 6 at TP16) cannot
        # be padded by repeat_interleave. Tile the query heads and slice to
        # exactly m; this reaches m for any 0 < num_heads < m (unlike a single
        # append, which under-pads when num_heads < m - num_heads). MLA
        # attention is independent per query head over the shared KV, so the
        # padding heads cannot affect heads [0:num_heads]; they are sliced back
        # off in get_mla_unpadded_o.
        reps = -(-m // num_heads)  # ceil(m / num_heads)
        return q.repeat(1, reps, 1)[:, :m, :]

    @staticmethod
    def get_mla_unpadded_o(num_heads: int, o: torch.Tensor) -> torch.Tensor:
        m = AiterMLAHelper._AITER_MIN_MLA_HEADS
        if num_heads >= m:
            return o
        if m % num_heads == 0:
            return o[:, :: m // num_heads, :]
        # Undo the tile-padding from get_mla_padded_q: the real heads are the
        # first num_heads.
        return o[:, :num_heads, :]

    @staticmethod
    def use_gluon_decode(num_heads: int, max_qo_len: int) -> bool:
        # Divisor head counts (<16) keep the Gluon decode. Non-divisor counts
        # (e.g. 12 heads/rank at TP8) are padded to 16 in get_mla_padded_q and
        # use the faster asm persistent decode instead of Gluon, which
        # parallelizes only over heads and scales linearly with KV length.
        m = AiterMLAHelper._AITER_MIN_MLA_HEADS
        if num_heads >= m or max_qo_len != 1:
            return False
        return m % num_heads == 0
'''

REL_PATH = "v1/attention/backends/mla/rocm_aiter_mla.py"
CLASS_START_RE = re.compile(r"^class AiterMLAHelper:\s*$", re.MULTILINE)
# A top-level construct ends the class body: another class/def, a decorator, or
# a module-level assignment/import at column 0.
NEXT_TOPLEVEL_RE = re.compile(r"^(?:@|class\s|def\s|import\s|from\s|\w+\s*=)", re.MULTILINE)


def locate_vllm() -> str:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("FATAL: cannot locate the installed vllm package")
    return list(spec.submodule_search_locations)[0]


def replace_class(src: str) -> str:
    start_m = CLASS_START_RE.search(src)
    if start_m is None:
        sys.exit(
            "FATAL: `class AiterMLAHelper:` not found in the installed "
            "rocm_aiter_mla.py -- this image's MLA backend does not match what "
            "PR 50578 patches. Refusing to run an unpatched cell."
        )
    body_start = start_m.end() + 1
    next_m = NEXT_TOPLEVEL_RE.search(src, body_start)
    end = next_m.start() if next_m else len(src)
    # Keep the blank lines that separate the class from whatever follows.
    tail = src[end:]
    trailing_blanks = ""
    while src[:end].endswith("\n\n"):
        end -= 1
        trailing_blanks += "\n"
    return src[: start_m.start()] + PR_CLASS + trailing_blanks + tail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None, help="path to rocm_aiter_mla.py")
    ap.add_argument("--diff-out", default=None, help="write the applied diff here")
    ap.add_argument(
        "--verify-only",
        action="store_true",
        help="only run the semantic checks against the file as-is",
    )
    args = ap.parse_args()

    target = args.target or os.path.join(locate_vllm(), REL_PATH)
    if not os.path.isfile(target):
        sys.exit(f"FATAL: {target} not found -- refusing to run an unpatched cell.")

    with open(target, encoding="utf-8") as fh:
        original = fh.read()

    if not args.verify_only:
        patched = replace_class(original)
        if patched == original:
            print(f"PR50578: {target} already carries the PR class (no-op)")
        else:
            diff = "".join(
                difflib.unified_diff(
                    original.splitlines(keepends=True),
                    patched.splitlines(keepends=True),
                    fromfile=f"a/{REL_PATH}",
                    tofile=f"b/{REL_PATH}",
                )
            )
            with open(target, "w", encoding="utf-8") as fh:
                fh.write(patched)
            print(f"PR50578: patched {target}")
            print(diff)
            if args.diff_out:
                with open(args.diff_out, "w", encoding="utf-8") as fh:
                    fh.write(diff)
        # Drop any stale bytecode so the workers cannot import the old class.
        pyc_dir = os.path.join(os.path.dirname(target), "__pycache__")
        if os.path.isdir(pyc_dir):
            for name in os.listdir(pyc_dir):
                if name.startswith("rocm_aiter_mla."):
                    os.remove(os.path.join(pyc_dir, name))

    verify(target)
    return 0


def verify(target: str) -> None:
    """Exec the installed class in isolation and assert the PR's semantics.

    Executing just the class body (rather than importing vllm) keeps this cheap
    and keeps the failure attributable: if this trips, the file on disk is not
    the code we think we are benchmarking.
    """
    import torch
    from typing import ClassVar, Final  # noqa: F401  (used by the class body)

    with open(target, encoding="utf-8") as fh:
        src = fh.read()
    start_m = CLASS_START_RE.search(src)
    if start_m is None:
        sys.exit("FATAL: AiterMLAHelper missing at verify time")
    next_m = NEXT_TOPLEVEL_RE.search(src, start_m.end() + 1)
    class_src = src[start_m.start() : (next_m.start() if next_m else len(src))]

    ns = {"torch": torch, "ClassVar": ClassVar, "Final": Final}
    exec(compile(class_src, target, "exec"), ns)  # noqa: S102
    helper = ns["AiterMLAHelper"]

    failures = []

    # Kimi-K3 at TP8: 96 heads / 8 ranks = 12, a non-divisor of 16.
    K3_TP8_HEADS = 12
    if helper.use_gluon_decode(K3_TP8_HEADS, 1) is not False:
        failures.append(
            f"use_gluon_decode({K3_TP8_HEADS}, 1) is not False -- K3 decode would "
            "still run Gluon, i.e. the PR is NOT in effect"
        )

    # Padding must reach exactly 16 for every small head count, and the real
    # heads must survive the round trip bit-exactly.
    for nh in range(1, 16):
        q = torch.arange(2 * nh * 4, dtype=torch.float32).reshape(2, nh, 4)
        padded = helper.get_mla_padded_q(nh, q)
        if padded.shape[1] != 16:
            failures.append(f"get_mla_padded_q({nh}) -> {padded.shape[1]} heads, want 16")
            continue
        back = helper.get_mla_unpadded_o(nh, padded)
        if back.shape != q.shape or not torch.equal(back, q):
            failures.append(f"round trip lost data at num_heads={nh}")

    # >=16 must be untouched.
    q16 = torch.randn(2, 16, 4)
    if helper.get_mla_padded_q(16, q16) is not q16:
        failures.append("get_mla_padded_q(16) should be a no-op")
    if helper.use_gluon_decode(16, 1) is not False:
        failures.append("use_gluon_decode(16, 1) should be False")

    if failures:
        for f in failures:
            print(f"FATAL: {f}", file=sys.stderr)
        sys.exit("FATAL: PR 50578 verification failed -- refusing to start the server.")

    print(
        "PR50578: verified -- 12 heads/rank pads to 16, round trip is exact, "
        "use_gluon_decode(12, 1) == False (asm persistent decode active)"
    )


if __name__ == "__main__":
    raise SystemExit(main())
