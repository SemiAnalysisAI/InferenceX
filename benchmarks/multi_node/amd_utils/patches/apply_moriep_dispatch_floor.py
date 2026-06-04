#!/usr/bin/env python3
"""Surgically floor the MoRI per-rank dispatch buffer to >=256 in the installed
sglang `moriep.py`, in place, inside the container.

Why in-place (not a bind-mount overlay): the lmsysorg/sglang-rocm image ships a
*downstream-patched* moriep.py (class `MoriEPDispatcher`, extra attrs such as
`expert_mask_gpu`) that diverges from the upstream v0.5.12.post1 tag. A full-file
overlay of the upstream file reverts those AMD additions and crashes the
scheduler at init (`AttributeError: ... 'expert_mask_gpu'`). So we patch the
image's own file and touch only the dispatch-token read.

The bug being fixed: at low concurrency the per-rank dispatch buffer
(num_max_dispatch_tokens_per_rank -> mori max_num_inp_token_per_rank) collapses
(conc-64/TP8/MTP3 -> 64/8*4 = 32). MoRI sizes its receive buffer
MaxNumTokensToRecv() = worldSize * maxNumInpTokenPerRank (dispatch_combine.hpp;
max_total_recv_tokens defaults to 0 -> that fallback, and it is a cap not a
floor). The intra-node dispatch kernel's per-dest atomic counter then overruns
the buffer; the only guard is assert(destTokId < MaxNumTokensToRecv()) which is
compiled out under -DNDEBUG -> silent out-of-bounds writes -> output that decodes
fine (high acceptance length) but is semantically garbage (gsm8k=0).

Empirically on MI355X (conc-64 DEP8+MTP3): dispatch 32 -> gsm8k 0.00,
64 -> 0.00 (one wavefront insufficient), 256 -> 0.94. We floor to 256.

Idempotent and fail-loud-but-non-fatal: a regex/structure miss prints a clear
marker and the surrounding source (for diagnosis) but does not abort the server.

Upstream: sgl-project/sglang#27194, ROCm/mori#356.
"""
import os
import re
import sys

FLOOR = 256
MARKER = "[InferenceX moriep dispatch floor]"
TAG = "[moriep-floor]"


def find_target():
    try:
        import sglang
    except Exception as e:  # pragma: no cover
        print(f"{TAG} ERROR: could not import sglang ({e}); NOT patched")
        return None

    # sglang may be a namespace package (no __init__.py) where __file__ is
    # None.  Fall through several strategies to locate the package root.
    pkg_dir = None
    if getattr(sglang, "__file__", None) is not None:
        pkg_dir = os.path.dirname(sglang.__file__)
    elif getattr(sglang, "__path__", None):
        pkg_dir = list(sglang.__path__)[0]
    else:
        try:
            import importlib.util
            spec = importlib.util.find_spec("sglang")
            if spec and spec.submodule_search_locations:
                pkg_dir = list(spec.submodule_search_locations)[0]
        except Exception:
            pass

    if pkg_dir is None:
        print(f"{TAG} ERROR: could not determine sglang install path "
              f"(__file__={getattr(sglang, '__file__', '?')}, "
              f"__path__={getattr(sglang, '__path__', '?')}); NOT patched")
        return None

    path = os.path.join(
        pkg_dir, "srt", "layers", "moe", "token_dispatcher", "moriep.py",
    )
    if not os.path.isfile(path):
        print(f"{TAG} ERROR: moriep.py not found at {path}; NOT patched")
        return None
    return path


def main():
    path = find_target()
    if path is None:
        return 0  # non-fatal

    with open(path) as f:
        src = f.read()
    lines = src.splitlines(keepends=True)

    # Diagnostic: always show where the dispatch-token count is read/used so the
    # CI log reveals the image's actual file shape even on a clean apply.
    for i, l in enumerate(lines):
        if "num_max_dispatch_tokens_per_rank" in l:
            print(f"{TAG}[diag] {path}:{i + 1}: {l.rstrip()}")

    if MARKER in src:
        print(f"{TAG} already applied; skipping")
        return 0

    # Find the assignment that reads the env var, regardless of class name or
    # formatting: `self.num_max_dispatch_tokens_per_rank = get_int_env_var(`.
    start = None
    for i, l in enumerate(lines):
        if re.search(
            r"self\.num_max_dispatch_tokens_per_rank\s*=\s*get_int_env_var\s*\(",
            l,
        ):
            start = i
            break
    if start is None:
        print(
            f"{TAG} ERROR: dispatch-token env read not found in {path}; "
            f"NOT patched (server will run UNPATCHED -> expect corruption at "
            f"low conc). See [diag] lines above for the actual source shape."
        )
        return 0  # non-fatal: surface loudly but let the run proceed

    # Walk forward to the end of the (possibly multi-line) call by balancing parens.
    depth = 0
    end = start
    for j in range(start, len(lines)):
        depth += lines[j].count("(") - lines[j].count(")")
        if depth <= 0:
            end = j
            break

    indent = re.match(r"\s*", lines[start]).group(0)
    floor_block = (
        f"{indent}# {MARKER} floor to {FLOOR} (warpSize/fan-in safe). MoRI recv buffer\n"
        f"{indent}# is worldSize*maxNumInpTokenPerRank; values below {FLOOR} silently\n"
        f"{indent}# corrupt the dispatch path (gsm8k=0). sgl#27194 / mori#356.\n"
        f"{indent}self.num_max_dispatch_tokens_per_rank = max(\n"
        f"{indent}    self.num_max_dispatch_tokens_per_rank, {FLOOR}\n"
        f"{indent})\n"
    )
    lines.insert(end + 1, floor_block)

    try:
        with open(path, "w") as f:
            f.write("".join(lines))
    except OSError as e:
        print(f"{TAG} ERROR: could not write {path} ({e}); NOT patched")
        return 0

    print(
        f"{TAG} applied: floored num_max_dispatch_tokens_per_rank to >= {FLOOR} "
        f"in {path} (after line {end + 1})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
