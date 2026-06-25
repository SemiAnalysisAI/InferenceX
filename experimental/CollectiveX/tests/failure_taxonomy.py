#!/usr/bin/env python3
"""CollectiveX failure taxonomy (goal Part 3: failure & reliability characterization).

A wedged or crashing EP run should become a CLASSIFIED, bounded record — not a silent hang or a
bare rc=1. classify() maps an exception (or a process return code from the timeout-wrapped driver)
onto a stable failure mode, so coverage/reliability views can keep failed cases instead of dropping
them. Pure stdlib.
"""
from __future__ import annotations

# Stable failure modes (goal Part 3). Order matters: classify() returns the first match.
MODES = [
    "unsupported",            # capability rejected the combo (run_ep exit 5)
    "initialization-failure",  # process group / buffer / NVSHMEM bring-up failed
    "out-of-memory",
    "registration-failure",    # MR / symmetric-heap registration (e.g. MoRI errno 22)
    "correctness-failure",     # ran but reconstruction gate failed
    "timeout",                 # killed by the timeout wrapper (rc 124) — bounded hang
    "deadlock",                # collective watchdog abort (NCCL SIGABRT / rc -6 after a stall)
    "teardown-failure",        # post-finalize / shmem_finalize assertion
    "infrastructure",          # slurm / container / FS / node failure
    "unknown",
]

_SIGNATURES = [
    ("unsupported", ("unsupported", "rejects", "not supported", "no fallback")),
    ("out-of-memory", ("out of memory", "outofmemory", "cuda oom", "cudaerrormemoryallocation")),
    ("registration-failure", ("errno 22", "registration", "register", "ibv_reg", "mr ")),
    ("initialization-failure", ("nvshmem", "init_process_group", "ncclcomminit", "bootstrap", "buffer(")),
    ("deadlock", ("watchdog", "sigabrt", "signal 6", "collective", "timed out waiting", "nccl timeout")),
    ("teardown-failure", ("shmem_finalize", "destroy_process_group", "teardown", "finalize")),
    ("correctness-failure", ("correct=false", "reconstruction", "max_rel", "assertion.*tol")),
    ("infrastructure", ("srun: error", "slurm", "node fail", "container", "no such file")),
]


def classify(text: str = "", rc: int | None = None) -> str:
    """Best-effort failure mode from captured stderr/stdout text and/or a process return code."""
    if rc is not None:
        if rc == 5:
            return "unsupported"
        if rc == 124:
            return "timeout"             # GNU timeout SIGTERM
        if rc in (137, -9):
            return "timeout"             # SIGKILL (timeout -k)
        if rc in (134, -6):
            return "deadlock"            # SIGABRT (NCCL watchdog / assertion)
    t = (text or "").lower()
    for mode, sigs in _SIGNATURES:
        if any(s in t for s in sigs):
            return mode
    if rc not in (None, 0):
        return "unknown"
    return "unknown"


def record(text="", rc=None, case=None) -> dict:
    """A classified failure record preserving the exact case + signal for reliability views."""
    return {"failure_mode": classify(text, rc), "return_code": rc,
            "case": case or {}, "evidence": (text or "")[-400:]}


if __name__ == "__main__":
    import sys
    cases = [
        ("RuntimeError: Unsupported number of EP ranks", None, "unsupported"),
        ("", 124, "timeout"),
        ("Signal 6 (SIGABRT) received ... NCCL watchdog", None, "deadlock"),
        ("", -6, "deadlock"),
        ("cuda out of memory", None, "out-of-memory"),
        ("ibv_reg_mr failed errno 22", None, "registration-failure"),
        ("shmem_finalize teardown assertion", None, "teardown-failure"),
        ("srun: error: node failed", None, "infrastructure"),
    ]
    ok = True
    for text, rc, want in cases:
        got = classify(text, rc)
        flag = "OK" if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"  [{flag}] rc={rc} text={text[:40]!r} -> {got} (want {want})")
    print("failure_taxonomy self-test:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
