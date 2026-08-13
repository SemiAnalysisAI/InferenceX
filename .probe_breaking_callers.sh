#!/usr/bin/env bash
# The 3 breaking signature changes are only safe if every caller is itself in
# the patch set. Find the real callers of each resigned function in post2, and
# check whether the vendor's copy of that caller passes the new argument.
set -u
python3 - <<'PY'
import os, re, subprocess
W = "/tmp/aitermain"
subprocess.run(["git", "-C", W, "checkout", "-q", "v0.1.19.post2"], check=True)
P = os.path.join(W, "aiter")
V = "/home/jiacao/3way-20260812-2214/vendor/usr/local/lib/python3.12/dist-packages/aiter"

PATCHSET = {
    "ops/flydsl/kernels/flydsl_dispatch_combine_intranode_op.py",
    "ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py",
    "ops/flydsl/kernels/communication_ops_utils.py",
    "ops/flydsl/kernels/mxfp4_gemm_common.py",
    "ops/flydsl/kernels/vector.py",
}
PATCHSET |= {"ops/flydsl/kernels/mega_moe/" + f for f in os.listdir(os.path.join(V, "ops/flydsl/kernels/mega_moe"))
             if f.endswith(".py")}

FUNCS = {
    "atomic_add_global_at": "syncscope",
    "make_combine_jit": "blockwise_fp8_transport",
    "make_combine_kernel": "blockwise_fp8_transport",
    "_lds_swizzle_mask": "row_bytes",
}

for fn, newarg in FUNCS.items():
    print(f"=== {fn}()  new arg: {newarg}")
    callers = []
    for dp, _, fns in os.walk(P):
        if "__pycache__" in dp: continue
        for f in fns:
            if not f.endswith(".py"): continue
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, P)
            src = open(p).read()
            # a call site, not the definition
            if re.search(r"(?<!def )\b" + re.escape(fn) + r"\s*\(", src):
                callers.append(rel)
    for c in sorted(set(callers)):
        inset = "IN-PATCHSET" if c in PATCHSET else "*** OUTSIDE PATCHSET ***"
        # does the vendor's copy of this caller pass the new arg?
        vp = os.path.join(V, c)
        passes = ""
        if os.path.exists(vp):
            vsrc = open(vp).read()
            hits = re.findall(r"\b" + re.escape(fn) + r"\s*\([^)]*\)", vsrc, re.S)
            if hits:
                passes = "  vendor-passes-newarg" if any(newarg in h for h in hits) else "  vendor-does-NOT-pass"
        else:
            passes = "  (absent in vendor)"
        print(f"    {inset:26} {c}{passes}")
    print()

print("=== does post2 have a DEFAULT for the new arg (making it back-compatible)? ===")
for fn, newarg in FUNCS.items():
    for dp, _, fns in os.walk(V):
        if "__pycache__" in dp: continue
        for f in fns:
            if not f.endswith(".py"): continue
            src = open(os.path.join(dp, f)).read()
            m = re.search(r"def\s+" + re.escape(fn) + r"\s*\(([^)]*)\)", src, re.S)
            if m:
                args = " ".join(m.group(1).split())
                has_default = re.search(re.escape(newarg) + r"\s*[:=]", args)
                dflt = "HAS-DEFAULT" if re.search(re.escape(newarg) + r"[^,]*=", args) else "NO-DEFAULT (required)"
                print(f"  {fn}: {dflt}")
                print(f"      {args[:200]}")
                break
        else:
            continue
        break
PY
