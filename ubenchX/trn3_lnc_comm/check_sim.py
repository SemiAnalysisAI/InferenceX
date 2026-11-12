"""Check the intra-LNC kernels' results in NKI's CPU simulator at LNC=2 (needs nki, ml_dtypes).

  NEURON_LOGICAL_NC_CONFIG=2 python ubenchX/trn3_lnc_comm/check_sim.py

Rank r's input tile is in_tensor[r]; after the loop rank r stores its last result to out[r]. Expected:
sendrecv (both engines) and shared HBM + core_barrier -> out[r] = in[1 - r] (the peer's tile);
sendrecv + add and sendrecv_compute -> out[r] = in[0] + in[1]; local copy -> out[r] = in[r].
"""
import os
import sys

os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")
import ml_dtypes
import numpy as np
import nki

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernels import get_kernel

CASES = [("sr_dma", "bf16", 32768), ("sr_gpsimd", "bf16", 32768), ("sr_add", "bf16", 32768), ("sr_add", "fp32", 32768),
         ("src_add", "bf16", 32768), ("src_add", "fp32", 32768), ("shm_barrier", "bf16", 32768),
         ("local_copy", "bf16", 32768)]
NP = {"bf16": ml_dtypes.bfloat16, "fp32": np.float32}
SIZE = {"bf16": 2, "fp32": 4}
ok_all = True
for kernel, dtype, nbytes in CASES:
    F = nbytes // (128 * SIZE[dtype])
    x = np.random.default_rng(1).standard_normal((2, 128, F)).astype(NP[dtype])
    k = get_kernel(kernel, 4, dtype)  # 4 operations is enough to check the result
    try:
        out = np.asarray(nki.simulate(k)(x), dtype=np.float32)
    except Exception as e:  # report and continue
        print(f"ERROR {kernel:12s} {dtype}: {type(e).__name__}: {str(e)[:200]}")
        ok_all = False
        continue
    xf = x.astype(np.float32)
    if kernel in ("sr_dma", "sr_gpsimd", "shm_barrier"):
        want = np.stack([xf[1], xf[0]])
    elif kernel in ("sr_add", "src_add"):
        s = (xf[0] + xf[1]).astype(NP[dtype]).astype(np.float32)
        want = np.stack([s, s])
    else:
        want = xf
    err = float(np.max(np.abs(out - want)))
    tol = 0.05 if dtype == "bf16" else 1e-5
    ok = out.shape == want.shape and err <= tol
    ok_all &= ok
    print(f"{'PASS' if ok else 'FAIL'} {kernel:12s} {dtype}  out {out.shape}  max |err| {err:.3g}")
sys.exit(0 if ok_all else 1)
