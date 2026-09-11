"""Offline check of the pieces that don't need a GPU: gate math, stitching, ranking."""
import os, sys, types, tempfile
import numpy as np, torch

sys.path.insert(0, os.path.abspath("analysis"))
from engram import gate_probe, scan

# --- 1. gate math matches the Triton kernel's arithmetic, computed by hand ---
T, HC, D, eps, clamp = 3, 4, 8, 1e-6, 1e-6
torch.manual_seed(0)
hidden = torch.randn(T, HC, D)
kv = torch.randn(T, (HC + 1) * D)
q = torch.randn(HC, D); k = torch.randn(HC, D)

mod = types.SimpleNamespace(
    wkv=lambda x: kv, embed=lambda h: torch.zeros(T, 1, 1),
    use_sequence_parallel=False, hc_mult=HC, dim=D, eps=eps, clamp_value=clamp,
    q_weight=q, k_weight=k, layer_hash_index=0,
)
gate, start = gate_probe._gate(mod, hidden, torch.zeros(T, 1, dtype=torch.long), None)

ref = np.empty((T, HC))
for t in range(T):
    for h in range(HC):
        hv = hidden[t, h].double().numpy(); kvv = kv[t, h*D:(h+1)*D].double().numpy()
        hr = 1/np.sqrt((hv**2).mean() + eps); kr = 1/np.sqrt((kvv**2).mean() + eps)
        dot = float((hv * q[h].double().numpy() * k[h].double().numpy() * kvv).sum())
        dot *= hr * kr / np.sqrt(D)
        gi = np.sign(dot) * np.sqrt(max(abs(dot), clamp))
        ref[t, h] = 1/(1+np.exp(-gi))
assert np.allclose(gate, ref.mean(axis=1), atol=1e-5), (gate, ref.mean(axis=1))
assert start == 0
print("gate math matches hand-computed kernel formula")

# --- 2. stitching: replicated ranks, and sequence-parallel disjoint windows ---
d = tempfile.mkdtemp()
np.save(os.path.join(d, "L0_r0_s0_n10_c1_111.npy"), np.arange(10, dtype=np.float32))
np.save(os.path.join(d, "L0_r1_s0_n10_c1_112.npy"), np.arange(10, dtype=np.float32))
np.save(os.path.join(d, "L1_r0_s0_n4_c1_111.npy"), np.zeros(4, dtype=np.float32))
np.save(os.path.join(d, "L1_r1_s4_n6_c1_112.npy"), np.ones(6, dtype=np.float32))
got = scan._collect(d, 10)
assert got[0].shape == (10,) and got[0][3] == 3, got[0]
assert got[1].shape == (10,) and got[1][0] == 0 and got[1][9] == 1, got[1]
print("stitching handles replicated and sequence-parallel layouts")

scan._clear(d); assert not os.listdir(d); print("clear works")

# --- 3. ranking: mean gate, min-count filter, ordering ---
import collections
stats = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0.0]))
stats[("chat", 0, 2)][(1, 2)] = [5, 4.75]     # avg .95
stats[("chat", 0, 2)][(3, 4)] = [10, 9.0]     # avg .90
stats[("chat", 0, 2)][(5, 6)] = [1, 0.99]     # below min-count
tok = types.SimpleNamespace(decode=lambda ids: "|".join(map(str, ids)))
rep = scan._render(stats, tok, top_k=10, min_count=3)
rows = rep["chat/engram0/2gram"]
assert [r["ngram"] for r in rows] == ["1|2", "3|4"], rows
assert rows[0]["avg_gate"] == 0.95 and rows[0]["count"] == 5
print("ranking: mean gate, ordering, and min-count filter all correct")
print("\nALL OFFLINE CHECKS PASSED")
