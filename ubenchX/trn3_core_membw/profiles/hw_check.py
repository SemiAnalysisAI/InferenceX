"""Run the chase on hardware through torch-neuronx and check the final index against numpy."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import kernels as K
N, W = 4096, 16
rng = np.random.default_rng(0); perm = rng.permutation(N)
chain = np.zeros((N, W), np.int32); chain[perm, 0] = np.roll(perm, -1)
idx = 0
for _ in range(K.HOPS):
    idx = chain[idx, 0]
print("expected after", K.HOPS, "hops:", idx, "(start slot 0 -> chain[0] first)")
import nki
sim = nki.simulate(K.chase_hwdge)(chain)
print("simulator:", np.asarray(sim).ravel())
dev = torch.device("neuron")
t = torch.from_numpy(chain).to(dev)
for name in ["chase_hwdge", "chase_swdge"]:
    out = K.KERNELS[name](t)
    print(name, "hardware:", out.cpu().numpy().ravel())
