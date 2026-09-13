"""CPU checks for the routing probe: the pinning math, the record/replay
hooks against a fake layer, the flush/load round trip, and the agreement
table. Run: python3 analysis/engram/test_route_offline.py"""
import json
import os
import sys
import tempfile
import types

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from engram import route_probe as rp  # noqa: E402

torch.manual_seed(0)
T, E, K = 7, 384, 6
logits = torch.randn(T, E) * 2
bias = torch.randn(E) * 0.3


def reference_gate(x, b, k, norm=True, scale=1.5):
    scores = torch.nn.functional.softplus(x.float()).sqrt()
    idx = (scores + b).topk(k, dim=-1)[1]
    w = scores.gather(1, idx)
    if norm:
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    return w * scale, idx


# 1. recomputed top-k matches the reference Gate, bias included
w_ref, idx_ref = reference_gate(logits, bias, K)
assert torch.equal(rp.reference_topk(logits, bias, K), idx_ref)
assert torch.allclose(rp.reference_weights(logits, idx_ref, True, 1.5), w_ref)

# 2. pinned logits with zero bias force an arbitrary expert set and keep weights
forced = torch.stack([torch.randperm(E)[:K] for _ in range(T)])
pinned = rp.pin_logits(logits, forced)
w_pin, idx_pin = reference_gate(pinned, torch.zeros(E), K)
assert set(map(tuple, np.sort(idx_pin.numpy(), 1))) == set(map(tuple, np.sort(forced.numpy(), 1)))
# the same experts, in whatever order the kernel ranks them, with the model's own weights
w_want, _ = reference_gate(logits, torch.zeros(E), K)  # not used: weights depend on set only
w_direct = rp.reference_weights(logits, idx_pin, True, 1.5)
assert torch.allclose(w_pin, w_direct, atol=1e-6)
# a very negative pinned logit still wins over every unpinned expert (score 0)
lo = logits.clone(); lo[0, forced[0]] = -30.0
_, idx_lo = reference_gate(rp.pin_logits(lo, forced), torch.zeros(E), K)
assert sorted(idx_lo[0].tolist()) == sorted(forced[0].tolist())
print("routing math: reference top-k reproduced, pinning forces the set and keeps weights")

# 3. record -> replay through the monolithic hook with a fake layer
d = tempfile.mkdtemp()
os.environ[rp.DIR_ENV] = d
calls = []


class FakeLayer:
    def __init__(self):
        self.e_score_correction_bias = torch.nn.Parameter(bias.clone(), requires_grad=False)
        self.top_k = K
        self.scoring_func = "sqrtsoftplus"

    def forward_monolithic(self, x, router_logits=None, input_ids=None):
        calls.append((router_logits.clone(), self.e_score_correction_bias.detach().clone()))
        return x


rp._wrap_monolithic(FakeLayer)
layer_a, layer_b = FakeLayer(), FakeLayer()
seq = torch.tensor([5, 9, 2, 8, 1, 3, 4])
other = torch.tensor([7, 7, 7, 7, 7, 7, 7])
x = torch.zeros(T, 4)

rp.set_mode(d, "record", "base")
for ids in (seq, other):          # two forwards, two MoE layers each
    rp._begin_forward(ids)
    layer_a.forward_monolithic(x, logits, ids)
    layer_b.forward_monolithic(x, logits + 1, ids)
assert rp._state["stats"]["record_calls"] == 4
k_seq = rp.seq_key(seq)
assert set(rp._state["store"]["base"][k_seq]) == {0, 1}
assert torch.equal(rp._state["store"]["base"][k_seq][0].long(), idx_ref)
assert torch.equal(calls[-1][1], bias)  # recording leaves the bias alone

rp.set_mode(d, "replay", "base")
rp._begin_forward(seq)
layer_a.forward_monolithic(x, logits + 0.5, seq)  # logits drifted, as under ablation
got_logits, got_bias = calls[-1]
assert torch.equal(got_bias, torch.zeros(E))            # bias zeroed for the kernel call
assert torch.equal(layer_a.e_score_correction_bias.detach(), bias)  # and restored after
_, idx_kernel = reference_gate(got_logits, torch.zeros(E), K)
assert set(map(tuple, np.sort(idx_kernel.numpy(), 1))) == set(map(tuple, np.sort(idx_ref.numpy(), 1)))
assert not os.path.exists(os.path.join(d, rp.MISS))
# a sequence never recorded is a miss, not a silent fall-through
rp._begin_forward(torch.tensor([1, 2, 3]))
layer_a.forward_monolithic(x[:3], logits[:3], torch.tensor([1, 2, 3]))
assert os.path.exists(os.path.join(d, rp.MISS))
os.unlink(os.path.join(d, rp.MISS))
print("hooks: record stores per (sequence, layer); replay pins and restores the bias; misses are flagged")

# 4. input_ids fallback boundary detection when the model forward is not wrapped
rp._state["model_wrapped"] = False
rp.set_mode(d, "free")
rp._state["key"] = None
rp._state["layer_calls"] = 0
rp._state["n_layers"] = 0
for ids in (other, seq):          # a completed forward teaches the layer count
    layer_a.forward_monolithic(x, logits, ids)
    layer_b.forward_monolithic(x, logits, ids)
assert rp._state["n_layers"] == 2 and rp._state["layer_calls"] == 2
before = rp._state["stats"]["forwards"]
layer_a.forward_monolithic(x, logits, seq)  # same sequence again: counter wrapped -> new forward
assert rp._state["layer_calls"] == 1 and rp._state["stats"]["forwards"] == before + 1
print("fallback: sequence boundary recovered from input_ids and the layer counter")

# 5. flush -> load -> agreement
rp._state["model_wrapped"] = True
rp.set_mode(d, "record", "abl")
rp._begin_forward(seq)
layer_a.forward_monolithic(x, logits - 0.2, seq)   # small drift: mostly the same experts
layer_b.forward_monolithic(x, -logits, seq)        # large drift: different experts
rp.set_mode(d, "flush")
rp._begin_forward(torch.tensor([42]))
layer_a.forward_monolithic(x[:1], logits[:1], torch.tensor([42]))
stats = rp.read_stats(d)
assert stats["ranks"] == 1 and not stats["miss"], stats
base_r, abl_r = rp.load_routes(d, "base"), rp.load_routes(d, "abl")
assert k_seq in base_r and k_seq in abl_r and base_r[k_seq][0].shape == (T, K)
# a second rank holding the other half of a sharded sequence is concatenated
import shutil
half = rp.load_routes(d, "base")
np.savez(os.path.join(d, "ROUTES_shard_r0.npz"), **{k_seq + "/N": np.array([T]), k_seq + "/L0": half[k_seq][0][:4]})
np.savez(os.path.join(d, "ROUTES_shard_r1.npz"), **{k_seq + "/N": np.array([T]), k_seq + "/L0": half[k_seq][0][4:]})
assert np.array_equal(rp.load_routes(d, "shard")[k_seq][0], half[k_seq][0])
table = rp.agreement(base_r, abl_r, {k_seq: 3})
assert table[0]["prompt"]["tokens"] == 3 and table[0]["answer"]["tokens"] == 4
assert table[0]["answer"]["exact_set"] >= table[1]["answer"]["exact_set"]
assert 0.0 <= table[1]["answer"]["mean_overlap"] <= 1.0
print("flush/load/agreement:", json.dumps(table))

# 6. driver helpers
sys.modules.setdefault("engram.gate_probe", types.ModuleType("engram.gate_probe"))
from engram import routing_ablation as ra  # noqa: E402

assert ra.scored_positions([1, 2, 3, 4, 5], [1, 2, 3]) == 3
assert ra.scored_positions([1, 2, 9, 4, 5], [1, 2, 3]) == 2  # join re-tokenised: back off


class LP:
    def __init__(self, logprob, rank):
        self.logprob, self.rank = logprob, rank


out = types.SimpleNamespace(
    prompt_token_ids=[10, 11, 12, 13],
    prompt_logprobs=[None, {11: LP(-0.1, 1)}, {12: LP(-0.7, 1)}, {13: LP(-2.0, 3)}],
)
s = ra.score_output(out, 2)
assert s["tokens"] == 2 and abs(s["nll"] - 2.7) < 1e-9 and s["greedy"] is False
s = ra.score_output(out, 1)
assert s["tokens"] == 3 and s["greedy"] is False
out.prompt_logprobs[3] = {13: LP(-0.3, 1)}
assert ra.score_output(out, 2)["greedy"] is True
print("driver: answer-span boundary and greedy/NLL scoring behave")
print("ALL ROUTE CHECKS PASS")
