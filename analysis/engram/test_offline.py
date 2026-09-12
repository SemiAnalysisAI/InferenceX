"""Offline check of the pieces that don't need a GPU: gate math, stitching, ranking."""
import json, os, sys, types, tempfile
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
assert gate.shape == (T, HC), gate.shape
assert np.allclose(gate, ref, atol=1e-5), (gate, ref)
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

# --- 4. corpora: rendering, per-language filtering, budget truncation ---
from engram import corpora

assert corpora._render({"text": "a"}, "text") == "a"
assert corpora._render({"dialog": [" hi ", " yo "]}, "dialog") == "hi\nyo"
assert corpora._render({"q": "Q", "a": "A"}, ("q", "a")) == "Q\nA"
assert corpora._render({"q": "Q", "a": ""}, ("q", "a")) == "Q"

import sys as _sys, types as _types
fake = _types.ModuleType("datasets")
rows = [{"code": "int main(){}"}, {"code": "auto x = 1;"}]
fake.load_dataset = lambda path, **kw: rows
_saved = _sys.modules.get("datasets")
_sys.modules["datasets"] = fake
try:
    text = corpora._load_one("x", {}, "code", None, budget=1_000)
    assert text == "int main(){}\nauto x = 1;", text
    assert len(corpora._load_one("x", {}, "code", None, budget=5)) == 5
finally:
    if _saved is not None:
        _sys.modules["datasets"] = _saved
    else:
        del _sys.modules["datasets"]

langs = [d for d in corpora.DOMAINS if d.startswith("code_")]
assert len(langs) == 7, langs  # 6 CodeSearchNet languages + MBPP
assert not any("rosetta" in str(v).lower() for v in corpora.DOMAINS.values())
# English and Chinese only: every natural-language domain is tagged or English.
zh = [d for d in corpora.all_domains() if d.endswith("_zh")]
assert sorted(zh) == ["chat_zh", "web_zh", "wiki_zh"], zh
print(f"corpora: {len(langs)} code domains, {len(zh)} Chinese domains, no Rosetta")
print("\nALL OFFLINE CHECKS PASSED")


def test_worker_bootstrap_is_importable_and_arms_only_when_asked(tmp_path, monkeypatch):
    """The sitecustomize must be valid at site time, when vllm is not importable."""
    import subprocess

    monkeypatch.delenv("ENGRAM_PROBE_DIR", raising=False)
    monkeypatch.setenv("PYTHONPATH", "")
    analysis_dir = str(tmp_path)
    bootstrap = gate_probe.install_in_workers(analysis_dir)
    assert bootstrap in os.environ["PYTHONPATH"].split(os.pathsep)

    # A fresh interpreter must import it cleanly with no probe dir set.
    env = dict(os.environ, PYTHONPATH=bootstrap)
    env.pop("ENGRAM_PROBE_DIR", None)
    done = subprocess.run(
        [sys.executable, "-c", "import sitecustomize, sys; print(sitecustomize.__file__)"],
        capture_output=True, text=True, env=env,
    )
    assert done.returncode == 0, done.stderr
    assert bootstrap in done.stdout


def test_dialogue_turns_render_from_role_content_dicts():
    """UltraChat rows are [{role, content}], not DailyDialog's plain strings."""
    row = {"messages": [{"role": "user", "content": " hi "}, {"role": "assistant", "content": "yo"}]}
    assert corpora._render(row, "messages") == "hi\nyo"
    assert corpora._render({"dialog": [" a ", "b"]}, "dialog") == "a\nb"


def test_take_chunks_shards_lazily_and_respects_the_cap():
    """The cap is the wall-clock knob, so it must bind exactly, per shard."""
    class Tok:
        def __call__(self, batch, add_special_tokens=False):
            return types.SimpleNamespace(input_ids=[[ord(c) for c in s] for s in batch])

    pieces = ["abcdefghij"] * 100  # 1000 tokens
    got = list(scan._take_chunks(iter(pieces), Tok(), 10, 0, 0, 1))
    assert len(got) == 100 and got[0] == [ord(c) for c in "abcdefghij"]

    capped = list(scan._take_chunks(iter(pieces), Tok(), 10, 7, 0, 1))
    assert len(capped) == 7

    # Three shards partition the chunk stream with no overlap and no loss.
    shards = [list(scan._take_chunks(iter(pieces), Tok(), 10, 0, s, 3)) for s in range(3)]
    assert sum(len(s) for s in shards) == 100
    assert [len(s) for s in shards] == [34, 33, 33]

    # A stream that ends mid-chunk drops the tail rather than emitting a short one.
    assert list(scan._take_chunks(iter(["abc"]), Tok(), 10, 0, 0, 1)) == []


def test_take_chunks_is_lazy():
    """It must not drain the stream past the cap; the streams are effectively infinite."""
    import itertools

    class Tok:
        def __call__(self, batch, add_special_tokens=False):
            return types.SimpleNamespace(input_ids=[[1] * 10 for _ in batch])

    pulled = 0

    def endless():
        nonlocal pulled
        for _ in itertools.count():
            pulled += 1
            yield "x"

    got = list(scan._take_chunks(endless(), Tok(), 10, 5, 0, 1))
    assert len(got) == 5
    assert pulled < 200, pulled


def test_parts_are_written_per_domain_and_merged(tmp_path):
    """A killed scan must keep completed domains; a rerun must reuse them."""
    out = str(tmp_path)
    a = {"wiki/engram0/2gram": [{"ngram": "x", "count": 3, "avg_gate": 0.5}], "_tokens": 100}
    b = {"math/engram0/2gram": [{"ngram": "y", "count": 4, "avg_gate": 0.6}], "_tokens": 200}
    for domain, part in (("wiki", a), ("math", b)):
        with open(scan._part_path(out, 0, domain), "w") as fh:
            json.dump(part, fh)

    report, tokens, dists = scan._merge_parts(out, 0, ["wiki", "math", "absent"])
    assert tokens == {"wiki": 100, "math": 200}
    assert dists == {"wiki": {}, "math": {}}
    assert set(report) == {"wiki/engram0/2gram", "math/engram0/2gram"}
    assert "_tokens" not in report
    # A shard's parts never collide with another shard's.
    assert scan._part_path(out, 1, "wiki") != scan._part_path(out, 0, "wiki")


def test_gate_probe_keeps_the_hyper_connection_axis():
    """Averaging the four copies was hiding the peak; the probe must not."""
    gate, start = gate_probe._gate(mod, hidden, torch.zeros(T, 1, dtype=torch.long), None)
    assert gate.shape == (T, HC), gate.shape
    assert start == 0
    assert ((gate > 0) & (gate < 1)).all(), "a sigmoid cannot leave (0, 1)"


def test_distribution_captures_max_quantiles_and_per_copy():
    acc = scan._new_dist()
    g = np.zeros((100, 4), dtype=np.float32)
    g[:, 0] = 0.02          # copy 0 shut
    g[:, 1] = 0.50
    g[:, 3] = 0.97          # copy 3 wide open
    scan._accumulate_dist(acc, g)
    out = scan._finalize_dist(acc)
    assert out["max"] == 0.97
    assert out["n_gate_values"] == 400
    assert out["per_copy_mean"] == [0.02, 0.5, 0.0, 0.97]
    assert out["per_copy_max"] == [0.02, 0.5, 0.0, 0.97]
    # The copy-max is 4x the copy-mean here, which is the error being fixed.
    assert abs(out["mean"] - 0.3725) < 1e-4
    assert sum(out["hist"]) == 400
    assert 0.9 < out["q0.99"] <= 1.0


def test_collect_stitches_two_dimensional_gates():
    d = tempfile.mkdtemp()
    np.save(os.path.join(d, "L0_r0_s0_n4_c1_1.npy"), np.zeros((4, 4), dtype=np.float32))
    np.save(os.path.join(d, "L0_r1_s4_n6_c1_2.npy"), np.ones((6, 4), dtype=np.float32))
    got = scan._collect(d, 10)
    assert got[0].shape == (10, 4), got[0].shape
    assert got[0][0].max() == 0 and got[0][9].max() == 1


def test_ablation_detects_the_return_convention():
    """Zeroing the wrong thing yields a broken model, not an ablated one."""
    h = torch.randn(6, 4, 8)
    delta = 0.02 * torch.randn_like(h)
    # Convention A: forward returns hidden + gate*value -> nearly parallel to h.
    assert gate_probe._looks_like_updated_hidden(h + delta, h) is True
    assert gate_probe._looks_like_updated_hidden(h, h) is True
    # Convention B: forward returns only the contribution -> not parallel.
    assert gate_probe._looks_like_updated_hidden(delta, h) is False
    assert gate_probe._looks_like_updated_hidden(torch.zeros_like(h), h) is False


def test_ablation_always_goes_through_the_shut_mask():
    """No convention guessing: every call runs the real forward with an
    all-False token_mask, which the module documents as shutting the gate."""
    masks = []

    class Fake:
        def forward(self, hidden_states, hash_ids, token_mask=None):
            masks.append(token_mask)
            if token_mask is not None and not token_mask.any():
                return hidden_states.clone()
            return hidden_states + 0.01 * torch.ones_like(hidden_states)

    saved = gate_probe._find_engram_class
    gate_probe._find_engram_class = lambda: Fake
    try:
        gate_probe.install_ablation()
        obj, h = Fake(), torch.randn(3, 2, 4)
        hash_ids = torch.zeros(3, 2, dtype=torch.long)
        for _ in range(3):
            out = Fake.forward(obj, h, hash_ids)
            assert torch.equal(out, h)
        assert len(masks) == 3
        assert all(m is not None and not m.any() and m.dtype == torch.bool for m in masks)
        # The mask length must follow hash_ids (pre sequence-parallel shard).
        assert all(m.shape == (3,) for m in masks)
    finally:
        gate_probe._find_engram_class = saved


def test_bootstrap_arms_the_ablation_when_only_ablate_is_set(tmp_path, monkeypatch):
    import subprocess

    boot = gate_probe.write_bootstrap(str(tmp_path))
    env = dict(os.environ, PYTHONPATH=boot, ENGRAM_ABLATE="1")
    env.pop("ENGRAM_PROBE_DIR", None)
    done = subprocess.run(
        [sys.executable, "-c", "import sitecustomize; print('ok')"],
        capture_output=True, text=True, env=env,
    )
    assert done.returncode == 0, done.stderr
    assert "ok" in done.stdout


def test_ablation_diagnostics_go_to_stderr_not_the_logger(capsys, monkeypatch):
    """vLLM's logging config swallowed logger.info in the server, which left the
    first ablation run unable to prove the patch was ever reached."""
    calls = []

    class Fake:
        def forward(self, hidden_states, hash_ids, token_mask=None):
            calls.append(1)
            return hidden_states + 0.01 * torch.ones_like(hidden_states)

    saved = gate_probe._find_engram_class
    gate_probe._find_engram_class = lambda: Fake
    try:
        gate_probe.install_ablation()
        obj, h = Fake(), torch.randn(3, 2, 4)
        hash_ids = torch.zeros(3, 2, dtype=torch.long)
        for _ in range(10):
            Fake.forward(obj, h, hash_ids)
    finally:
        gate_probe._find_engram_class = saved

    err = capsys.readouterr().err
    assert "gate-shut forward call count = 1" in err
    assert "gate-shut forward call count = 10" in err


def test_meter_records_contribution_and_ablation_zeroes_it(tmp_path, monkeypatch):
    """The meter measures ||returned - hidden|| / ||hidden||, which is the
    contribution the rest of the model actually receives."""
    meter = str(tmp_path)
    monkeypatch.setenv(gate_probe.METER_DIR_ENV, meter)

    masks = []

    class Fake:
        layer_hash_index = 0

        def forward(self, hidden_states, hash_ids, token_mask=None):
            masks.append(token_mask)
            # Mirrors the real module: an all-False mask passes through.
            if token_mask is not None and not token_mask.any():
                return hidden_states.clone()
            return hidden_states + 0.05 * torch.ones_like(hidden_states)

    saved = gate_probe._find_engram_class
    gate_probe._find_engram_class = lambda: Fake
    try:
        gate_probe.install_meter()
        obj, h = Fake(), torch.ones(4, 2, 8)

        gate_probe.set_ablate(meter, False)
        obj_hash = torch.zeros(4, 2, dtype=torch.long)
        out = Fake.forward(obj, h, obj_hash)
        assert not torch.equal(out, h), "baseline must pass the real output through"
        base = gate_probe.read_meter(meter)
        assert base["calls"] == 1 and base["mean_rel_norm"] > 1e-6

        gate_probe.clear_meter(meter)
        gate_probe.set_ablate(meter, True)
        out = Fake.forward(obj, h, obj_hash)
        assert torch.equal(out, h), "ablated must hand back the input untouched"
        assert masks and masks[-1] is not None and not masks[-1].any(), \
            "ablation must go through the module's all-False token_mask path"
        abl = gate_probe.read_meter(meter)
        assert abl["calls"] == 1 and abl["max_rel_norm"] == 0.0
        assert "engram0" in abl["per_layer"]
    finally:
        gate_probe._find_engram_class = saved


def test_bootstrap_arms_for_every_mode_env_var(tmp_path):
    """The meter has its own env var, and the bootstrap once gated only on the
    other two -- so it never armed in the workers and measured nothing."""
    boot = gate_probe.write_bootstrap(str(tmp_path))
    src = open(os.path.join(boot, "sitecustomize.py")).read()
    for var in ("ENGRAM_PROBE_DIR", "ENGRAM_ABLATE", "ENGRAM_METER_DIR"):
        assert var in src, var
    # The meter must win when set: it does the ablation itself, via its toggle.
    assert src.index("ENGRAM_METER_DIR") < src.index("install_ablation")


def test_ablation_verdict_is_shared_and_tolerant_of_last_bit_residue():
    ok = gate_probe.ablation_verdict({"mean_rel_norm": 0.45}, {"max_rel_norm": 0.0})
    assert ok["ok"] and ok["engram_used_in_baseline"] and ok["contribution_removed"]
    # Bitwise-zero is expected, but a last-bit residue must not fail the run.
    assert gate_probe.ablation_verdict({"mean_rel_norm": 0.45}, {"max_rel_norm": 1e-9})["ok"]
    # The two failure modes that produced uninterpretable numbers today.
    assert not gate_probe.ablation_verdict({}, {})["ok"]                      # never armed
    assert not gate_probe.ablation_verdict(
        {"mean_rel_norm": 0.45}, {"max_rel_norm": 1.0})["ok"]                 # wiped the stream


def test_gate_matches_the_kernels_sign_and_clamp_branches():
    """Pin the two branches that the Triton source makes explicit: the clamp
    floor, and negation driven by the sign of dot (not of the clamped value)."""
    import math

    def kernel_gate(dot, clamp=1e-6):
        gi = math.sqrt(max(abs(dot), clamp))
        if dot < 0:
            gi = -gi
        return 1.0 / (1.0 + math.exp(-gi))

    for dot in (5.0, 0.5, 1e-9, -1e-9, -0.5, -5.0):
        mine = float(
            torch.sigmoid(
                torch.sqrt(torch.tensor(abs(dot)).clamp_min(1e-6))
                * torch.sign(torch.tensor(dot))
            )
        )
        assert abs(mine - kernel_gate(dot)) < 1e-6, (dot, mine, kernel_gate(dot))
    # Documented divergence at exactly zero: sign(0)==0 here, kernel goes positive.
    assert abs(float(torch.sigmoid(torch.tensor(0.0))) - 0.5) < 1e-9
    assert abs(kernel_gate(0.0) - 0.5) < 1e-3


def test_phase_mask_splits_on_the_boundary():
    """prefix/suffix are complements at the boundary and cover every token."""
    pm = gate_probe.phase_mask
    dev = torch.device("cpu")
    pre = pm("prefix", 4, 10, dev)
    suf = pm("suffix", 4, 10, dev)
    assert pre.dtype == torch.bool and pre.shape == (10,)
    assert pre.sum() == 4 and suf.sum() == 6
    assert bool((pre ^ suf).all()), "prefix and suffix must partition the call"
    assert not pm("none", 4, 10, dev).any()
    # An incoming token_mask is intersected, never widened.
    incoming = torch.zeros(10, dtype=torch.bool)
    incoming[:2] = True
    assert int(pm("prefix", 4, 10, dev, incoming).sum()) == 2
    assert int(pm("suffix", 4, 10, dev, incoming).sum()) == 0


def test_meter_applies_the_selected_phase_mask(tmp_path):
    """The meter must honour the mode file, not just the legacy boolean."""
    masks = []

    class Fake:
        def forward(self, hidden_states, hash_ids, token_mask=None):
            masks.append(None if token_mask is None else token_mask.clone())
            return hidden_states + 0.01 * torch.ones_like(hidden_states)

    meter = str(tmp_path)
    saved = gate_probe._find_engram_class
    gate_probe._find_engram_class = lambda: Fake
    os.environ[gate_probe.METER_DIR_ENV] = meter
    try:
        gate_probe.install_meter()
        obj = Fake()
        obj.layer_hash_index = 0
        h = torch.randn(8, 2, 4)
        hash_ids = torch.zeros(8, 2, dtype=torch.long)
        for mode, boundary, expect in (
            ("all", 0, None), ("none", 0, 0), ("prefix", 3, 3), ("suffix", 3, 5),
        ):
            masks.clear()
            gate_probe.set_mode(meter, mode, boundary)
            Fake.forward(obj, h, hash_ids)
            if expect is None:
                # "all" must not re-run the forward with a mask at all.
                assert masks == [None], masks
            else:
                assert masks[-1] is not None and int(masks[-1].sum()) == expect, (mode, masks)
    finally:
        gate_probe._find_engram_class = saved
        os.environ.pop(gate_probe.METER_DIR_ENV, None)


def _crux():
    import importlib
    return importlib.import_module("engram.cruxeval_ablation")


def test_cruxeval_grades_literals_without_executing_anything():
    crux = _crux()
    ok = lambda text, ref: crux.equivalent(crux.extract(text), ref)
    assert ok("[ANSWER]\nassert f(17) == 17\n[/ANSWER]", "17")
    assert ok("[ANSWER]\nassert f(x) == {'a': 1}\n[/ANSWER]", "{'a':1}")
    assert ok('[ANSWER]\nassert f("s") == "s"\n[/ANSWER]', "'s'")
    assert ok("[ANSWER]\n[1, 2, 3]\n[/ANSWER]", "[1, 2, 3]")
    assert not ok("[ANSWER]\nassert f(x) == 'abc'\n[/ANSWER]", "'abd'")
    assert not ok("", "3")
    # An unsimplified expression is wrong, not evaluated -- the task forbids it
    # and we refuse to execute model output to find out.
    assert not ok("[ANSWER]\nassert f(x) == 1 + 1\n[/ANSWER]", "2")
    # A call in the answer must never be invoked.
    assert not ok("[ANSWER]\nassert f(x) == __import__('os').getpid()\n[/ANSWER]", "1")


def test_cruxeval_prompt_shape():
    crux = _crux()
    prompt = crux.build_prompt("def f(a):\n    return a * 2", "3")
    assert "assert f(3) == ??" in prompt
    assert prompt.rstrip().endswith("[ANSWER]")
    assert "[/ANSWER]" in prompt  # the two-shot exemplars are present
