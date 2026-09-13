"""Record and replay MoE expert routing across Engram arms.

Question: when Engram is removed, how much of the damage is the *routing
shift* -- tokens whose residual changed at layers 1 and 14 now get sent to
different experts in every MoE layer downstream -- and how much is the lost
features themselves? The two are confounded in a plain ablation. This module
separates them by pinning the expert choice.

Mechanism. The B200 MXFP4 backend routes inside the TRT-LLM kernel
(`RoutedExperts.forward_monolithic` hands it raw router logits), so there are
no Python-side `topk_ids` to overwrite. The kernel is steered through its own
inputs instead. The reference Gate (inference/model.py) is:

    scores  = sqrt(softplus(logits))            # strictly positive
    indices = topk(scores + bias)               # bias picks, never scales
    weights = scores[indices] / sum * scale

so with the bias swapped for zeros and every non-pinned logit set to a large
negative value (softplus -> 0, score -> 0) the kernel *must* select the pinned
experts, and their weights are the model's own scores at those experts,
renormalised exactly as before. Nothing about the expert computation changes;
only which experts run.

Recording on this path recomputes the top-k from the same logits with the
reference arithmetic. A self-pin arm (pin an arm to its own recording) checks
that the recomputation matches the kernel: it must reproduce the unpinned arm
to within numerical noise, or the run is not a measurement.

Keys. Runs are teacher-forced with `max_num_seqs=1`, so each forward call is
one whole sequence. A sequence is identified by the hash of its token ids
(taken from the model forward, not from the driver's tokenisation) and an
MoE layer by its call index within that forward. The driver hashes
`prompt_token_ids` the same way.

Modes are switched by a JSON file in the probe dir, like the Engram meter:
    free            -- untouched
    record <tag>    -- store this arm's routing under <tag>
    replay <src>    -- force the routing stored under <src>
    flush           -- dump stores/stats to disk (driver sends a dummy prompt)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading

import torch

DIR_ENV = "ENGRAM_ROUTE_DIR"
_MODE_FILE = "ROUTING_MODE.json"
MISS = "ROUTE_MISS"
NEG = -1.0e4  # softplus(-1e4) is exactly 0 in fp32; sqrt(0) = 0

def _fresh_stats() -> dict:
    return {"record_calls": 0, "replay_calls": 0, "free_calls": 0, "misses": 0,
            "layers": 0, "bias_none": 0, "forwards": 0, "path": None}


_lock = threading.Lock()
_state = {
    "installed": False,
    "model_wrapped": False,   # DeepseekV4Model.forward stashes the key
    "n_layers": 0,            # MoE calls per forward, learned from the first
    "path": None,             # "monolithic" | "modular"
    "key": None,              # sha1 of the current forward's token ids
    "ntok": 0,
    "layer_calls": 0,
    "store": {},              # tag -> key -> {layer_idx: int16 [T, k] cpu}
    "ntok_by_key": {},        # key -> tokens in that forward (full, pre-shard)
    "stats": _fresh_stats(),
    "flushed_key": None,
    "said": set(),
}


def _say(msg: str) -> None:
    sys.stderr.write("engram-route: %s\n" % msg)
    sys.stderr.flush()


def _say_once(tag: str, msg: str) -> None:
    if tag not in _state["said"]:
        _state["said"].add(tag)
        _say(msg)


def seq_key(token_ids) -> str:
    """Hash of a token sequence, identical on the driver and in the workers."""
    arr = torch.as_tensor(token_ids).reshape(-1).to(torch.int64).cpu().numpy()
    return hashlib.sha1(arr.tobytes()).hexdigest()


# ---------------------------------------------------------------- mode file

def set_mode(route_dir: str, mode: str, tag: str | None = None) -> None:
    if mode not in ("free", "record", "replay", "flush"):
        raise ValueError(mode)
    path = os.path.join(route_dir, _MODE_FILE)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"mode": mode, "tag": tag}, fh)
    os.replace(tmp, path)


def _read_mode(route_dir: str) -> tuple[str, str | None]:
    try:
        with open(os.path.join(route_dir, _MODE_FILE)) as fh:
            spec = json.load(fh)
        return spec.get("mode", "free"), spec.get("tag")
    except (OSError, ValueError):
        return "free", None


def _miss(route_dir: str, why: str) -> None:
    _state["stats"]["misses"] += 1
    _say_once("miss:" + why, "replay miss (%s); run is invalid" % why)
    try:
        open(os.path.join(route_dir, MISS), "w").close()
    except OSError:
        pass


# ---------------------------------------------------------------- routing math

@torch.no_grad()
def reference_topk(logits: torch.Tensor, bias: torch.Tensor | None, k: int,
                   scoring: str = "sqrtsoftplus") -> torch.Tensor:
    """Expert indices the reference Gate would pick. [T, k] int64."""
    x = logits.float()
    if scoring == "softmax":
        scores = x.softmax(dim=-1)
    elif scoring == "sigmoid":
        scores = x.sigmoid()
    else:
        scores = torch.nn.functional.softplus(x).sqrt()
    if bias is not None:
        scores = scores + bias.float().reshape(1, -1).to(scores.device)
    return scores.topk(k, dim=-1)[1]


@torch.no_grad()
def reference_weights(logits: torch.Tensor, ids: torch.Tensor, renormalize: bool,
                      scale: float, scoring: str = "sqrtsoftplus") -> torch.Tensor:
    x = logits.float()
    if scoring == "softmax":
        scores = x.softmax(dim=-1)
    elif scoring == "sigmoid":
        scores = x.sigmoid()
    else:
        scores = torch.nn.functional.softplus(x).sqrt()
    w = scores.gather(1, ids.to(torch.int64))
    if renormalize and ids.shape[1] > 1:
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return w * scale


@torch.no_grad()
def pin_logits(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Logits that force `ids` under sqrt-softplus selection with zero bias.

    Pinned experts keep their logits (so their weights are unchanged); every
    other expert gets NEG, whose score is exactly 0 and can never beat a
    strictly positive pinned score."""
    out = torch.full_like(logits, NEG)
    ids = ids.to(logits.device, torch.int64)
    out.scatter_(1, ids, logits.gather(1, ids))
    return out


# ---------------------------------------------------------------- hooks

def _begin_forward(input_ids: torch.Tensor) -> None:
    with _lock:
        _state["key"] = seq_key(input_ids)
        _state["ntok"] = int(input_ids.numel())
        _state["ntok_by_key"][_state["key"]] = _state["ntok"]
        _state["layer_calls"] = 0
        _state["stats"]["forwards"] += 1


def _next_layer_idx() -> int:
    idx = _state["layer_calls"]
    _state["layer_calls"] = idx + 1
    _state["stats"]["layers"] = max(_state["stats"]["layers"], idx + 1)
    return idx


def _maybe_begin_from_input_ids(input_ids) -> None:
    """Fallback sequence boundary when the model forward is not wrapped.

    DeepSeek V4 hands `input_ids` to every MoE call, so a new sequence shows
    up as a new hash; the same sequence twice in a row (the flush prompt) is
    caught by the layer counter wrapping past the count seen so far."""
    if _state["model_wrapped"] or input_ids is None:
        return
    key = seq_key(input_ids)
    if key != _state["key"]:
        # The previous forward is complete; its call count is the layer count.
        if _state["key"] is not None and _state["layer_calls"]:
            _state["n_layers"] = _state["layer_calls"]
        _begin_forward(input_ids)
    elif _state["n_layers"] and _state["layer_calls"] >= _state["n_layers"]:
        _begin_forward(input_ids)


def _tp_rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        return int(get_tensor_model_parallel_rank())
    except Exception:
        return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) or 0)


def _flush(route_dir: str) -> None:
    """Dump stores (rank 0) and stats (every rank), then reset the stats."""
    import numpy as np

    rank = _tp_rank()
    # Every rank dumps its own copy. Without sequence parallelism the copies
    # are identical and the driver keeps rank 0; with it each rank holds a
    # token shard and the driver concatenates them in rank order.
    for tag, seqs in _state["store"].items():
        if not seqs:
            continue
        path = os.path.join(route_dir, "ROUTES_%s_r%d.npz" % (tag, rank))
        if os.path.exists(path):
            continue  # written at the previous flush; stores are append-only
        arrays = {}
        for key, layers in seqs.items():
            arrays["%s/N" % key] = np.array([_state["ntok_by_key"].get(key, -1)], dtype=np.int64)
            for idx, ids in layers.items():
                arrays["%s/L%d" % (key, idx)] = ids.numpy()
        tmp = path + ".tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
        _say("flushed %d arrays for tag %r to %s" % (len(arrays), tag, path))
    stats = dict(_state["stats"], path=_state["path"], rank=rank, pid=os.getpid())
    with open(os.path.join(route_dir, "ROUTE_STATS_r%d_%d.json" % (rank, os.getpid())), "w") as fh:
        json.dump(stats, fh)
    _state["stats"] = _fresh_stats()


def _handle(route_dir: str, layer, router_logits: torch.Tensor):
    """Decide what this MoE call does. Returns (logits_to_use, bias_override).

    bias_override is None (leave the layer alone) or a tensor to swap in for
    `layer.e_score_correction_bias` for the duration of the kernel call."""
    mode, tag = _read_mode(route_dir)
    key = _state["key"]
    if key is None:
        # No model-forward stash: the input_ids argument (DeepSeek V4 passes it
        # for hash routing) is the fallback, handled by the caller.
        _miss(route_dir, "no sequence key")
        return router_logits, None
    idx = _next_layer_idx()

    if mode == "flush":
        if _state["flushed_key"] != key:
            _state["flushed_key"] = key
            _flush(route_dir)
        _state["stats"]["free_calls"] += 1
        return router_logits, None

    if mode == "free":
        _state["stats"]["free_calls"] += 1
        return router_logits, None

    bias = getattr(layer, "e_score_correction_bias", None)
    k = int(getattr(layer, "top_k", 6))
    scoring = getattr(layer, "scoring_func", "sqrtsoftplus")

    if mode == "record":
        if bias is None:
            _state["stats"]["bias_none"] += 1
        ids = reference_topk(router_logits, bias, k, scoring)
        with _lock:
            _state["store"].setdefault(tag, {}).setdefault(key, {})[idx] = (
                ids.to(torch.int16).cpu()
            )
        _state["stats"]["record_calls"] += 1
        return router_logits, None

    # replay
    ids = _state["store"].get(tag, {}).get(key, {}).get(idx)
    if ids is None:
        _miss(route_dir, "no recording for tag=%r layer=%d" % (tag, idx))
        return router_logits, None
    if ids.shape[0] != router_logits.shape[0]:
        _miss(route_dir, "token count %d != recorded %d" % (router_logits.shape[0], ids.shape[0]))
        return router_logits, None
    _state["stats"]["replay_calls"] += 1
    pinned = pin_logits(router_logits, ids)
    zero_bias = None
    if bias is not None:
        zero_bias = torch.zeros_like(bias)
    return pinned, zero_bias


def _wrap_monolithic(cls) -> None:
    original = cls.forward_monolithic

    def forward_monolithic(self, x, router_logits=None, input_ids=None, *a, **kw):
        route_dir = os.environ.get(DIR_ENV)
        if not route_dir or router_logits is None:
            return original(self, x, router_logits, input_ids, *a, **kw)
        _maybe_begin_from_input_ids(input_ids)
        _state["path"] = "monolithic"
        logits, bias_override = _handle(route_dir, self, router_logits)
        if bias_override is None:
            return original(self, x, logits, input_ids, *a, **kw)
        saved = self.e_score_correction_bias
        # Swap the attribute rather than a kwarg: the quant method reads the
        # bias off the layer when it builds the kernel call.
        try:
            if isinstance(saved, torch.nn.Parameter):
                self.e_score_correction_bias = torch.nn.Parameter(bias_override, requires_grad=False)
            else:
                self.e_score_correction_bias = bias_override
            return original(self, x, logits, input_ids, *a, **kw)
        finally:
            self.e_score_correction_bias = saved

    cls.forward_monolithic = forward_monolithic
    cls._route_probe_installed = True


def _wrap_modular(router_cls) -> None:
    """Python-side routing (non-B200 backends). Best effort, same semantics."""
    original = router_cls.select_experts

    def select_experts(self, hidden_states, router_logits, *a, **kw):
        route_dir = os.environ.get(DIR_ENV)
        if not route_dir:
            return original(self, hidden_states, router_logits, *a, **kw)
        _state["path"] = "modular"
        mode, tag = _read_mode(route_dir)
        out = original(self, hidden_states, router_logits, *a, **kw)
        weights, ids = out[0], out[1]
        key = _state["key"]
        if key is None:
            _miss(route_dir, "no sequence key (modular)")
            return out
        idx = _next_layer_idx()
        if mode == "flush":
            if _state["flushed_key"] != key:
                _state["flushed_key"] = key
                _flush(route_dir)
            return out
        if mode == "free":
            _state["stats"]["free_calls"] += 1
            return out
        if mode == "record":
            with _lock:
                _state["store"].setdefault(tag, {}).setdefault(key, {})[idx] = (
                    ids.to(torch.int16).cpu()
                )
            _state["stats"]["record_calls"] += 1
            return out
        rec = _state["store"].get(tag, {}).get(key, {}).get(idx)
        if rec is None or rec.shape[0] != ids.shape[0]:
            _miss(route_dir, "no usable recording (modular) layer=%d" % idx)
            return out
        _state["stats"]["replay_calls"] += 1
        new_ids = rec.to(ids.device, ids.dtype)
        new_w = reference_weights(
            router_logits, new_ids, bool(getattr(self, "renormalize", True)),
            float(getattr(self, "routed_scaling_factor", 1.0)),
            getattr(self, "scoring_func", "sqrtsoftplus"),
        ).to(weights.dtype)
        return (new_w, new_ids) + tuple(out[2:])

    router_cls.select_experts = select_experts
    router_cls._route_probe_installed = True


def _wrap_model_forward(cls) -> None:
    original = cls.forward

    def forward(self, input_ids, *a, **kw):
        if os.environ.get(DIR_ENV) and input_ids is not None:
            _begin_forward(input_ids)
        return original(self, input_ids, *a, **kw)

    cls.forward = forward
    cls._route_probe_installed = True
    _state["model_wrapped"] = True


def _find_model_classes() -> list:
    found = []
    for name, mod in list(sys.modules.items()):
        if not name.startswith("vllm.") or "deepseek_v4" not in name:
            continue
        cls = getattr(mod, "DeepseekV4Model", None)
        if cls is not None and hasattr(cls, "forward") and cls not in found:
            found.append(cls)
    return found


def install(deadline: float = 600.0, interval: float = 0.2) -> None:
    """Arm the hooks in this process. Safe to call more than once."""
    import importlib
    import time

    if _state["installed"]:
        return
    wrapped = False
    try:
        re_mod = importlib.import_module("vllm.model_executor.layers.fused_moe.routed_experts")
        cls = getattr(re_mod, "RoutedExperts", None)
        if cls is not None and hasattr(cls, "forward_monolithic") and not getattr(cls, "_route_probe_installed", False):
            _wrap_monolithic(cls)
            wrapped = True
            _say("wrapped RoutedExperts.forward_monolithic in pid %d" % os.getpid())
    except Exception as exc:
        _say("RoutedExperts import failed (%r); scanning loaded modules" % (exc,))
    if not wrapped:
        # The image may predate the RoutedExperts split: wrap any loaded
        # fused-MoE class that exposes the monolithic entry point.
        for name, mod in list(sys.modules.items()):
            if not name.startswith("vllm.model_executor.layers.fused_moe"):
                continue
            for attr in dir(mod):
                obj = getattr(mod, attr, None)
                if isinstance(obj, type) and "forward_monolithic" in obj.__dict__ \
                        and not getattr(obj, "_route_probe_installed", False):
                    _wrap_monolithic(obj)
                    wrapped = True
                    _say("wrapped %s.%s.forward_monolithic (scan)" % (name, attr))
        if not wrapped:
            _say("no monolithic MoE entry point found; only the modular router is hooked")
    try:
        r_mod = importlib.import_module("vllm.model_executor.layers.fused_moe.router.fused_moe_router")
        rcls = getattr(r_mod, "FusedMoERouter", None)
        if rcls is not None and not getattr(rcls, "_route_probe_installed", False):
            _wrap_modular(rcls)
            _say("wrapped FusedMoERouter.select_experts in pid %d" % os.getpid())
    except Exception as exc:
        _say("FusedMoERouter wrap failed: %r" % (exc,))

    def _arm_model():
        end = time.time() + deadline
        while time.time() < end:
            classes = [c for c in _find_model_classes() if not getattr(c, "_route_probe_installed", False)]
            if classes:
                for c in classes:
                    _wrap_model_forward(c)
                    _say("wrapped %s.%s.forward" % (c.__module__, c.__name__))
                return
            time.sleep(interval)
        _say("DeepseekV4Model not found within %.0fs; relying on input_ids fallback" % deadline)

    threading.Thread(target=_arm_model, daemon=True, name="engram-route-arm").start()
    _state["installed"] = True


# ---------------------------------------------------------------- driver side

def clear(route_dir: str) -> None:
    for name in os.listdir(route_dir):
        if name.startswith(("ROUTE_STATS_", "ROUTES_", _MODE_FILE)) or name == MISS:
            os.unlink(os.path.join(route_dir, name))


def read_stats(route_dir: str) -> dict:
    """Aggregate and remove the per-rank stats files from the last flush."""
    ranks = []
    for name in sorted(os.listdir(route_dir)):
        if name.startswith("ROUTE_STATS_"):
            with open(os.path.join(route_dir, name)) as fh:
                ranks.append(json.load(fh))
            os.unlink(os.path.join(route_dir, name))
    agg = {"ranks": len(ranks), "miss": os.path.exists(os.path.join(route_dir, MISS))}
    for k in ("record_calls", "replay_calls", "free_calls", "misses", "forwards", "bias_none"):
        agg[k] = [r.get(k, 0) for r in ranks]
    agg["layers"] = max([r.get("layers", 0) for r in ranks] or [0])
    agg["path"] = sorted({r.get("path") for r in ranks if r.get("path")})
    return agg


def load_routes(route_dir: str, tag: str) -> dict:
    """{key: {layer_idx: int16 array [T, k]}} from the flushed per-rank stores.

    Ranks holding a full copy (no sequence parallelism) are deduplicated to
    rank 0; ranks holding token shards are concatenated in rank order."""
    import numpy as np

    per_rank: dict[int, dict] = {}
    for name in sorted(os.listdir(route_dir)):
        if not (name.startswith("ROUTES_%s_r" % tag) and name.endswith(".npz")):
            continue
        rank = int(name[len("ROUTES_%s_r" % tag):-4])
        data: dict = {}
        with np.load(os.path.join(route_dir, name)) as z:
            for arr in z.files:
                key, _, rest = arr.partition("/")
                if rest == "N":
                    data.setdefault(key, {})["N"] = int(z[arr][0])
                else:
                    data.setdefault(key, {})[int(rest[1:])] = z[arr]
        per_rank[rank] = data
    if not per_rank:
        return {}
    ranks = sorted(per_rank)
    out: dict = {}
    for key, layers0 in per_rank[ranks[0]].items():
        n = layers0.get("N", -1)
        for idx, ids0 in layers0.items():
            if idx == "N":
                continue
            shards = [per_rank[r].get(key, {}).get(idx) for r in ranks]
            if all(sh is not None for sh in shards) and n > 0 and ids0.shape[0] < n \
                    and sum(sh.shape[0] for sh in shards) == n:
                out.setdefault(key, {})[idx] = np.concatenate(shards, axis=0)
            else:
                out.setdefault(key, {})[idx] = ids0
    return out


def agreement(a: dict, b: dict, boundaries: dict) -> dict:
    """Per-layer routing agreement between two recordings of the same sequences.

    `boundaries` maps key -> first scored (answer) position, so the prompt and
    the answer span are reported separately."""
    import numpy as np

    per_layer: dict = {}
    for key, layers in a.items():
        other = b.get(key)
        if other is None:
            continue
        cut = boundaries.get(key, 0)
        for idx, ids_a in layers.items():
            ids_b = other.get(idx)
            if ids_b is None or ids_b.shape != ids_a.shape:
                continue
            sa = np.sort(ids_a.astype(np.int64), axis=1)
            sb = np.sort(ids_b.astype(np.int64), axis=1)
            exact = (sa == sb).all(axis=1)
            # |A ∩ B| / k via sorted merge: count elements of sa present in sb.
            overlap = np.array([np.intersect1d(x, y).size for x, y in zip(sa, sb)]) / sa.shape[1]
            top1 = ids_a[:, 0].astype(np.int64) == ids_b[:, 0].astype(np.int64)
            slot = per_layer.setdefault(idx, {
                "prompt_exact": 0, "prompt_overlap": 0.0, "prompt_top1": 0, "prompt_n": 0,
                "answer_exact": 0, "answer_overlap": 0.0, "answer_top1": 0, "answer_n": 0,
            })
            for span, sel in (("prompt", slice(0, cut)), ("answer", slice(cut, None))):
                n = exact[sel].size
                slot[span + "_n"] += n
                slot[span + "_exact"] += int(exact[sel].sum())
                slot[span + "_overlap"] += float(overlap[sel].sum())
                slot[span + "_top1"] += int(top1[sel].sum())
    table = {}
    for idx in sorted(per_layer):
        s = per_layer[idx]
        row = {}
        for span in ("prompt", "answer"):
            n = max(s[span + "_n"], 1)
            row[span] = {
                "tokens": s[span + "_n"],
                "exact_set": round(s[span + "_exact"] / n, 4),
                "mean_overlap": round(s[span + "_overlap"] / n, 4),
                "top1_same": round(s[span + "_top1"] / n, 4),
            }
        table[idx] = row
    return table
