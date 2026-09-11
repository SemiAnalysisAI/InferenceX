"""Materialize DeepSeek-V4.1-Flash Engram gates, which the fused kernel hides.

The shipped path computes the gate inside a Triton kernel and consumes it in
the same store (`hidden + gate * value`), so no hook can observe it. We wrap
`Engram.forward`, recompute the gate in torch from the module's own public
tensors, and mirror the kernel's arithmetic exactly:

    hidden_rms = rsqrt(mean(hidden^2) + eps)
    key_rms    = rsqrt(mean(key^2) + eps)
    dot        = sum(hidden * q * k * key) * hidden_rms * key_rms * dim^-0.5
    gate       = sigmoid(sign(dot) * sqrt(max(|dot|, clamp)))

Only public attributes are touched (wkv, embed, q_weight, k_weight, hc_mult,
dim, eps, clamp_value), so this survives the module-path and kernel-name
churn between the 0909 image and vLLM main.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)

# Module path moved between the 0909 image and vLLM main; try both, then scan.
_CANDIDATE_MODULES = (
    "vllm.model_executor.layers.engram",
    "vllm.models.deepseek_v4_1.common.engram",
)

# With TP the model runs in worker subprocesses, so an in-memory buffer is
# unreachable from the driver. Each worker writes rank-tagged .npy files into
# ENGRAM_PROBE_DIR and the driver collects them after every prefill.
PROBE_DIR_ENV = "ENGRAM_PROBE_DIR"
_CALL_COUNTER = 0


def _find_engram_class():
    for name in _CANDIDATE_MODULES:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        cls = getattr(mod, "Engram", None)
        if cls is not None and hasattr(cls, "forward"):
            logger.info("engram-probe: found Engram in %s", name)
            return cls
    import pkgutil

    import vllm

    for info in pkgutil.walk_packages(vllm.__path__, "vllm."):
        if "engram" not in info.name.rsplit(".", 1)[-1]:
            continue
        try:
            mod = importlib.import_module(info.name)
        except Exception:
            continue
        cls = getattr(mod, "Engram", None)
        if cls is not None and hasattr(cls, "forward"):
            logger.info("engram-probe: found Engram in %s (scan)", info.name)
            return cls
    raise RuntimeError("engram-probe: could not locate the Engram module")


def install() -> None:
    """Wrap Engram.forward so every call also records its gate."""
    cls = _find_engram_class()
    if getattr(cls, "_gate_probe_installed", False):
        return
    original = cls.forward

    def forward(self, hidden_states, hash_ids, token_mask=None):
        out = original(self, hidden_states, hash_ids, token_mask)
        probe_dir = os.environ.get(PROBE_DIR_ENV)
        if not probe_dir:
            return out
        try:
            _dump(self, hidden_states, hash_ids, token_mask, probe_dir)
        except Exception:
            # Never let the probe take down a forward pass.
            logger.exception("engram-probe: gate recompute failed")
        return out

    cls.forward = forward
    cls._gate_probe_installed = True


def _dump(self, hidden_states, hash_ids, token_mask, probe_dir) -> None:
    """Write this rank's slice of the gate, tagged so the driver can stitch."""
    global _CALL_COUNTER
    import numpy as np

    gate, start = _gate(self, hidden_states, hash_ids, token_mask)
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        rank = get_tensor_model_parallel_rank()
    except Exception:
        rank = 0
    _CALL_COUNTER += 1
    os.makedirs(probe_dir, exist_ok=True)
    name = (
        f"L{self.layer_hash_index}_r{rank}_s{start}"
        f"_n{gate.shape[0]}_c{_CALL_COUNTER}_{os.getpid()}.npy"
    )
    # Write then rename so the driver never reads a partial file.
    tmp = os.path.join(probe_dir, "." + name)  # np.save keeps the .npy suffix
    np.save(tmp, gate)
    os.replace(tmp, os.path.join(probe_dir, name))


@torch.no_grad()
def _gate(self, hidden_states, hash_ids, token_mask):
    """Recompute the per-(token, hyper-connection-copy) gate."""
    kv = self.wkv(self.embed(hash_ids).flatten(-2))
    num_kv_tokens = hash_ids.shape[0]
    start = 0

    # Mirror forward()'s sequence-parallel slicing so rows line up with kv.
    if getattr(self, "use_sequence_parallel", False):
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        shard = (num_kv_tokens + tp_size - 1) // tp_size
        start = min(tp_rank * shard, num_kv_tokens)
        num_kv_tokens = min(shard, num_kv_tokens - start)
        if token_mask is not None:
            token_mask = token_mask[start : start + num_kv_tokens]

    hc, dim = self.hc_mult, self.dim
    h = hidden_states[:num_kv_tokens].float()
    key = kv[:num_kv_tokens, : hc * dim].view(-1, hc, dim).float()
    q = self.q_weight.float().unsqueeze(0)
    k = self.k_weight.float().unsqueeze(0)

    hidden_rms = torch.rsqrt(h.pow(2).mean(-1) + self.eps)
    key_rms = torch.rsqrt(key.pow(2).mean(-1) + self.eps)
    dot = (h * q * k * key).sum(-1) * hidden_rms * key_rms * (dim**-0.5)
    gate = torch.sigmoid(torch.sqrt(dot.abs().clamp_min(self.clamp_value)) * dot.sign())
    if token_mask is not None:
        gate = gate * token_mask.view(-1, 1).to(gate.dtype)
    # Keep the hyper-connection axis: [tokens, hc]. Averaging over the four
    # copies was suppressing the peak -- if one copy opens and three stay shut,
    # the mean understates the gate by ~4x.
    return gate.to(torch.float32).cpu().numpy(), start


def install_in_workers(analysis_dir: str) -> str:
    """Arm the patch inside vLLM's worker processes and return the bootstrap dir.

    With TP the engine spawns fresh interpreters (`VLLM_WORKER_MULTIPROC_METHOD`
    is forced to 'spawn' once CUDA is initialized), so a patch applied in the
    driver is not inherited -- the first run captured nothing for exactly this
    reason. CPython imports `sitecustomize` in *every* interpreter it starts, so
    a sitecustomize on PYTHONPATH reaches the workers. It cannot import vllm at
    that point (site time is far too early), so it arms a tiny daemon thread
    that installs the wrapper the moment the Engram module shows up in
    sys.modules -- which happens during model load, minutes before any forward.
    """
    bootstrap = write_bootstrap(analysis_dir)
    existing = os.environ.get("PYTHONPATH", "")
    parts = [bootstrap, analysis_dir] + ([existing] if existing else [])
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)
    return bootstrap


# Kept as a template rather than a file so the probe stays a single module.
_BOOTSTRAP = '''\
"""Generated by gate_probe.install_in_workers -- installs the Engram gate probe.

This shadows any container-provided sitecustomize, so chain to it first.
"""
import os
import sys
import threading

_self = os.path.dirname(os.path.abspath(__file__))
for _entry in [p for p in sys.path if os.path.abspath(p) != _self]:
    _candidate = os.path.join(_entry, "sitecustomize.py")
    if os.path.exists(_candidate):
        import importlib.util

        _spec = importlib.util.spec_from_file_location("_real_sitecustomize", _candidate)
        _mod = importlib.util.module_from_spec(_spec)
        try:
            _spec.loader.exec_module(_mod)
        except Exception:
            pass
        break

if any(os.environ.get(v) for v in ("ENGRAM_PROBE_DIR", "ENGRAM_ABLATE", "ENGRAM_METER_DIR")):
    sys.path.insert(0, {analysis_dir!r})

    def _arm(deadline=3600.0, interval=0.05):
        import time

        end = time.time() + deadline
        while time.time() < end:
            if any(name.rsplit(".", 1)[-1] == "engram" for name in list(sys.modules)):
                try:
                    from engram import gate_probe

                    if os.environ.get("ENGRAM_METER_DIR"):
                        gate_probe.install_meter()
                        sys.stderr.write("engram-meter: armed in pid %d\\n" % os.getpid())
                    elif os.environ.get("ENGRAM_ABLATE"):
                        gate_probe.install_ablation()
                        sys.stderr.write("engram-ablate: armed in pid %d\\n" % os.getpid())
                    else:
                        gate_probe.install()
                        sys.stderr.write("engram-probe: armed in pid %d\\n" % os.getpid())
                except Exception as exc:  # never break a worker
                    sys.stderr.write("engram-probe: arm failed: %r\\n" % (exc,))
                return
            time.sleep(interval)

    threading.Thread(target=_arm, daemon=True, name="engram-probe-arm").start()
'''


ABLATE_ENV = "ENGRAM_ABLATE"


def install_ablation() -> None:
    """Force the Engram contribution to zero, leaving the rest of the model intact.

    The gate is consumed inside the fused kernel, so it cannot be set to zero
    directly; the contribution is removed by not applying it at all.

    `Engram.forward` could return either the updated hidden states
    (`hidden + gate * value`) or just the delta (`gate * value`), and the
    correct ablation differs -- return the input unchanged in the first case,
    zeros in the second. Returning the wrong one would leave a model that
    still runs and quietly produces garbage, which would look like a dramatic
    ablation result. So the convention is detected on the first call, from the
    cosine similarity between the real output and the input: the mean gate is
    ~0.02, so an updated-hidden return is nearly parallel to its input, while
    a delta return is not.
    """
    cls = _find_engram_class()
    if getattr(cls, "_gate_ablation_installed", False):
        return
    original = cls.forward
    try:
        import inspect

        _say("Engram.forward source:\n" + inspect.getsource(original))
    except Exception:
        _say("Engram.forward source unavailable")

    state = {"returns_updated_hidden": None, "calls": 0}

    def forward(self, hidden_states, hash_ids, token_mask=None):
        if state["returns_updated_hidden"] is None:
            out = original(self, hidden_states, hash_ids, token_mask)
            state["returns_updated_hidden"] = _looks_like_updated_hidden(out, hidden_states)
            _say(
                "forward returns %s; ablating by returning %s"
                % (
                    "updated hidden states" if state["returns_updated_hidden"] else "the delta",
                    "the input" if state["returns_updated_hidden"] else "zeros",
                )
            )
            # Discard this one real output so even the first token is ablated.
        # The call count is the evidence that the ablation engaged at all. A
        # zero count means the patch was installed but never reached, and any
        # eval delta would then be run-to-run noise rather than an ablation.
        state["calls"] += 1
        if state["calls"] in (1, 10, 100) or state["calls"] % 5000 == 0:
            _say("forward call count = %d" % state["calls"])
        if state["returns_updated_hidden"]:
            return hidden_states
        return torch.zeros_like(hidden_states)

    cls.forward = forward
    cls._gate_ablation_installed = True


def _say(message: str) -> None:
    """Diagnostics that must survive the server's logging config."""
    sys.stderr.write("engram-ablate: %s\n" % message)
    sys.stderr.flush()


@torch.no_grad()
def _looks_like_updated_hidden(out, hidden_states) -> bool:
    if out is hidden_states:
        return True
    try:
        a = out.reshape(-1).float()
        b = hidden_states.reshape(-1).float()
        if a.shape != b.shape:
            return False
        cos = torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)
        _say("cos(out, hidden) = %.6f" % float(cos))
        return bool(cos > 0.5)
    except Exception as exc:
        _say("convention detection failed (%r); assuming updated hidden" % (exc,))
        return True


def write_bootstrap(analysis_dir: str) -> str:
    """Write the worker bootstrap and return its directory, without touching env.

    `vllm serve` is launched from a shell, so the shell needs the path to put
    on PYTHONPATH itself.
    """
    bootstrap = os.path.join(analysis_dir, "_probe_bootstrap")
    os.makedirs(bootstrap, exist_ok=True)
    with open(os.path.join(bootstrap, "sitecustomize.py"), "w") as fh:
        fh.write(_BOOTSTRAP.format(analysis_dir=analysis_dir))
    return bootstrap


if __name__ == "__main__":  # `python3 -m engram.gate_probe <analysis_dir>`
    import sys as _sys

    print(write_bootstrap(_sys.argv[1] if len(_sys.argv) > 1 else "analysis"))


METER_DIR_ENV = "ENGRAM_METER_DIR"
_ABLATE_SENTINEL = "ABLATE_NOW"


def set_ablate(meter_dir: str, on: bool) -> None:
    """Toggle ablation for the worker processes via a sentinel file.

    The workers are separate interpreters, so the driver cannot flip a global
    in them. A file check per call costs microseconds and keeps baseline and
    ablated measurements inside one process lifetime, which is the whole point
    -- same weights, same allocation, one variable.
    """
    path = os.path.join(meter_dir, _ABLATE_SENTINEL)
    if on:
        open(path, "w").close()
    elif os.path.exists(path):
        os.unlink(path)


def clear_meter(meter_dir: str) -> None:
    for name in os.listdir(meter_dir):
        if name.startswith("M") and name.endswith(".npy"):
            os.unlink(os.path.join(meter_dir, name))


def read_meter(meter_dir: str) -> dict:
    """Aggregate the per-call contribution records the workers wrote."""
    import numpy as np

    rows = []
    for name in sorted(os.listdir(meter_dir)):
        if not name.startswith("M") or not name.endswith(".npy"):
            continue
        try:
            rows.append(np.load(os.path.join(meter_dir, name)))
        except Exception:
            continue
    if not rows:
        return {}
    arr = np.concatenate([r.reshape(-1, 3) for r in rows])
    by_layer = {}
    for layer in sorted({int(v) for v in arr[:, 0]}):
        sel = arr[arr[:, 0] == layer]
        by_layer[f"engram{layer}"] = {
            "calls": int(sel.shape[0]),
            "mean_rel_norm": round(float(sel[:, 1].mean()), 8),
            "max_rel_norm": round(float(sel[:, 1].max()), 8),
            "mean_max_gate": round(float(sel[:, 2].mean()), 6),
        }
    return {
        "calls": int(arr.shape[0]),
        "mean_rel_norm": round(float(arr[:, 1].mean()), 8),
        "max_rel_norm": round(float(arr[:, 1].max()), 8),
        "per_layer": by_layer,
    }


def install_meter() -> None:
    """Wrap Engram.forward to record the contribution it hands back."""
    cls = _find_engram_class()
    if getattr(cls, "_gate_meter_installed", False):
        return
    original = cls.forward
    state = {"returns_updated_hidden": None}

    def forward(self, hidden_states, hash_ids, token_mask=None):
        meter_dir = os.environ.get(METER_DIR_ENV)
        out = original(self, hidden_states, hash_ids, token_mask)
        if state["returns_updated_hidden"] is None:
            state["returns_updated_hidden"] = _looks_like_updated_hidden(out, hidden_states)
            _say("meter: rel-norm of the real contribution = %.6f"
                 % float((out.float() - hidden_states.float()).norm()
                         / (hidden_states.float().norm() + 1e-12)))
        ablate = bool(meter_dir) and os.path.exists(
            os.path.join(meter_dir, _ABLATE_SENTINEL)
        )
        if ablate:
            # Use the module's own gate-shutting path rather than guessing what
            # forward returns. Its docstring is explicit: "token_mask: [T],
            # False shuts the gate so those positions pass through untouched."
            # An all-False mask is therefore an exact, supported ablation, and
            # it is applied by the same fused kernel that normally consumes the
            # gate -- no assumption about return conventions at all.
            shut = torch.zeros(
                hash_ids.shape[0], dtype=torch.bool, device=hidden_states.device
            )
            returned = original(self, hidden_states, hash_ids, shut)
        else:
            returned = out
        if meter_dir:
            try:
                _record(self, meter_dir, returned, hidden_states)
            except Exception:
                _say("meter failed: %r" % (sys.exc_info()[1],))
        return returned

    cls.forward = forward
    cls._gate_meter_installed = True
    _say("meter installed")


@torch.no_grad()
def _record(self, meter_dir, returned, hidden_states) -> None:
    import numpy as np

    delta = (returned.float() - hidden_states.float()).norm()
    rel = float(delta / (hidden_states.float().norm() + 1e-12))
    row = np.array(
        [[float(self.layer_hash_index), rel, 0.0]], dtype=np.float64
    )
    name = f"M{self.layer_hash_index}_{os.getpid()}_{len(os.listdir(meter_dir))}.npy"
    tmp = os.path.join(meter_dir, "." + name)
    np.save(tmp, row)
    os.replace(tmp, os.path.join(meter_dir, name))
