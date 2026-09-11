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
    # One scalar per token: average over the hyper-connection copies.
    return gate.mean(dim=1).to(torch.float32).cpu().numpy(), start


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
    bootstrap = os.path.join(analysis_dir, "_probe_bootstrap")
    os.makedirs(bootstrap, exist_ok=True)
    with open(os.path.join(bootstrap, "sitecustomize.py"), "w") as fh:
        fh.write(_BOOTSTRAP.format(analysis_dir=analysis_dir))
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

if os.environ.get("ENGRAM_PROBE_DIR"):
    sys.path.insert(0, {analysis_dir!r})

    def _arm(deadline=3600.0, interval=0.05):
        import time

        end = time.time() + deadline
        while time.time() < end:
            if any(name.rsplit(".", 1)[-1] == "engram" for name in list(sys.modules)):
                try:
                    from engram import gate_probe

                    gate_probe.install()
                    sys.stderr.write("engram-probe: armed in pid %d\\n" % os.getpid())
                except Exception as exc:  # never break a worker
                    sys.stderr.write("engram-probe: arm failed: %r\\n" % (exc,))
                return
            time.sleep(interval)

    threading.Thread(target=_arm, daemon=True, name="engram-probe-arm").start()
'''
