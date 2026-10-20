"""MoE routing: each step's per-layer top-k expert ids."""

import os

from . import runtime

IDS_MAX_TOKENS = 1024


class _State:
    buffer = None
    rank = None
    write = False
    num_experts = None
    captured = False
    pending = []


_s = _State()


def _define_op():
    import torch

    if hasattr(torch.ops.infx, "capture_routing"):
        return

    @torch.library.custom_op("infx::capture_routing", mutates_args=("buf",))
    def capture_routing(buf: torch.Tensor, ids: torch.Tensor, layer: int) -> None:
        _copy(buf, ids, layer)

    @capture_routing.register_fake
    def _(buf, ids, layer):
        return None


def _copy(buf, ids, layer):
    n = min(ids.shape[0], buf.shape[0])
    buf[:n, layer, :ids.shape[1]] = ids[:n]
    _s.captured = True


def setup(rank, max_tokens, num_layers, topk, num_experts, write=True, bound=None, failed=None, extra=None):
    """Allocate the buffer and write routing/<rank>/meta.json. `write`: this rank records (one per TP group)."""
    import torch

    if _s.buffer is not None:
        return
    try:
        _define_op()
    except Exception:
        runtime.write_error("routing custom op")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _s.buffer = torch.full((max_tokens, num_layers, topk), -1, dtype=torch.int32, device=device)
    _s.rank, _s.write, _s.num_experts = rank, write, num_experts
    meta = {"rank": rank, "write": write, "layers": num_layers, "topk": topk, "num_experts": num_experts,
            "max_tokens": max_tokens, "bound": {str(k): v for k, v in sorted((bound or {}).items())},
            "failed": failed or {}}
    meta.update(extra or {})
    runtime.write_json(os.path.join("routing", rank, "meta.json"), meta)


def update_meta(**fields):
    import json

    path = os.path.join(runtime.DIR, "routing", _s.rank, "meta.json")
    try:
        with open(path) as f:
            meta = json.load(f)
    except Exception:
        meta = {}
    meta.update(fields)
    runtime.write_json(os.path.join("routing", _s.rank, "meta.json"), meta)


def active():
    return _s.buffer is not None


def capture(layer_id, topk_ids):
    """Record layer `layer_id`'s top-k ids [tokens, k]; traceable by Dynamo (fullgraph)."""
    import torch

    buf = _s.buffer
    if buf is None or topk_ids is None or not isinstance(topk_ids, torch.Tensor) or topk_ids.dim() != 2:
        return
    if not 0 <= layer_id < buf.shape[1]:
        return
    if topk_ids.shape[1] > buf.shape[2]:
        topk_ids = topk_ids[:, :buf.shape[2]]
    if torch.compiler.is_compiling():
        torch.ops.infx.capture_routing(buf, topk_ids, layer_id)
    else:
        _copy(buf, topk_ids, layer_id)


def reset():
    """Clear the buffer before a step, so a step never reports rows an earlier one wrote."""
    if _s.buffer is None or not _s.write:
        return
    import torch

    with torch.autograd.profiler.record_function("infx_routing_reset"):
        _s.buffer.fill_(-1)


def snapshot(step, num_tokens, t0_ns):
    """Queue this step's routing rows for the host (asynchronous)."""
    if _s.buffer is None or not _s.write or not num_tokens:
        return
    import torch

    with torch.autograd.profiler.record_function("infx_routing_copy"):
        rows = _s.buffer[:num_tokens].to(torch.int16)
        host = torch.empty(rows.shape, dtype=torch.int16, pin_memory=rows.is_cuda)
        host.copy_(rows, non_blocking=True)
    _s.pending.append((step, num_tokens, t0_ns, host, rows))


def flush():
    """Write the window's queued routing: per-layer expert counts, and ids for small steps."""
    pending, _s.pending = _s.pending, []
    if not pending:
        return
    if not _s.captured:
        update_meta(unavailable="no MoE layer exposed its top-k ids (fused routing inside the MoE kernel)")
        return
    import numpy as np
    import torch

    if torch.cuda.is_available():
        torch.cuda.current_stream().synchronize()
    out_dir = os.path.join(runtime.DIR, "routing", _s.rank)
    os.makedirs(out_dir, exist_ok=True)
    for step, num_tokens, t0_ns, host, _ in pending:
        ids = host.numpy().astype(np.int32)
        experts = max(_s.num_experts or 0, int(ids.max()) + 1)
        counts = np.zeros((ids.shape[1], experts), dtype=np.int32)
        for layer in range(ids.shape[1]):
            valid = ids[:, layer][ids[:, layer] >= 0]
            counts[layer] = np.bincount(valid, minlength=experts)[:experts]
        valid_rows = int((ids >= 0).any(axis=(1, 2)).sum())
        arrays = {"counts": counts, "t0_ns": np.array(t0_ns), "tokens": np.array(num_tokens),
                  "valid_rows": np.array(valid_rows)}
        if num_tokens <= IDS_MAX_TOKENS:
            arrays["ids"] = ids.astype(np.int16)
        np.savez_compressed(os.path.join(out_dir, f"step{step:06d}.npz"), **arrays)
