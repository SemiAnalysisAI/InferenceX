"""neuronx_distributed (NxD) backend for Trainium collectives.

NxD wraps torch_xla's xm.* primitives but, before any collective runs, calls
parallel_state.initialize_model_parallel(...) to register a tensor-parallel
ProcessGroup. Every NxD collective then passes that group's replica-mesh
through to xm.* as the `groups=` arg. That populates `replica_groups` on
the emitted HLO instead of leaving it empty, which lets the Neuron PJRT
compile passes attach the `collective_type`/`stream_id` frontend_attributes
the runtime needs to fill in `channel_n` and pick a working algorithm at
ws >= 8.

The bare-xm path (operatorx torch backend) emits collectives with
`replica_groups={}` and fails at ws >= 8 with `Invalid channel_n(0) for
kangaring algorithm` (see compute-comm-overlap and intranode-collective
docs).
"""
from __future__ import annotations

import os

import torch
import torch_neuronx  # noqa: F401  registers the neuron device
import torch_xla.core.xla_model as xm

from operatorx.core import BackendImpl, Op, UnsupportedOpError, lookup_versions


def versions() -> dict[str, str]:
    return lookup_versions("neuronx_distributed", "torch_neuronx", "torch_xla")


_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _resolve(dtype: str) -> torch.dtype:
    if dtype not in _DTYPES:
        raise UnsupportedOpError(f"nxd doesn't support dtype={dtype!r}")
    return _DTYPES[dtype]


def _device():
    return xm.xla_device()


_NXD_READY = False


def _ensure_ready() -> None:
    """Bring up torch.distributed (xla backend) + NxD's parallel_state with
    a single TP group spanning all ranks."""
    global _NXD_READY
    if _NXD_READY:
        return
    import torch.distributed as dist
    if not dist.is_initialized():
        import torch_xla.distributed.xla_backend  # noqa: F401
        dist.init_process_group("xla")

    from neuronx_distributed.parallel_layers import parallel_state
    if not parallel_state.model_parallel_is_initialized():
        ws = dist.get_world_size()
        lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=ws,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            lnc_size=lnc,
        )
    _NXD_READY = True


def _tp_groups() -> list[list[int]]:
    from neuronx_distributed.parallel_layers import parallel_state
    return parallel_state.get_tensor_model_parallel_replica_groups()


def _prepare_allreduce(op: Op) -> dict:
    _ensure_ready()
    dt = _resolve(op.args["dtype"])
    t = torch.randn(op.args["num_elements"], dtype=dt, device=_device())
    return {"t": t}


def _kernel_allreduce(ctx: dict) -> None:
    from neuronx_distributed.parallel_layers.comm import all_reduce
    all_reduce(xm.REDUCE_SUM, ctx["t"], groups=_tp_groups())
    ctx["out"] = ctx["t"]
    xm.mark_step()


def _prepare_allgather(op: Op) -> dict:
    _ensure_ready()
    dt = _resolve(op.args["dtype"])
    t = torch.randn(op.args["num_elements_per_rank"], dtype=dt, device=_device())
    return {"t": t}


def _kernel_allgather(ctx: dict) -> None:
    from neuronx_distributed.parallel_layers.comm import all_gather
    ctx["out"] = all_gather(ctx["t"], dim=0, groups=_tp_groups())
    xm.mark_step()


def _prepare_reduce_scatter(op: Op) -> dict:
    _ensure_ready()
    ws = int(op.args["world_size"])
    if op.args["num_elements"] % ws != 0:
        raise UnsupportedOpError(
            f"reduce_scatter requires num_elements divisible by world_size "
            f"({op.args['num_elements']} % {ws} != 0)"
        )
    dt = _resolve(op.args["dtype"])
    t = torch.randn(op.args["num_elements"], dtype=dt, device=_device())
    return {"t": t, "shard_count": ws}


def _kernel_reduce_scatter(ctx: dict) -> None:
    from neuronx_distributed.parallel_layers.comm import reduce_scatter
    ctx["out"] = reduce_scatter(
        reduce_type=xm.REDUCE_SUM,
        input=ctx["t"],
        scale=1.0,
        scatter_dim=0,
        shard_count=ctx["shard_count"],
        groups=_tp_groups(),
    )
    xm.mark_step()


# NxD doesn't expose an all_to_all wrapper. Call xm.all_to_all directly with
# the explicit TP group so the HLO has populated replica_groups (same pattern
# nccom-test uses).
def _prepare_alltoall(op: Op) -> dict:
    _ensure_ready()
    ws = int(op.args["world_size"])
    n_per = op.args["num_elements_per_rank"]
    if n_per % ws != 0:
        raise UnsupportedOpError(
            f"alltoall requires num_elements_per_rank divisible by world_size "
            f"({n_per} % {ws} != 0)"
        )
    dt = _resolve(op.args["dtype"])
    t = torch.randn(n_per, dtype=dt, device=_device())
    return {"t": t, "split_count": ws}


def _kernel_alltoall(ctx: dict) -> None:
    ctx["out"] = xm.all_to_all(
        ctx["t"], split_dimension=0, concat_dimension=0,
        split_count=ctx["split_count"], groups=_tp_groups(),
    )
    xm.mark_step()


IMPLS = [
    BackendImpl(op_type="allreduce", prepare=_prepare_allreduce, kernel=_kernel_allreduce),
    BackendImpl(op_type="allgather", prepare=_prepare_allgather, kernel=_kernel_allgather),
    BackendImpl(op_type="reduce_scatter", prepare=_prepare_reduce_scatter, kernel=_kernel_reduce_scatter),
    BackendImpl(op_type="alltoall", prepare=_prepare_alltoall, kernel=_kernel_alltoall),
]
