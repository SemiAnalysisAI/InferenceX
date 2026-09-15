#!/usr/bin/env python3
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import os
from threading import Event
import time

import torch
import torch.distributed as dist

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    ensure_model_parallel_initialized,
    get_pp_group,
    get_tp_group,
    init_distributed_environment,
)


POLICY_KEY = "__k3_pp_sender_all_gather__"
STOP_KEY = "__k3_pp_prepost_stop__"
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])
assert world_size == 16

torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
with set_current_vllm_config(VllmConfig()):
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method=(
            f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
        ),
        local_rank=local_rank,
        backend="nccl",
    )
    ensure_model_parallel_initialized(8, 2)

pp_group = get_pp_group()
tp_group = get_tp_group()
sibling_group = pp_group.make_sibling_device_group(
    group_desc="test_pp_prepost_activation"
)
send_slots: list[dict[str, torch.Tensor] | None] = [None, None]
send_works = [[], []]
slot_index = 0
executor = (
    ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-pp-recv")
    if pp_group.is_last_rank
    else None
)
recv_future: Future | None = None


def receive(entered: Event | None = None):
    if entered is not None:
        # The sender-side barrier below cannot pass until every PP1 lane has
        # entered its background receive task. The deliberate sender delay
        # then leaves a wide window in which metadata receive is already posted.
        entered.set()
    torch.cuda.set_device(device)
    return pp_group.irecv_tensor_dict(
        all_gather_group=tp_group,
        # Intentionally wrong for half the steps: sender metadata must win.
        all_gather_tensors={"hidden_states": False, "residual": False},
        device_group=sibling_group,
        use_sender_all_gather_metadata=True,
    )


dist.barrier()
for step in range(100):
    rows = (47, 1, 24, 13, 31, 7, 18)[step % 7]
    shape = (rows, 128)
    replicated = step % 2 == 0
    gather_policy = (
        {"hidden_states": False, "residual": False} if replicated else {}
    )

    if step > 0:
        if pp_group.is_last_rank:
            assert executor is not None
            entered = Event()
            recv_future = executor.submit(receive, entered)
            assert entered.wait(timeout=10), "background receive did not start"
        # Establish that PP1 entered receive before PP0 begins delayed work.
        dist.barrier()
        if pp_group.is_first_rank and step == 1:
            time.sleep(0.25)

    if pp_group.is_first_rank:
        base = torch.arange(
            rows * 128, dtype=torch.float32, device=device
        ).reshape(shape)
        tensors = {
            "hidden_states": base + step * 1000,
            "residual": base + step * 1000 + 1,
        }
        for handle in send_works[slot_index]:
            handle.wait()
        slot = send_slots[slot_index]
        schema = tuple(
            (key, tensor.shape, tensor.dtype)
            for key, tensor in sorted(tensors.items())
        )
        slot_schema = (
            None
            if slot is None
            else tuple(
                (key, tensor.shape, tensor.dtype)
                for key, tensor in sorted(slot.items())
            )
        )
        if slot_schema != schema:
            slot = {
                key: torch.empty_like(tensor) for key, tensor in tensors.items()
            }
            send_slots[slot_index] = slot
        assert slot is not None
        for key, tensor in tensors.items():
            slot[key].copy_(tensor)
        send_works[slot_index] = pp_group.isend_tensor_dict(
            slot,
            all_gather_group=tp_group,
            all_gather_tensors=gather_policy,
            device_group=sibling_group,
            send_all_gather_metadata=True,
        )
        slot_index = (slot_index + 1) % 2
        for tensor in tensors.values():
            tensor.fill_(-999)
    else:
        if step == 0:
            # Production's first-step fallback has no prior call to prepost.
            received, handles, postprocess = receive()
        else:
            assert recv_future is not None
            received, handles, postprocess = recv_future.result(timeout=30)
            recv_future = None
        assert received is not None
        for handle in handles:
            handle.wait()
        for fn in postprocess:
            fn()
        expected = torch.arange(
            rows * 128, dtype=torch.float32, device=device
        ).reshape(shape)
        torch.testing.assert_close(
            received["hidden_states"], expected + step * 1000
        )
        torch.testing.assert_close(
            received["residual"], expected + step * 1000 + 1
        )

# Mirror production: PP1 has already posted step 101, and PP0 releases it with
# metadata rather than leaving a background Gloo receive blocked at teardown.
if pp_group.is_last_rank:
    assert executor is not None
    entered = Event()
    recv_future = executor.submit(receive, entered)
    assert entered.wait(timeout=10)
dist.barrier()
if pp_group.is_first_rank:
    stop_handle = pp_group.isend_object(
        [(STOP_KEY, True), (POLICY_KEY, {})],
        dst=1,
    )
    stop_handle.wait()
else:
    assert recv_future is not None
    received, handles, postprocess = recv_future.result(timeout=30)
    assert received is not None and received.pop(STOP_KEY) is True
    assert not handles
    assert not postprocess
    executor.shutdown(wait=True)

for handles in send_works:
    for handle in handles:
        handle.wait()
pp_group.drain_pending_isends()
dist.barrier()
if rank == 0:
    print(
        "PP_PREPOST_ACTIVATION_TEST_OK steps=100 tp=8 pp=2 "
        "shape_churn=7 policies=sender fifo=ok prepost_before_delay=ok"
    )
dist.destroy_process_group()
