#!/usr/bin/env python3
"""Prepost the next dynamic PP receive while preserving sender metadata."""

from __future__ import annotations

import os
from pathlib import Path
import py_compile
import runpy


SITE = Path("/usr/local/lib/python3.12/dist-packages")
if configured_base := os.environ.get("K3_PP_ASYNC_V2_PATCH"):
    BASE_PATCH = Path(configured_base)
else:
    BASE_PATCH = next(
        path
        for path in (
            Path("/ppasync/apply_pp_async_activation_v2.py"),
            Path("/pppatch/apply_pp_async_activation_v2.py"),
        )
        if path.is_file()
    )
MARKER = "K3 sender-metadata PP receive prepost"
POLICY_KEY = "__k3_pp_sender_all_gather__"
STOP_KEY = "__k3_pp_prepost_stop__"


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text()
    if new in source:
        return
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected one anchor, got {count}: {old[:120]!r}"
        )
    path.write_text(source.replace(old, new, 1))


def replace_in_function(path: Path, function: str, old: str, new: str) -> None:
    source = path.read_text()
    start = source.index(f"    def {function}(")
    next_def = source.find("\n    def ", start + 8)
    end = len(source) if next_def < 0 else next_def
    block = source[start:end]
    if new in block:
        return
    count = block.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}:{function}: expected one anchor, got {count}: {old[:120]!r}"
        )
    block = block.replace(old, new, 1)
    path.write_text(source[:start] + block + source[end:])


if not BASE_PATCH.is_file():
    raise FileNotFoundError(f"missing v2 patch: {BASE_PATCH}")
runpy.run_path(str(BASE_PATCH), run_name="__main__")

parallel_state = SITE / "vllm/distributed/parallel_state.py"
replace_in_function(
    parallel_state,
    "isend_tensor_dict",
    """        device_group: ProcessGroup | None = None,
    ) -> list[Handle]:
""",
    """        device_group: ProcessGroup | None = None,
        send_all_gather_metadata: bool = False,
    ) -> list[Handle]:
""",
)
replace_in_function(
    parallel_state,
    "isend_tensor_dict",
    """        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)

        tensor_keys = [k for k, v in tensor_dict.items() if isinstance(v, torch.Tensor)]
        assert len(tensor_keys) == len(tensor_list)
""",
    f"""        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)

        tensor_keys = [k for k, v in tensor_dict.items() if isinstance(v, torch.Tensor)]
        assert len(tensor_keys) == len(tensor_list)
        if send_all_gather_metadata:
            # {MARKER}: the receiver may run before its next scheduler call,
            # so carry the sender's per-key shard decision with the schema.
            if any(key == "{POLICY_KEY}" for key, _ in metadata_list):
                raise ValueError("reserved PP sender metadata key")
            metadata_list.append(
                (
                    "{POLICY_KEY}",
                    {{
                        key: self._should_use_all_gather(
                            key,
                            tensor.numel(),
                            all_gather_group,
                            all_gather_tensors,
                        )
                        for key, tensor in zip(tensor_keys, tensor_list)
                    }},
                )
            )
""",
)
replace_in_function(
    parallel_state,
    "irecv_tensor_dict",
    """        device_group: ProcessGroup | None = None,
    ) -> tuple[
""",
    """        device_group: ProcessGroup | None = None,
        use_sender_all_gather_metadata: bool = False,
    ) -> tuple[
""",
)
replace_in_function(
    parallel_state,
    "irecv_tensor_dict",
    """        recv_metadata_list = self.recv_object(src=src)
        tensor_dict: dict[str, Any] = {}
""",
    f"""        recv_metadata_list = self.recv_object(src=src)
        if use_sender_all_gather_metadata:
            # {MARKER}: use the policy that shaped the sender's wire payload.
            # This preserves TP shard/all-gather correctness without needing
            # the next SchedulerOutput in the prepost thread.
            policy_entries = [
                value
                for key, value in recv_metadata_list
                if key == "{POLICY_KEY}"
            ]
            if len(policy_entries) != 1:
                raise RuntimeError(
                    "preposted PP receive requires exactly one sender policy"
                )
            all_gather_tensors = policy_entries[0]
            recv_metadata_list = [
                (key, value)
                for key, value in recv_metadata_list
                if key != "{POLICY_KEY}"
            ]
        tensor_dict: dict[str, Any] = {{}}
""",
)

gpu_worker = SITE / "vllm/v1/worker/gpu_worker.py"
replace_once(
    gpu_worker,
    "from collections.abc import Callable\n",
    """from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
""",
)
replace_once(
    gpu_worker,
    """        self._pp_activation_slot_index = 0
""",
    f"""        self._pp_activation_slot_index = 0
        # {MARKER}
        self._pp_prepost_recv_enabled = (
            os.environ.get("K3_PP_PREPOST_RECV", "0") == "1"
        )
        self._pp_prepost_executor: ThreadPoolExecutor | None = None
        self._pp_prepost_future: Future | None = None
""",
)
replace_once(
    gpu_worker,
    """            if self._pp_async_activation_enabled:
                if not self.use_v2_model_runner:
""",
    f"""            if self._pp_prepost_recv_enabled:
                if not self._pp_async_activation_enabled:
                    raise ValueError(
                        "K3_PP_PREPOST_RECV requires K3_PP_ASYNC_ACTIVATION=1"
                    )
                if self.parallel_config.pipeline_parallel_size != 2:
                    raise ValueError("K3 PP receive prepost currently requires PP2")

            if self._pp_async_activation_enabled:
                if not self.use_v2_model_runner:
""",
)
replace_once(
    gpu_worker,
    """                logger.info_once("Enabled K3 dynamic TP8 PP activation ring")

            # Set random seed.
""",
    f"""                logger.info_once("Enabled K3 dynamic TP8 PP activation ring")
                if (
                    self._pp_prepost_recv_enabled
                    and get_pp_group().is_last_rank
                ):
                    self._pp_prepost_executor = ThreadPoolExecutor(
                        max_workers=1,
                        thread_name_prefix="k3-pp-recv",
                    )
                    logger.info_once(
                        "Enabled K3 sender-metadata PP receive prepost"
                    )

            # Set random seed.
""",
)

helper = f"""    def _recv_pp_preposted_activation(
        self,
    ) -> tuple[
        dict[str, torch.Tensor],
        list[Handle],
        list[Callable[[], None]],
    ]:
        # {MARKER}: this runs while PP1 is outside its next execute_model.
        assert self._pp_activation_group is not None
        assert self._pp_activation_stream is not None
        torch.cuda.set_device(self.device)
        with torch.cuda.stream(self._pp_activation_stream):
            tensor_dict, handles, postprocess = (
                get_pp_group().irecv_tensor_dict(
                    all_gather_group=get_tp_group(),
                    device_group=self._pp_activation_group,
                    use_sender_all_gather_metadata=True,
                )
            )
        assert tensor_dict is not None
        return tensor_dict, handles, postprocess

    def _start_pp_preposted_activation(self) -> None:
        if not self._pp_prepost_recv_enabled:
            return
        if self._pp_prepost_future is not None:
            raise RuntimeError("PP receive prepost already pending")
        assert self._pp_prepost_executor is not None
        self._pp_prepost_future = self._pp_prepost_executor.submit(
            self._recv_pp_preposted_activation
        )

    def _take_pp_preposted_activation(
        self,
    ) -> tuple[
        dict[str, torch.Tensor],
        list[Handle],
        list[Callable[[], None]],
    ]:
        # The first forward has no predecessor from which to prepost.
        if self._pp_prepost_future is None:
            return self._recv_pp_preposted_activation()
        future = self._pp_prepost_future
        self._pp_prepost_future = None
        result = future.result()
        if result[0].pop("{STOP_KEY}", False):
            raise RuntimeError("PP receive prepost stopped before execute_model")
        return result

    def _shutdown_pp_preposted_activation(self) -> None:
        if not self._pp_prepost_recv_enabled:
            return
        pp_group = get_pp_group()
        if pp_group.is_first_rank:
            # Release the final PP1 metadata receive. Include a policy entry so
            # the normal metadata parser remains the only receive path.
            handle = pp_group.isend_object(
                [
                    ("{STOP_KEY}", True),
                    ("{POLICY_KEY}", {{}}),
                ],
                dst=1,
            )
            handle.wait()
            return

        # There can be no prepost after a no-forward first step. Start one so
        # the sender's shutdown sentinel always has a matching receive.
        if self._pp_prepost_future is None:
            self._start_pp_preposted_activation()
        while self._pp_prepost_future is not None:
            future = self._pp_prepost_future
            self._pp_prepost_future = None
            tensor_dict, handles, postprocess = future.result()
            for handle in handles:
                handle.wait()
            for fn in postprocess:
                fn()
            if tensor_dict.pop("{STOP_KEY}", False):
                break
            # Drain an unexpected queued activation, then receive the sentinel.
            self._start_pp_preposted_activation()
        assert self._pp_prepost_executor is not None
        self._pp_prepost_executor.shutdown(wait=True)
        self._pp_prepost_executor = None

"""
replace_once(
    gpu_worker,
    """    def _send_pp_async_activation(
""",
    helper + """    def _send_pp_async_activation(
""",
)
replace_in_function(
    gpu_worker,
    "execute_model",
    """                with torch.cuda.stream(self._pp_activation_stream):
                    tensor_dict, comm_handles, comm_postprocess = (
                        get_pp_group().irecv_tensor_dict(
                            all_gather_group=get_tp_group(),
                            all_gather_tensors=all_gather_tensors,
                            device_group=self._pp_activation_group,
                        )
                    )
""",
    f"""                if self._pp_prepost_recv_enabled:
                    # {MARKER}
                    tensor_dict, comm_handles, comm_postprocess = (
                        self._take_pp_preposted_activation()
                    )
                else:
                    with torch.cuda.stream(self._pp_activation_stream):
                        tensor_dict, comm_handles, comm_postprocess = (
                            get_pp_group().irecv_tensor_dict(
                                all_gather_group=get_tp_group(),
                                all_gather_tensors=all_gather_tensors,
                                device_group=self._pp_activation_group,
                            )
                        )
""",
)
replace_in_function(
    gpu_worker,
    "execute_model",
    """                return output

        assert isinstance(output, IntermediateTensors)
""",
    f"""                if (
                    self._pp_prepost_recv_enabled
                    and forward_pass
                    and get_pp_group().is_last_rank
                ):
                    # {MARKER}: arm the next receive before returning PP1.
                    self._start_pp_preposted_activation()
                return output

        assert isinstance(output, IntermediateTensors)
""",
)
replace_in_function(
    gpu_worker,
    "_send_pp_async_activation",
    """                    all_gather_tensors=all_gather_tensors,
                    device_group=self._pp_activation_group,
                )
""",
    """                    all_gather_tensors=all_gather_tensors,
                    device_group=self._pp_activation_group,
                    send_all_gather_metadata=self._pp_prepost_recv_enabled,
                )
""",
)
replace_in_function(
    gpu_worker,
    "shutdown",
    """        # K3 dynamic TP8 PP activation ring
        self._drain_pp_async_activation()
""",
    f"""        # {MARKER}: release/drain the receiver before destroying groups.
        self._shutdown_pp_preposted_activation()
        self._drain_pp_async_activation()
""",
)

for relative in (
    "vllm/distributed/parallel_state.py",
    "vllm/v1/worker/gpu_worker.py",
):
    py_compile.compile(str(SITE / relative), doraise=True)
    print(f"py_compile OK: {relative}")
print("K3 sender-metadata PP receive prepost ready")
