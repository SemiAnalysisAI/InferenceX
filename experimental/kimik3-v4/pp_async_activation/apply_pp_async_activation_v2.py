#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import py_compile
import subprocess


SITE = Path("/usr/local/lib/python3.12/dist-packages")
if configured_root := os.environ.get("K3_PP_PATCH_ROOT"):
    PATCH_ROOT = Path(configured_root)
else:
    PATCH_ROOT = next(
        root
        for root in (Path("/ppasync/upstream"), Path("/pppatch/upstream"))
        if root.is_dir()
    )
MARKER = "K3 dynamic TP8 PP activation ring"


def apply_patch(name: str) -> None:
    patch = PATCH_ROOT / name
    check = subprocess.run(
        ["git", "apply", "--include=vllm/**", "--check", str(patch)],
        cwd=SITE,
        capture_output=True,
        text=True,
    )
    if check.returncode == 0:
        subprocess.run(
            ["git", "apply", "--include=vllm/**", str(patch)],
            cwd=SITE,
            check=True,
        )
        print(f"applied {name}")
        return
    reverse = subprocess.run(
        [
            "git",
            "apply",
            "--include=vllm/**",
            "--reverse",
            "--check",
            str(patch),
        ],
        cwd=SITE,
        capture_output=True,
        text=True,
    )
    if reverse.returncode != 0:
        raise RuntimeError(f"cannot apply {name}: {check.stderr}\n{reverse.stderr}")
    print(f"already applied {name}")


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text()
    if new in source:
        return
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one anchor, got {count}: {old[:120]!r}")
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


# Restore the correctness ordering for the generic fallback first.
apply_patch("vllm-54962.patch")

parallel_state = SITE / "vllm/distributed/parallel_state.py"
replace_once(
    parallel_state,
    """    def isend_tensor_dict(
""",
    """    def drain_pending_isends(self) -> None:
        \"\"\"Wait for and release all retained tensor-dict sends.\"\"\"
        pending = getattr(self, "_pending_isends", None)
        if pending is None:
            return
        while pending:
            handles, _ = pending.popleft()
            for handle in handles:
                handle.wait()

    def isend_tensor_dict(
""",
)
replace_in_function(
    parallel_state,
    "isend_tensor_dict",
    """        all_gather_tensors: dict[str, bool] | None = None,
    ) -> list[Handle]:
""",
    """        all_gather_tensors: dict[str, bool] | None = None,
        device_group: ProcessGroup | None = None,
    ) -> list[Handle]:
""",
)
replace_in_function(
    parallel_state,
    "isend_tensor_dict",
    "        group = self.device_group\n",
    "        group = self.device_group if device_group is None else device_group\n",
)
replace_in_function(
    parallel_state,
    "irecv_tensor_dict",
    """        all_gather_tensors: dict[str, bool] | None = None,
    ) -> tuple[
""",
    """        all_gather_tensors: dict[str, bool] | None = None,
        device_group: ProcessGroup | None = None,
    ) -> tuple[
""",
)
replace_in_function(
    parallel_state,
    "irecv_tensor_dict",
    "        group = self.device_group\n",
    "        group = self.device_group if device_group is None else device_group\n",
)

gpu_worker = SITE / "vllm/v1/worker/gpu_worker.py"
replace_once(
    gpu_worker,
    """        # Device handles of the previous step's PP intermediate-tensor send.
        self._pp_send_work: list[Handle] = []
""",
    f"""        # Device handles of the previous step's generic PP send.
        self._pp_send_work: list[Handle] = []
        # {MARKER}: opt-in dynamic-schema sender ring.
        self._pp_async_activation_enabled = (
            os.environ.get("K3_PP_ASYNC_ACTIVATION", "0") == "1"
        )
        self._pp_activation_group = None
        self._pp_activation_stream = None
        self._pp_activation_slots: list[dict[str, torch.Tensor] | None] = [
            None,
            None,
        ]
        self._pp_activation_slot_work: list[list[Handle]] = [[], []]
        self._pp_activation_slot_events: list[torch.cuda.Event] = []
        self._pp_activation_slot_index = 0
""",
)
replace_once(
    gpu_worker,
    """            if self.use_v2_model_runner:
                logger.info_once("Using V2 Model Runner")

            # Set random seed.
""",
    f"""            if self.use_v2_model_runner:
                logger.info_once("Using V2 Model Runner")

            if self._pp_async_activation_enabled:
                if not self.use_v2_model_runner:
                    raise ValueError("K3 async PP activation requires MRV2")
                if self.parallel_config.pipeline_parallel_size != 2:
                    raise ValueError("K3 async PP activation currently requires PP2")
                # {MARKER}: every rank collectively creates the same sibling
                # PP groups; each worker retains the group for its TP lane.
                self._pp_activation_group = (
                    get_pp_group().make_sibling_device_group(
                        group_desc="k3_pp_activation"
                    )
                )
                self._pp_activation_stream = torch.cuda.Stream(device=self.device)
                self._pp_activation_slot_events = [
                    torch.cuda.Event(),
                    torch.cuda.Event(),
                ]
                logger.info_once("Enabled K3 dynamic TP8 PP activation ring")

            # Set random seed.
""",
)

helper = f"""    def _drain_pp_async_activation(self) -> None:
        # {MARKER}
        for handles in self._pp_activation_slot_work:
            for handle in handles:
                handle.wait()
            handles.clear()
        if self._pp_async_activation_enabled:
            get_pp_group().drain_pending_isends()

    def _send_pp_async_activation(
        self,
        tensors: dict[str, torch.Tensor],
        all_gather_tensors: dict[str, bool],
    ) -> None:
        assert self._pp_activation_group is not None
        assert self._pp_activation_stream is not None
        index = self._pp_activation_slot_index
        for handle in self._pp_activation_slot_work[index]:
            handle.wait()
        self._pp_activation_slot_work[index] = []

        slot = self._pp_activation_slots[index]
        schema = tuple(
            (key, tensor.shape, tensor.dtype, tensor.device)
            for key, tensor in sorted(tensors.items())
        )
        if slot is None or tuple(
            (key, tensor.shape, tensor.dtype, tensor.device)
            for key, tensor in sorted(slot.items())
        ) != schema:
            slot = {{key: torch.empty_like(tensor) for key, tensor in tensors.items()}}
            self._pp_activation_slots[index] = slot

        # Copy static CUDA-graph outputs on the main stream before it can run
        # the next forward and overwrite those addresses.
        for key, tensor in tensors.items():
            slot[key].copy_(tensor)
        event = self._pp_activation_slot_events[index]
        event.record(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self._pp_activation_stream):
            self._pp_activation_stream.wait_event(event)
            self._pp_activation_slot_work[index] = (
                get_pp_group().isend_tensor_dict(
                    slot,
                    all_gather_group=get_tp_group(),
                    all_gather_tensors=all_gather_tensors,
                    device_group=self._pp_activation_group,
                )
            )
        self._pp_activation_slot_index = (index + 1) % 2

"""
replace_once(
    gpu_worker,
    """    @torch.inference_mode()
    @with_gpu_sync_check
    def sample_tokens(
""",
    helper
    + """    @torch.inference_mode()
    @with_gpu_sync_check
    def sample_tokens(
""",
)

replace_once(
    gpu_worker,
    """        if forward_pass and not get_pp_group().is_first_rank:
            tensor_dict, comm_handles, comm_postprocess = (
                get_pp_group().irecv_tensor_dict(
                    all_gather_group=get_tp_group(),
                    all_gather_tensors=all_gather_tensors,
                )
            )
""",
    f"""        if forward_pass and not get_pp_group().is_first_rank:
            if self._pp_async_activation_enabled:
                # {MARKER}: post P2P receives on the sibling communication
                # stream, preserving the existing TP shard/all-gather path.
                assert self._pp_activation_group is not None
                assert self._pp_activation_stream is not None
                with torch.cuda.stream(self._pp_activation_stream):
                    tensor_dict, comm_handles, comm_postprocess = (
                        get_pp_group().irecv_tensor_dict(
                            all_gather_group=get_tp_group(),
                            all_gather_tensors=all_gather_tensors,
                            device_group=self._pp_activation_group,
                        )
                    )
            else:
                tensor_dict, comm_handles, comm_postprocess = (
                    get_pp_group().irecv_tensor_dict(
                        all_gather_group=get_tp_group(),
                        all_gather_tensors=all_gather_tensors,
                    )
                )
""",
)
replace_once(
    gpu_worker,
    """        # Non-blocking send of the intermediate tensors. The metadata handle
        # is reaped lazily by the GroupCoordinator; the device handles are
        # waited at the top of the next step.
        handles = get_pp_group().isend_tensor_dict(
            output.tensors,
            all_gather_group=get_tp_group(),
            all_gather_tensors=all_gather_tensors,
        )
        self._pp_send_work = handles[1:]
""",
    f"""        if self._pp_async_activation_enabled:
            # {MARKER}
            self._send_pp_async_activation(output.tensors, all_gather_tensors)
        else:
            # Generic correctness path from #54962.
            handles = get_pp_group().isend_tensor_dict(
                output.tensors,
                all_gather_group=get_tp_group(),
                all_gather_tensors=all_gather_tensors,
            )
            self._pp_send_work = handles[1:]
""",
)
replace_once(
    gpu_worker,
    """    def shutdown(self) -> None:
        gc.unfreeze()
""",
    f"""    def shutdown(self) -> None:
        # {MARKER}
        self._drain_pp_async_activation()
        gc.unfreeze()
""",
)

for relative in (
    "vllm/distributed/parallel_state.py",
    "vllm/v1/worker/gpu_worker.py",
):
    py_compile.compile(str(SITE / relative), doraise=True)
    print(f"py_compile OK: {relative}")
print("K3 dynamic TP8 PP activation ring ready")
