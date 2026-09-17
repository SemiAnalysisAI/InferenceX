#!/usr/bin/env python3
"""Isolate PP draft transfer from sampled-token and activation streams."""

from pathlib import Path
import py_compile


path = Path(
    "/usr/local/lib/python3.12/dist-packages/"
    "vllm/v1/worker/gpu/pp_utils.py"
)
source = path.read_text()

replacements = (
    (
        """        self.main_stream = torch.cuda.current_stream(device)
        self.broadcast_stream = torch.cuda.Stream(device)

        # On non-last ranks, a FIFO with one entry per in-flight step: the entry
""",
        """        self.main_stream = torch.cuda.current_stream(device)
        self.broadcast_stream = torch.cuda.Stream(device)
        # K3 PP draft collective stream isolation: proposals use a separate
        # stream so sampled-token receives cannot form a dependency cycle.
        self.draft_broadcast_stream = torch.cuda.Stream(device)

        # On non-last ranks, a FIFO with one entry per in-flight step: the entry
""",
    ),
    (
        """        self.broadcast_group = get_pp_group().make_sibling_device_group(
            group_desc="pp_broadcast"
        )

        # K3 PP draft transfer uses the exact sampled-token decision captured
""",
        """        self.broadcast_group = get_pp_group().make_sibling_device_group(
            group_desc="pp_broadcast"
        )
        # K3 PP draft collective stream isolation: use a distinct communicator
        # from both sampled tokens and the dynamic activation ring.
        self.draft_broadcast_group = (
            get_pp_group().make_sibling_device_group(
                group_desc="pp_draft_broadcast"
            )
        )

        # K3 PP draft transfer uses the exact sampled-token decision captured
""",
    ),
    (
        """        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            send = draft_tokens.contiguous()
            torch.distributed.broadcast(
                send, src=self.last_rank, group=self.broadcast_group
            )
            send.record_stream(self.broadcast_stream)
""",
        """        with torch.cuda.stream(self.draft_broadcast_stream):
            # Sender proposals are produced on main_stream.
            self.draft_broadcast_stream.wait_stream(self.main_stream)
            send = draft_tokens.contiguous()
            torch.distributed.broadcast(
                send, src=self.last_rank, group=self.draft_broadcast_group
            )
            send.record_stream(self.draft_broadcast_stream)
            if os.environ.get("K3_PP_DRAFT_TRANSFER_LOG", "0") == "1":
                logger.info(
                    "K3 PP draft collective posted rank=last "
                    "num_reqs=%d width=%d",
                    send.shape[0],
                    send.shape[1],
                )
""",
    ),
    (
        """        num_reqs = input_batch.num_reqs
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            draft_tokens = torch.empty(
                num_reqs,
                self.num_speculative_steps,
                dtype=torch.int64,
                device=self.device,
            )
            torch.distributed.broadcast(
                draft_tokens, src=self.last_rank, group=self.broadcast_group
            )
            # Replace the slot event so one wait covers sampled + draft recv.
            event = self.broadcast_stream.record_event()
            draft_tokens.record_stream(self.main_stream)
""",
        """        num_reqs = input_batch.num_reqs
        with torch.cuda.stream(self.draft_broadcast_stream):
            # Receiver data has no dependency on main_stream. Waiting on main
            # here can create main -> sampled stream -> main cycles at phase
            # handoff, so post the receive immediately on its own stream.
            draft_tokens = torch.empty(
                num_reqs,
                self.num_speculative_steps,
                dtype=torch.int64,
                device=self.device,
            )
            torch.distributed.broadcast(
                draft_tokens,
                src=self.last_rank,
                group=self.draft_broadcast_group,
            )
            event = self.draft_broadcast_stream.record_event()
            draft_tokens.record_stream(self.main_stream)
            if os.environ.get("K3_PP_DRAFT_TRANSFER_LOG", "0") == "1":
                logger.info(
                    "K3 PP draft collective posted rank=first "
                    "num_reqs=%d width=%d",
                    draft_tokens.shape[0],
                    draft_tokens.shape[1],
                )
""",
    ),
)

for old, new in replacements:
    if old not in source:
        raise RuntimeError(
            f"PP draft collective stream patch anchor missing: {old[:100]!r}"
        )
    source = source.replace(old, new, 1)

path.write_text(source)
py_compile.compile(str(path), doraise=True)
print("K3 PP draft collective uses an isolated communicator and stream")
