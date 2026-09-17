#!/usr/bin/env python3
"""Pair PP draft broadcasts with the pre-mutation sampled-token decision."""

from pathlib import Path
import py_compile


path = Path(
    "/usr/local/lib/python3.12/dist-packages/"
    "vllm/v1/worker/gpu/pp_utils.py"
)
source = path.read_text()

replacements = (
    (
        """from collections import deque
from dataclasses import dataclass

import numpy as np
""",
        """from collections import deque
from dataclasses import dataclass
import os

import numpy as np
""",
    ),
    (
        """from vllm.distributed.parallel_state import get_pp_group
from vllm.platforms import current_platform
""",
        """from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.platforms import current_platform
""",
    ),
    (
        """from vllm.v1.worker.gpu.input_batch import InputBatch


@dataclass
""",
        """from vllm.v1.worker.gpu.input_batch import InputBatch


logger = init_logger(__name__)


@dataclass
""",
    ),
    (
        """        self.broadcast_group = get_pp_group().make_sibling_device_group(
            group_desc="pp_broadcast"
        )

    def on_req_idx_freed(self, req_idx: int) -> None:
""",
        """        self.broadcast_group = get_pp_group().make_sibling_device_group(
            group_desc="pp_broadcast"
        )

        # K3 PP draft transfer uses the exact sampled-token decision captured
        # before postprocess_sampled mutates request state on the last stage.
        self._draft_transfer_seq = 0
        self._draft_transfer_decision: tuple[
            int, int, np.ndarray | None
        ] | None = None

    def on_req_idx_freed(self, req_idx: int) -> None:
""",
    ),
    (
        """    def broadcast_drafts(
        self, draft_tokens: torch.Tensor, input_batch: InputBatch
    ) -> None:
""",
        """    def _record_draft_transfer_decision(
        self,
        input_batch: InputBatch,
        need_sampled_mask: np.ndarray | None,
    ) -> None:
        assert self._draft_transfer_decision is None
        seq = self._draft_transfer_seq
        self._draft_transfer_seq += 1
        self._draft_transfer_decision = (
            seq,
            input_batch.num_reqs,
            need_sampled_mask,
        )
        if os.environ.get("K3_PP_DRAFT_TRANSFER_LOG", "0") == "1":
            logger.info(
                "K3 PP draft decision rank=%s seq=%d num_reqs=%d mask_count=%d",
                "last" if self.is_last_rank else "first",
                seq,
                input_batch.num_reqs,
                0 if need_sampled_mask is None else int(need_sampled_mask.sum()),
            )

    def _consume_draft_transfer_decision(
        self,
        input_batch: InputBatch,
        operation: str,
    ) -> np.ndarray | None:
        decision = self._draft_transfer_decision
        assert decision is not None, "missing paired PP draft-transfer decision"
        seq, num_reqs, need_sampled_mask = decision
        self._draft_transfer_decision = None
        assert num_reqs == input_batch.num_reqs, (
            f"PP draft-transfer seq {seq} changed num_reqs: "
            f"{num_reqs} -> {input_batch.num_reqs}"
        )
        if os.environ.get("K3_PP_DRAFT_TRANSFER_LOG", "0") == "1":
            logger.info(
                "K3 PP draft %s rank=%s seq=%d num_reqs=%d mask_count=%d",
                operation,
                "last" if self.is_last_rank else "first",
                seq,
                num_reqs,
                0 if need_sampled_mask is None else int(need_sampled_mask.sum()),
            )
        return need_sampled_mask

    def broadcast_drafts(
        self, draft_tokens: torch.Tensor, input_batch: InputBatch
    ) -> None:
""",
    ),
    (
        """        assert self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:
            return
        with torch.cuda.stream(self.broadcast_stream):
""",
        """        assert self.is_last_rank
        need_sampled_mask = self._consume_draft_transfer_decision(
            input_batch, "broadcast"
        )
        if need_sampled_mask is None:
            return
        with torch.cuda.stream(self.broadcast_stream):
""",
    ),
    (
        """        assert not self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:
            return
        num_reqs = input_batch.num_reqs
""",
        """        assert not self.is_last_rank
        need_sampled_mask = self._consume_draft_transfer_decision(
            input_batch, "receive"
        )
        if need_sampled_mask is None:
            return
        num_reqs = input_batch.num_reqs
""",
    ),
    (
        """        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if need_sampled_mask is None:
            # Leave this step's reserved slot as None.
""",
        """        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if self.num_speculative_steps > 0:
            self._record_draft_transfer_decision(input_batch, need_sampled_mask)
        if need_sampled_mask is None:
            # Leave this step's reserved slot as None.
""",
    ),
    (
        """        assert self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:
            # No request needs sampled outputs for a subsequent decode step.
            return

        assert sampled_token_ids.dtype == torch.int64
""",
        """        assert self.is_last_rank
        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if self.num_speculative_steps > 0:
            self._record_draft_transfer_decision(input_batch, need_sampled_mask)
        if need_sampled_mask is None:
            # No request needs sampled outputs for a subsequent decode step.
            return

        assert sampled_token_ids.dtype == torch.int64
""",
    ),
)

for old, new in replacements:
    if old not in source:
        raise RuntimeError(f"PP draft decision patch anchor missing: {old[:100]!r}")
    source = source.replace(old, new, 1)

path.write_text(source)
py_compile.compile(str(path), doraise=True)
print("K3 PP draft transfer uses one pre-mutation decision")
