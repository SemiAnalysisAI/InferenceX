#!/usr/bin/env python3
"""Send sampled tokens and DSpark proposals in one PP token packet."""

from pathlib import Path
import py_compile


site = Path("/usr/local/lib/python3.12/dist-packages")
pp_utils = site / "vllm/v1/worker/gpu/pp_utils.py"
model_runner = site / "vllm/v1/worker/gpu/model_runner.py"

pp_source = pp_utils.read_text()
model_source = model_runner.read_text()

pp_replacements = (
    (
        """        self.broadcast_stream = torch.cuda.Stream(device)
        # K3 PP draft collective stream isolation: proposals use a separate
        # stream so sampled-token receives cannot form a dependency cycle.
        self.draft_broadcast_stream = torch.cuda.Stream(device)
""",
        """        self.broadcast_stream = torch.cuda.Stream(device)
""",
    ),
    (
        """        # K3 PP draft collective stream isolation: use a distinct communicator
        # from both sampled tokens and the dynamic activation ring.
        self.draft_broadcast_group = (
            get_pp_group().make_sibling_device_group(
                group_desc="pp_draft_broadcast"
            )
        )

""",
        "",
    ),
    (
        """    def broadcast_drafts(
        self, draft_tokens: torch.Tensor, input_batch: InputBatch
    ) -> None:
        \"\"\"Broadcast draft proposals so non-last ranks can embed real token ids.\"\"\"
        assert self.is_last_rank
        need_sampled_mask = self._consume_draft_transfer_decision(
            input_batch, "broadcast"
        )
        if need_sampled_mask is None:
            return
        with torch.cuda.stream(self.draft_broadcast_stream):
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

    def receive_drafts(self, input_batch: InputBatch) -> None:
        \"\"\"Recv draft proposals onto the same deferred slot as sampled tokens.\"\"\"
        assert not self.is_last_rank
        need_sampled_mask = self._consume_draft_transfer_decision(
            input_batch, "receive"
        )
        if need_sampled_mask is None:
            return
        num_reqs = input_batch.num_reqs
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
        slot = self.queue[-1]
        if slot is not None:
            slot.draft_tokens = draft_tokens
            slot.event = event

""",
        """    def prepare_draft_transfer(self, input_batch: InputBatch) -> None:
        \"\"\"Capture one pre-mutation decision for the combined PP packet.\"\"\"
        assert self.num_speculative_steps > 0
        need_sampled_mask = compute_need_sampled_mask(input_batch)
        self._record_draft_transfer_decision(input_batch, need_sampled_mask)

""",
    ),
    (
        """        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if self.num_speculative_steps > 0:
            self._record_draft_transfer_decision(input_batch, need_sampled_mask)
        if need_sampled_mask is None:
""",
        """        if self.num_speculative_steps > 0:
            need_sampled_mask = self._consume_draft_transfer_decision(
                input_batch, "combined receive"
            )
        else:
            need_sampled_mask = compute_need_sampled_mask(input_batch)
        if need_sampled_mask is None:
""",
    ),
    (
        """        num_reqs = input_batch.num_reqs
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            sampled_tokens = torch.empty(
                num_reqs, self.max_sample_len, dtype=torch.int64, device=self.device
            )
            combined = torch.empty(2, num_reqs, dtype=torch.int32, device=self.device)
            torch.distributed.broadcast(
                sampled_tokens, src=self.last_rank, group=self.broadcast_group
            )
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            event = self.broadcast_stream.record_event()
            num_sampled, num_rejected = combined.unbind(dim=0)
            # Must record_stream since these were allocated on broadcast stream but
            # later used on the main stream.
            sampled_tokens.record_stream(self.main_stream)
            combined.record_stream(self.main_stream)
        self.queue[-1] = PendingRecv(
            event,
            sampled_tokens,
            num_sampled,
            num_rejected,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            need_sampled_mask,
            gen_at_receive_np,
        )
""",
        """        num_reqs = input_batch.num_reqs
        packet_width = self.max_sample_len + self.num_speculative_steps
        with torch.cuda.stream(self.broadcast_stream):
            token_packet = torch.empty(
                num_reqs,
                packet_width,
                dtype=torch.int64,
                device=self.device,
            )
            combined = torch.empty(2, num_reqs, dtype=torch.int32, device=self.device)
            torch.distributed.broadcast(
                token_packet, src=self.last_rank, group=self.broadcast_group
            )
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            event = self.broadcast_stream.record_event()
            num_sampled, num_rejected = combined.unbind(dim=0)
            token_packet.record_stream(self.main_stream)
            combined.record_stream(self.main_stream)
        sampled_tokens = token_packet[:, : self.max_sample_len]
        draft_tokens = (
            token_packet[:, self.max_sample_len :]
            if self.num_speculative_steps > 0
            else None
        )
        self.queue[-1] = PendingRecv(
            event,
            sampled_tokens,
            num_sampled,
            num_rejected,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            need_sampled_mask,
            gen_at_receive_np,
            draft_tokens=draft_tokens,
        )
""",
    ),
    (
        """    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        assert self.is_last_rank
        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if self.num_speculative_steps > 0:
            self._record_draft_transfer_decision(input_batch, need_sampled_mask)
        if need_sampled_mask is None:
""",
        """    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
        draft_tokens: torch.Tensor | None = None,
    ) -> None:
        assert self.is_last_rank
        if self.num_speculative_steps > 0:
            need_sampled_mask = self._consume_draft_transfer_decision(
                input_batch, "combined broadcast"
            )
            assert draft_tokens is not None
        else:
            need_sampled_mask = compute_need_sampled_mask(input_batch)
            assert draft_tokens is None
        if need_sampled_mask is None:
""",
    ),
    (
        """                padded[:, :width] = send_tokens
                send_tokens = padded
            torch.distributed.broadcast(
                send_tokens.contiguous(),
                src=self.last_rank,
                group=self.broadcast_group,
            )
""",
        """                padded[:, :width] = send_tokens
                send_tokens = padded
            if self.num_speculative_steps > 0:
                assert draft_tokens is not None
                assert draft_tokens.shape == (
                    send_tokens.shape[0],
                    self.num_speculative_steps,
                )
                send_tokens = torch.cat((send_tokens, draft_tokens), dim=1)
            torch.distributed.broadcast(
                send_tokens.contiguous(),
                src=self.last_rank,
                group=self.broadcast_group,
            )
""",
    ),
    (
        """            for tensor in (sampled_token_ids, num_sampled, num_rejected):
                tensor.record_stream(self.broadcast_stream)
""",
        """            for tensor in (sampled_token_ids, num_sampled, num_rejected):
                tensor.record_stream(self.broadcast_stream)
            if draft_tokens is not None:
                draft_tokens.record_stream(self.broadcast_stream)
            if os.environ.get("K3_PP_DRAFT_TRANSFER_LOG", "0") == "1":
                logger.info(
                    "K3 PP combined sample+draft packet posted rank=last "
                    "num_reqs=%d width=%d",
                    send_tokens.shape[0],
                    send_tokens.shape[1],
                )
""",
    ),
)

for old, new in pp_replacements:
    if old not in pp_source:
        raise RuntimeError(f"combined PP packet anchor missing: {old[:100]!r}")
    pp_source = pp_source.replace(old, new, 1)

model_replacements = (
    (
        """        routed_experts = self.execute_model_state.routed_experts
        self.execute_model_state = None

        if not self.is_last_pp_rank:
""",
        """        routed_experts = self.execute_model_state.routed_experts
        self.execute_model_state = None

        if self.pp_handler is not None and self.num_speculative_steps > 0:
            # Capture before postprocess_sampled mutates the last-stage state.
            self.pp_handler.prepare_draft_transfer(input_batch)

        if not self.is_last_pp_rank:
""",
    ),
    (
        """            all_decode_next = self.pp_handler.receive(input_batch)
            # Pair the last rank's post-propose draft send.
            if self.num_speculative_steps > 0:
                self.pp_handler.receive_drafts(input_batch)
            # Optimistically update num_computed_tokens for entire batch here.
""",
        """            all_decode_next = self.pp_handler.receive(input_batch)
            # Optimistically update num_computed_tokens for entire batch here.
""",
    ),
    (
        """        if self.pp_handler is not None:
            # Broadcast to non-last PP ranks (handles spec decode multi-token).
            self.pp_handler.broadcast(
                sampler_output.sampled_token_ids,
                num_sampled,
                num_rejected,
                input_batch,
            )
""",
        """        if self.pp_handler is not None and self.num_speculative_steps == 0:
            self.pp_handler.broadcast(
                sampler_output.sampled_token_ids,
                num_sampled,
                num_rejected,
                input_batch,
            )
""",
    ),
    (
        """            # The other PP ranks have no drafter, so hand them the proposals
            # they must feed the target on the next step.
            if self.pp_handler is not None:
                self.pp_handler.broadcast_drafts(
                    self.req_states.draft_tokens[input_batch.idx_mapping],
                    input_batch,
                )
""",
        """            # Send sampled outputs and proposals in one ordered PP packet.
            if self.pp_handler is not None:
                self.pp_handler.broadcast(
                    sampler_output.sampled_token_ids,
                    num_sampled,
                    num_rejected,
                    input_batch,
                    draft_tokens=self.req_states.draft_tokens[
                        input_batch.idx_mapping
                    ],
                )
""",
    ),
)

for old, new in model_replacements:
    if old not in model_source:
        raise RuntimeError(f"combined model-runner anchor missing: {old[:100]!r}")
    model_source = model_source.replace(old, new, 1)

pp_utils.write_text(pp_source)
model_runner.write_text(model_source)
for path in (pp_utils, model_runner):
    py_compile.compile(str(path), doraise=True)
print("K3 PP sampled tokens and draft proposals share one token packet")
