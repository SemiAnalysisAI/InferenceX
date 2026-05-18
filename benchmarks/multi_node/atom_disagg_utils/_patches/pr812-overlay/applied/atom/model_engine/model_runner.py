# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import math
import os
import time
from contextlib import nullcontext
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.profiler as torch_profiler
import tqdm
from aiter import destroy_dist_env, init_dist_env
from aiter.dist.parallel_state import (
    get_dp_group,
    get_pp_group,
    get_tp_group,
    graph_capture,
)
from aiter.dist.utils import get_distributed_init_method
from atom.config import Config, set_current_atom_config
from atom.model_engine.scheduler import ScheduledBatch, ScheduledBatchOutput
from atom.model_engine.sequence import Sequence, SequenceStatus, SequenceType
from atom.model_loader.loader import load_model
from atom.model_ops.rejection_sampler import RejectionSampler
from atom.model_ops.sampler import SAMPLER_EPS, Sampler
from atom.spec_decode.eagle import EagleProposer
from atom.utils import (
    CpuGpuBuffer,
    envs,
    get_hf_text_config,
    init_exit_handler,
    resolve_obj_by_qualname,
)
from atom.kv_transfer.disaggregation import KVConnectorOutput
from atom.utils.forward_context import get_kvconnector
from atom.utils.tbo import UBatchWrapper, maybe_create_ubatch_slices
from atom.utils.forward_context import (
    Context,
    DPMetadata,
    get_forward_context,
    reset_forward_context,
    set_forward_context,
    set_kv_cache_data,
)
from atom.utils.selector import get_attn_backend
from torch.profiler import record_function

logger = logging.getLogger("atom")

support_model_arch_dict = {
    "Qwen3ForCausalLM": "atom.models.qwen3.Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM": "atom.models.qwen3_moe.Qwen3MoeForCausalLM",
    "LlamaForCausalLM": "atom.models.llama.LlamaForCausalLM",
    "MixtralForCausalLM": "atom.models.mixtral.MixtralForCausalLM",
    "DeepseekV3ForCausalLM": "atom.models.deepseek_v2.DeepseekV2ForCausalLM",
    "DeepseekV32ForCausalLM": "atom.models.deepseek_v2.DeepseekV2ForCausalLM",
    "DeepseekV4ForCausalLM": "atom.models.deepseek_v4.DeepseekV4ForCausalLM",
    "GptOssForCausalLM": "atom.models.gpt_oss.GptOssForCausalLM",
    "GlmMoeDsaForCausalLM": "atom.models.deepseek_v2.GlmMoeDsaForCausalLM",
    "Glm4MoeForCausalLM": "atom.models.glm4_moe.Glm4MoeForCausalLM",
    "Qwen3NextForCausalLM": "atom.models.qwen3_next.Qwen3NextForCausalLM",
    "Qwen3_5ForConditionalGeneration": "atom.models.qwen3_5.Qwen3_5ForConditionalGenerationTextOnly",
    "Qwen3_5MoeForConditionalGeneration": "atom.models.qwen3_5.Qwen3_5MoeForConditionalGenerationTextOnly",
    "KimiK25ForConditionalGeneration": "atom.models.kimi_k25.KimiK25ForCausalLM",
    "MiniMaxM2ForCausalLM": "atom.models.minimax_m2.MiniMaxM2ForCausalLM",
    "MiMoV2FlashForCausalLM": "atom.models.mimo_v2_flash.MiMoV2FlashForCausalLM",
}
# seed = 34567
# np.random.seed(seed)
# torch.cuda.manual_seed_all(seed)


class tokenIDProcessor:

    def __init__(
        self,
        runner: "ModelRunner",
        max_num_batched_tokens: int,
        use_spec: bool = False,
        num_spec_tokens: int = 0,
    ):
        """Asynchronously copy the sampled_token_ids tensor to the host."""
        # Deferred output is disabled when running in P/D disaggregation mode
        # (kv_transfer_config is set), enabled otherwise.
        # TODO: In P/D disaggregation mode, if have issue, we can disable it
        self.is_deferred_out = True

        self.runner = runner
        device = runner.device
        self.input_ids = CpuGpuBuffer(
            max_num_batched_tokens + 1, dtype=torch.int32, device=device
        )
        self.input_ids_loc = CpuGpuBuffer(
            max_num_batched_tokens, dtype=torch.int64, device=device
        )
        self.use_spec = use_spec
        self.num_spec_tokens = num_spec_tokens

        self.async_copy_stream = torch.cuda.Stream(runner.device)
        self.default_num_rejected_tokens = torch.zeros(
            max_num_batched_tokens, dtype=torch.int32, device=device
        )
        self.clean()

    def send_to_cpu_async(
        self,
        gpu_tensor: torch.Tensor,
        cpu_tensor_handle,
        data_ready: torch.cuda.Event,
        copy_done: Optional[torch.cuda.Event] = None,
    ):
        copy_done = copy_done or torch.cuda.Event()
        with torch.cuda.stream(self.async_copy_stream):
            data_ready.wait(stream=self.async_copy_stream)
            cpu_tensor = gpu_tensor.to("cpu", non_blocking=True)
            copy_done.record(self.async_copy_stream)
        cpu_tensor_handle.append((cpu_tensor, copy_done))

    def recv_async_output(self, cpu_tensor_handle) -> torch.Tensor:
        if not cpu_tensor_handle:
            return torch.empty(0, dtype=torch.int32, device="cpu")
        cpu_tensor, event = cpu_tensor_handle.pop(0)
        event.synchronize()
        return cpu_tensor

    def send_to_cpu_async_draft(self, gpu_tensor: torch.Tensor):
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self.async_copy_stream):
            self.async_copy_stream.wait_stream(default_stream)
            cpu_tensor = gpu_tensor.to("cpu", non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.async_copy_stream)
        self.draft_token_ids_cpu.append((cpu_tensor, event))

    def recv_async_output_draft(self) -> np.ndarray:
        if not self.draft_token_ids_cpu:
            return np.array([], dtype=np.int32)
        token_ids, event = self.draft_token_ids_cpu.pop(0)
        event.synchronize()
        return token_ids.numpy()

    def send_mtp_status_to_cpu_async(
        self,
        num_rejected: torch.Tensor,
        num_bonus: torch.Tensor,
        data_ready: torch.cuda.Event,
    ):
        # rejected num and bonus num are slightly different info for mtp
        # take mtp=1 for example:
        #   first decode after prefill have 0 rej, 0 bonus
        #   prev acc decode have 0 rej, 1 bonus
        #   prev rej decode have 1 rej, 0 bonus
        # It is clear that only rejected number is not sufficient for all status tracking, bonus number is also needed.
        self.send_to_cpu_async(num_rejected, self.rejected_tokens_cpu, data_ready)
        self.send_to_cpu_async(num_bonus, self.bonus_tokens_cpu, data_ready)

    def recv_mtp_status_async(
        self,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.rejected_tokens_cpu:
            return None, None
        return (
            self.recv_async_output(self.rejected_tokens_cpu).numpy(),
            self.recv_async_output(self.bonus_tokens_cpu).numpy(),
        )

    def clean(self):
        self.token_ids_cpu: list[torch.Tensor] = []

        self.prev_batch: Optional[ScheduledBatch] = None
        self.prev_token_ids: Optional[torch.Tensor] = None

        self.pre_num_decode_token_per_seq = 1
        self.draft_token_ids: Optional[torch.Tensor] = None
        self.draft_token_ids_cpu: list[torch.Tensor] = []
        self.rejected_tokens_cpu: list[torch.Tensor] = []
        self.bonus_tokens_cpu: list[torch.Tensor] = []
        self.mapped_bonus_list: Optional[list[int]] = (
            None  # Mapped to current batch order
        )
        self.num_rejected: Optional[np.ndarray] = None
        self.num_bonus: Optional[np.ndarray] = None

    @staticmethod
    def _batch_process_token_ids(token_ids: list) -> list[tuple[int, ...]]:
        """Batch process token_ids: vectorized -1 truncation using numpy."""
        arr = np.array(token_ids, dtype=np.int64)
        mask = arr == -1
        if not mask.any():
            # No -1 sentinel in any row, convert each row to tuple directly
            return [tuple(row) for row in arr.tolist()]
        # Per-row: find first -1, truncate
        # Use argmax on mask; rows without -1 get 0, disambiguate with ~mask.any(axis=1)
        has_sentinel = mask.any(axis=1)
        first_neg = mask.argmax(axis=1)
        result = []
        rows = arr.tolist()
        for i, row in enumerate(rows):
            if has_sentinel[i]:
                result.append(tuple(row[: first_neg[i]]))
            else:
                result.append(tuple(row))
        return result

    def prepare_sampled_ids(
        self,
        batch: ScheduledBatch,
        sampled_token_ids: torch.Tensor,
        sync_event: torch.cuda.Event,
    ) -> tuple[list[int], list[tuple[int, ...]]]:
        if not self.is_deferred_out:
            token_ids = sampled_token_ids.tolist()
            req_ids = batch.req_ids
            if token_ids and isinstance(token_ids[0], list):
                processed = self._batch_process_token_ids(token_ids)
            else:
                processed = [(tid,) for tid in token_ids]
            return req_ids, processed

        token_ids = self.recv_async_output(self.token_ids_cpu).tolist()
        self.send_to_cpu_async(sampled_token_ids, self.token_ids_cpu, sync_event)
        req_ids_out: list[int] = []
        processed_out: list[tuple[int, ...]] = []
        self.prev_req_ids = None
        if self.prev_batch is not None:
            self.prev_req_ids = self.prev_batch.req_ids
            req_ids_out = self.prev_req_ids
            if token_ids and isinstance(token_ids[0], list):
                processed_out = self._batch_process_token_ids(token_ids)
            else:
                processed_out = [(tid,) for tid in token_ids]

        self.prev_batch = batch
        self.prev_token_ids = sampled_token_ids

        return req_ids_out, processed_out

    def get_token_locations(
        self, batch: ScheduledBatch
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        prev_req_ids = self.prev_batch.req_ids
        cur_req_ids = batch.req_ids
        num_prev = len(prev_req_ids)
        num_cur = len(cur_req_ids)

        prev_id_to_idx = dict(zip(prev_req_ids, range(num_prev)))

        deferred_curr = np.empty(num_cur, dtype=np.intp)
        deferred_prev = np.empty(num_cur, dtype=np.intp)
        new_curr = np.empty(num_cur, dtype=np.intp)
        n_deferred = 0
        n_new = 0

        for cur_idx in range(num_cur):
            prev_idx = prev_id_to_idx.get(cur_req_ids[cur_idx])
            if prev_idx is not None:
                deferred_curr[n_deferred] = cur_idx
                deferred_prev[n_deferred] = prev_idx
                n_deferred += 1
            else:
                new_curr[n_new] = cur_idx
                n_new += 1

        deferred_curr = deferred_curr[:n_deferred]
        deferred_prev = deferred_prev[:n_deferred]
        new_curr = new_curr[:n_new]

        is_all_same = (
            n_new == 0
            and n_deferred == num_prev
            and np.array_equal(deferred_curr, deferred_prev)
        )

        return deferred_curr, deferred_prev, new_curr, is_all_same

    def prepare_input_ids(
        self,
        batch: ScheduledBatch,
    ) -> torch.Tensor:
        """Prepare the input IDs for the current batch.

        Carefully handles the `prev_sampled_token_ids` which can be cached
        from the previous engine iteration, in which case those tokens on the
        GPU need to be copied into the corresponding slots into input_ids.
        """
        scheduled_tokens = batch.scheduled_tokens  # tokens per req
        total_tokens = batch.total_tokens_num
        total_tokens_prefill = batch.total_tokens_num_prefill
        total_tokens_decode = batch.total_tokens_num_decode
        total_reqs_prefill = batch.total_seqs_num_prefill
        """for prefill: all input ids are new"""
        self.input_ids.np[:total_tokens_prefill] = scheduled_tokens[
            :total_tokens_prefill
        ]
        self.input_ids.copy_to_gpu(total_tokens_prefill)

        self.prev_rejected_num, self.prev_bonus_num = self.recv_mtp_status_async()

        # TODO: remove this when we support mixed prefill and decode in one batch
        if total_reqs_prefill > 0:
            return self.input_ids.gpu[:total_tokens_prefill]

        if not self.is_deferred_out:
            token_ids = scheduled_tokens[
                total_tokens_prefill : total_tokens_prefill + total_tokens_decode
            ]
            if self.use_spec:
                token_ids[:, 1:] = batch.scheduled_spec_decode_tokens

            self.input_ids.np[:total_tokens_decode] = token_ids
            return self.input_ids.copy_to_gpu(total_tokens_decode)

        # PD consumer first decode: no prior prefill step initialized
        # prev_batch, so use scheduled_tokens directly for this step.
        if self.prev_batch is None:
            token_ids = scheduled_tokens[
                total_tokens_prefill : total_tokens_prefill + total_tokens_decode
            ]
            self.input_ids.np[:total_tokens_decode] = token_ids
            return self.input_ids.copy_to_gpu(total_tokens_decode)

        """for decode: input ids are from prev_sampled_token_ids"""
        deferred_curr_indices, deferred_prev_indices, new_curr_indices, is_all_same = (
            self.get_token_locations(batch)
        )
        num_deferred_seqs = len(deferred_curr_indices)
        num_new_seqs = len(new_curr_indices)

        # Calculate token counts: in MTP mode, each seq has multiple tokens
        if self.use_spec:
            tokens_per_seq = self.num_spec_tokens + 1
            num_deferred_tokens = num_deferred_seqs * tokens_per_seq
            num_new_tokens = num_new_seqs * tokens_per_seq
        else:
            tokens_per_seq = 1
            num_deferred_tokens = num_deferred_seqs
            num_new_tokens = num_new_seqs

        # Receive and map bonus_list to current batch order
        self.num_rejected = batch.num_rejected
        self.num_bonus = batch.num_bonus
        if num_deferred_seqs > 0 and self.prev_rejected_num is not None:
            # Map: prev_bonus_list[prev_idx] → mapped_bonus_list[curr_idx]
            self.num_rejected[deferred_curr_indices] = self.prev_rejected_num[
                deferred_prev_indices
            ]
            self.num_bonus[deferred_curr_indices] = self.prev_bonus_num[
                deferred_prev_indices
            ]

        if is_all_same:
            # All requests are the same, only deferred tokens
            if self.use_spec:
                # MTP mode: combine prev_token_ids and draft_token_ids
                if (
                    self.draft_token_ids is not None
                    and self.pre_num_decode_token_per_seq > 1
                ):
                    combined = torch.cat(
                        [
                            self.prev_token_ids.unsqueeze(1),  # (num_seqs, 1)
                            self.draft_token_ids,  # (num_seqs, mtp_n_grams-1)
                        ],
                        dim=1,
                    ).reshape(
                        -1
                    )  # (num_deferred_tokens,)
                else:
                    combined = self.prev_token_ids
                self.input_ids.gpu[:num_deferred_tokens] = combined
            else:
                # Non-MTP mode: only prev_token_ids
                self.input_ids.gpu[:num_deferred_tokens] = self.prev_token_ids
        else:
            """
            (1) prev_batch=[301], cur_batch=[0..255, 301] → Layout: [301 prefill | new | deferred]
            (2) prev_batch=[0..255], cur_batch=[0..253, 256, 257] → Layout: [deferred | new 256, 257] when conc > max_num_seq
            """
            is_prev_prefill = self.prev_batch.total_tokens_num_prefill > 0
            new_decode_front = (
                is_prev_prefill
                and np.array_equal(new_curr_indices, np.arange(num_new_seqs))
                and np.array_equal(
                    deferred_curr_indices,
                    np.arange(num_new_seqs, num_new_seqs + num_deferred_seqs),
                )
            )

            gathered_tokens = None
            # old requests (deferred)
            if num_deferred_seqs > 0:
                self.input_ids_loc.np[:num_deferred_seqs] = deferred_prev_indices
                deferred_indices_gpu = self.input_ids_loc.copy_to_gpu(num_deferred_seqs)
                gathered_prev = torch.gather(
                    self.prev_token_ids,
                    0,
                    deferred_indices_gpu,
                )
                if self.use_spec:
                    # MTP mode: combine prev_token_ids and draft_token_ids
                    if (
                        self.draft_token_ids is not None
                        and self.pre_num_decode_token_per_seq > 1
                    ):
                        # draft_token_ids is 2D (num_seqs, mtp_n_grams-1), use direct indexing
                        gathered_draft = self.draft_token_ids[deferred_indices_gpu]
                        gathered_tokens = torch.cat(
                            [
                                gathered_prev.unsqueeze(1),  # (num_deferred_seqs, 1)
                                gathered_draft,  # (num_deferred_seqs, mtp_n_grams-1)
                            ],
                            dim=1,
                        ).reshape(
                            -1
                        )  # (num_deferred_tokens,)
                    else:
                        # normal decode (fallback)
                        gathered_tokens = gathered_prev
                else:
                    # Non-MTP mode: only prev_token_ids
                    gathered_tokens = gathered_prev

            if new_decode_front:
                # Layout: [new | deferred]
                if gathered_tokens is not None:
                    self.input_ids.gpu[
                        num_new_tokens : num_new_tokens + num_deferred_tokens
                    ] = gathered_tokens
                if num_new_tokens > 0:
                    token_ids = scheduled_tokens[
                        total_tokens_prefill : total_tokens_prefill + num_new_tokens
                    ].reshape(num_new_seqs, tokens_per_seq)
                    if self.use_spec:
                        token_ids[:, 1:] = batch.scheduled_spec_decode_tokens[
                            :num_new_seqs
                        ]
                    self.input_ids.np[:num_new_tokens] = token_ids.flatten()
                    self.input_ids.copy_to_gpu(num_new_tokens)
            else:
                # Layout: [deferred | new] - deferred at front, new is from previous finished prefill and waiting for decode
                if num_new_tokens > 0:
                    new_token_ids = scheduled_tokens[new_curr_indices].reshape(
                        num_new_seqs, tokens_per_seq
                    )
                    if self.use_spec:
                        # MTP mode: combine scheduled_tokens and draft_tokens
                        # For new_decode_front=False, use new_curr_indices to get the right sequences
                        draft_tokens = batch.scheduled_spec_decode_tokens[
                            new_curr_indices
                        ]
                        new_token_ids[:, 1:] = draft_tokens
                    self.input_ids.np[:num_new_tokens] = new_token_ids.flatten()
                    self.input_ids.gpu[
                        num_deferred_tokens : num_deferred_tokens + num_new_tokens
                    ].copy_(self.input_ids.cpu[:num_new_tokens], non_blocking=True)
                if gathered_tokens is not None:
                    self.input_ids.gpu[:num_deferred_tokens] = gathered_tokens
        input_ids = self.input_ids.gpu[:total_tokens]
        return input_ids

    def prepare_draft_ids(
        self, batch: ScheduledBatch, draft_token_ids: torch.Tensor
    ) -> np.ndarray:
        if not self.is_deferred_out:
            ret = draft_token_ids.numpy()
        else:
            self.draft_token_ids = draft_token_ids
            self.pre_num_decode_token_per_seq = self.num_spec_tokens + 1
            token_ids = self.recv_async_output_draft()
            self.send_to_cpu_async_draft(draft_token_ids)
            ret = (
                token_ids
                if self.prev_req_ids is not None
                else np.array([], dtype=np.int32)
            )
        return ret


class ModelRunner:

    def __init__(self, rank: int, config: Config):
        self.config = config
        self.mark_trace = getattr(config, "mark_trace", False)
        from atom.utils.graph_marker import set_graph_marker_enabled

        set_graph_marker_enabled(self.mark_trace)
        set_current_atom_config(config)
        hf_config = config.hf_config
        self.block_size = config.kv_cache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.label = f"Model Runner{rank}/{self.world_size}"
        self.hf_text_config = get_hf_text_config(hf_config)
        if self.hf_text_config.model_type in ["llama"] and self.config.torch_dtype in [
            torch.bfloat16,
            torch.float16,
        ]:
            os.environ["AITER_QUICK_REDUCE_QUANTIZATION"] = "INT4"
        self.use_mla = self.is_deepseek_mla()
        self.use_gdn = self.is_qwen_next()
        self.use_v4 = self.is_deepseek_v4()
        self.is_deepseek_v32 = (
            hasattr(hf_config, "index_topk") if self.use_mla else False
        )
        # Calculate local device rank considering both TP and DP
        # When data parallelism is enabled on the same node, different DP ranks
        # need to use different sets of GPUs
        dp_rank_local = config.parallel_config.data_parallel_rank_local
        if dp_rank_local is None:
            dp_rank_local = 0
        local_device_rank = dp_rank_local * config.tensor_parallel_size + rank
        num_gpus = torch.cuda.device_count()
        if local_device_rank >= num_gpus:
            raise ValueError(
                f"Calculated local_device_rank={local_device_rank} exceeds available GPUs ({num_gpus}). "
            )

        device = torch.device(f"cuda:{local_device_rank}")
        logger.info(
            f"ModelRunner rank={rank}, dp_rank_local={dp_rank_local}, local_device_rank={local_device_rank}, device={device}"
        )
        self.device = device

        # Initialize profiler for this rank
        self.profiler = None
        self.profiler_dir = None
        if config.torch_profiler_dir is not None:
            # Create rank-specific profiler directory
            if dp_rank_local > 0 or config.parallel_config.data_parallel_size > 1:
                rank_name = f"dp{dp_rank_local}_tp{rank}"
            else:
                rank_name = f"rank_{rank}"
            self.profiler_dir = os.path.join(config.torch_profiler_dir, rank_name)
            os.makedirs(self.profiler_dir, exist_ok=True)

        self.graph_bs = [0]  # for eager fallback

        torch.cuda.set_device(self.device)
        os.environ["MASTER_ADDR"] = self.config.master_addr
        os.environ["MASTER_PORT"] = str(self.config.port)
        distributed_init_method = get_distributed_init_method(
            config.parallel_config.data_parallel_master_ip,
            config.parallel_config.data_parallel_base_port,
        )
        init_dist_env(
            config.tensor_parallel_size,
            rankID=rank,
            backend="nccl",
            distributed_init_method=distributed_init_method,
            data_parallel_size=config.parallel_config.data_parallel_size,
            data_parallel_rank=config.parallel_config.data_parallel_rank,
        )
        init_exit_handler(self)
        default_dtype = self.config.torch_dtype
        torch.set_default_dtype(default_dtype)
        torch.set_default_device(self.device)
        self.attn_backend = get_attn_backend(
            self.block_size,
            use_mla=self.use_mla,
            use_gdn=self.use_gdn,
            use_v4=self.use_v4,
        )
        use_spec = bool(self.config.speculative_config) and get_pp_group().is_last_rank
        self.num_spec_tokens = (
            self.config.speculative_config.num_speculative_tokens if use_spec else 0
        )
        self.eagle3_mode = (
            self.config.speculative_config is not None
            and self.config.speculative_config.method == "eagle3"
        )

        self.use_aux_hidden_state_outputs = False
        self._aux_hidden_states = None
        self.tokenID_processor = tokenIDProcessor(
            self,
            self.config.max_num_batched_tokens,
            use_spec,
            self.num_spec_tokens,
        )
        self.sampler = Sampler()
        self.arange_np = np.arange(
            max(
                self.config.max_num_seqs + 1,
                self.config.max_model_len,
                self.config.max_num_batched_tokens,
            ),
            dtype=np.int64,
        )

        model_class = resolve_obj_by_qualname(support_model_arch_dict[hf_config.architectures[0]])  # type: ignore
        # The model construction depends on quant_config,
        # so we must complete the remapping for layers before constructing the model.
        config.quant_config.remap_layer_name(
            config.hf_config,
            packed_modules_mapping=getattr(model_class, "packed_modules_mapping", {}),
            quant_exclude_name_mapping=getattr(
                model_class, "quant_exclude_name_mapping", {}
            ),
        )
        self.model = model_class(config)
        fused_shared_expert_load_fn = None
        if hasattr(self.model, "load_fused_expert_weights"):
            fused_shared_expert_load_fn = self.model.load_fused_expert_weights
        torch.set_default_device(None)
        load_model(
            self.model,
            config.model,
            config.hf_config,
            config.load_dummy,
            load_fused_expert_weights_fn=fused_shared_expert_load_fn,
        )
        logger.info(f"Model load done: {config.model}")

        # Optional debug instrumentation; no-op when env vars unset.
        # See atom/utils/debug_helper/.
        from atom.utils.debug_helper import (
            install_block_forward_hooks,
            maybe_dump_weights_and_exit,
        )

        _n_fwd_hooks = install_block_forward_hooks(self.model)
        if _n_fwd_hooks > 0:
            logger.info(f"[ATOM_FWD_DUMP] {_n_fwd_hooks} Block forward hooks installed")
        maybe_dump_weights_and_exit(self.model)

        if self.config.speculative_config and get_pp_group().is_last_rank:
            from atom.utils.backends import set_model_tag

            torch.set_default_device(self.device)
            with set_model_tag("drafter"):
                self.drafter = EagleProposer(self.config, self.device, self)
            self.rejection_sampler = RejectionSampler()
            torch.set_default_device(None)
            logger.info("Loading drafter model...")
            self.drafter.load_model(self.model)

        if self.eagle3_mode and self.config.speculative_config.use_aux_hidden_state:
            aux_ids = self.config.speculative_config.eagle3_aux_layer_ids
            if not aux_ids and hasattr(
                self.model, "get_eagle3_aux_hidden_state_layers"
            ):
                aux_ids = list(self.model.get_eagle3_aux_hidden_state_layers())
            if aux_ids:
                self.model.set_aux_hidden_state_layers(tuple(aux_ids))
                self.use_aux_hidden_state_outputs = True
                logger.info(f"Eagle3 aux hidden state layers: {aux_ids}")

        torch.set_default_device(self.device)
        self.async_execute_stream = torch.cuda.Stream(self.device)
        self.allocate_forward_vars()
        self.attn_metadata_builder = self.attn_backend.get_builder_cls()(
            model_runner=self
        )
        self.physical_block_size = self.attn_metadata_builder.block_size
        # Sanity-check: any builder that allocates a per-request cache must
        # have its model_type listed in `InputOutputProcessor`'s
        # `per_req_cache_model_types` set; otherwise sequences will be
        # constructed with `has_per_req_cache=False`, the BlockManager will
        # never assign them a slot, and the builder will silently read
        # tensor[-1] on first decode. Catch the misconfiguration up front
        # rather than producing wrong outputs at inference time.
        if self.attn_metadata_builder.compute_per_req_cache_bytes() > 0:
            from atom.model_engine.llm_engine import (
                InputOutputProcessor as _IOProc,
            )

            mt = self.config.hf_config.model_type
            known = _IOProc._per_req_cache_model_types()  # noqa: SLF001
            assert mt in known, (
                f"Attention builder {type(self.attn_metadata_builder).__name__} "
                f"reports per_req_cache_bytes>0 but model_type={mt!r} is not in "
                f"InputOutputProcessor.per_req_cache_model_types ({sorted(known)}). "
                "Add it to the set or sequences will not be assigned slots "
                "(silent corruption)."
            )
        if config.enable_tbo:
            dp_gather_scatter = (
                config.enable_dp_attention and not config.enable_expert_parallel
            )
            self.model = UBatchWrapper(
                self.model,
                attn_metadata_builder=self.attn_metadata_builder,
                dp_gather_scatter=dp_gather_scatter,
            )
            logger.info("TBO enabled: model wrapped with UBatchWrapper")
        self.forward_done_event = torch.cuda.Event()
        self.warmup_model()
        logger.info(f"Model warmup done: {config.model}")

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.config.compilation_config.level == 1:
            self.model = torch.compile(self.model, fullgraph=True, backend="eager")
            if hasattr(self, "drafter"):
                self.drafter.model = torch.compile(
                    self.drafter.model, fullgraph=True, backend="eager"
                )

    def is_deepseek_mla(self) -> bool:
        if not hasattr(self.hf_text_config, "model_type"):
            return False
        elif self.hf_text_config.model_type in (
            "deepseek_v2",
            "deepseek_v3",
            "deepseek_v32",
            "deepseek_mtp",
            "glm_moe_dsa",
            "kimi_k2",
        ):
            return self.hf_text_config.kv_lora_rank is not None
        elif self.hf_text_config.model_type == "eagle":
            # if the model is an EAGLE module, check for the
            # underlying architecture
            return (
                self.hf_text_config.model.model_type in ("deepseek_v2", "deepseek_v3")
                and self.hf_text_config.kv_lora_rank is not None
            )
        return False

    def is_qwen_next(self) -> bool:
        if not hasattr(self.hf_text_config, "model_type"):
            return False
        elif self.hf_text_config.model_type in (
            "qwen3_next",
            "qwen3_next_mtp",
            "qwen3_5_text",
            "qwen3_5_moe_text",
        ):
            return True
        return False

    def is_deepseek_v4(self) -> bool:
        if not hasattr(self.hf_text_config, "model_type"):
            return False
        return self.hf_text_config.model_type == "deepseek_v4"

    def is_mimo_v2(self) -> bool:
        if not hasattr(self.hf_text_config, "model_type"):
            return False
        elif self.hf_text_config.model_type in ("mimo_v2_flash"):
            return True
        return False

    def _make_buffer(
        self, *size: Union[int, torch.SymInt], dtype: torch.dtype, numpy: bool = True
    ) -> CpuGpuBuffer:
        # Bfloat16 torch tensors cannot be directly cast to a numpy array, so
        # if a bfloat16 buffer is needed without a corresponding numpy array,
        # don't bother instantiating the numpy array.
        return CpuGpuBuffer(
            *size, dtype=dtype, device=self.device, pin_memory=True, with_numpy=numpy
        )

    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        cumsum_dtype: Optional[np.dtype] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the cumulative sum and batched arange of the given array.
        # E.g., [2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        # Equivalent to but faster than:
        # np.concatenate([np.arange(n) for n in num_tokens])
        """
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = self.arange_np[:total_num_tokens] - cumsums_offsets

        return cu_num_tokens, arange

    def exit(self):
        if not self.still_running:
            return
        self.still_running = False
        # 1. Destroy distributed env (NCCL + CustomAllreduce + process groups)
        #    Must happen while ops module is still alive for CustomAllreduce cleanup.
        destroy_dist_env()
        # 2. Release CUDA graphs
        if not self.enforce_eager:
            self.graphs = self.graph_pool = None  # type: ignore
        if isinstance(self.model, UBatchWrapper):
            self.model.tbo_graphs.clear()
        # 3. Release GPU tensors
        for attr in (
            "kv_cache",
            "kv_scale",
            "index_cache",
            "mamba_k_cache",
            "mamba_v_cache",
        ):
            if hasattr(self, attr):
                delattr(self, attr)
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "drafter"):
            del self.drafter
        torch.cuda.empty_cache()
        return True

    def start_profiler(self, trace_name: Optional[str] = None):
        """
        Start profiling for this rank.

        The ATOM_PROFILER_MORE environment variable controls detailed profiling features:
        - Set to "1" to enable record_shapes, with_stack, and profile_memory.
        - Set to "0" or unset to disable these features (default).
        """
        if self.profiler_dir is not None and self.profiler is None:
            enable_detailed_profiling = envs.ATOM_PROFILER_MORE
            model_name = os.path.basename(self.config.model.rstrip("/"))
            safe_model_name = "".join(
                c if c.isalnum() or c in ("_", "-", ".") else "_" for c in model_name
            )
            worker_name = safe_model_name or "trace"
            if isinstance(trace_name, str) and trace_name:
                worker_name = "".join(
                    c if c.isalnum() or c in ("_", "-", ".") else "_"
                    for c in trace_name
                )
            if worker_name == "capture_graph":
                if safe_model_name:
                    worker_name = f"{worker_name}_{safe_model_name}"
            output_prefix = os.path.join(self.profiler_dir, worker_name)

            def _on_trace_ready(prof):
                import gzip as _gzip

                # Use a short human-readable timestamp in file name.
                ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                ms = int((time.time() % 1) * 1000)
                output_path = f"{output_prefix}_ts_{ts}_{ms:03d}.pt.trace.json.gz"
                tmp_json_path = output_path[:-3]
                try:
                    t0 = time.monotonic()
                    prof.export_chrome_trace(tmp_json_path)
                    # Chunked gzip: read 64 MB at a time to avoid loading
                    # the entire JSON (~30 GB) into memory at once.
                    with (
                        open(tmp_json_path, "rb") as src,
                        _gzip.open(output_path, "wb") as dst,
                    ):
                        while chunk := src.read(64 * 1024 * 1024):
                            dst.write(chunk)
                    os.remove(tmp_json_path)
                    sz = os.path.getsize(output_path)
                    logger.info(
                        "Rank %d: trace exported to %s (%.1f MB, %.1fs)",
                        self.rank,
                        output_path,
                        sz / 1e6,
                        time.monotonic() - t0,
                    )
                except Exception:
                    logger.exception(
                        "Rank %d: failed to export trace to %s",
                        self.rank,
                        output_path,
                    )
                    for p in (tmp_json_path, output_path):
                        if os.path.exists(p):
                            os.remove(p)

            self.profiler = torch_profiler.profile(
                activities=[
                    torch_profiler.ProfilerActivity.CPU,
                    torch_profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=enable_detailed_profiling,
                with_stack=enable_detailed_profiling,
                profile_memory=enable_detailed_profiling,
                on_trace_ready=_on_trace_ready,
            )
            self.profiler.__enter__()
            logger.info(
                "Rank %d: profiler started (detailed=%s, dir=%s)",
                self.rank,
                enable_detailed_profiling,
                self.profiler_dir,
            )
        return True

    def stop_profiler(self):
        """Stop profiling for this rank."""
        if self.profiler is None:
            return True
        t0 = time.monotonic()
        logger.info("Rank %d: stopping profiler...", self.rank)
        try:
            self.profiler.__exit__(None, None, None)
        except Exception:
            logger.exception("Rank %d: profiler stop failed", self.rank)
        finally:
            self.profiler = None
        logger.info(
            "Rank %d: profiler stop completed in %.1fs",
            self.rank,
            time.monotonic() - t0,
        )
        return True

    def debug(self, *args: Any):
        if self.rank == 0:
            logger.info(*args)

    def _run_dummy_drafter(self, hidden_states, draft_bs=None):
        """Run drafter forward for DP synchronization (no real proposal)."""
        if not hasattr(self, "drafter"):
            return
        forward_context = get_forward_context()
        forward_context.context.is_draft = True
        if draft_bs is None:
            draft_bs = forward_context.context.graph_bs
        for i in range(self.drafter.mtp_k):
            hidden_states = self.drafter.model(
                input_ids=torch.zeros(
                    hidden_states.shape[0],
                    dtype=torch.int32,
                    device=self.device,
                ),
                positions=torch.zeros(
                    hidden_states.shape[0],
                    dtype=torch.int64,
                    device=self.device,
                ),
                hidden_states=hidden_states,
            )
            if i == 0:
                hidden_states = hidden_states[:draft_bs]
                # pad_for_all_gather uses graph_bs * 1, consistent with
                # ranks running propose
                forward_context.attn_metadata.max_seqlen_q = 1

    def dummy_execution(self):
        """Execute dummy decode batch for DP synchronization."""
        # num_tokens_original = 1
        mtp_factor = (self.drafter.mtp_k + 1) if hasattr(self, "drafter") else 1
        num_tokens_original = mtp_factor

        seq = Sequence([0] * num_tokens_original, block_size=self.block_size)
        seq.status = SequenceStatus.RUNNING
        seq.type = SequenceType.DECODE
        seq.block_table = [0]

        dummy_batch = ScheduledBatch(
            seqs={seq.id: seq},
            num_scheduled_tokens=np.array([num_tokens_original], dtype=np.int32),
            total_tokens_num=num_tokens_original,
            total_tokens_num_decode=num_tokens_original,
            total_seqs_num=1,
            total_seqs_num_decode=1,
            is_dummy_run=True,
        )

        bs = self.prepare_inputs(dummy_batch)
        self.forward_vars["input_ids"].gpu[:bs].zero_()
        input_ids = self.forward_vars["input_ids"].gpu[:bs]

        logits, hidden_states = self.run_model(input_ids)
        self._run_dummy_drafter(hidden_states)

        reset_forward_context()
        logger.debug(
            f"{self.label}: dummy batch executed with {dummy_batch.total_tokens_num} tokens"
        )
        return True

    def dummy_prefill_execution(self, num_tokens: int, num_reqs: int = 1):
        """Execute dummy prefill batch for DP synchronization."""
        if num_reqs < 1:
            num_reqs = 1
        if num_tokens < num_reqs:
            num_tokens = num_reqs
        # Distribute tokens evenly across requests
        base = num_tokens // num_reqs
        remainder = num_tokens % num_reqs
        tokens_per_seq = [base + (1 if i < remainder else 0) for i in range(num_reqs)]

        seqs = {}
        for t in tokens_per_seq:
            seq = Sequence([0] * t, block_size=self.block_size)
            seqs[seq.id] = seq

        dummy_batch = ScheduledBatch(
            seqs=seqs,
            num_scheduled_tokens=np.array(tokens_per_seq, dtype=np.int32),
            total_tokens_num=num_tokens,
            total_tokens_num_prefill=num_tokens,
            total_seqs_num=num_reqs,
            total_seqs_num_prefill=num_reqs,
            is_dummy_run=True,
        )

        bs = self.prepare_inputs(dummy_batch)
        self.forward_vars["input_ids"].gpu[:bs].zero_()
        input_ids = self.forward_vars["input_ids"].gpu[:bs]

        with torch.no_grad():
            logits, hidden_states = self.run_model(input_ids)
            self._run_dummy_drafter(hidden_states, draft_bs=1)

        torch.cuda.synchronize()
        reset_forward_context()

        logger.info(
            f"{self.label}: dummy PREFILL batch executed with {num_tokens} tokens, {num_reqs} reqs"
        )
        # TODO: initialize KV connector during warmup
        return True

    def warmup_model(self):
        start_time = time.time()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = (
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )
        dp_size = get_dp_group().world_size
        warmup_max_tokens = max_num_batched_tokens // dp_size

        num_seqs = min(warmup_max_tokens // max_model_len, self.config.max_num_seqs)

        if num_seqs == 0:
            num_seqs = 1
            seq_len = min(warmup_max_tokens, max_model_len)
            if seq_len == 0:
                seq_len = 1
            logger.warning(
                f"{self.label}: DP size={dp_size} too large, warmup_max_tokens={warmup_max_tokens} < max_model_len={max_model_len}. "
                f"Using {num_seqs} seq with length {seq_len} for warmup."
            )
        else:
            seq_len = max_model_len

        seqs = [
            Sequence([0] * seq_len, block_size=self.block_size) for _ in range(num_seqs)
        ]
        seqs = {seq.id: seq for seq in seqs}

        num_scheduled_tokens = np.array([seq_len] * num_seqs, dtype=np.int32)
        total_tokens_num = int(num_scheduled_tokens.sum())

        dummy_batch = ScheduledBatch(
            seqs=seqs,
            num_scheduled_tokens=num_scheduled_tokens,
            total_tokens_num=total_tokens_num,
            total_tokens_num_prefill=total_tokens_num,
            total_seqs_num=num_seqs,
            total_seqs_num_prefill=num_seqs,
            is_dummy_run=True,
        )
        self.forward(dummy_batch)
        self.tokenID_processor.clean()
        torch.cuda.empty_cache()
        logger.info(
            f"{self.label}: warmup_model {time.time() - start_time:.2f} seconds with {num_seqs} reqs {total_tokens_num} tokens"
        )

    def allocate_forward_vars(self):
        config = self.config
        hidden_size = config.hf_config.hidden_size
        hidden_type = config.torch_dtype
        self.max_bs = self.config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        i64_kwargs = {"dtype": torch.int64, "device": self.device}
        i32_kwargs = {"dtype": torch.int32, "device": self.device}
        f32_kwargs = {"dtype": torch.float, "device": self.device}

        # TODO: remove it in forward_context
        self.forward_vars = {
            "input_ids": self.tokenID_processor.input_ids,
            "positions": CpuGpuBuffer(self.max_num_batched_tokens, **i64_kwargs),
            "temperatures": CpuGpuBuffer(self.max_bs, **f32_kwargs),
            "top_ks": CpuGpuBuffer(self.max_bs, **i32_kwargs),
            "top_ps": CpuGpuBuffer(self.max_bs, **f32_kwargs),
            # Keep enough space for MTP decode (max_q_len > 1).
            "outputs": torch.empty(
                self.max_num_batched_tokens, hidden_size, dtype=hidden_type
            ),
        }
        if hasattr(self, "drafter"):
            self.forward_vars["mtp_k"] = self.drafter.mtp_k
            self.forward_vars["num_accepted_tokens"] = CpuGpuBuffer(
                self.max_bs, **i32_kwargs
            )

    def _get_num_kv_heads(self):
        """Return the per-rank number of KV heads."""
        hf_config = self.config.hf_config
        if hf_config.num_key_value_heads >= self.world_size:
            assert hf_config.num_key_value_heads % self.world_size == 0
            return hf_config.num_key_value_heads // self.world_size
        else:
            assert self.world_size % hf_config.num_key_value_heads == 0
            return 1

    def _get_total_num_layers(self):
        """Return total layer count including draft (MTP) layers.

        Drafts that own an independent KV cache via their own builder
        (e.g. Eagle3 MHA draft on an MLA target) account for their layers
        through that builder, so they are NOT added here. Only MTP-style
        drafts that share the target's KV pool contribute.
        """
        total = self.config.hf_config.num_hidden_layers
        if self.config.speculative_config and hasattr(self, "drafter"):
            if not hasattr(self, "eagle3_draft_builder"):
                draft_hf = self.config.speculative_config.draft_model_hf_config
                total += getattr(draft_hf, "num_nextn_predict_layers", 1)
        return total

    def _compute_block_bytes(self):
        """Per-block bytes for the unified KV pool budget.

        Sum across all attention builders attached to this runner: the
        target builder always, plus an optional `eagle3_draft_builder`
        when a heterogeneous spec-decode draft owns its own KV pool. Each
        builder knows its own tensor layout (MLA 576-dim packed, GDN-hybrid
        full-attn-only, MiMo-V2 per-layer-type, standard MHA split-K/V,
        Eagle3 independent MHA). Per-request cache bytes are accounted
        for separately via `compute_per_req_cache_bytes()`.
        """
        block_bytes = self.attn_metadata_builder.compute_block_bytes()
        if hasattr(self, "eagle3_draft_builder"):
            block_bytes += self.eagle3_draft_builder.compute_block_bytes()
        return block_bytes

    def _estimate_cudagraph_overhead(self):
        """Estimate GPU memory consumed by CUDA graph capture.

        CUDA graphs allocate a shared memory pool for intermediate activations.
        The pool size is roughly the peak activation memory during a single
        forward pass. We estimate this from the gap between warmup peak and
        current (steady-state) allocation.

        Returns 0 when enforce_eager is set (no CUDA graphs).
        """
        if self.config.enforce_eager:
            return 0
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        activation_bytes = max(peak - current, 0)
        # CUDA graph pool overhead is roughly 20% of single-pass activation
        # memory due to pooling across multiple captured batch sizes.
        return int(activation_bytes * 0.2)

    def get_num_blocks(self) -> dict[str, int]:
        torch.set_default_device(self.device)
        config = self.config
        hf_config = config.hf_config
        if not hasattr(hf_config, "head_dim") or hf_config.head_dim is None:
            hf_config.head_dim = hf_config.hidden_size // hf_config.num_attention_heads

        free, total = torch.cuda.mem_get_info()
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        # Peak PyTorch usage (high watermark during warmup) — this is memory
        # consumed by THIS process only (model weights + peak activations).
        peak_torch = max(peak, current)

        # CUDA graph capture overhead estimate
        cudagraph_overhead = self._estimate_cudagraph_overhead()

        # Safety margin (2% of total)
        safety_margin = int(total * 0.02)

        # Budget: this server may use up to gpu_memory_utilization * total.
        # Subtract our own PyTorch usage + CUDA graph estimate + safety.
        # This is independent of other processes on the GPU.
        budget = int(total * config.gpu_memory_utilization)
        available_for_kv = budget - peak_torch - cudagraph_overhead - safety_margin

        # Physical clamp: never exceed what's actually free on the GPU.
        # This prevents OOM when other processes share the GPU.
        available_for_kv = min(available_for_kv, free)

        torch.set_default_device("cpu")

        block_bytes = self._compute_block_bytes()

        # Per-request cache (e.g. GDN recurrent state, future DeepseekV4 ring
        # buffer + compressor state): deduct its tensor memory from the KV
        # pool budget. The actual layout / shape is owned by the attention
        # builder; ModelRunner only does sizing math.
        per_req_cache_bytes = self.attn_metadata_builder.compute_per_req_cache_bytes()
        slots_per_req = self.attn_metadata_builder.slots_per_req()
        max_per_req_cache_slots = (
            config.max_num_seqs * slots_per_req if per_req_cache_bytes > 0 else 0
        )
        per_req_cache_tensor_bytes = max_per_req_cache_slots * per_req_cache_bytes
        available_for_pool = available_for_kv - per_req_cache_tensor_bytes
        if available_for_pool <= 0:
            raise RuntimeError(
                f"Per-request cache tensor "
                f"({per_req_cache_tensor_bytes / (1 << 30):.2f}GB for "
                f"{max_per_req_cache_slots} slots) exceeds available KV budget "
                f"({available_for_kv / (1 << 30):.2f}GB). "
                f"Reduce --max-num-seqs or increase gpu_memory_utilization."
            )
        per_req_cache_equiv_blocks = (
            math.ceil(per_req_cache_bytes / block_bytes)
            if per_req_cache_bytes > 0
            else 0
        )

        # Store for BlockManager and allocate_kv_cache.
        # Note the distinction:
        #   - per_req_cache_equiv_blocks: block-equivalents charged to the
        #     unified pool per request (memory accounting)
        #   - num_per_req_cache_groups: BlockManager free-list size; one
        #     group == one request occupies `slots_per_req` contiguous
        #     tensor slots
        #   - max_per_req_cache_slots (runner-only): TENSOR slot dimension
        #     == groups × slots_per_req (groups != slots in general)
        config.per_req_cache_equiv_blocks = per_req_cache_equiv_blocks
        config.num_per_req_cache_groups = (
            config.max_num_seqs if per_req_cache_bytes > 0 else 0
        )
        self.max_per_req_cache_slots = max_per_req_cache_slots

        num_kvcache_blocks = available_for_pool // block_bytes

        logger.info(
            f"Memory budget: total_gpu={total / (1 << 30):.2f}GB, "
            f"free={free / (1 << 30):.2f}GB, "
            f"utilization={config.gpu_memory_utilization}, "
            f"budget={budget / (1 << 30):.2f}GB, "
            f"peak_torch={peak_torch / (1 << 30):.2f}GB, "
            f"cudagraph_est={cudagraph_overhead / (1 << 30):.2f}GB, "
            f"safety={safety_margin / (1 << 30):.2f}GB, "
            f"available_for_kv={available_for_kv / (1 << 30):.2f}GB, "
            f"block_bytes={block_bytes}, "
            f"num_kvcache_blocks={num_kvcache_blocks}"
        )
        if per_req_cache_bytes > 0:
            logger.info(
                f"Per-req cache pool: bytes_per_slot="
                f"{per_req_cache_bytes / (1 << 20):.2f}MB, "
                f"max_slots={max_per_req_cache_slots}, "
                f"tensor_total={per_req_cache_tensor_bytes / (1 << 30):.2f}GB, "
                f"equiv_blocks_per_req={per_req_cache_equiv_blocks}, "
                f"pool_blocks={num_kvcache_blocks}"
            )

        assert num_kvcache_blocks > 0, (
            f"Not enough memory for KV cache with block size({self.block_size}). "
            f"At least 1 block ({block_bytes / (1 << 20):.2f}MB) is required, "
            f"but available_for_kv={available_for_kv / (1 << 20):.2f}MB "
            f"(budget={budget / (1 << 30):.2f}GB, "
            f"peak_torch={peak_torch / (1 << 30):.2f}GB, "
            f"cudagraph_est={cudagraph_overhead / (1 << 30):.2f}GB, "
            f"safety={safety_margin / (1 << 30):.2f}GB, "
            f"free={free / (1 << 30):.2f}GB)"
        )
        return {
            "num_kvcache_blocks": num_kvcache_blocks,
            "per_req_cache_equiv_blocks": per_req_cache_equiv_blocks,
            "num_per_req_cache_groups": (
                config.max_num_seqs if per_req_cache_bytes > 0 else 0
            ),
        }

    def allocate_kv_cache(self, num_kvcache_blocks):
        pre_alloc = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        config = self.config
        config.num_kvcache_blocks = num_kvcache_blocks
        hf_config = config.hf_config
        self.num_physical_kvcache_blocks = (
            num_kvcache_blocks * self.attn_metadata_builder.block_ratio
        )
        if hf_config.num_key_value_heads >= self.world_size:
            assert hf_config.num_key_value_heads % self.world_size == 0
            num_kv_heads = hf_config.num_key_value_heads // self.world_size
        else:
            assert self.world_size % hf_config.num_key_value_heads == 0
            num_kv_heads = 1
        # Promote to self so attention builders' build_kv_cache_tensor()
        # hooks can access it without re-deriving from hf_config.
        self.num_kv_heads = num_kv_heads
        self.aligned_index_dim = None  # set below for DeepSeek-V3.2

        # Calculate total number of layers (target + draft)
        total_num_layers = hf_config.num_hidden_layers
        num_draft_layers = 0
        if self.config.speculative_config and hasattr(self, "drafter"):
            draft_hf_config = self.config.speculative_config.draft_model_hf_config
            if hasattr(self, "eagle3_draft_builder"):
                # Heterogeneous draft (e.g. Eagle3 MHA on MLA target) owns
                # its own KV pool via its builder; don't add to target's count.
                num_draft_layers = draft_hf_config.num_hidden_layers
                logger.info(
                    f"Allocating KV cache for {hf_config.num_hidden_layers} target layers + "
                    f"{num_draft_layers} Eagle3 draft layers (separate non-MLA cache)"
                )
            else:
                # For MTP, use num_nextn_predict_layers instead of num_hidden_layers
                num_draft_layers = getattr(
                    draft_hf_config, "num_nextn_predict_layers", 1
                )
                total_num_layers += num_draft_layers
                logger.info(
                    f"Allocating KV cache for {hf_config.num_hidden_layers} target layers + "
                    f"{num_draft_layers} draft (MTP) layers = {total_num_layers} total layers"
                )

        # Primary KV cache allocation (model-agnostic, delegated to the
        # attention builder). Each builder owns its tensor layout: MLA →
        # single 576-dim per layer; GDN-hybrid → only num_full_attn rows;
        # MiMo-V2 → defer per-module; standard MHA → split-K/V `[2, L, ...]`.
        # Returned tensors are setattr'd on `self` under their conventional
        # names (kv_cache, kv_scale, index_cache, aligned_index_dim,
        # _kv_layer_cache_store) so binding code and downstream consumers
        # find them where they expect.
        main_kv = self.attn_metadata_builder.allocate_kv_cache_tensors(
            num_kv_heads, num_draft_layers
        )
        for name, value in main_kv.items():
            setattr(self, name, value)

        # Heterogeneous draft (e.g. Eagle3 MHA alongside an MLA target) owns
        # its own KV pool through a sibling builder; same protocol as above,
        # tensors land under namespaced keys (eagle3_kv_cache, eagle3_kv_scale).
        if hasattr(self, "eagle3_draft_builder"):
            draft_kv = self.eagle3_draft_builder.allocate_kv_cache_tensors(
                num_kv_heads, num_draft_layers
            )
            for name, value in draft_kv.items():
                setattr(self, name, value)

        # Per-request cache allocation (model-agnostic, delegated to the
        # attention metadata builder). For GDN this returns
        # `{"mamba_k_cache": ..., "mamba_v_cache": ...}`; for stateless
        # attentions it returns an empty dict (no-op). Tensors are setattr'd
        # on `self` so model layers can access them as `model_runner.<name>`.
        if self.max_per_req_cache_slots > 0:
            per_req_tensors = self.attn_metadata_builder.allocate_per_req_cache(
                self.max_per_req_cache_slots
            )
            for name, tensor in per_req_tensors.items():
                setattr(self, name, tensor)

        # Build KVCacheConfig
        # lirong TODO: This is a simple solution to build KVCacheConfig,
        # models with only one type of attention, but not support multi-type of attention models.
        # We need to support it by kv_cache_group in the future.

        # Prepare list of models to bind KV cache
        models_to_bind = [("target", self.model)]
        if self.config.speculative_config and hasattr(self, "drafter"):
            models_to_bind.append(("draft", self.drafter.model))

        kv_cache_tensors = []
        layer_id = 0
        # Promote to self so the attention builder's build_kv_cache_tensor()
        # can access it without recomputing from drafter state. Heterogeneous
        # drafts (Eagle3) own their own layer space via their builder, so
        # leave mtp_start_layer_idx at hf_config.num_hidden_layers in that mode.
        self.mtp_start_layer_idx = (
            self.drafter.model.model.mtp_start_layer_idx
            if hasattr(self, "drafter") and not hasattr(self, "eagle3_draft_builder")
            else hf_config.num_hidden_layers
        )
        for model_name, model in models_to_bind:
            logger.info(
                f"Binding KV cache for {model_name} model starting at layer_id={layer_id}"
            )

            for module in model.modules():
                # Drafts that own an independent KV pool (Eagle3) bind through
                # their sibling builder first; for unrecognized modules it
                # returns None and we fall through to the target builder.
                if model_name == "draft" and hasattr(self, "eagle3_draft_builder"):
                    kv_cache_tensor = self.eagle3_draft_builder.build_kv_cache_tensor(
                        layer_id, module
                    )
                    if kv_cache_tensor is not None:
                        kv_cache_tensors.append(kv_cache_tensor)
                        layer_id += 1
                        continue

                # Per-attention-type binding is owned by the attention
                # metadata builder; ModelRunner only walks modules and
                # collects the resulting KVCacheTensor entries. The builder
                # returns None for modules it does not recognize (so a
                # sibling module like nn.LayerNorm is silently skipped),
                # and increments through MHA / MLA / GDN / V3.2-indexer
                # internally.
                kv_cache_tensor = self.attn_metadata_builder.build_kv_cache_tensor(
                    layer_id, module
                )
                if kv_cache_tensor is not None:
                    kv_cache_tensors.append(kv_cache_tensor)
                    layer_id += 1

        # Store KVCacheConfig
        kv_cache_data = {
            f"layer_{i}": kv_cache_tensor
            for i, kv_cache_tensor in enumerate(kv_cache_tensors)
        }
        transfer_tensors = self.attn_metadata_builder.get_kv_transfer_tensors()
        set_kv_cache_data(kv_cache_data, config, transfer_tensors)

        # Cross-validate: compare estimated vs actual KV cache allocation.
        # `actual_kv_bytes` includes BOTH the unified pool tensors (counted by
        # `block_bytes × num_blocks`) AND the per-request cache tensors (state
        # buffers + SWA window prefix embedded in unified_kv). The budget
        # math in `get_num_blocks()` reserves both separately, so the cross-
        # check must mirror that — otherwise it spuriously fires for any
        # backend with non-zero `compute_per_req_cache_bytes()` (V4, GDN).
        post_alloc = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        actual_kv_bytes = post_alloc - pre_alloc
        expected_kv_bytes = (
            self._compute_block_bytes() * num_kvcache_blocks
            + self.attn_metadata_builder.compute_per_req_cache_bytes()
            * self.max_per_req_cache_slots
        )
        if expected_kv_bytes > 0:
            diff_pct = abs(actual_kv_bytes - expected_kv_bytes) / expected_kv_bytes
            # 3% threshold: budget formula matches allocation exactly, but the
            # measured `post_alloc - pre_alloc` includes allocator alignment
            # (round to 256 B / 16 MiB segments) and ephemeral init buffers
            # from `_zero_state` / `_neg_inf_state` views, accounting for ~2%
            # noise on multi-GiB pools. Lower thresholds spuriously fire.
            if diff_pct > 0.03:
                logger.warning(
                    f"KV cache allocation mismatch: "
                    f"expected={expected_kv_bytes / (1 << 30):.3f}GB, "
                    f"actual={actual_kv_bytes / (1 << 30):.3f}GB, "
                    f"diff={diff_pct:.1%}"
                )

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return True

    def get_dp_padding(self, num_tokens: int) -> tuple[int, Optional[torch.Tensor]]:
        dp_size = self.config.parallel_config.data_parallel_size
        dp_rank = self.config.parallel_config.data_parallel_rank

        # For DP: Don't pad when setting enforce_eager.
        # This lets us set enforce_eager on the prefiller in a P/D setup and
        # still use CUDA graphs (enabled by this padding) on the decoder.
        #
        # TODO(tms) : There are many cases where padding is enabled for
        # prefills, causing unnecessary and excessive padding of activations.

        if dp_size == 1:
            # Early exit.
            return 0, None
        num_tokens_across_dp = DPMetadata.num_tokens_across_dp(
            num_tokens, dp_size, dp_rank
        )
        max_tokens_across_dp_cpu = torch.max(num_tokens_across_dp).item()

        return max_tokens_across_dp_cpu - num_tokens, num_tokens_across_dp

    def _maybe_create_tbo_slices(
        self,
        batch,
        is_prefill,
        scheduled_bs,
        actual_num_tokens,
        num_scheduled_tokens,
        reqs_across_dp,
    ):
        """Create TBO ubatch slices if conditions are met."""
        if not self.config.enable_tbo:
            return None
        if not is_prefill and not self.config.enable_tbo_decode:
            return None
        if not is_prefill and batch.is_dummy_run:
            return None

        tbo_num_reqs = batch.total_seqs_num_prefill if is_prefill else scheduled_bs
        if reqs_across_dp is not None:
            can_tbo = int(torch.min(reqs_across_dp).item()) >= 2
        else:
            can_tbo = tbo_num_reqs >= 2
        if not can_tbo:
            return None

        ubatch_slices = maybe_create_ubatch_slices(
            num_reqs=tbo_num_reqs,
            num_tokens=actual_num_tokens,
            is_prefill=is_prefill,
            num_scheduled_tokens=num_scheduled_tokens if is_prefill else None,
        )
        if ubatch_slices is not None:
            logger.debug(
                f"[TBO] splitting {'prefill' if is_prefill else 'decode'} batch: "
                f"num_reqs={tbo_num_reqs}, ubatches={len(ubatch_slices)}"
            )
        return ubatch_slices

    def _preprocess(self, batch: ScheduledBatch):
        num_input_tokens = batch.total_tokens_num
        is_prefill = batch.total_tokens_num_prefill > 0

        dp_size = self.config.parallel_config.data_parallel_size
        dp_rank = self.config.parallel_config.data_parallel_rank

        if dp_size <= 1:
            return num_input_tokens, None, None

        reqs_across_dp = None
        if self.config.enable_tbo:
            from atom.utils.tbo.ubatching import sync_dp_for_tbo

            sync_reqs = (
                batch.total_seqs_num_prefill
                if is_prefill
                else batch.total_seqs_num_decode
            )
            num_tokens_across_dp, reqs_across_dp = sync_dp_for_tbo(
                dp_size,
                dp_rank,
                num_input_tokens,
                sync_reqs,
            )
        else:
            _, num_tokens_across_dp = self.get_dp_padding(num_input_tokens)

        num_input_tokens = int(torch.max(num_tokens_across_dp).item())
        return num_input_tokens, num_tokens_across_dp, reqs_across_dp

    def prepare_inputs(self, batch: ScheduledBatch, input_ids: torch.Tensor = None):
        is_prefill = batch.total_tokens_num_prefill > 0
        bs = batch.total_seqs_num
        num_scheduled_tokens = np.asarray(batch.num_scheduled_tokens)
        cu_seqlens_q, arange = self._get_cumsum_and_arange(num_scheduled_tokens)
        num_input_tokens, num_tokens_across_dp, reqs_across_dp = self._preprocess(batch)
        self.forward_vars["cu_seqlens_q"].np[1 : bs + 1] = cu_seqlens_q
        if not is_prefill:
            scheduled_bs = batch.total_seqs_num_decode
            # num_pad, num_tokens_across_dp = self.get_dp_padding(scheduled_bs)
            # padded_scheduled_bs = scheduled_bs + num_pad
            # TODO rename num_input_tokens to actual bs in currrent rank?
            padded_scheduled_bs = num_input_tokens
            # for MTP, we need to divide by (mtp_k + 1) to get the actual batch size
            if hasattr(self, "drafter"):
                mtp_step = self.drafter.mtp_k + 1
                padded_scheduled_bs = (padded_scheduled_bs + mtp_step - 1) // mtp_step
            bs = (
                padded_scheduled_bs
                if self.enforce_eager
                else next(
                    (x for x in self.graph_bs if x >= padded_scheduled_bs),
                    padded_scheduled_bs,
                )
                # Use cudagraph and padding to batch_size, if bs > graph_bs, use eager mode
            )
            assert (
                bs >= padded_scheduled_bs
            ), f"current decode {padded_scheduled_bs=} > max graph_bs{bs}"
            self.forward_vars["cu_seqlens_q"].np[scheduled_bs + 1 : bs + 1] = (
                self.forward_vars["cu_seqlens_q"].np[scheduled_bs]
            )
        attn_metadata, positions = self.attn_metadata_builder.build(batch=batch, bs=bs)
        context_bs = batch.total_seqs_num_prefill if is_prefill else scheduled_bs

        # graph_bs should be batch size (number of sequences), not token count
        graph_bs = num_input_tokens if is_prefill else bs
        context = Context(
            positions=positions,
            is_prefill=is_prefill,
            is_dummy_run=batch.is_dummy_run,
            batch_size=context_bs,
            graph_bs=graph_bs,
        )
        actual_num_tokens = batch.total_tokens_num

        spec_decode_metadata = None
        if not is_prefill and hasattr(self, "drafter") and not batch.is_dummy_run:
            scheduled_bs = batch.total_seqs_num_decode
            spec_decode_metadata = self.drafter.calc_spec_decode_metadata(
                num_scheduled_tokens[:scheduled_bs],
                cu_seqlens_q[:scheduled_bs],
                input_ids,
            )

        ubatch_slices = self._maybe_create_tbo_slices(
            batch,
            is_prefill,
            scheduled_bs if not is_prefill else 0,
            actual_num_tokens,
            num_scheduled_tokens,
            reqs_across_dp,
        )

        set_forward_context(
            attn_metadata=attn_metadata,
            atom_config=self.config,
            context=context,
            num_tokens=actual_num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            spec_decode_metadata=spec_decode_metadata,
            ubatch_slices=ubatch_slices,
        )
        return graph_bs

    def prepare_sample(
        self, batch: ScheduledBatch
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, bool, bool]:
        bs = batch.total_seqs_num

        # Check on CPU whether all requests are greedy (temperature=0)
        all_greedy = (batch.temperatures == 0).all()

        # Check on CPU whether any fan-out sibling needs per-row random noise.
        # Missing attribute (e.g. dummy runs, older callers) -> False.
        needs_independent_noise = bool(
            getattr(batch, "needs_independent_noise", np.zeros(0, dtype=bool)).any()
        )

        temp_buffer = self.forward_vars["temperatures"]
        # Clamp temperatures on CPU to avoid division by zero in sampler
        temp_buffer.np[:bs] = np.maximum(batch.temperatures, SAMPLER_EPS)
        temperatures = temp_buffer.copy_to_gpu(bs)

        # Check on CPU whether filtering is needed to avoid GPU sync in sampler.
        # If no filtering needed, return None to skip GPU copy entirely.
        needs_topk = (batch.top_ks != -1).any()
        needs_topp = (batch.top_ps < 1.0).any()

        if needs_topk:
            top_k_buffer = self.forward_vars["top_ks"]
            top_k_buffer.np[:bs] = batch.top_ks
            # If all values are the same, only copy one element to save bandwidth
            if bs > 1 and (batch.top_ks == batch.top_ks[0]).all():
                top_ks = top_k_buffer.copy_to_gpu(1)
            else:
                top_ks = top_k_buffer.copy_to_gpu(bs)
        else:
            top_ks = None

        if needs_topp:
            top_p_buffer = self.forward_vars["top_ps"]
            top_p_buffer.np[:bs] = batch.top_ps
            # If all values are the same, only copy one element to save bandwidth
            if bs > 1 and (batch.top_ps == batch.top_ps[0]).all():
                top_ps = top_p_buffer.copy_to_gpu(1)
            else:
                top_ps = top_p_buffer.copy_to_gpu(bs)
        else:
            top_ps = None

        return temperatures, top_ks, top_ps, all_greedy, needs_independent_noise

    def prepare_model(self, batch: ScheduledBatch):
        total_tokens_num = batch.total_tokens_num
        assert total_tokens_num > 0

        temperatures, top_ks, top_ps, all_greedy, needs_independent_noise = (
            self.prepare_sample(batch)
        )
        input_ids = self.tokenID_processor.prepare_input_ids(batch)
        self.prepare_inputs(batch, input_ids)
        return (
            input_ids,
            temperatures,
            top_ks,
            top_ps,
            all_greedy,
            needs_independent_noise,
        )

    def run_model(
        self,
        input_ids: torch.Tensor,
        batch: Optional[ScheduledBatch] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        forward_context = get_forward_context()
        context = forward_context.context
        bs = context.batch_size
        is_prefill = context.is_prefill
        positions = context.positions

        if is_prefill or self.enforce_eager or bs > self.graph_bs[-1]:
            # prefill[bs=1 tok=115 ctx=115]
            label = f"prefill[bs={bs}"
            if batch is not None:
                ctx = batch.context_lens
                if len(ctx) == 1:
                    ctx_str = str(ctx[0])
                elif len(ctx) <= 5:
                    ctx_str = str(ctx.tolist())
                else:
                    ctx_str = f"{ctx[:3].tolist()}...+{len(ctx)-3}"
                label += f" tok={batch.total_tokens_num} ctx={ctx_str}"
            label += "]"
            with record_function(label):
                model_output = self.model(input_ids, positions)
                if self.use_aux_hidden_state_outputs:
                    hidden_states, self._aux_hidden_states = model_output
                else:
                    hidden_states = model_output
                    self._aux_hidden_states = None
                logits = self.model.compute_logits(hidden_states)
        else:
            # decode[bs=128 tok=128 d=128]  or  decode[bs=128 tok=128 p=2 d=126 spec=3]
            label = f"decode[bs={bs}"
            if batch is not None:
                label += f" tok={batch.total_tokens_num}"
                if batch.total_seqs_num_prefill > 0:
                    label += f" p={batch.total_seqs_num_prefill}"
                label += f" d={batch.total_seqs_num_decode}"
                if batch.num_spec_step > 0:
                    label += f" spec={batch.num_spec_step}"
            label += "]"
            with record_function(label):
                graph_bs = context.graph_bs
                max_q_len = forward_context.attn_metadata.max_seqlen_q
                graph_key = (graph_bs, max_q_len)
                self.graphs[graph_key].replay()
                num_tokens = context.batch_size * max_q_len
                hidden_states = self.forward_vars["outputs"][:num_tokens]
                if graph_key in self.graph_aux_hidden:
                    self._aux_hidden_states = [
                        aux[:num_tokens] for aux in self.graph_aux_hidden[graph_key]
                    ]
                else:
                    self._aux_hidden_states = None
                if self.logits_in_graph:
                    logits = self.graph_logits[graph_key][:num_tokens]
                else:
                    logits = self.model.compute_logits(hidden_states)

        return logits, hidden_states

    def postprocess(
        self,
        batch: ScheduledBatch,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor | None,
        top_ps: torch.Tensor | None,
        all_greedy: bool,
        # following for draft
        hidden_states: torch.Tensor,
        needs_independent_noise: bool = False,
    ) -> ScheduledBatchOutput:
        spec_decode_metadata = get_forward_context().spec_decode_metadata
        bs = batch.total_seqs_num
        if spec_decode_metadata is None:
            sampled_tokens = self.sampler(
                logits,
                temperatures,
                top_ks,
                top_ps,
                all_greedy,
                needs_independent_noise=needs_independent_noise,
            )
            num_reject_tokens = self.tokenID_processor.default_num_rejected_tokens[:bs]
            next_token_locs = num_reject_tokens
        else:
            assert logits is not None
            bonus_logits_indices = spec_decode_metadata.bonus_logits_indices
            target_logits_indices = spec_decode_metadata.target_logits_indices

            bonus_logits = torch.index_select(logits, 0, bonus_logits_indices)
            target_logits = torch.index_select(logits, 0, target_logits_indices)
            bonus_token_ids = self.sampler(
                logits=bonus_logits,
                temperatures=temperatures,
                top_ks=top_ks,
                top_ps=top_ps,
                all_greedy=all_greedy,
                needs_independent_noise=needs_independent_noise,
            )
            # Validate shapes match expectations
            if target_logits.shape[0] != len(spec_decode_metadata.draft_token_ids):
                raise ValueError(
                    f"Shape mismatch: target_logits.shape[0]={target_logits.shape[0]} "
                    f"but len(draft_token_ids)={len(spec_decode_metadata.draft_token_ids)}. "
                    f"target_logits_indices shape={spec_decode_metadata.target_logits_indices.shape}, "
                    f"logits.shape[0]={logits.shape[0]}"
                )

            sampled_tokens, num_bonus_tokens = self.rejection_sampler.forward(
                spec_decode_metadata,
                target_logits,
                bonus_token_ids,
            )
            num_reject_tokens = self.drafter.mtp_k - num_bonus_tokens
            next_token_locs = num_bonus_tokens

        if get_tp_group().world_size > 1 and self.tokenID_processor.is_deferred_out:
            sampled_tokens = get_tp_group().broadcast(sampled_tokens, src=0)

        self.forward_done_event.record()
        # Capture before prepare_sampled_ids(), which advances self.prev_batch to current batch.
        prev_batch = self.tokenID_processor.prev_batch
        req_ids_out, token_ids_out = self.tokenID_processor.prepare_sampled_ids(
            batch, sampled_tokens, self.forward_done_event
        )

        draft_token_ids: Optional[np.ndarray] = None
        if self.tokenID_processor.is_deferred_out:
            if hasattr(self, "drafter"):
                prev_rejected_num = self.tokenID_processor.prev_rejected_num
                prev_bonus_num = self.tokenID_processor.prev_bonus_num
                self.tokenID_processor.send_mtp_status_to_cpu_async(
                    num_reject_tokens, next_token_locs, self.forward_done_event
                )  # Async copy to CPU
                next_token_ids = torch.gather(
                    sampled_tokens.view(bs, -1), 1, next_token_locs.view(-1, 1)
                ).view(bs)
                self.tokenID_processor.prev_token_ids = next_token_ids
                # self.debug(f"{sampled_tokens=}")
                # self.debug(f"{next_token_locs=}")
                draft_token_ids = self.propose_draft_token_ids(
                    batch,
                    self.tokenID_processor.input_ids.gpu[
                        1 : batch.total_tokens_num + 1
                    ],
                    hidden_states,
                    next_token_ids,
                    num_reject_tokens,
                )
                # self.debug(f"{num_bonus_tokens=}")

            elif prev_batch is not None:
                prev_rejected_num = np.zeros(prev_batch.total_seqs_num, dtype=np.int32)
                prev_bonus_num = np.zeros(prev_batch.total_seqs_num, dtype=np.int32)
            else:
                # First forward pass: no deferred output yet, req_ids_out is empty
                prev_rejected_num = np.zeros(0, dtype=np.int32)
                prev_bonus_num = np.zeros(0, dtype=np.int32)
        else:
            prev_rejected_num = np.zeros(batch.total_seqs_num, dtype=np.int32)
            prev_bonus_num = np.zeros(batch.total_seqs_num, dtype=np.int32)

        return ScheduledBatchOutput(
            req_ids=req_ids_out,
            token_ids=token_ids_out,
            draft_token_ids=draft_token_ids,
            is_deferred_out=self.tokenID_processor.is_deferred_out,
            num_rejected=prev_rejected_num,
            num_bonus=prev_bonus_num,
        )

    @torch.inference_mode()
    def forward(self, batch: ScheduledBatch) -> ScheduledBatchOutput:
        (
            input_ids,
            temperatures,
            top_ks,
            top_ps,
            all_greedy,
            needs_independent_noise,
        ) = self.prepare_model(batch)
        logits, hidden_states = self.run_model(input_ids, batch)
        fwd_output = self.postprocess(
            batch,
            logits,
            temperatures,
            top_ks,
            top_ps,
            all_greedy,
            hidden_states,
            needs_independent_noise=needs_independent_noise,
        )
        reset_forward_context()
        return fwd_output

    @torch.inference_mode()
    def process_kvconnector_output(self, connector_meta_output):
        """Dispatch KV connector metadata to initiate async KV loading."""
        if connector_meta_output is not None:
            connector = get_kvconnector()
            if connector is not None:
                connector.start_load_kv(connector_meta_output)

    @torch.inference_mode()
    def async_proc_aggregation(self) -> KVConnectorOutput:
        """Collect finished send/recv status from the KV connector."""
        connector = get_kvconnector()
        if connector is None:
            return KVConnectorOutput(finished_sending=[], finished_recving=[])
        done_sending, done_recving = connector.get_finished()

        return KVConnectorOutput(
            finished_sending=done_sending, finished_recving=done_recving
        )

    def propose_draft_token_ids(
        self,
        batch: ScheduledBatch,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        num_reject_tokens: torch.Tensor,
    ):
        forward_context = get_forward_context()

        positions = forward_context.context.positions
        last_token_offset = 1 + num_reject_tokens

        assert isinstance(self.drafter, EagleProposer)

        last_token_indices = self.drafter.prepare_inputs(
            batch.total_seqs_num, last_token_offset
        )

        draft_token = self.drafter.propose(
            target_token_ids=input_ids,
            target_positions=positions,
            target_hidden_states=hidden_states,
            num_reject_tokens=num_reject_tokens,
            next_token_ids=next_token_ids,
            last_token_indices=last_token_indices,
            aux_hidden_states=self._aux_hidden_states,
        )
        return self.tokenID_processor.prepare_draft_ids(batch, draft_token)

    @torch.inference_mode()
    def capture_cudagraph(self):
        start_time = time.time()
        # self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        if self.config.compilation_config.cudagraph_capture_sizes:
            self.graph_bs = self.config.compilation_config.cudagraph_capture_sizes
        else:
            cuda_graph_sizes = self.config.compilation_config.cuda_graph_sizes
            if len(cuda_graph_sizes) == 1:
                self.graph_bs = [1, 2, 4, 8] + [
                    i for i in range(16, cuda_graph_sizes[0] + 1, 16)
                ]
            elif len(cuda_graph_sizes) > 1:
                self.graph_bs = cuda_graph_sizes
        self.graph_bs.sort(reverse=True)

        # Drop any capture size that exceeds max_num_seqs — those graphs would
        # never be replayed since the scheduler can't produce a batch larger
        # than max_num_seqs. Warn so the user notices a misconfig (default
        # cuda_graph_sizes=[512] vs e.g. max_num_seqs=16) without crashing.
        max_bs = self.config.max_num_seqs
        oversized = [s for s in self.graph_bs if s > max_bs]
        if oversized:
            self.graph_bs = [s for s in self.graph_bs if s <= max_bs]
            logger.warning(
                "cudagraph capture sizes %s exceed max_num_seqs=%d; dropping. "
                "Remaining: %s",
                oversized,
                max_bs,
                self.graph_bs,
            )
        assert self.graph_bs, (
            f"no cudagraph capture sizes left after filtering by "
            f"max_num_seqs={max_bs}; pass --cudagraph-capture-sizes or raise "
            f"--max-num-seqs."
        )

        input_ids = self.forward_vars["input_ids"].gpu
        positions = self.forward_vars["positions"].gpu
        outputs = self.forward_vars["outputs"]
        self.forward_vars["kv_indptr"].gpu.zero_()
        if self.is_deepseek_v32 and "sparse_kv_indptr" in self.forward_vars:
            self.forward_vars["sparse_kv_indptr"].gpu.zero_()

        self.graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = dict()
        self.graph_logits: dict[tuple[int, int], torch.Tensor] = dict()
        self.graph_aux_hidden: dict[tuple[int, int], list[torch.Tensor]] = dict()
        self.graph_pool = None
        is_tbo = self.config.enable_tbo and isinstance(self.model, UBatchWrapper)
        # TBO graphs don't capture compute_logits, so disable logits_in_graph.
        self.logits_in_graph = self.world_size == 1 and not is_tbo

        with graph_capture() as gc:
            capture_range = (
                tqdm.tqdm(self.graph_bs) if self.rank == 0 else self.graph_bs
            )
            max_q_len = self.drafter.mtp_k + 1 if hasattr(self, "drafter") else 1
            for bs in capture_range:
                if self.rank == 0:
                    capture_range.set_description(f"Capturing {bs=}, {max_q_len=}")
                graph = torch.cuda.CUDAGraph()

                cu_seqlens_q = np.arange(
                    0, (bs + 1) * max_q_len, max_q_len, dtype=np.int32
                )
                self.forward_vars["cu_seqlens_q"].np[: bs + 1] = cu_seqlens_q
                self.forward_vars["cu_seqlens_q"].copy_to_gpu(bs + 1)

                num_tokens = bs * max_q_len
                # Use a simple, safe position pattern for capture.
                self.forward_vars["positions"].np[:num_tokens] = (
                    np.arange(num_tokens, dtype=np.int64) % max_q_len
                )

                attn_metadata, context = (
                    self.attn_metadata_builder.build_for_cudagraph_capture(bs=bs)
                )
                num_pad, num_tokens_across_dp = self.get_dp_padding(num_tokens)
                num_tokens += num_pad
                # Create ubatch slices for TBO capture (need >= 2 requests)
                ubatch_slices = None
                if is_tbo and self.config.enable_tbo_decode and bs >= 2:
                    ubatch_slices = maybe_create_ubatch_slices(
                        num_reqs=bs,
                        num_tokens=num_tokens,
                    )

                set_forward_context(
                    attn_metadata=attn_metadata,
                    atom_config=self.config,
                    context=context,
                    num_tokens=num_tokens,
                    num_tokens_across_dp=num_tokens_across_dp,
                    ubatch_slices=ubatch_slices,
                )

                # Warmup
                model_output = self.model(
                    input_ids[:num_tokens], positions[:num_tokens]
                )
                if self.use_aux_hidden_state_outputs:
                    outputs[:num_tokens] = model_output[0]
                else:
                    outputs[:num_tokens] = model_output
                if self.logits_in_graph:
                    self.model.compute_logits(outputs[:num_tokens])

                # Capture
                with (
                    record_function(f"capture_graph_bs_{bs}")
                    if self.mark_trace
                    else nullcontext()
                ):
                    if ubatch_slices is not None:
                        # TBO capture: threads + multi-stream captured in graph.
                        graph, graph_output = self.model.capture_tbo_graph(
                            input_ids[:num_tokens],
                            positions[:num_tokens],
                            self.graph_pool,
                            gc.stream,
                            output_buffer=outputs[:num_tokens],
                        )
                        graph_aux = None
                    else:
                        # Standard single-stream capture
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, self.graph_pool, stream=gc.stream):
                            model_output = self.model(
                                input_ids[:num_tokens], positions[:num_tokens]
                            )
                            if self.use_aux_hidden_state_outputs:
                                outputs[:num_tokens] = model_output[0]
                                graph_aux = model_output[1]
                            else:
                                outputs[:num_tokens] = model_output
                                graph_aux = None
                            if self.logits_in_graph:
                                graph_logits = self.model.compute_logits(
                                    outputs[:num_tokens]
                                )
                if self.graph_pool is None:
                    self.graph_pool = graph.pool()
                self.graphs[(bs, max_q_len)] = graph
                if self.logits_in_graph and ubatch_slices is None:
                    self.graph_logits[(bs, max_q_len)] = graph_logits
                if graph_aux is not None:
                    self.graph_aux_hidden[(bs, max_q_len)] = graph_aux
                torch.cuda.synchronize()
        self.graph_bs.sort(reverse=False)

        # Post-init memory validation
        free_after, total_after = torch.cuda.mem_get_info()
        actual_usage = total_after - free_after
        target_usage = int(total_after * self.config.gpu_memory_utilization)
        usage_ratio = actual_usage / total_after
        logger.info(
            f"Post-init memory: "
            f"actual={actual_usage / (1 << 30):.2f}GB ({usage_ratio:.1%}), "
            f"target={target_usage / (1 << 30):.2f}GB "
            f"({self.config.gpu_memory_utilization:.0%})"
        )
        if usage_ratio > self.config.gpu_memory_utilization + 0.02:
            logger.warning(
                f"Actual GPU memory usage ({usage_ratio:.1%}) exceeds target "
                f"({self.config.gpu_memory_utilization:.0%}) by "
                f"{(usage_ratio - self.config.gpu_memory_utilization):.1%}. "
                f"Consider reducing gpu_memory_utilization."
            )

        return time.time() - start_time, self.graph_bs
