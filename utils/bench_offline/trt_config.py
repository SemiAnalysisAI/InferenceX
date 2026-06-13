"""Pure configuration helpers for the B300 TRT offline benchmark."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Optional


WORLD_SIZE = 8
INPUT_TOKENS = 8192
OUTPUT_TOKENS = 625
MTP_DRAFT_TOKENS = 3
MAX_SEQ_LEN = 9216
SAMPLING_TEMPERATURE = 1.0
SAMPLING_TOP_P = 1.0
SAMPLING_TOP_K = 0
PINNED_TRT_GLOBAL_SEED = 42
TUNING_MEASURED_PASSES = 1
FINAL_MEASURED_PASSES = 1
MIN_WINNER_IMPROVEMENT = 0.03


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    batching_wait_iters: int
    attention_dp_balance: bool = True
    attention_dp_timeout_iters: int | None = 60
    overlap_scheduler: bool = True
    cuda_graph: bool = True
    enable_lm_head_tp_in_adp: bool = False
    mtp_draft_tokens: int = MTP_DRAFT_TOKENS
    kind: str = "scheduler"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.name):
            raise ValueError(f"Invalid candidate name: {self.name!r}")
        if (
            not isinstance(self.batching_wait_iters, int)
            or isinstance(self.batching_wait_iters, bool)
            or self.batching_wait_iters < 0
        ):
            raise ValueError("batching_wait_iters must be non-negative")
        if (
            self.attention_dp_timeout_iters is not None
            and (
                not isinstance(self.attention_dp_timeout_iters, int)
                or isinstance(self.attention_dp_timeout_iters, bool)
                or self.attention_dp_timeout_iters < 0
            )
        ):
            raise ValueError(
                "attention_dp_timeout_iters must be non-negative"
            )
        if (
            not isinstance(self.mtp_draft_tokens, int)
            or isinstance(self.mtp_draft_tokens, bool)
            or not 1 <= self.mtp_draft_tokens <= MTP_DRAFT_TOKENS
        ):
            raise ValueError(
                "mtp_draft_tokens must be between 1 and "
                f"{MTP_DRAFT_TOKENS}"
            )
        for field_name in (
            "attention_dp_balance",
            "overlap_scheduler",
            "cuda_graph",
            "enable_lm_head_tp_in_adp",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be boolean")
        if not isinstance(self.kind, str):
            raise ValueError("kind must be a string")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CandidateConfig":
        return cls(**value)

    def without_cuda_graph(self) -> "CandidateConfig":
        suffix = "" if self.name.endswith("-graph-off") else "-graph-off"
        return replace(self, name=f"{self.name}{suffix}", cuda_graph=False)


def scheduler_candidates() -> list[CandidateConfig]:
    # Start from the existing B300 recipe's wait=30 default.
    return [
        CandidateConfig(name="wait30", batching_wait_iters=30),
        CandidateConfig(name="wait0", batching_wait_iters=0),
        CandidateConfig(name="wait10", batching_wait_iters=10),
        CandidateConfig(name="wait60", batching_wait_iters=60),
    ]


def balance_candidate(base: CandidateConfig) -> CandidateConfig:
    return replace(
        base,
        name=f"{base.name}-balance-off",
        attention_dp_balance=False,
        kind="balance",
    )


def overlap_candidate(base: CandidateConfig) -> CandidateConfig:
    return replace(
        base,
        name=f"{base.name}-overlap-off",
        overlap_scheduler=False,
        kind="overlap",
    )


def max_batch_size(concurrency: int) -> int:
    return max(16, concurrency)


def max_num_tokens(
    concurrency: int,
    mtp_draft_tokens: int = MTP_DRAFT_TOKENS,
) -> int:
    # Mirrors the working B300 DeepSeek-V4 TRT MTP recipe.
    return max(
        INPUT_TOKENS
        + (mtp_draft_tokens + 1) * max_batch_size(concurrency)
        + 256,
        INPUT_TOKENS,
    )


def build_llm_kwargs(
    model_path: str,
    concurrency: int,
    candidate: CandidateConfig,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model_path,
        "backend": "pytorch",
        "trust_remote_code": True,
        # TRT uses TP size to establish the eight-rank world. Attention and
        # expert execution are data/expert parallel below.
        "tensor_parallel_size": WORLD_SIZE,
        "moe_expert_parallel_size": WORLD_SIZE,
        "moe_tensor_parallel_size": 1,
        "enable_attention_dp": True,
        "enable_lm_head_tp_in_adp": candidate.enable_lm_head_tp_in_adp,
        "max_batch_size": max_batch_size(concurrency),
        "max_seq_len": MAX_SEQ_LEN,
        "max_num_tokens": max_num_tokens(
            concurrency,
            candidate.mtp_draft_tokens,
        ),
        "custom_tokenizer": "deepseek_v4",
        "return_perf_metrics": True,
        "print_iter_log": True,
        "stream_interval": 100,
        "num_postprocess_workers": 4,
        "disable_overlap_scheduler": not candidate.overlap_scheduler,
        "attention_dp_config": {
            "batching_wait_iters": candidate.batching_wait_iters,
            "enable_balance": candidate.attention_dp_balance,
        },
        "kv_cache_config": {
            "tokens_per_block": 128,
            "dtype": "fp8",
            "free_gpu_memory_fraction": 0.60,
            "enable_block_reuse": False,
        },
        "moe_config": {
            "backend": "TRTLLM",
            "use_low_precision_moe_combine": True,
        },
        "speculative_config": {
            "decoding_type": "MTP",
            "max_draft_len": candidate.mtp_draft_tokens,
        },
    }
    if candidate.attention_dp_timeout_iters is not None:
        kwargs["attention_dp_config"]["timeout_iters"] = (
            candidate.attention_dp_timeout_iters
        )
    if candidate.cuda_graph:
        kwargs["cuda_graph_config"] = {
            "batch_sizes": [concurrency],
            "max_batch_size": concurrency,
            "enable_padding": True,
        }
    else:
        kwargs["cuda_graph_config"] = None
    return kwargs


def objective(result: dict[str, Any]) -> Optional[float]:
    if result.get("status") != "success":
        return None
    aggregate = result.get("aggregate") or {}
    value = aggregate.get("derived_output_tput_per_gpu")
    if value is None:
        return None
    return float(value)


def choose_winner(
    results: Iterable[dict[str, Any]],
    minimum_improvement: float = MIN_WINNER_IMPROVEMENT,
) -> Optional[dict[str, Any]]:
    """Choose the earliest result unless a later one is at least 3% faster."""
    winner: Optional[dict[str, Any]] = None
    winner_value: Optional[float] = None
    for result in results:
        value = objective(result)
        if value is None:
            continue
        if winner is None:
            winner = result
            winner_value = value
            continue
        assert winner_value is not None
        if value >= winner_value * (1.0 + minimum_improvement):
            winner = result
            winner_value = value
    return winner


def resolved_parallelism(
    llm_args: Any,
    candidate: CandidateConfig | None = None,
) -> dict[str, Any]:
    if candidate is None:
        candidate = CandidateConfig(name="default", batching_wait_iters=30)
    parallel = llm_args.parallel_config
    resolved = {
        "world_size": int(parallel.world_size),
        "tensor_parallel_size": int(parallel.tp_size),
        "moe_expert_parallel_size": int(parallel.moe_ep_size),
        "moe_tensor_parallel_size": int(parallel.moe_tp_size),
        "enable_attention_dp": bool(parallel.enable_attention_dp),
        "enable_lm_head_tp_in_adp": bool(
            parallel.enable_lm_head_tp_in_adp
        ),
        "effective_parallelism": "DEP8",
    }
    expected = {
        "world_size": WORLD_SIZE,
        "tensor_parallel_size": WORLD_SIZE,
        "moe_expert_parallel_size": WORLD_SIZE,
        "moe_tensor_parallel_size": 1,
        "enable_attention_dp": True,
        "enable_lm_head_tp_in_adp": candidate.enable_lm_head_tp_in_adp,
    }
    for key, expected_value in expected.items():
        if resolved[key] != expected_value:
            raise RuntimeError(
                f"Resolved TRT parallelism mismatch for {key}: "
                f"{resolved[key]!r} != {expected_value!r}"
            )

    speculative = llm_args.speculative_config
    draft_len = int(speculative.max_draft_len)
    if draft_len != candidate.mtp_draft_tokens:
        raise RuntimeError(
            f"Resolved MTP draft length {draft_len} != "
            f"{candidate.mtp_draft_tokens}"
        )
    resolved["mtp_max_draft_len"] = draft_len
    return resolved
