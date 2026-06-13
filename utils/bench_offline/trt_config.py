"""Pure configuration helpers for the B300 TRT offline benchmark."""

from __future__ import annotations

import math
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
class ParallelismConfig:
    name: str
    label: str
    world_size: int
    tensor_parallel_size: int
    moe_expert_parallel_size: int
    moe_tensor_parallel_size: int
    enable_attention_dp: bool


PARALLELISM_CONFIGS = {
    "dep8": ParallelismConfig(
        name="dep8",
        label="DEP8",
        world_size=8,
        tensor_parallel_size=8,
        moe_expert_parallel_size=8,
        moe_tensor_parallel_size=1,
        enable_attention_dp=True,
    ),
    "tp4": ParallelismConfig(
        name="tp4",
        label="TP4",
        world_size=4,
        tensor_parallel_size=4,
        moe_expert_parallel_size=1,
        moe_tensor_parallel_size=4,
        enable_attention_dp=False,
    ),
    "dep4": ParallelismConfig(
        name="dep4",
        label="DEP4",
        world_size=4,
        tensor_parallel_size=4,
        moe_expert_parallel_size=4,
        moe_tensor_parallel_size=1,
        enable_attention_dp=True,
    ),
}


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    batching_wait_iters: int
    attention_dp_balance: bool = True
    attention_dp_timeout_iters: int | None = 60
    overlap_scheduler: bool = True
    cuda_graph: bool = True
    enable_lm_head_tp_in_adp: bool = False
    attention_dp_batch_mode: str = "global"
    use_cute_dsl_topk: bool = False
    use_cute_dsl_paged_mqa_logits: bool = False
    enable_heuristic_topk: bool = False
    indexer_k_dtype: str | None = None
    print_iter_log: bool = True
    max_seq_len: int = MAX_SEQ_LEN
    mtp_draft_tokens: int = MTP_DRAFT_TOKENS
    parallelism: str = "dep8"
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
            "use_cute_dsl_topk",
            "use_cute_dsl_paged_mqa_logits",
            "enable_heuristic_topk",
            "print_iter_log",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be boolean")
        if self.indexer_k_dtype not in {None, "fp4", "fp8"}:
            raise ValueError("indexer_k_dtype must be fp4, fp8, or null")
        minimum_seq_len = INPUT_TOKENS + OUTPUT_TOKENS + MTP_DRAFT_TOKENS
        if (
            not isinstance(self.max_seq_len, int)
            or isinstance(self.max_seq_len, bool)
            or self.max_seq_len < minimum_seq_len
        ):
            raise ValueError(
                f"max_seq_len must be an integer >= {minimum_seq_len}"
            )
        if not isinstance(self.kind, str):
            raise ValueError("kind must be a string")
        if (
            not isinstance(self.attention_dp_batch_mode, str)
            or self.attention_dp_batch_mode not in {"global", "local-rank"}
        ):
            raise ValueError(
                "attention_dp_batch_mode must be global or local-rank"
            )
        if (
            not isinstance(self.parallelism, str)
            or self.parallelism not in PARALLELISM_CONFIGS
        ):
            raise ValueError(
                "parallelism must be one of "
                f"{sorted(PARALLELISM_CONFIGS)}"
            )
        if (
            self.enable_lm_head_tp_in_adp
            and not self.parallelism_config.enable_attention_dp
        ):
            raise ValueError(
                "enable_lm_head_tp_in_adp requires attention DP"
            )
        if (
            self.attention_dp_batch_mode == "local-rank"
            and not self.parallelism_config.enable_attention_dp
        ):
            raise ValueError(
                "attention_dp_batch_mode=local-rank requires attention DP"
            )

    @property
    def parallelism_config(self) -> ParallelismConfig:
        return PARALLELISM_CONFIGS[self.parallelism]

    @property
    def active_gpu_count(self) -> int:
        return self.parallelism_config.world_size

    @property
    def effective_parallelism(self) -> str:
        return self.parallelism_config.label

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


def local_attention_dp_batch_size(
    concurrency: int,
    candidate: CandidateConfig,
) -> int:
    return math.ceil(
        concurrency / candidate.parallelism_config.tensor_parallel_size
    )


def max_batch_size(
    concurrency: int,
    candidate: CandidateConfig | None = None,
) -> int:
    if (
        candidate is not None
        and candidate.attention_dp_batch_mode == "local-rank"
    ):
        return local_attention_dp_batch_size(concurrency, candidate)
    return max(16, concurrency)


def cuda_graph_batch_size(
    concurrency: int,
    candidate: CandidateConfig,
) -> int:
    if candidate.attention_dp_batch_mode == "local-rank":
        return local_attention_dp_batch_size(concurrency, candidate)
    return concurrency


def max_num_tokens(
    concurrency: int,
    mtp_draft_tokens: int = MTP_DRAFT_TOKENS,
    candidate: CandidateConfig | None = None,
) -> int:
    # Mirrors the working B300 DeepSeek-V4 TRT MTP recipe.
    return max(
        INPUT_TOKENS
        + (mtp_draft_tokens + 1) * max_batch_size(concurrency, candidate)
        + 256,
        INPUT_TOKENS,
    )


def build_llm_kwargs(
    model_path: str,
    concurrency: int,
    candidate: CandidateConfig,
) -> dict[str, Any]:
    parallelism = candidate.parallelism_config
    graph_batch_size = cuda_graph_batch_size(concurrency, candidate)
    kwargs: dict[str, Any] = {
        "model": model_path,
        "backend": "pytorch",
        "trust_remote_code": True,
        "tensor_parallel_size": parallelism.tensor_parallel_size,
        "moe_expert_parallel_size": (
            parallelism.moe_expert_parallel_size
        ),
        "moe_tensor_parallel_size": parallelism.moe_tensor_parallel_size,
        "enable_attention_dp": parallelism.enable_attention_dp,
        "enable_lm_head_tp_in_adp": (
            candidate.enable_lm_head_tp_in_adp
            if parallelism.enable_attention_dp
            else False
        ),
        "max_batch_size": max_batch_size(concurrency, candidate),
        "max_seq_len": candidate.max_seq_len,
        "max_num_tokens": max_num_tokens(
            concurrency,
            candidate.mtp_draft_tokens,
            candidate,
        ),
        "custom_tokenizer": "deepseek_v4",
        "return_perf_metrics": True,
        "print_iter_log": candidate.print_iter_log,
        "stream_interval": 100,
        "num_postprocess_workers": 4,
        "disable_overlap_scheduler": not candidate.overlap_scheduler,
        "kv_cache_config": {
            "tokens_per_block": 128,
            "dtype": "fp8",
            "free_gpu_memory_fraction": (
                0.60 if parallelism.enable_attention_dp else 0.90
            ),
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
    sparse_attention_config: dict[str, Any] = {
        "algorithm": "deepseek_v4",
    }
    if candidate.use_cute_dsl_topk:
        sparse_attention_config["use_cute_dsl_topk"] = True
    if candidate.use_cute_dsl_paged_mqa_logits:
        sparse_attention_config["use_cute_dsl_paged_mqa_logits"] = True
    if candidate.enable_heuristic_topk:
        sparse_attention_config["enable_heuristic_topk"] = True
    if candidate.indexer_k_dtype is not None:
        sparse_attention_config["indexer_k_dtype"] = (
            candidate.indexer_k_dtype
        )
    if len(sparse_attention_config) > 1:
        kwargs["sparse_attention_config"] = sparse_attention_config
    if parallelism.enable_attention_dp:
        kwargs["attention_dp_config"] = {
            "batching_wait_iters": candidate.batching_wait_iters,
            "enable_balance": candidate.attention_dp_balance,
        }
    if (
        parallelism.enable_attention_dp
        and candidate.attention_dp_timeout_iters is not None
    ):
        kwargs["attention_dp_config"]["timeout_iters"] = (
            candidate.attention_dp_timeout_iters
        )
    if candidate.cuda_graph:
        kwargs["cuda_graph_config"] = {
            "batch_sizes": [graph_batch_size],
            "max_batch_size": graph_batch_size,
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
    concurrency: int | None = None,
) -> dict[str, Any]:
    if candidate is None:
        candidate = CandidateConfig(name="default", batching_wait_iters=30)
    expected_parallelism = candidate.parallelism_config
    parallel = llm_args.parallel_config
    resolved = {
        "world_size": int(parallel.world_size),
        "tensor_parallel_size": int(parallel.tp_size),
        "moe_expert_parallel_size": int(parallel.moe_ep_size),
        "moe_tensor_parallel_size": int(parallel.moe_tp_size),
        "enable_attention_dp": bool(parallel.enable_attention_dp),
        "enable_lm_head_tp_in_adp": bool(
            getattr(parallel, "enable_lm_head_tp_in_adp", False)
        ),
        "effective_parallelism": expected_parallelism.label,
    }
    expected = {
        "world_size": expected_parallelism.world_size,
        "tensor_parallel_size": (
            expected_parallelism.tensor_parallel_size
        ),
        "moe_expert_parallel_size": (
            expected_parallelism.moe_expert_parallel_size
        ),
        "moe_tensor_parallel_size": (
            expected_parallelism.moe_tensor_parallel_size
        ),
        "enable_attention_dp": expected_parallelism.enable_attention_dp,
        "enable_lm_head_tp_in_adp": (
            candidate.enable_lm_head_tp_in_adp
            if expected_parallelism.enable_attention_dp
            else False
        ),
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
    resolved["attention_dp_batch_mode"] = (
        candidate.attention_dp_batch_mode
    )
    resolved["max_seq_len"] = int(llm_args.max_seq_len)
    if resolved["max_seq_len"] != candidate.max_seq_len:
        raise RuntimeError(
            "Resolved TRT max_seq_len mismatch: "
            f"{resolved['max_seq_len']} != {candidate.max_seq_len}"
        )
    resolved["print_iter_log"] = bool(llm_args.print_iter_log)
    if resolved["print_iter_log"] != candidate.print_iter_log:
        raise RuntimeError(
            "Resolved TRT print_iter_log mismatch: "
            f"{resolved['print_iter_log']} != {candidate.print_iter_log}"
        )
    sparse_config = getattr(llm_args, "sparse_attention_config", None)
    resolved["use_cute_dsl_topk"] = bool(
        sparse_config is not None
        and getattr(sparse_config, "use_cute_dsl_topk", False)
    )
    resolved["use_cute_dsl_paged_mqa_logits"] = bool(
        sparse_config is not None
        and getattr(
            sparse_config,
            "use_cute_dsl_paged_mqa_logits",
            False,
        )
    )
    resolved["enable_heuristic_topk"] = bool(
        sparse_config is not None
        and getattr(sparse_config, "enable_heuristic_topk", False)
    )
    resolved["indexer_k_dtype"] = (
        getattr(sparse_config, "indexer_k_dtype", None)
        if sparse_config is not None
        else None
    )
    resolved["index_topk"] = (
        getattr(sparse_config, "index_topk", None)
        if sparse_config is not None
        else None
    )
    sparse_flags = (
        "use_cute_dsl_topk",
        "use_cute_dsl_paged_mqa_logits",
        "enable_heuristic_topk",
    )
    for field_name in sparse_flags:
        expected_value = getattr(candidate, field_name)
        if resolved[field_name] != expected_value:
            raise RuntimeError(
                f"Resolved TRT {field_name} mismatch: "
                f"{resolved[field_name]} != {expected_value}"
            )
    if (
        candidate.indexer_k_dtype is not None
        and resolved["indexer_k_dtype"] != candidate.indexer_k_dtype
    ):
        raise RuntimeError(
            "Resolved TRT indexer_k_dtype mismatch: "
            f"{resolved['indexer_k_dtype']!r} != "
            f"{candidate.indexer_k_dtype!r}"
        )
    if (
        candidate.enable_heuristic_topk
        and resolved["index_topk"] not in {512, 1024, 2048}
    ):
        raise RuntimeError(
            "Resolved TRT heuristic top-k is unsupported: "
            f"index_topk={resolved['index_topk']!r}"
        )

    if concurrency is not None:
        resolved["global_concurrency"] = concurrency
        resolved["max_batch_size"] = int(llm_args.max_batch_size)
        resolved["max_num_tokens"] = int(llm_args.max_num_tokens)
        expected_max_batch_size = max_batch_size(concurrency, candidate)
        expected_max_num_tokens = max_num_tokens(
            concurrency,
            candidate.mtp_draft_tokens,
            candidate,
        )
        if resolved["max_batch_size"] != expected_max_batch_size:
            raise RuntimeError(
                "Resolved TRT max_batch_size mismatch: "
                f"{resolved['max_batch_size']} != "
                f"{expected_max_batch_size}"
            )
        if resolved["max_num_tokens"] != expected_max_num_tokens:
            raise RuntimeError(
                "Resolved TRT max_num_tokens mismatch: "
                f"{resolved['max_num_tokens']} != "
                f"{expected_max_num_tokens}"
            )

        graph_config = llm_args.cuda_graph_config
        if candidate.cuda_graph:
            if graph_config is None:
                raise RuntimeError(
                    "Resolved TRT cuda_graph_config is unexpectedly disabled"
                )
            if isinstance(graph_config, dict):
                graph_batch_sizes = graph_config["batch_sizes"]
                graph_max_batch_size = graph_config["max_batch_size"]
            else:
                graph_batch_sizes = graph_config.batch_sizes
                graph_max_batch_size = graph_config.max_batch_size
            resolved["cuda_graph_batch_sizes"] = [
                int(value) for value in graph_batch_sizes
            ]
            resolved["cuda_graph_max_batch_size"] = int(
                graph_max_batch_size
            )
            expected_graph_batch_size = cuda_graph_batch_size(
                concurrency,
                candidate,
            )
            if resolved["cuda_graph_batch_sizes"] != [
                expected_graph_batch_size
            ]:
                raise RuntimeError(
                    "Resolved TRT CUDA graph batch sizes mismatch: "
                    f"{resolved['cuda_graph_batch_sizes']} != "
                    f"{[expected_graph_batch_size]}"
                )
            if (
                resolved["cuda_graph_max_batch_size"]
                != expected_graph_batch_size
            ):
                raise RuntimeError(
                    "Resolved TRT CUDA graph max batch size mismatch: "
                    f"{resolved['cuda_graph_max_batch_size']} != "
                    f"{expected_graph_batch_size}"
                )
        elif graph_config is not None:
            raise RuntimeError(
                "Resolved TRT cuda_graph_config is unexpectedly enabled"
            )
    return resolved
