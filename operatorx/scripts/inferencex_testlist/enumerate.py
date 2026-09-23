"""Per-row enumeration of canonical OperatorX ops.

Input: a `WorkloadRow` (one InferenceX matrix entry) and the model `Arch`.
Output: a list of `(op_type, args_dict, name)` triples that can be folded
into the testlist JSONs.
"""
from __future__ import annotations

from typing import Any, Iterable

from .dtypes import OpDtype, RowDtypes, resolve_dtypes
from .matrix import WorkloadRow
from .models import Arch, AttentionArch, MoeArch
from .parallelism import Parallelism


OpTriple = tuple[str, dict[str, Any], str]


# Existing testlists name entries by model (e.g. "dsv3", "llama3-8b") for
# layer-shaped ops (attention/moe) and leave `name` empty for shape-only ops
# (gemm/collectives/memory). That convention drives dedupe behavior: identical
# (m,n,k,dtype) GEMMs that arise from different models should collapse to one
# entry in gemm.json.
_NAMED_OP_TYPES = {"attention_mha", "attention_mla", "moe_forward", "topk_routing"}


def _name(op_type: str, model_prefix: str) -> str:
    return model_prefix if op_type in _NAMED_OP_TYPES else ""


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------


def _mtp_factor(row: WorkloadRow) -> int:
    """How many draft tokens decode evaluates per step. 1 if no spec-decoding.

    InferenceX scripts uniformly run `--speculative-num-steps 1` for MTP and
    most draft-model setups, so a decode step verifies 1 token + 1 draft.
    """
    if row.spec_decoding in ("mtp", "draft_model"):
        return 2
    return 1


# Per-kernel chunked-prefill cap. The InferenceX launch scripts pass
# --chunked-prefill-size in the 8192..32768 range; we use the upper bound as a
# realistic per-kernel ceiling. Without this cap, multinode disagg rows
# (conc=21504, ISL=8192) would emit 176M-token MoE ops that no kernel can run —
# the framework chunks them into ~5000 waves of `chunked-prefill-size` each.
CHUNKED_PREFILL_MAX = 32768


def _prefill_tokens(row: WorkloadRow, par: Parallelism) -> int:
    """Per-kernel tokens-per-step at prefill, per attention rank.

    Prefill chunking + DP-attn sharding + multinode disagg per-worker split:
      1. Each prefill worker handles `conc * ISL / num_workers` total tokens.
      2. The engine streams these in waves of `--chunked-prefill-size`
         (capped at CHUNKED_PREFILL_MAX). One kernel call sees one wave.
      3. Under DP-attn each rank handles `1/attn_dp` of the wave (token-axis
         sharding); without DP-attn each rank sees the full wave (heads-axis
         sharding handles parallelism instead).
    """
    per_worker_total = (row.conc * row.isl) // max(1, row.num_workers)
    capped = min(per_worker_total, CHUNKED_PREFILL_MAX)
    return max(1, capped // par.attn_dp)


def _decode_tokens(row: WorkloadRow, par: Parallelism) -> int:
    """Per-kernel tokens-per-step at decode, per attention rank.

    Multinode disagg: each decode worker handles `1/num_workers` of total
    concurrent requests. DP-attn: each rank handles `1/attn_dp` of the batch.
    """
    per_worker = (row.conc * _mtp_factor(row)) // max(1, row.num_workers)
    return max(1, per_worker // par.attn_dp)


def _moe_tokens_prefill(row: WorkloadRow) -> int:
    """Per-kernel tokens visible to the MoE block at prefill.

    MoE runs on the *full* batch regardless of DP-attn (AllGather brings the
    sharded attention output back to full batch before dispatch). For multinode
    disagg, the matrix `conc` is the total target throughput; each prefill
    worker sees `conc / num_workers` worth of requests. The per-kernel batch is
    further bounded by `--chunked-prefill-size` (typically 32768). Floored to
    1 so int-division rounding at low conc still emits a benchable shape.
    """
    per_worker_total = (row.conc * row.isl) // max(1, row.num_workers)
    return max(1, min(per_worker_total, CHUNKED_PREFILL_MAX))


def _moe_tokens_decode(row: WorkloadRow) -> int:
    """Per-kernel tokens visible to the MoE block at decode.

    Single-node: `conc * mtp_factor` (one MTP-extended decode step on the full
    batch). Multinode disagg: divided by `num_workers` (each decode worker
    handles its share of the total concurrency). Floored to 1 so
    `(conc=1, mtp=1, num_workers=2..5)` still emits a benchable shape rather
    than rounding to 0.
    """
    return max(1, (row.conc * _mtp_factor(row)) // max(1, row.num_workers))


# ---------------------------------------------------------------------------
# GEMM enumeration
# ---------------------------------------------------------------------------


def _gemm(
    M: int, N: int, K: int, row: WorkloadRow, dt: OpDtype,
    role: str, shard: int, bias: bool = False,
) -> OpTriple:
    """Emit a canonical GEMM op.

    `role` is a short tag identifying which projection this is (e.g.
    "q_proj", "mla_q_b", "mlp_gate_up"). `shard` records the TP shard factor
    applied to the N-axis weight of this GEMM so the name field captures the
    sharding context (and identical (M,N,K,dtype) GEMMs from different
    shardings stay distinguishable when wanted)."""
    name = f"{row.model_prefix}-{role}" + (f"-tp{shard}" if shard > 1 else "")
    args: dict[str, Any] = {
        "m": int(M),
        "n": int(N),
        "k": int(K),
        "dtype_a": dt.activation,
        "dtype_b": dt.weight,
        "dtype_out": dt.output,
    }
    if bias:
        args["bias"] = True
    return ("gemm", args, name)


def _attn_block_gemms(
    M: int, arch: Arch, par: Parallelism, row: WorkloadRow, phase: str, dt: OpDtype
) -> Iterable[OpTriple]:
    """Emit projection GEMMs for one attention block.

    Conventions:
      - MHA: q_proj [H, num_heads*head_dim], k_proj/v_proj [H, num_kv_heads*head_dim],
        o_proj [num_heads*head_dim, H]. With TP=N, heads are sharded N-way.
      - MLA (DeepSeek/Kimi/GLM): q_a_proj [H, q_lora_rank], q_b_proj [q_lora_rank, num_heads*(qk_nope+qk_rope)],
        kv_a_proj_with_mqa [H, kv_lora_rank+qk_rope], kv_b_proj [kv_lora_rank, num_heads*(qk_nope+v_head_dim)],
        o_proj [num_heads*v_head_dim, H].
    """
    H = arch.hidden_size
    a = arch.attention
    tp = par.attn_tp

    # GPT-OSS has `attention_bias=True` in its HF config; QKV/O projections
    # all carry biases. Other InferenceX models leave attention_bias=False.
    attn_bias = (row.model_prefix == "gptoss")

    if a.kind == "mha":
        n_q = a.num_heads * a.head_dim
        n_kv = a.num_kv_heads * a.head_dim
        kv_shard = max(1, min(tp, a.num_kv_heads))
        # SGLang/vLLM both fuse Q+K+V into one GEMM with width
        # (num_heads + 2*num_kv_heads)*head_dim. We emit that fused shape since
        # it's what the kernel actually runs.
        fused_qkv_n = (n_q + 2 * n_kv) // tp if tp == kv_shard else None
        if fused_qkv_n is not None:
            yield _gemm(M, fused_qkv_n, H, row, dt, f"attn_qkv_proj_{phase}", tp, bias=attn_bias)
        else:
            yield _gemm(M, n_q // tp, H, row, dt, f"attn_q_proj_{phase}", tp, bias=attn_bias)
            yield _gemm(M, n_kv // kv_shard, H, row, dt, f"attn_k_proj_{phase}", kv_shard, bias=attn_bias)
            yield _gemm(M, n_kv // kv_shard, H, row, dt, f"attn_v_proj_{phase}", kv_shard, bias=attn_bias)
        yield _gemm(M, H, n_q // tp, row, dt, f"attn_o_proj_{phase}", tp, bias=attn_bias)
        return

    # MLA path: emit canonical 5 projections (q_a, q_b, kv_a, kv_b, o).
    q_lora = a.q_lora_rank
    kv_lora = a.kv_lora_rank
    head_qk = a.qk_nope_head_dim + a.qk_rope_head_dim
    yield _gemm(M, q_lora, H, row, dt, f"mla_q_a_proj_{phase}", 1)
    yield _gemm(M, (a.num_heads * head_qk) // tp, q_lora, row, dt, f"mla_q_b_proj_{phase}", tp)
    yield _gemm(M, kv_lora + a.qk_rope_head_dim, H, row, dt, f"mla_kv_a_proj_{phase}", 1)
    yield _gemm(M, (a.num_heads * (a.qk_nope_head_dim + a.v_head_dim)) // tp, kv_lora, row, dt,
                f"mla_kv_b_proj_{phase}", tp)
    yield _gemm(M, H, (a.num_heads * a.v_head_dim) // tp, row, dt, f"mla_o_proj_{phase}", tp)


def _dense_mlp_gemms(
    M: int, arch: Arch, par: Parallelism, row: WorkloadRow, phase: str, dt: OpDtype
) -> Iterable[OpTriple]:
    if arch.moe.dense_intermediate_size <= 0:
        return
    H = arch.hidden_size
    interm = arch.moe.dense_intermediate_size
    tp = par.tp
    # SGLang/vLLM fuse gate + up into one GEMM with width 2*intermediate, then
    # apply silu_and_mul. Emit the fused shape — that's what the kernel runs.
    yield _gemm(M, (2 * interm) // tp, H, row, dt, f"mlp_gate_up_proj_{phase}", tp)
    yield _gemm(M, H, interm // tp, row, dt, f"mlp_down_proj_{phase}", tp)


# ---------------------------------------------------------------------------
# Attention op enumeration
# ---------------------------------------------------------------------------


def _attention_ops(
    arch: Arch, par: Parallelism, row: WorkloadRow, phase: str, dts: RowDtypes
) -> Iterable[OpTriple]:
    a = arch.attention
    if phase == "prefill":
        # Treat as one big extend step at full ISL (chunked prefill produces
        # the same canonical shape modulo the chunk size, which we don't
        # model explicitly here).
        s_q = row.isl
        s_kv = row.isl
        batch = max(1, row.conc // par.attn_dp)
    else:
        s_q = _mtp_factor(row)
        # Sweep S_kv over representative points to cover the decode trajectory.
        # We yield only the worst case (full prefix + half output); other
        # points get emitted at the per-row level by callers if they want.
        s_kv = row.isl + row.osl // 2
        batch = max(1, row.conc // par.attn_dp)

    out_dtype = dts.attn.output
    q_dtype = dts.attn.activation
    kvd = dts.kv

    if a.kind == "mha":
        yield (
            "attention_mha",
            {
                "batch_size": int(batch),
                "seq_len_q": int(s_q),
                "seq_len_kv": int(s_kv),
                "num_heads": int(a.num_heads // par.attn_tp),
                "num_heads_kv": int(max(1, a.num_kv_heads // par.attn_tp)),
                "head_dim": int(a.head_dim),
                "dtype_q": q_dtype,
                "dtype_k": kvd,
                "dtype_v": kvd,
                "dtype_o": out_dtype,
                "causal": True,
                **({"sliding_window": int(a.sliding_window)} if a.sliding_window else {}),
            },
            _name("attention_mha", row.model_prefix),
        )
        return

    # MLA
    yield (
        "attention_mla",
        {
            "batch_size": int(batch),
            "seq_len_q": int(s_q),
            "seq_len_kv": int(s_kv),
            "num_heads": int(a.num_heads // par.attn_tp),
            "head_dim_qk_nope": int(a.qk_nope_head_dim),
            "head_dim_qk_rope": int(a.qk_rope_head_dim),
            "head_dim_v": int(a.v_head_dim),
            "kv_lora_rank": int(a.kv_lora_rank),
            "dtype_q": q_dtype,
            "dtype_kv": kvd,
            "dtype_o": out_dtype,
            "rope": True,
            "causal": True,
        },
        _name("attention_mla", row.model_prefix),
    )


# ---------------------------------------------------------------------------
# MoE op enumeration
# ---------------------------------------------------------------------------


def _shared_tp(row: WorkloadRow, par: Parallelism) -> int:
    """Effective TP shard for the SHARED expert MLP under SGLang.

    SGLang's behavior (`deepseek_v2.py` / `qwen2_moe.py`):
      - DSv4 sets `disable_shared_experts_fusion=True`. The shared expert is a
        separate MLP and runs with `tp_size=1` so its (clamped SwiGLU)
        activation can be kept distinct from the routed experts' (un-clamped).
      - When `--moe-a2a-backend deepep` is selected (full EP path), the shared
        expert is either DeepEP-fused as one extra slot per EP rank, or — for
        the non-fused path — explicitly created with `tp_rank=0, tp_size=1`.
        Either way, weights are replicated per rank: `shared_tp = 1`.
      - Otherwise the shared expert is folded into the routed FusedMoE kernel
        as an extra expert slot, and its weights are TP-sharded along
        intermediate the same way routed weights are: `shared_tp = moe_tp`.
    """
    if row.model_prefix == "dsv4":
        return 1
    if par.moe_ep > 1:
        return 1
    return par.moe_tp


def _moe_ops(
    arch: Arch, par: Parallelism, row: WorkloadRow, phase: str, tokens: int, dts: RowDtypes
) -> Iterable[OpTriple]:
    moe = arch.moe
    if moe.num_experts <= 0:
        return

    H = arch.hidden_size
    yield (
        "topk_routing",
        {
            "num_tokens": int(tokens),
            "num_experts": int(moe.num_experts),
            "top_k": int(moe.num_experts_per_tok),
            "dtype": dts.routing,
        },
        _name("topk_routing", row.model_prefix),
    )
    yield (
        "moe_forward",
        {
            "num_tokens": int(tokens),
            "hidden": int(H),
            "intermediate": int(moe.moe_intermediate_size),
            "num_experts": int(moe.num_experts),
            "top_k": int(moe.num_experts_per_tok),
            "dtype_act": dts.moe.activation,
            "dtype_weight": dts.moe.weight,
            "world_size": int(par.world_size),
            "expert_parallel_size": int(par.moe_ep),
            "routed_tensor_parallel_size": int(par.moe_tp),
            "shared_tensor_parallel_size": int(_shared_tp(row, par)),
            "n_shared_experts": int(moe.n_shared_experts),
        },
        _name("moe_forward", row.model_prefix),
    )


# ---------------------------------------------------------------------------
# Collective enumeration
# ---------------------------------------------------------------------------


def _collective_ops(
    arch: Arch, par: Parallelism, row: WorkloadRow, phase: str, tokens: int, dts: RowDtypes
) -> Iterable[OpTriple]:
    if par.world_size <= 1:
        return

    H = arch.hidden_size
    moe = arch.moe
    # Collectives carry the MoE-layer activations, which are exchanged in the
    # MoE op's activation dtype (post-attention output is dequantized to bf16
    # before MoE input). Use the attention output dtype as the comm dtype since
    # that's what's in flight during AG/RS/AR. For dispatch/combine the MoE
    # activation dtype applies.
    comm_dtype = dts.attn.output
    moe_comm_dtype = dts.moe.activation

    if par.dp_attn and par.attn_dp > 1:
        # Under DP-attn, attention output is sharded by tokens; an AllGather
        # along the token axis brings activations back to full batch before
        # MoE dispatch. Conversely, after MoE combine, a ReduceScatter
        # re-shards by token. SGLang's deepseek path does this.
        yield (
            "allgather",
            {
                "num_elements_per_rank": int(tokens * H),
                "dtype": comm_dtype,
                "world_size": int(par.attn_dp),
            },
            _name("allgather", row.model_prefix),
        )
        yield (
            "reduce_scatter",
            {
                "num_elements": int(tokens * par.attn_dp * H),
                "dtype": comm_dtype,
                "world_size": int(par.attn_dp),
            },
            _name("reduce_scatter", row.model_prefix),
        )
    elif par.tp > 1:
        # Plain TP: one AllReduce after attention out-proj, another after MLP.
        yield (
            "allreduce",
            {
                "num_elements": int(tokens * H),
                "dtype": comm_dtype,
                "world_size": int(par.tp),
            },
            _name("allreduce", row.model_prefix),
        )

    if moe.num_experts > 0 and par.moe_ep > 1:
        # MoE expert-parallel dispatch + combine.
        yield (
            "dispatch",
            {
                "num_tokens": int(tokens),
                "num_experts": int(moe.num_experts),
                "top_k": int(moe.num_experts_per_tok),
                "hidden": int(H),
                "dtype": moe_comm_dtype,
                "world_size": int(par.moe_ep),
                "mode": "low_latency" if phase == "decode" else "normal",
            },
            _name("dispatch", row.model_prefix),
        )
        yield (
            "combine",
            {
                "num_tokens": int(tokens),
                "num_experts": int(moe.num_experts),
                "top_k": int(moe.num_experts_per_tok),
                "hidden": int(H),
                "dtype": moe_comm_dtype,
                "world_size": int(par.moe_ep),
                "mode": "low_latency" if phase == "decode" else "normal",
            },
            _name("combine", row.model_prefix),
        )


# ---------------------------------------------------------------------------
# Top-level row enumeration
# ---------------------------------------------------------------------------


def ops_for_row(row: WorkloadRow, arch: Arch) -> list[OpTriple]:
    par = Parallelism(tp=row.tp, ep=row.ep, dp_attn=row.dp_attn)
    dts = resolve_dtypes(row)

    # Attention runs per-rank (DP-attn slices tokens, plain TP shards heads).
    prefill_M = _prefill_tokens(row, par)
    decode_M = _decode_tokens(row, par)
    # MoE runs on the full batch after AllGather under DP-attn; EP only shards weights.
    prefill_moe_tokens = _moe_tokens_prefill(row)
    decode_moe_tokens = _moe_tokens_decode(row)

    out: list[OpTriple] = []

    # Multinode disagg rows are tagged with the phase the worker handles; emit
    # only that phase's ops. Single-node rows (phase=None) emit both phases.
    emit_prefill = row.phase in (None, "prefill")
    emit_decode = row.phase in (None, "decode")

    if emit_prefill:
        out.extend(_attn_block_gemms(prefill_M, arch, par, row, "prefill", dts.attn))
        out.extend(_attention_ops(arch, par, row, "prefill", dts))
        out.extend(_dense_mlp_gemms(prefill_M, arch, par, row, "prefill", dts.dense))
        out.extend(_moe_ops(arch, par, row, "prefill", prefill_moe_tokens, dts))
        out.extend(_collective_ops(arch, par, row, "prefill", prefill_moe_tokens, dts))

    if emit_decode:
        out.extend(_attn_block_gemms(decode_M, arch, par, row, "decode", dts.attn))
        out.extend(_attention_ops(arch, par, row, "decode", dts))
        out.extend(_dense_mlp_gemms(decode_M, arch, par, row, "decode", dts.dense))
        out.extend(_moe_ops(arch, par, row, "decode", decode_moe_tokens, dts))
        out.extend(_collective_ops(arch, par, row, "decode", decode_moe_tokens, dts))

    return out
