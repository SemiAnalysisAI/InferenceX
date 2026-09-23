"""HuggingFace model config loader + per-architecture shape descriptors.

`load_arch()` returns an `Arch` dataclass that exposes everything the
enumerator needs about a model's structure (hidden dims, attention shape,
MoE config, attention type). Per-arch wrappers translate the various
HF config layouts (top-level, `text_config`, etc.) into a uniform shape.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


_DEFAULT_LOCAL_DIRS = (
    "/models",
    "/scratch/fsw/models",
)


@dataclass(frozen=True)
class AttentionArch:
    kind: str                   # "mha" | "mla"
    num_heads: int
    num_kv_heads: int
    head_dim: int               # MHA: per-head dim. MLA: not directly used (see MLA fields).
    # MLA-only:
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    kv_lora_rank: int = 0
    q_lora_rank: int = 0
    # Optional: per-layer attention types ("full" vs "sliding"/"linear"). When
    # heterogeneous, the canonical shape set should include all variants.
    sliding_window: int | None = None


@dataclass(frozen=True)
class MoeArch:
    """Empty / disabled when this model has no MoE layers."""
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    n_shared_experts: int = 0
    # If the model has dense (non-MoE) layers in addition to MoE layers, set
    # `dense_intermediate_size` so we also emit MLP-GEMM shapes for those.
    dense_intermediate_size: int = 0
    num_dense_layers: int = 0   # number of leading dense layers (first_k_dense_replace etc.)


@dataclass(frozen=True)
class Arch:
    name: str
    family: str                 # "deepseek" | "glm" | "kimi" | "minimax" | "gptoss" | "qwen3moe"
    hidden_size: int
    num_layers: int
    attention: AttentionArch
    moe: MoeArch
    mtp_num_layers: int = 0     # additional speculative-decode layers (DeepSeek MTP, Qwen MTP)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _candidate_paths(model_id: str, extra_dirs: tuple[str, ...]) -> list[str]:
    """Where to look for a config.json for `model_id`."""
    paths: list[str] = []
    # Bare ids from the master configs are usually "<org>/<name>"; the local
    # caches store them under "<name>".
    short = model_id.split("/")[-1]
    for base in extra_dirs:
        paths.append(os.path.join(base, short, "config.json"))
        paths.append(os.path.join(base, model_id.replace("/", "_"), "config.json"))
    # HF default cache layout:
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    paths.append(os.path.join(hf_home, "hub", f"models--{model_id.replace('/', '--')}", "snapshots"))
    return paths


def _load_hf_config(model_id: str, extra_dirs: tuple[str, ...]) -> dict[str, Any]:
    for p in _candidate_paths(model_id, extra_dirs):
        if p.endswith("snapshots") and os.path.isdir(p):
            # pick first snapshot dir
            for d in sorted(os.listdir(p)):
                cand = os.path.join(p, d, "config.json")
                if os.path.isfile(cand):
                    with open(cand) as f:
                        return json.load(f)
        elif os.path.isfile(p):
            with open(p) as f:
                return json.load(f)
    raise FileNotFoundError(
        f"Could not locate config.json for {model_id!r}. Searched: {_candidate_paths(model_id, extra_dirs)}"
    )


def _text_subconfig(raw: dict[str, Any]) -> dict[str, Any]:
    """Some configs nest the LM under `text_config` (Kimi, Qwen3.5 multimodal).

    Top-level architectures/model_type wins so family detection sees the brand
    (e.g. KimiK25ForConditionalGeneration) rather than the LM-class it reuses
    (Kimi K2.5 reuses DeepseekV3ForCausalLM under the hood)."""
    if "text_config" in raw and isinstance(raw["text_config"], dict):
        merged = dict(raw["text_config"])
        if raw.get("architectures"):
            merged["architectures"] = raw["architectures"]
        if raw.get("model_type"):
            merged["model_type"] = raw["model_type"]
        return merged
    return raw


# ---------------------------------------------------------------------------
# Family detection
# ---------------------------------------------------------------------------


def _family(cfg: dict[str, Any]) -> str:
    arch = (cfg.get("architectures") or [""])[0]
    mt = cfg.get("model_type", "")
    if "Deepseek" in arch or mt.startswith("deepseek"):
        return "deepseek"
    if "Glm" in arch or mt.startswith("glm"):
        return "glm"
    if "Kimi" in arch or mt.startswith("kimi"):
        return "kimi"
    if "MiniMax" in arch or mt.startswith("minimax"):
        return "minimax"
    if "GptOss" in arch or "gpt_oss" in mt or "gptoss" in arch.lower():
        return "gptoss"
    if "Qwen" in arch or mt.startswith("qwen"):
        return "qwen3moe"
    raise ValueError(f"Unsupported model architecture: arch={arch!r} model_type={mt!r}")


# ---------------------------------------------------------------------------
# Per-family builders
# ---------------------------------------------------------------------------


def _build_deepseek(cfg: dict[str, Any], name: str) -> Arch:
    # DeepSeek V3 / R1: qk_nope_head_dim, qk_rope_head_dim, v_head_dim, kv_lora_rank
    #   are all explicit; `head_dim` is absent.
    # DeepSeek V4: head_dim is set (= v_head_dim), qk_rope_head_dim is set,
    #   qk_nope_head_dim is implicit (= head_dim - qk_rope_head_dim),
    #   kv_lora_rank is named `o_lora_rank`.
    qk_rope = cfg.get("qk_rope_head_dim", 0)
    head_dim_explicit = cfg.get("head_dim") or 0
    v_head_dim = cfg.get("v_head_dim") or head_dim_explicit
    qk_nope = cfg.get("qk_nope_head_dim")
    if qk_nope is None:
        qk_nope = head_dim_explicit - qk_rope if head_dim_explicit else 0
    kv_lora = cfg.get("kv_lora_rank") or cfg.get("o_lora_rank") or 0
    return Arch(
        name=name,
        family="deepseek",
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_hidden_layers"],
        attention=AttentionArch(
            kind="mla",
            num_heads=cfg["num_attention_heads"],
            num_kv_heads=cfg.get("num_key_value_heads", 1),
            head_dim=v_head_dim,
            qk_nope_head_dim=qk_nope,
            qk_rope_head_dim=qk_rope,
            v_head_dim=v_head_dim,
            kv_lora_rank=kv_lora,
            q_lora_rank=cfg["q_lora_rank"],
            sliding_window=cfg.get("sliding_window"),
        ),
        moe=MoeArch(
            num_experts=cfg["n_routed_experts"],
            num_experts_per_tok=cfg["num_experts_per_tok"],
            moe_intermediate_size=cfg["moe_intermediate_size"],
            n_shared_experts=cfg.get("n_shared_experts", 0),
            dense_intermediate_size=cfg.get("intermediate_size", 0),
            num_dense_layers=cfg.get("first_k_dense_replace", 0) or 0,
        ),
        mtp_num_layers=cfg.get("num_nextn_predict_layers", 0) or 0,
    )


def _build_glm(cfg: dict[str, Any], name: str) -> Arch:
    # GLM-5 MoE+DSA: it has MLA-style attention (qk_nope/qk_rope/v_head_dim/kv_lora).
    return Arch(
        name=name,
        family="glm",
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_hidden_layers"],
        attention=AttentionArch(
            kind="mla",
            num_heads=cfg["num_attention_heads"],
            num_kv_heads=cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
            head_dim=cfg.get("head_dim", 0),
            qk_nope_head_dim=cfg["qk_nope_head_dim"],
            qk_rope_head_dim=cfg["qk_rope_head_dim"],
            v_head_dim=cfg["v_head_dim"],
            kv_lora_rank=cfg["kv_lora_rank"],
            q_lora_rank=cfg["q_lora_rank"],
        ),
        moe=MoeArch(
            num_experts=cfg["n_routed_experts"],
            num_experts_per_tok=cfg["num_experts_per_tok"],
            moe_intermediate_size=cfg["moe_intermediate_size"],
            n_shared_experts=cfg.get("n_shared_experts", 0),
            dense_intermediate_size=cfg.get("intermediate_size", 0),
            num_dense_layers=cfg.get("first_k_dense_replace", 0) or 0,
        ),
        mtp_num_layers=cfg.get("num_nextn_predict_layers", 0) or 0,
    )


def _build_kimi(cfg: dict[str, Any], name: str) -> Arch:
    # Kimi K2.5: MLA + MoE. Same shape as DeepSeek.
    return Arch(
        name=name,
        family="kimi",
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_hidden_layers"],
        attention=AttentionArch(
            kind="mla",
            num_heads=cfg["num_attention_heads"],
            num_kv_heads=cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
            head_dim=cfg.get("head_dim", 0),
            qk_nope_head_dim=cfg["qk_nope_head_dim"],
            qk_rope_head_dim=cfg["qk_rope_head_dim"],
            v_head_dim=cfg["v_head_dim"],
            kv_lora_rank=cfg["kv_lora_rank"],
            q_lora_rank=cfg["q_lora_rank"],
        ),
        moe=MoeArch(
            num_experts=cfg["n_routed_experts"],
            num_experts_per_tok=cfg["num_experts_per_tok"],
            moe_intermediate_size=cfg["moe_intermediate_size"],
            n_shared_experts=cfg.get("n_shared_experts", 0),
            dense_intermediate_size=cfg.get("intermediate_size", 0),
            num_dense_layers=cfg.get("first_k_dense_replace", 0) or 0,
        ),
        mtp_num_layers=cfg.get("num_nextn_predict_layers", 0) or 0,
    )


def _build_minimax(cfg: dict[str, Any], name: str) -> Arch:
    # MiniMax-M2.5: GQA-style MHA + MoE (no shared experts in published config).
    return Arch(
        name=name,
        family="minimax",
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_hidden_layers"],
        attention=AttentionArch(
            kind="mha",
            num_heads=cfg["num_attention_heads"],
            num_kv_heads=cfg["num_key_value_heads"],
            head_dim=cfg["head_dim"],
        ),
        moe=MoeArch(
            num_experts=cfg.get("num_local_experts", cfg.get("num_experts", 0)),
            num_experts_per_tok=cfg["num_experts_per_tok"],
            moe_intermediate_size=cfg.get("moe_intermediate_size", cfg.get("intermediate_size", 0)),
            n_shared_experts=cfg.get("n_shared_experts", 0),
            dense_intermediate_size=cfg.get("intermediate_size", 0),
            num_dense_layers=0,
        ),
    )


def _build_gptoss(cfg: dict[str, Any], name: str) -> Arch:
    # GPT-OSS-120B: GQA MHA + MoE. layer_types alternates sliding/full.
    sw = cfg.get("sliding_window")
    return Arch(
        name=name,
        family="gptoss",
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_hidden_layers"],
        attention=AttentionArch(
            kind="mha",
            num_heads=cfg["num_attention_heads"],
            num_kv_heads=cfg["num_key_value_heads"],
            head_dim=cfg["head_dim"],
            sliding_window=sw,
        ),
        moe=MoeArch(
            num_experts=cfg.get("num_local_experts", cfg.get("num_experts", 0)),
            num_experts_per_tok=cfg["num_experts_per_tok"],
            moe_intermediate_size=cfg.get("intermediate_size", 0),
            n_shared_experts=0,
            dense_intermediate_size=0,
            num_dense_layers=0,
        ),
    )


def _build_qwen3moe(cfg: dict[str, Any], name: str) -> Arch:
    # Qwen3.5-MoE: hybrid linear/full attention, MoE with `num_experts` and
    # `shared_expert_intermediate_size`. We model the full-attention layers
    # only (the linear-attention layers don't fit the canonical attention op
    # cleanly yet — TODO).
    return Arch(
        name=name,
        family="qwen3moe",
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_hidden_layers"],
        attention=AttentionArch(
            kind="mha",
            num_heads=cfg["num_attention_heads"],
            num_kv_heads=cfg["num_key_value_heads"],
            head_dim=cfg["head_dim"],
        ),
        moe=MoeArch(
            num_experts=cfg.get("num_experts", cfg.get("num_local_experts", 0)),
            num_experts_per_tok=cfg["num_experts_per_tok"],
            moe_intermediate_size=cfg.get("moe_intermediate_size", 0),
            n_shared_experts=1 if cfg.get("shared_expert_intermediate_size") else 0,
            dense_intermediate_size=cfg.get("shared_expert_intermediate_size", 0),
            num_dense_layers=0,
        ),
        mtp_num_layers=cfg.get("mtp_num_hidden_layers", 0) or 0,
    )


_BUILDERS = {
    "deepseek": _build_deepseek,
    "glm": _build_glm,
    "kimi": _build_kimi,
    "minimax": _build_minimax,
    "gptoss": _build_gptoss,
    "qwen3moe": _build_qwen3moe,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_arch(model_id: str, extra_dirs: tuple[str, ...] = _DEFAULT_LOCAL_DIRS) -> Arch:
    """Load the architecture descriptor for a HuggingFace model id."""
    raw = _load_hf_config(model_id, extra_dirs)
    cfg = _text_subconfig(raw)
    family = _family(cfg)
    return _BUILDERS[family](cfg, name=model_id)
