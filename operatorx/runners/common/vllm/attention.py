"""Attention modules through vLLM's own model code, KV cache and scheduling.

The op's checkpoint family (attention_models.json) with its sizes, cut to the layers the module
needs, dummy weights; the batch is scheduled through the model runner with ctx tokens already
computed, and the module's forward is timed with its decoder layer's arguments and forward context.
"""
from __future__ import annotations

import copy
import gc
import json
import os
import random
import sys
import tempfile
from dataclasses import dataclass, field, fields
from pathlib import Path

import torch

from operatorx.core import BackendImpl, Op, UnsupportedOpError
from operatorx.runners.common import ranks
from operatorx.runners.common.vllm import engine as vllm_engine
from operatorx.runners.common.vllm import linear as vllm_linear

_MODELS = json.loads((Path(__file__).with_name("attention_models.json")).read_text())
_MLP = 256  # width of the (untimed) MLP in the cut-down model
_VOCAB = 1024
_MAX_BATCHED_TOKENS = 32768  # the largest max-num-batched-tokens of InferenceX's recipes
_MAX_SEQS = 1024
_KV_FRACTION = float(os.environ.get("OPERATORX_ATTN_KV_FRACTION", "0.5"))  # of device memory
# startup headroom check only; the KV cache is sized by _KV_FRACTION
_GPU_UTIL = float(os.environ.get("OPERATORX_ATTN_GPU_UTIL", "0.6"))
_KV_DTYPES = {"auto": "auto", "bf16": "bfloat16", "fp8": "fp8", "fp8_ds_mla": "fp8_ds_mla"}


@dataclass
class _Build:
    family: str
    config: dict
    module: str  # module path suffix, e.g. "layers.0.self_attn"
    engine: dict = field(default_factory=dict)
    without: tuple = ()  # recipe arguments for model parts the cut-down model leaves out


def _family(name: str) -> dict:
    return copy.deepcopy(_MODELS[name]["config"])


def _yarn(s: dict | None) -> dict | None:
    if s is None:
        return None
    out = {"type": "yarn", "factor": s["factor"], "original_max_position_embeddings": s["original_max"]}
    for k in ("beta_fast", "beta_slow", "mscale", "mscale_all_dim"):
        if k in s:
            out[k] = s[k]
    return out


def _quant(op: Op, family: str) -> dict | None:
    """The family checkpoint variant's quantization_config whose projections match proj."""
    proj = op.args.get("proj") or {}
    if not proj:
        return None
    for v in _MODELS[family].get("variants", []):
        if all(v["proj"].get(name) == pair for name, pair in proj.items()):
            return copy.deepcopy(v["quantization_config"])
    raise UnsupportedOpError(f"no {family} checkpoint quantizes these projections: {sorted(proj)}")


def _kimi_k3(mla: dict, kda: dict, target: str) -> _Build:
    """Kimi-K3: one KDA layer then one MLA layer (1-based layer lists), dense MLPs."""
    c = _family("kimi_k3")
    t = c["text_config"]
    lin = dict(t["linear_attn_config"], kda_layers=[1], full_attn_layers=[2], **kda)
    t.update(num_hidden_layers=2, linear_attn_config=lin, first_k_dense_replace=2, intermediate_size=_MLP,
             num_nextn_predict_layers=0, vocab_size=_VOCAB, **mla)
    return _Build("kimi_k3", c, target, {"language_model_only": True})


def _mla_dims(a: dict) -> dict:
    return dict(hidden_size=a["hidden"], num_attention_heads=a["heads"], num_key_value_heads=a["heads"],
                q_lora_rank=a["q_lora_rank"], kv_lora_rank=a["kv_lora_rank"], qk_nope_head_dim=a["nope"],
                qk_rope_head_dim=a["rope_dim"], v_head_dim=a["v"])


def _build_mla(op: Op) -> _Build:
    a = op.args
    if not a.get("rope", True) or a.get("gate"):
        if a.get("rope", True):
            raise UnsupportedOpError("the gated MLA module (Kimi-K3) has no RoPE")
        return _kimi_k3(dict(_mla_dims(a), mla_use_nope=True, mla_use_output_gate=bool(a.get("gate"))), {},
                        "layers.1.self_attn")
    c = _family("deepseek_v3")
    c.update(hidden_size=a["hidden"], num_attention_heads=a["heads"], num_key_value_heads=a["heads"],
             q_lora_rank=a["q_lora_rank"], kv_lora_rank=a["kv_lora_rank"], qk_nope_head_dim=a["nope"],
             qk_rope_head_dim=a["rope_dim"], v_head_dim=a["v"], rope_theta=a["rope_theta"],
             rope_scaling=_yarn(a.get("rope_scaling")), num_hidden_layers=1, first_k_dense_replace=1,
             intermediate_size=_MLP, num_nextn_predict_layers=0, vocab_size=_VOCAB)
    return _Build("deepseek_v3", c, "layers.0.self_attn", {})


def _qwen35(a: dict, full: dict, linear: dict, target: str) -> _Build:
    """Qwen3.5: one Gated DeltaNet then one gated full-attention layer (serving's hybrid KV layout)."""
    c = _family("qwen3_5_moe")
    t = c["text_config"]
    t.update(num_hidden_layers=2, layer_types=["linear_attention", "full_attention"], mtp_num_hidden_layers=0,
             num_experts=8, num_experts_per_tok=2, moe_intermediate_size=_MLP, shared_expert_intermediate_size=_MLP,
             vocab_size=_VOCAB, **full, **linear)
    return _Build("qwen3_5_moe", c, target, {"language_model_only": True})


def _qwen35_full(a: dict) -> dict:
    rope = {"rope_type": "default", "rope_theta": a["rope_theta"], "partial_rotary_factor": a["rope_dim"] / a["head_dim"]}
    if a.get("mrope_section"):
        rope.update(mrope_section=a["mrope_section"], mrope_interleaved=True)
    return dict(hidden_size=a["hidden"], num_attention_heads=a["q_heads"], num_key_value_heads=a["kv_heads"],
                head_dim=a["head_dim"], attn_output_gate=True, rope_parameters=rope)


def _build_gqa(op: Op) -> _Build:
    a = op.args
    if a.get("gate"):
        if not a.get("qk_norm"):
            raise UnsupportedOpError("the gated GQA module (Qwen3.5) always has QK norm")
        return _qwen35(a, _qwen35_full(a), {}, "layers.1.self_attn")
    if a.get("mrope_section"):
        raise UnsupportedOpError("M-RoPE without the output gate is not a module of these models")
    c = _family("minimax_m3")
    t = c["text_config"]
    sparse = dict(t["sparse_attention_config"], sparse_attention_freq=[0], sparse_disable_index_value=[0])
    t.update(hidden_size=a["hidden"], num_attention_heads=a["q_heads"], num_key_value_heads=a["kv_heads"],
             head_dim=a["head_dim"], rotary_dim=a["rope_dim"], partial_rotary_factor=a["rope_dim"] / a["head_dim"],
             rope_theta=a["rope_theta"], use_qk_norm=bool(a.get("qk_norm")), attention_output_gate=False,
             num_hidden_layers=1, moe_layer_freq=[0], dense_intermediate_size=_MLP, num_mtp_modules=0,
             sparse_attention_config=sparse, vocab_size=_VOCAB)
    return _Build("minimax_m3", c, "layers.0.self_attn", {"language_model_only": True})


def _build_gdn(op: Op) -> _Build:
    """silu: Qwen3.5's Gated DeltaNet; sigmoid: Qwen3.8's."""
    a = op.args
    linear = dict(hidden_size=a["hidden"], linear_num_key_heads=a["qk_heads"], linear_num_value_heads=a["v_heads"],
                  linear_key_head_dim=a["head_dim"], linear_value_head_dim=a["head_dim"],
                  linear_conv_kernel_dim=a["conv_kernel"],
                  mamba_ssm_dtype={"fp32": "float32", "bf16": "bfloat16"}[a.get("state_dtype", "fp32")])
    if a.get("norm_act", "silu") == "sigmoid":
        return _build_gdn_qwen38(a, linear)
    b = _qwen35(a, {}, linear, "layers.0.linear_attn")
    b.engine["mamba_ssm_cache_dtype"] = linear["mamba_ssm_dtype"]
    return b


def _build_kda(op: Op) -> _Build:
    a = op.args
    b = _kimi_k3({"hidden_size": a["hidden"]},
                 {"num_heads": a["heads"], "head_dim": a["head_dim"], "short_conv_kernel_size": a["conv_kernel"]},
                 "layers.0.self_attn")
    b.engine["mamba_ssm_cache_dtype"] = {"fp32": "float32", "bf16": "bfloat16"}[a.get("state_dtype", "fp32")]
    return b


def _build_mla_dsa(op: Op) -> _Build:
    """GLM-5.x: a layer with its own indexer, then one reusing its top-k ("FS")."""
    a = op.args
    c = _family("glm_moe_dsa")
    rope = {"rope_type": "default", "rope_theta": a["rope_theta"]}
    if a.get("rope_scaling"):
        raise UnsupportedOpError("rope scaling for the GLM sparse MLA module is not wired")
    c.update(_mla_dims(a), qk_head_dim=a["nope"] + a["rope_dim"], rope_parameters=rope, index_topk=a["topk"],
             index_n_heads=a["index_heads"], index_head_dim=a["index_dim"], num_hidden_layers=2,
             index_topk_pattern="FS", indexer_types=["full", "shared"], mlp_layer_types=["dense", "dense"],
             first_k_dense_replace=2, intermediate_size=_MLP, num_nextn_predict_layers=0, vocab_size=_VOCAB)
    return _Build("glm_moe_dsa", c, "layers.0.self_attn" if a.get("indexer", "own") == "own" else "layers.1.self_attn")


def _qwen38(text: dict, target: str) -> _Build:
    """Qwen3.8-Flash-Next: one Gated DeltaNet layer then one QSA layer."""
    c = _family("qwen4_exp")
    t = c["text_config"]
    t.update(num_hidden_layers=2, layer_types=["linear_attention", "full_attention"], mtp_num_hidden_layers=0,
             ple_layer_ids=[], num_experts=8, num_experts_per_tok=2, moe_intermediate_size=_MLP,
             shared_expert_intermediate_size=_MLP, **text)
    return _Build("qwen4_exp", c, target, {"language_model_only": True})


def _build_qsa(op: Op) -> _Build:
    a = op.args
    rope = {"rope_type": "default", "rope_theta": a["rope_theta"], "partial_rotary_factor": a["rope_dim"] / a["head_dim"]}
    if a.get("mrope_section"):
        rope.update(mrope_section=a["mrope_section"], mrope_interleaved=True)
    return _qwen38(dict(hidden_size=a["hidden"], num_attention_heads=a["q_heads"], num_key_value_heads=a["kv_heads"],
                        head_dim=a["head_dim"], partial_rotary_factor=rope["partial_rotary_factor"],
                        rope_parameters=rope, indexer_n_heads=a["index_heads"], indexer_kv_heads=1,
                        indexer_head_dim=a["index_dim"], indexer_budget=a["budget"],
                        indexer_compress_ratio=a["compress"]), "layers.1.self_attn")


# DeepSeek-V4.1's layer roles, in checkpoint order: window-only; ratio-2 source, top-k reuser;
# ratio-1 source (publishes candidate blocks), consumer, index source over the shared index cache
_V41_RATIOS = [0, 2, 2, 1, 1, 1]
_V41_LAYER = {(0, True, None): 0, (2, True, "own"): 1, (2, False, "reuse"): 2, (1, True, "own"): 3,
              (1, False, "reuse"): 4, (1, False, "shared"): 5}


def _build_dsv41(a: dict) -> _Build:
    layer = _V41_LAYER.get((a["compress_ratio"], a.get("source", True), a.get("indexer")))
    if layer is None:
        raise UnsupportedOpError("DeepSeek-V4.1 has no layer with this compress ratio / source / indexer")
    c = _family("deepseek_v41")
    t = c["text_config"]
    t.update(hidden_size=a["hidden"], num_attention_heads=a["heads"], head_dim=a["head_dim"],
             qk_rope_head_dim=a["rope_dim"], q_lora_rank=a["q_lora_rank"], sliding_window=a["window"],
             o_groups=a["o_groups"], o_lora_rank=a["o_lora_rank"], compress_ratios=_V41_RATIOS,
             kv_source_layer_ids=[1, 3], index_source_layer_ids=[1, 3, 5], candidate_source_layer_id=3,
             rope_theta=a.get("rope_theta", 10000.0), num_hidden_layers=len(_V41_RATIOS),
             num_nextn_predict_layers=0, engram_layer_ids=[], dspark_target_layer_ids=[], n_routed_experts=8,
             num_experts_per_tok=2, moe_intermediate_size=_MLP, vocab_size=_VOCAB)
    if a.get("compress_rope_theta"):
        t["compress_rope_theta"] = a["compress_rope_theta"]
    if a.get("rope_scaling"):
        t["rope_scaling"] = dict(_yarn(a["rope_scaling"]), rope_type="yarn")
    if a.get("indexer"):
        t.update(index_topk=a["topk"], index_n_heads=a["index_heads"], index_head_dim=a["index_dim"])
    return _Build("deepseek_v41", c, f"layers.{layer}.attn", {"language_model_only": True},
                  without=("engram-config",))


def _build_dsv4(op: Op) -> _Build:
    """compress ratio 4 / 128: DeepSeek-V4-Pro; 0 / 1 / 2: DeepSeek-V4.1-Flash."""
    a = op.args
    if a["compress_ratio"] in (0, 1, 2):
        b = _build_dsv41(a)
    else:
        b = _build_dsv4_pro(a)
    if a.get("index_cache_dtype"):
        b.engine["attention_config"] = {"indexer_kv_dtype": a["index_cache_dtype"]}
    return b


def _build_dsv4_pro(a: dict) -> _Build:
    c = _family("deepseek_v4")
    c.pop("expert_dtype", None)
    # the C4A and C128A layers share one KV cache layout, so either is built next to the other
    ratios = [4, 128]
    c.update(hidden_size=a["hidden"], num_attention_heads=a["heads"], head_dim=a["head_dim"],
             qk_rope_head_dim=a["rope_dim"], q_lora_rank=a["q_lora_rank"], sliding_window=a["window"],
             o_groups=a["o_groups"], o_lora_rank=a["o_lora_rank"], compress_ratios=ratios,
             rope_theta=a.get("rope_theta", 10000.0), num_hidden_layers=len(ratios), num_nextn_predict_layers=0,
             num_hash_layers=0, n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=_MLP,
             vocab_size=_VOCAB)
    if a["compress_ratio"]:
        c["compress_rope_theta"] = a["compress_rope_theta"]
        if a.get("rope_scaling"):
            c["rope_scaling"] = _yarn(a["rope_scaling"])
    if a.get("indexer"):
        c.update(index_topk=a["topk"], index_n_heads=a["index_heads"], index_head_dim=a["index_dim"])
    return _Build("deepseek_v4", c, f"layers.{ratios.index(a['compress_ratio'])}.attn")


def _build_gdn_qwen38(a: dict, linear: dict) -> _Build:
    b = _qwen38(linear, "layers.0.linear_attn")
    b.engine["mamba_ssm_cache_dtype"] = linear["mamba_ssm_dtype"]
    return b


_BUILDERS = {"mla": _build_mla, "mla_dsa": _build_mla_dsa, "dsv4_attn": _build_dsv4, "gqa": _build_gqa,
             "qsa": _build_qsa, "gdn": _build_gdn, "kda": _build_kda}


class _Engine:
    """One vLLM engine (in-process) for one module config; reused while ops share it."""

    def __init__(self, key: str, b: _Build, split: dict | None):
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        from vllm import LLM
        self.key = key
        self.dir = tempfile.mkdtemp(prefix="opx-attn-")
        Path(self.dir, "config.json").write_text(json.dumps(b.config))
        vllm_engine.launch()
        kwargs, self.recipe = vllm_engine.engine_args(
            split, without=b.without, eager=True, defaults={"max_num_batched_tokens": _MAX_BATCHED_TOKENS},
            model=self.dir, load_format="dummy", skip_tokenizer_init=True, enforce_eager=True,
            enable_prefix_caching=False, max_num_seqs=_MAX_SEQS, kv_cache_memory_bytes=_kv_bytes(),
            gpu_memory_utilization=_GPU_UTIL, **b.engine)
        # JIT kernels on first use (the untimed step), not every shape up front, where the
        # image's vLLM has that setting
        from vllm.config.kernel import KernelConfig
        if "enable_jit_warmup" in {f.name for f in fields(KernelConfig)}:
            kernel = kwargs.get("kernel_config")
            if kernel is None or isinstance(kernel, dict):
                kwargs["kernel_config"] = {**(kernel or {}), "enable_jit_warmup": False}
            else:
                kernel.enable_jit_warmup = False
        # the engine runs eagerly: the full graphs serving would capture, from vLLM
        self.capture = vllm_engine.full_graph_sizes(kwargs, self.recipe["compilation_config"])
        self.reqs: list = []
        self.n = 0
        try:
            self.llm = LLM(**kwargs)
            core = self.llm.llm_engine.engine_core.engine_core
            self.runner = core.model_executor.driver_worker.worker.model_runner
            self.kvm = core.scheduler.kv_cache_manager
            _fill_caches(self.runner)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        # keep the process group for the next engine
        from vllm.distributed.parallel_state import destroy_model_parallel
        try:
            self.release()
            if hasattr(self, "llm"):
                self.llm.llm_engine.engine_core.shutdown()
        except Exception as e:  # noqa: BLE001 - teardown is best effort
            print(f"[vllm.attention] engine shutdown: {type(e).__name__}: {e}", file=sys.stderr)
        for k in ("llm", "runner", "kvm"):
            self.__dict__.pop(k, None)
        destroy_model_parallel()
        gc.collect()
        torch.cuda.empty_cache()

    def _output(self, new: list, scheduled: dict, finished: set):
        from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
        return SchedulerOutput(
            scheduled_new_reqs=new, scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens=scheduled, total_num_scheduled_tokens=sum(scheduled.values()),
            scheduled_spec_decode_tokens={}, scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0] * len(self.runner.kv_cache_config.kv_cache_groups),
            finished_req_ids=finished, free_encoder_mm_hashes=[])

    def _configured(self):
        from vllm.config import set_current_vllm_config
        return set_current_vllm_config(self.runner.vllm_config)

    def release(self) -> None:
        """Finish the previous op's requests in the runner and free their blocks."""
        if not self.reqs or not hasattr(self, "runner"):
            return
        with self._configured():
            self.runner.execute_model(self._output([], {}, {r.request_id for r in self.reqs}))
        for r in self.reqs:
            self.kvm.free(r)
        self.reqs = []

    def step(self, batch: dict, path: str) -> dict:
        """Schedule the batch; return the model's call to the module at path, with its forward context."""
        from vllm import SamplingParams
        from vllm.forward_context import get_forward_context
        from vllm.v1.core.sched.output import NewRequestData
        from vllm.v1.request import Request
        self.release()
        budget = self.runner.vllm_config.scheduler_config.max_num_batched_tokens
        if sum(g["count"] * g["q"] for g in batch["groups"]) > budget:
            raise UnsupportedOpError(f"the batch schedules more than max-num-batched-tokens={budget} new tokens")
        rng = random.Random(batch.get("seed", 0))
        for g in batch["groups"]:
            for _ in range(g["count"]):
                ctx = g["ctx"] if isinstance(g["ctx"], int) else rng.randint(g["ctx"]["min"], g["ctx"]["max"])
                self.n += 1
                r = Request(f"opx{self.n}", [0] * (ctx + g["q"]), SamplingParams(max_tokens=1), None)
                # as the scheduler allocates computed ctx: every block (full), window only (sliding)
                r.num_computed_tokens = ctx
                self.reqs.append(r)
                if self.kvm.allocate_slots(r, g["q"]) is None:
                    raise UnsupportedOpError(f"the batch needs more KV cache than {_kv_bytes() >> 30} GiB")
        blocks = [self.kvm.get_block_ids(r.request_id) for r in self.reqs]
        if batch.get("pages", "contiguous") == "shuffled":
            blocks = _shuffle(blocks, rng)
        new = [NewRequestData.from_request(r, b, r._all_token_ids) for r, b in zip(self.reqs, blocks)]
        module = next(m for n, m in self.runner.model.named_modules() if n.endswith(path))
        forward = module.forward
        seen: dict = {"forward": forward}

        def capture(*args, **kwargs):
            if "fc" not in seen:
                seen.update(args=args, kwargs=kwargs, fc=get_forward_context())
            return forward(*args, **kwargs)

        module.forward = capture
        try:
            with self._configured():
                self.runner.execute_model(self._output(new, {r.request_id: r.num_tokens - r.num_computed_tokens
                                                             for r in self.reqs}, set()))
        finally:
            module.forward = forward
            self.runner.execute_model_state = None
        if "fc" not in seen:
            raise RuntimeError("the model never called the attention module")
        return seen


def _kv_bytes() -> int:
    return int(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory * _KV_FRACTION)


def _shuffle(blocks: list, rng: random.Random) -> list:
    """Permute each KV cache group's blocks across the batch's requests."""
    out = [list(map(list, b)) for b in blocks]
    for gi in range(len(blocks[0])):
        pool = [x for b in blocks for x in b[gi]]
        rng.shuffle(pool)
        it = iter(pool)
        for b in out:
            b[gi] = [next(it) for _ in b[gi]]
    return [tuple(b) for b in out]


def _fill_caches(runner) -> None:
    """Random bytes below 0x40 in every KV cache/state buffer: small and finite as fp32, bf16,
    fp8 e4m3 / ue8m0 and int8 alike (caches are multi-dtype views over raw bytes)."""
    ctx = runner.vllm_config.compilation_config.static_forward_context
    seen: set[int] = set()
    with torch.no_grad():
        for layer in ctx.values():
            kc = getattr(layer, "kv_cache", None)
            for t in kc if isinstance(kc, (list, tuple)) else [kc]:
                if not isinstance(t, torch.Tensor):
                    continue
                storage = t.untyped_storage()
                if storage.data_ptr() in seen:
                    continue
                seen.add(storage.data_ptr())
                raw = torch.empty(0, dtype=torch.uint8, device=t.device).set_(storage)
                for c in raw.split(1 << 30):
                    c.random_(0, 0x40)


_ENGINE: _Engine | None = None


def _engine(b: _Build, split: dict | None) -> _Engine:
    global _ENGINE
    key = json.dumps([b.family, b.config, b.engine, split], sort_keys=True)
    if _ENGINE is not None and _ENGINE.key == key:
        return _ENGINE
    if _ENGINE is not None:
        _ENGINE.close()
        _ENGINE = None
    _ENGINE = _Engine(key, b, split)
    return _ENGINE


def _backends(runner) -> dict[str, str]:
    return {", ".join(g.layer_names): getattr(g.backend, "__name__", type(g.backend).__name__)
            for gs in runner.attn_groups for g in gs}


def _prepare(op: Op) -> dict:
    a = op.args
    if a.get("selection", "natural") != "natural":
        raise UnsupportedOpError("forced token selection is not wired yet")
    b = _BUILDERS[op.type](op)
    q = _quant(op, b.family)
    if q is not None:
        b.config["quantization_config"] = q
    kv = a.get("kv_cache_dtype")
    if kv is not None:
        b.engine["kv_cache_dtype"] = _KV_DTYPES[kv]
    split = a.get("parallel")
    with ranks.together("build this engine"):
        try:
            eng = _engine(b, split)
        # vLLM's startup (build, profiling, warmup) failing on this config
        except (ValueError, NotImplementedError, AssertionError, RuntimeError) as e:
            if vllm_linear._is_fault(e):
                raise
            raise UnsupportedOpError(f"vLLM rejected the {b.family} module: {type(e).__name__}: {e}"[:400]) from e
    with ranks.together("schedule this batch"):
        try:
            seen = eng.step(a["batch"], b.module)
        except (ValueError, NotImplementedError, AssertionError) as e:
            if vllm_linear._is_fault(e):
                raise
            raise UnsupportedOpError(f"vLLM rejected the batch: {type(e).__name__}: {e}"[:400]) from e
    ctx = {"engine": eng, **seen,
           "meta": {"vllm_engine": vllm_engine.meta(), "vllm_family": b.family, "vllm_repo": _MODELS[b.family]["repo"], "vllm_module": b.module,
                    "vllm_backends": _backends(eng.runner),
                    "vllm_attn_metadata": {k: type(v).__name__ for k, v in (seen["fc"].attn_metadata or {}).items()},
                    "kv_cache_groups": [
                        {"spec": type(g.kv_cache_spec).__name__, "block_size": g.kv_cache_spec.block_size,
                         "dtype": str(getattr(g.kv_cache_spec, "dtype", "")).removeprefix("torch.")}
                        for g in eng.runner.kv_cache_config.kv_cache_groups]}}
    _kernel(ctx)
    torch.cuda.synchronize()
    return ctx


def _replaying(ctx: dict):
    """The model runner step's context: inference mode, vLLM's current config, forward context."""
    import contextlib

    from vllm.config import set_current_vllm_config
    from vllm.forward_context import override_forward_context
    stack = contextlib.ExitStack()
    stack.enter_context(torch.inference_mode())
    stack.enter_context(set_current_vllm_config(ctx["engine"].runner.vllm_config))
    stack.enter_context(override_forward_context(ctx["fc"]))
    return stack


def _kernel(ctx: dict) -> None:
    with _replaying(ctx):
        ctx["out"] = ctx["forward"](*ctx["args"], **ctx["kwargs"])


def _cudagraph(ctx: dict) -> bool:
    """vLLM would replay this as a full CUDA graph: uniform decode within capture sizes, backend support."""
    with ctx["engine"]._configured():  # backends read vLLM's current config to answer
        return _full_graph(ctx["engine"])


def _full_graph(eng: _Engine) -> bool:
    from vllm.v1.attention.backend import AttentionCGSupport
    qs = {r.num_tokens - r.num_computed_tokens for r in eng.reqs}
    if len(qs) != 1:
        return False
    q = qs.pop()
    if not eng.capture or q * len(eng.reqs) > eng.capture[-1]:
        return False
    need = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE if q == 1 else AttentionCGSupport.UNIFORM_BATCH
    for gs in eng.runner.attn_groups:
        for g in gs:
            support = g.backend.get_builder_cls().get_cudagraph_support(eng.runner.vllm_config, g.kv_cache_spec)
            if support.value < need.value:
                return False
    return True


def _ready_upstream(ctx: dict) -> None:
    """A top-k reusing layer waits on events its source layer records in the same graph;
    captured alone, record them here (already satisfied) instead."""
    stream = torch.cuda.current_stream()
    for m in ctx["engine"].runner.model.modules():
        group = getattr(getattr(m, "impl", None), "index_group", None) or getattr(m, "index_group", None)
        for name in ("logical_topk_ready", "physical_topk_ready"):
            ev = getattr(group, name, None)
            if ev is not None:
                ev.record(stream)


def _launcher(ctx: dict):
    from vllm.distributed.parallel_state import graph_capture
    eager = (lambda: _kernel(ctx)), False
    if not _cudagraph(ctx):
        return eager
    err = None
    try:
        with _replaying(ctx):
            for _ in range(2):
                ctx["forward"](*ctx["args"], **ctx["kwargs"])
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with graph_capture(torch.device("cuda", torch.cuda.current_device())) as gc_, \
                    torch.cuda.graph(g, pool=torch.cuda.graph_pool_handle(), stream=gc_.stream):
                _ready_upstream(ctx)
                ctx["graph_out"] = ctx["forward"](*ctx["args"], **ctx["kwargs"])
        torch.cuda.synchronize()
    except Exception as e:  # fall back to eager, as the linear launcher does
        if vllm_linear._is_fault(e):
            raise
        torch.cuda.synchronize()
        err = e
    if not ranks.agree(err is None):  # every rank replays, or every rank runs eagerly
        print(f"[vllm.attention] CUDA-graph capture failed, timing eagerly: {type(err).__name__}: {err}"[:300]
              if err else "[vllm.attention] another rank's CUDA-graph capture failed, timing eagerly",
              file=sys.stderr)
        return eager
    ctx["graph"] = g
    return g.replay, True


IMPLS = [BackendImpl(op_type=t, prepare=_prepare, kernel=_kernel, launcher=_launcher) for t in _BUILDERS]
