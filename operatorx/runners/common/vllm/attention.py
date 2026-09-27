"""Attention modules through vLLM's own model code, KV cache and scheduling.

An op picks the checkpoint family whose module it describes (attention_models.json holds
each family's config.json), overrides the module's sizes, cuts the model to the layers
that module needs, and shrinks the MLP. vLLM builds that model with dummy weights,
allocates and lays out its KV cache, and picks the attention backend. The op's batch is
scheduled through vLLM's model runner as requests whose ctx tokens are already computed
(the cache holds random, format-valid data); the timed call is the module's forward,
with the arguments and forward context the model's own decoder layer gives it.
"""
from __future__ import annotations

import copy
import gc
import json
import os
import random
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import torch

from operatorx.core import BackendImpl, Op, UnsupportedOpError
from operatorx.ops import attention as schema
from operatorx.runners.common.vllm import linear as vllm_linear

_MODELS = json.loads((Path(__file__).with_name("attention_models.json")).read_text())
_MLP = 256  # width of the (untimed) MLP in the cut-down model
_VOCAB = 1024
_MAX_BATCHED_TOKENS = 65536
_MAX_SEQS = 1024
_KV_BYTES = int(os.environ.get("OPERATORX_ATTN_KV_BYTES", str(48 << 30)))
# startup headroom check only; the KV cache is sized by _KV_BYTES
_GPU_UTIL = float(os.environ.get("OPERATORX_ATTN_GPU_UTIL", "0.6"))
_KV_DTYPES = {"auto": "auto", "bf16": "bfloat16", "fp8": "fp8", "fp8_ds_mla": "fp8_ds_mla"}


@dataclass
class _Build:
    family: str
    config: dict
    module: str  # module path suffix, e.g. "layers.0.self_attn"
    engine: dict = field(default_factory=dict)


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


def _quant(op: Op, names: tuple[str, ...]) -> dict | None:
    """The checkpoint quantization_config for the op's projection operands."""
    proj = op.args.get("proj") or {}
    if not proj:
        return None
    pairs = {json.dumps(v, sort_keys=True) for v in proj.values()}
    if len(pairs) > 1 or set(proj) != set(names):
        raise UnsupportedOpError("projections with different operands are not wired; give all or none")
    pair = next(iter(proj.values()))
    scheme = vllm_linear._scheme(pair["a"], pair["b"])
    if not scheme:
        raise UnsupportedOpError(f"no vLLM quantization config for projections {pair}")
    return scheme[1]


def _build_mla(op: Op) -> _Build:
    a = op.args
    if not a.get("rope", True) or a.get("gate"):
        raise UnsupportedOpError("MLA without RoPE or with an output gate (Kimi-K3) is not wired yet")
    c = _family("deepseek_v3")
    c.update(hidden_size=a["hidden"], num_attention_heads=a["heads"], num_key_value_heads=a["heads"],
             q_lora_rank=a["q_lora_rank"], kv_lora_rank=a["kv_lora_rank"], qk_nope_head_dim=a["nope"],
             qk_rope_head_dim=a["rope_dim"], v_head_dim=a["v"], rope_theta=a["rope_theta"],
             rope_scaling=_yarn(a.get("rope_scaling")), num_hidden_layers=1, first_k_dense_replace=1,
             intermediate_size=_MLP, num_nextn_predict_layers=0, vocab_size=_VOCAB)
    return _Build("deepseek_v3", c, "layers.0.self_attn", {})


def _qwen35(a: dict, full: dict, linear: dict, target: str) -> _Build:
    """Qwen3.5: one Gated DeltaNet layer then one gated full-attention layer, so the KV
    cache has the hybrid layout serving uses."""
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
    # MiniMax-M3 serves with 128-token pages (its sparse layers need them), dense layers included
    return _Build("minimax_m3", c, "layers.0.self_attn", {"language_model_only": True, "block_size": 128})


def _build_gdn(op: Op) -> _Build:
    a = op.args
    if a.get("norm_act", "silu") != "silu":
        raise UnsupportedOpError("the sigmoid-gated GDN (Qwen3.8) is not wired yet")
    linear = dict(hidden_size=a["hidden"], linear_num_key_heads=a["qk_heads"], linear_num_value_heads=a["v_heads"],
                  linear_key_head_dim=a["head_dim"], linear_value_head_dim=a["head_dim"],
                  linear_conv_kernel_dim=a["conv_kernel"],
                  mamba_ssm_dtype={"fp32": "float32", "bf16": "bfloat16"}[a.get("state_dtype", "fp32")])
    b = _qwen35(a, {}, linear, "layers.0.linear_attn")
    b.engine["mamba_ssm_cache_dtype"] = linear["mamba_ssm_dtype"]
    return b


_BUILDERS = {"mla": (_build_mla, schema.MLA_PROJ[:-1]), "gqa": (_build_gqa, schema.GQA_PROJ),
             "gdn": (_build_gdn, schema.GDN_PROJ)}


class _Engine:
    """One vLLM engine (in-process) for one module config; reused while ops share it."""

    def __init__(self, key: str, b: _Build):
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        from vllm import LLM
        self.key = key
        self.dir = tempfile.mkdtemp(prefix="opx-attn-")
        Path(self.dir, "config.json").write_text(json.dumps(b.config))
        kwargs = dict(model=self.dir, load_format="dummy", skip_tokenizer_init=True, enforce_eager=True,
                      enable_prefix_caching=False, max_num_seqs=_MAX_SEQS,
                      max_num_batched_tokens=_MAX_BATCHED_TOKENS, kv_cache_memory_bytes=_KV_BYTES,
                      gpu_memory_utilization=_GPU_UTIL)
        kwargs.update(b.engine)
        self.reqs: list = []
        self.n = 0
        try:
            self.llm = LLM(**kwargs)
            core = self.llm.llm_engine.engine_core.engine_core
            self.runner = core.model_executor.driver_worker.worker.model_runner
            self.kvm = core.scheduler.kv_cache_manager
            self.module = next(m for n, m in self.runner.model.named_modules() if n.endswith(b.module))
            self.forward = self.module.forward
            _fill_caches(self.runner)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
        try:
            self.release()
            self.llm.llm_engine.engine_core.shutdown()
        except Exception as e:  # noqa: BLE001 - teardown is best effort
            print(f"[vllm.attention] engine shutdown: {type(e).__name__}: {e}", file=sys.stderr)
        for k in ("llm", "runner", "kvm", "module", "forward"):
            self.__dict__.pop(k, None)
        cleanup_dist_env_and_memory()
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

    def release(self) -> None:
        """Finish the previous op's requests in the runner and free their blocks."""
        if not self.reqs:
            return
        self.runner.execute_model(self._output([], {}, {r.request_id for r in self.reqs}))
        for r in self.reqs:
            self.kvm.free(r)
        self.reqs = []

    def step(self, batch: dict) -> dict:
        """Schedule the batch; return the module call the model made, and its context."""
        from vllm import SamplingParams
        from vllm.forward_context import get_forward_context
        from vllm.v1.core.sched.output import NewRequestData
        from vllm.v1.request import Request
        self.release()
        rng = random.Random(batch.get("seed", 0))
        for g in batch["groups"]:
            for _ in range(g["count"]):
                ctx = g["ctx"] if isinstance(g["ctx"], int) else rng.randint(g["ctx"]["min"], g["ctx"]["max"])
                self.n += 1
                r = Request(f"opx{self.n}", [0] * (ctx + g["q"]), SamplingParams(max_tokens=1), None)
                if self.kvm.allocate_slots(r, ctx + g["q"]) is None:
                    self.reqs.append(r)
                    raise UnsupportedOpError(f"the batch needs more KV cache than {_KV_BYTES >> 30} GiB")
                r.num_computed_tokens = ctx
                self.reqs.append(r)
        blocks = [self.kvm.get_block_ids(r.request_id) for r in self.reqs]
        if batch.get("pages", "contiguous") == "shuffled":
            blocks = _shuffle(blocks, rng)
        new = [NewRequestData.from_request(r, b, r._all_token_ids) for r, b in zip(self.reqs, blocks)]
        seen: dict = {}

        def capture(*args, **kwargs):
            if not seen:
                seen.update(args=args, kwargs=kwargs, fc=get_forward_context())
            return self.forward(*args, **kwargs)

        self.module.forward = capture
        try:
            self.runner.execute_model(self._output(new, {r.request_id: r.num_tokens - r.num_computed_tokens
                                                         for r in self.reqs}, set()))
        finally:
            self.module.forward = self.forward
            self.runner.execute_model_state = None
        if not seen:
            raise RuntimeError("the model never called the attention module")
        return seen


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
    """Random, format-valid contents for every KV cache and state buffer. Caches are
    views (often several dtypes, packed layouts with scales) over raw byte storage; every
    byte is drawn below 0x40, which decodes to a small finite value in fp32, bf16, fp8
    e4m3 / ue8m0 and int8 alike."""
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


def _engine(b: _Build) -> _Engine:
    global _ENGINE
    key = json.dumps([b.family, b.config, b.engine], sort_keys=True)
    if _ENGINE is not None and _ENGINE.key == key:
        return _ENGINE
    if _ENGINE is not None:
        _ENGINE.close()
        _ENGINE = None
    _ENGINE = _Engine(key, b)
    return _ENGINE


def _backends(runner) -> dict[str, str]:
    return {", ".join(g.layer_names): getattr(g.backend, "__name__", type(g.backend).__name__)
            for gs in runner.attn_groups for g in gs}


def _prepare(op: Op) -> dict:
    a = op.args
    if a.get("selection", "natural") != "natural":
        raise UnsupportedOpError("forced token selection is not wired yet")
    build, names = _BUILDERS[op.type]
    b = build(op)
    q = _quant(op, names)
    if q is not None:
        b.config["quantization_config"] = q
    kv = a.get("kv_cache_dtype")
    if kv is not None:
        b.engine["kv_cache_dtype"] = _KV_DTYPES[kv]
    eng = _engine(b)
    seen = eng.step(a["batch"])
    ctx = {"engine": eng, **seen,
           "meta": {"vllm_family": b.family, "vllm_repo": _MODELS[b.family]["repo"],
                    "vllm_backends": _backends(eng.runner),
                    "vllm_attn_metadata": {k: type(v).__name__ for k, v in (seen["fc"].attn_metadata or {}).items()},
                    "kv_cache_groups": [
                        {"spec": type(g.kv_cache_spec).__name__, "block_size": g.kv_cache_spec.block_size,
                         "dtype": str(getattr(g.kv_cache_spec, "dtype", "")).removeprefix("torch.")}
                        for g in eng.runner.kv_cache_config.kv_cache_groups]}}
    _kernel(ctx)
    torch.cuda.synchronize()
    return ctx


def _kernel(ctx: dict) -> None:
    from vllm.forward_context import override_forward_context
    with override_forward_context(ctx["fc"]):
        ctx["out"] = ctx["engine"].forward(*ctx["args"], **ctx["kwargs"])


def _cudagraph(ctx: dict) -> bool:
    """Whether vLLM would replay this batch as a full CUDA graph: a uniform decode batch
    within the capture sizes, on backends that support one."""
    from vllm.v1.attention.backend import AttentionCGSupport
    groups = ctx["engine"].reqs
    qs = {r.num_tokens - r.num_computed_tokens for r in groups}
    if len(qs) != 1:
        return False
    q = qs.pop()
    tokens = q * len(groups)
    from vllm.config import set_current_vllm_config
    with set_current_vllm_config(ctx["engine"].runner.vllm_config):
        top = vllm_linear._capture_sizes()[-1]
    if tokens > top:
        return False
    need = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE if q == 1 else AttentionCGSupport.UNIFORM_BATCH
    runner = ctx["engine"].runner
    for gs in runner.attn_groups:
        for g in gs:
            builder = g.backend.get_builder_cls()
            support = builder.get_cudagraph_support(runner.vllm_config, g.kv_cache_spec)
            if support.value < need.value:
                return False
    return True


def _launcher(ctx: dict):
    eager = (lambda: _kernel(ctx)), False  # noqa: E731
    if not _cudagraph(ctx):
        return eager
    from vllm.forward_context import override_forward_context
    try:
        with override_forward_context(ctx["fc"]):
            for _ in range(2):
                ctx["engine"].forward(*ctx["args"], **ctx["kwargs"])
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=torch.cuda.graph_pool_handle()):
                ctx["graph_out"] = ctx["engine"].forward(*ctx["args"], **ctx["kwargs"])
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001 - fall back to eager, as the linear launcher does
        if vllm_linear._is_fault(e):
            raise
        torch.cuda.synchronize()
        print(f"[vllm.attention] CUDA-graph capture failed, timing eagerly: {type(e).__name__}: {e}"[:300],
              file=sys.stderr)
        return eager
    ctx["graph"] = g
    return g.replay, True


IMPLS = [BackendImpl(op_type=t, prepare=_prepare, kernel=_kernel, launcher=_launcher) for t in _BUILDERS]
