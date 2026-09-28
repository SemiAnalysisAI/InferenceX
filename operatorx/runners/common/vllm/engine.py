"""The vLLM engine a case runs under, configured as `vllm serve` configures it.

engine_args(): the InferenceX recipe's serve arguments (OPERATORX_ENGINE_ARGS, less those
recipes.serve_argv does not apply) through vLLM's own parser, the op's parallel split and
the backend's overrides, with the external_launcher executor (one srun task per rank,
joined through env://). Layer ops enter that config once per process (context());
attention builds an LLM from it.
"""
from __future__ import annotations

import dataclasses
import json
import os
import tempfile

import torch

from operatorx.core import UnsupportedOpError, parallel

# for layer cases without a checkpoint config; MoE, so vLLM builds the expert-parallel group
_STAND_IN = {"architectures": ["MixtralForCausalLM"], "model_type": "mixtral", "hidden_size": 4096,
             "intermediate_size": 512, "num_attention_heads": 64, "num_key_value_heads": 8,
             "num_hidden_layers": 1, "num_local_experts": 8, "num_experts_per_tok": 2,
             "vocab_size": 1024, "max_position_embeddings": 65536, "torch_dtype": "bfloat16"}

_STATE: dict | None = None
_META: dict = {}


def launch() -> None:
    """The env:// rendezvous external_launcher reads (one device by default)."""
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 1000))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def recipe_kwargs(without: tuple = ()) -> tuple[dict, dict]:
    """The recipe's server arguments as EngineArgs kwargs (non-default fields only), and the
    attention and compilation configs apart from them: attention is merged with the op's, and
    an eager engine applies only the compilation config's custom-op selection."""
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    from operatorx.recipes import serve_argv
    args = {k: v for k, v in json.loads(os.environ.get("OPERATORX_ENGINE_ARGS") or "{}").items()
            if k not in without}
    attention = json.loads(args.pop("attention-config", None) or "{}")
    compilation = json.loads(args.pop("compilation-config", None) or "{}")
    if args.get("max-cudagraph-capture-size"):  # moved: vLLM rejects it in both places
        compilation.setdefault("max_cudagraph_capture_size", int(args.pop("max-cudagraph-capture-size")))
    info = {"attention_config": attention, "compilation_config": compilation}
    if not args:
        return {}, info
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    base = vars(parser.parse_known_args(["--model", "m"])[0])
    ns, frontend_only = parser.parse_known_args(["--model", "m", *serve_argv(args)])
    fields = {f.name for f in dataclasses.fields(EngineArgs)} - {"model"}
    kwargs = {k: v for k, v in vars(ns).items() if k in fields and v != base.get(k)}
    info["ignored_args"] = frontend_only
    return kwargs, info


def engine_args(split: dict | None, without: tuple = (), eager: bool = False, defaults: dict | None = None,
                **overrides) -> tuple[dict, dict]:
    """EngineArgs kwargs for a case: defaults, the recipe's, the split, the overrides (an
    attention_config override merges into the recipe's); plus the recipe's attention and
    compilation configs."""
    kw, info = recipe_kwargs(without)
    kw = {**(defaults or {}), **kw}
    compilation = info["compilation_config"]
    if not eager and compilation:
        kw["compilation_config"] = compilation
    elif compilation.get("custom_ops"):  # custom-op selection holds without compilation
        kw["compilation_config"] = {"custom_ops": compilation["custom_ops"]}
    key = parallel.normalize(split)
    kw.update(tensor_parallel_size=key["tp"], data_parallel_size=key["dp"],
              decode_context_parallel_size=key["dcp"], enable_expert_parallel=key["ep"] > 1,
              distributed_executor_backend="external_launcher")
    attention = {**info["attention_config"], **overrides.pop("attention_config", {})}
    kw.update(overrides)
    if attention:
        kw["attention_config"] = attention
    _META.update(engine_args=kw, ignored_args=info.get("ignored_args", []))
    return kw, info


def full_graph_sizes(kw: dict, compilation: dict) -> list[int]:
    """The sizes vLLM captures full CUDA graphs at for these arguments and the recipe's
    compilation config, as a non-eager engine would; none when it captures no full graphs."""
    from vllm.engine.arg_utils import EngineArgs
    cfg = EngineArgs(**{**kw, "enforce_eager": False, "compilation_config": compilation}).create_engine_config()
    cc = cfg.compilation_config
    return sorted(cc.cudagraph_capture_sizes or []) if cc.cudagraph_mode.has_full_cudagraphs() else []


def _model_dir() -> str:
    path = os.environ.get("OPERATORX_MODEL_CONFIG")
    if path:
        return path
    path = tempfile.mkdtemp(prefix="opx_vllm_cfg_")
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(_STAND_IN, f)
    return path


def context(split: dict | None = None):
    """Enter vLLM's config and the worker's distributed environment, once per process."""
    global _STATE
    key = parallel.normalize(split)
    if _STATE is not None:
        if _STATE["split"] != key:
            raise UnsupportedOpError(f"this process runs parallel={_STATE['split']}, not {key}")
        return _STATE["config"]
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.worker.gpu_worker import init_worker_distributed_environment
    from vllm.v1.worker.workspace import init_workspace_manager

    launch()
    kw, _ = engine_args(split, model=_model_dir(), skip_tokenizer_init=True, load_format="dummy")
    cfg = EngineArgs(**kw).create_engine_config()
    ctx = set_current_vllm_config(cfg)
    ctx.__enter__()
    local_rank = int(os.environ["LOCAL_RANK"])
    init_worker_distributed_environment(cfg, int(os.environ["RANK"]), "env://", local_rank)
    init_workspace_manager(torch.device("cuda", local_rank))
    _STATE = {"split": key, "config": cfg, "ctx": ctx}
    return cfg


def meta() -> dict:
    return {k: json.loads(json.dumps(v, default=str)) for k, v in _META.items()}
