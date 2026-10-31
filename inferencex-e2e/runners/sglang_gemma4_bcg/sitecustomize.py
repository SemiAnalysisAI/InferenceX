"""Let SGLang's prefill CUDA graph (--cuda-graph-backend-prefill breakable|full) capture Gemma-4.

SGLang auto-disables the prefill graph for multimodal archs, so every Gemma-4 extend step runs
eager (~88 ms forward for a 22-token suffix at TP4). Forcing the backend crashes at capture: the
runner calls Gemma4TextModel.forward with input_ids AND its static input_embeds slot (refilled from
input_ids on replay), and the model accepts exactly one. Without per-layer inputs
(hidden_size_per_layer_input 0, e.g. 31B) the ids add nothing, so drop them when embeds are given.

Loaded via PYTHONPATH (inherited by the scheduler subprocesses); patches the module on import.
"""
import functools
import importlib.abc
import importlib.util
import sys

TARGET = "sglang.srt.models.gemma4_causal"


def _patch(mod):
    cls = mod.Gemma4TextModel
    orig = cls.forward

    @functools.wraps(orig)
    def forward(self, input_ids, positions, forward_batch, input_embeds=None, *args, **kwargs):
        if input_ids is not None and input_embeds is not None and not self.hidden_size_per_layer_input:
            input_ids = None
        return orig(self, input_ids, positions, forward_batch, input_embeds, *args, **kwargs)

    cls.forward = forward
    print(f"[gemma4_bcg_patch] patched {TARGET}.Gemma4TextModel.forward", file=sys.stderr, flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


sys.meta_path.insert(0, _Finder())
