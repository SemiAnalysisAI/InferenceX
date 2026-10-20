"""Module markers: every launch carries the module that issued it."""

import re
import threading
import weakref

from . import runtime

_names = weakref.WeakKeyDictionary()
_hooks = []
_tls = threading.local()
_compiled_names = {}


def register_root(prefix, root):
    if root is None or not hasattr(root, "named_modules"):
        return
    for name, mod in root.named_modules(prefix=prefix):
        _names.setdefault(mod, name.replace("._orig_mod", ""))
    _mark_direct_calls(root, methods=True)
    for mod in root.children():
        _mark_direct_calls(mod)
    _mark_layer_methods(root)
    _compiled_names.clear()
    _resolved.clear()


_resolved = {}


def _marked_call(self, call, args):
    """Run call() as module `self`'s, unless `self` is already the innermost marked module."""
    import torch

    if torch.compiler.is_compiling() or not _hooks:
        return call()
    stack = getattr(_tls, "stack", None)
    if stack is None:
        stack = _tls.stack = []
    if stack and stack[-1][0] is self:
        return call()
    rf = torch.autograd.profiler.record_function(
        f"infx_mod#{_names.get(self) or type(self).__name__}#{tensor_signature(args)}")
    rf.__enter__()
    entry = (self, rf)
    stack.append(entry)
    try:
        return call()
    finally:
        if entry in stack:
            while stack:
                top = stack.pop()
                top[1].__exit__(None, None, None)
                if top is entry:
                    break


def _marking(fn):
    """`fn` (a model class's own forward or method) marked as its module when called directly."""
    import functools

    @functools.wraps(fn)
    def method(self, *args, **kwargs):
        return _marked_call(self, lambda: fn(self, *args, **kwargs), args)

    method._infx_marking = True
    return method


def _own_methods(cls, private=False):
    """Functions a model class defines itself (not nn.Module's), e.g. compute_logits; with `private`, its single-underscore helpers too."""
    import types

    import torch

    base = set(dir(torch.nn.Module))
    return {name for name, obj in vars(cls).items()
            if isinstance(obj, types.FunctionType) and name not in base and not name.startswith("__")
            and (private or not name.startswith("_"))}


def _mark_layer_methods(root):
    """A layer stack's layers (a ModuleList's entries): models call their methods directly (DSV4: forward_hc_pre_from_prev, _hc_post_with_combine) past the module hooks, which would leave that work on the model root; mark those calls as the layer."""
    import types

    import torch

    done = set()
    for mod in root.modules():
        if not isinstance(mod, torch.nn.ModuleList):
            continue
        for layer in mod:
            for cls in type(layer).__mro__:
                if cls in done or cls is torch.nn.Module or not issubclass(cls, torch.nn.Module) \
                        or cls.__module__.startswith("torch."):
                    continue
                done.add(cls)
                for name in _own_methods(cls, private=True):
                    fn = vars(cls).get(name)
                    if isinstance(fn, types.FunctionType) and not getattr(fn, "_infx_marking", False):
                        try:
                            setattr(cls, name, _marking(fn))
                        except Exception:
                            pass


def _mark_direct_calls(module, methods=False):
    """Engines call a model's forward directly (SGLang: model.forward, then model.model.forward), and its other methods (vLLM: compute_logits; a speculative head's logits step), which skips the module hooks; mark such calls. The module's classes are patched, not the instance: engines copy models (SGLang's breakable prefill graphs), and an instance attribute would carry the original module into the copy."""
    import types

    import torch

    if isinstance(module, (torch.nn.ModuleList, torch.nn.ModuleDict)) or "forward" in vars(module):
        return
    for cls in type(module).__mro__:
        if cls is torch.nn.Module or not issubclass(cls, torch.nn.Module) or cls.__module__.startswith("torch."):
            continue
        names = (_own_methods(cls) if methods else set()) | {"forward"}
        for name in names:
            fn = vars(cls).get(name)
            if not isinstance(fn, types.FunctionType) or getattr(fn, "_infx_marking", False):
                continue
            try:
                setattr(cls, name, _marking(fn))
            except Exception:
                pass


def resolve(name, root="model"):
    """The registered qualified name ending with `name` (an engine's own layer prefix), shortest first."""
    key = (root, name)
    if key not in _resolved:
        suffix = f".{name}"
        matches = sorted((n for n in list(_names.values())
                          if (n.startswith(root + ".") or not root) and (n.endswith(suffix) or n == name)), key=len)
        _resolved[key] = matches[0] if matches else name
    return _resolved[key]


class marked:
    """Mark the enclosed launches as module `name`'s (for code that runs a module without calling it)."""

    def __init__(self, name, args=()):
        self.name, self.args, self.rf = name, args, None

    def __enter__(self):
        if runtime.profiling() and not runtime.compiling():
            try:
                import torch

                self.rf = torch.autograd.profiler.record_function(
                    f"infx_mod#{self.name}#{tensor_signature(self.args)}")
                self.rf.__enter__()
            except Exception:
                self.rf = None
        return self

    def __exit__(self, *exc):
        if self.rf is not None:
            try:
                self.rf.__exit__(None, None, None)
            except Exception:
                pass
        return False


def tensor_signature(args):
    """'7x7168:bfloat16;1x2:int64' for the tensors among a call's (one-level nested) args."""
    import torch

    sig = []
    for arg in args:
        items = arg if isinstance(arg, (list, tuple)) else (arg,)
        for item in items:
            if isinstance(item, torch.Tensor):
                dims = "x".join(str(d) for d in item.shape)
                sig.append(f"{dims}:{str(item.dtype).removeprefix('torch.')}")
    return ";".join(sig)


def _pre_hook(module, args):
    if runtime.compiling():
        return
    try:
        import torch

        name = _names.get(module) or type(module).__name__
        rf = torch.autograd.profiler.record_function(f"infx_mod#{name}#{tensor_signature(args)}")
        rf.__enter__()
        stack = getattr(_tls, "stack", None)
        if stack is None:
            stack = _tls.stack = []
        stack.append((module, rf))
    except Exception:
        pass


def _post_hook(module, args, output):
    if runtime.compiling():
        return
    stack = getattr(_tls, "stack", None)
    while stack:
        mod, rf = stack.pop()
        try:
            rf.__exit__(None, None, None)
        except Exception:
            pass
        if mod is module:
            break


def enable():
    if _hooks:
        return
    from torch.nn.modules import module as nn_module

    _hooks.append(nn_module.register_module_forward_pre_hook(_pre_hook))
    _hooks.append(nn_module.register_module_forward_hook(_post_hook, always_call=True))


def disable():
    while _hooks:
        _hooks.pop().remove()
    stack = getattr(_tls, "stack", None)
    while stack:
        try:
            stack.pop()[1].__exit__(None, None, None)
        except Exception:
            pass



_PLACEHOLDER_MODULE = re.compile(r"^l_self_(?:modules_)?(.*?)_(?:parameters|buffers)_")
_compile_inputs = threading.local()


def _relative_module_path(fqn):
    """"L['self'].layers[3].self_attn" -> "layers.3.self_attn"."""
    fqn = re.sub(r"^L\['self'\]\.?", "", fqn)
    fqn = re.sub(r"_modules\['([^']+)'\]", r"\1", fqn)
    fqn = re.sub(r"\[(\d+)\]", r".\1", fqn)
    fqn = re.sub(r"\['([^']+)'\]", r".\1", fqn)
    return fqn.strip(".").replace("..", ".")


def _stack_modules(node):
    paths = []
    for snode in node.get_nodes():
        if snode.node is None:
            continue
        for origin in snode.node.get_origins():
            stack = origin.meta.get("nn_module_stack")
            if stack:
                paths.append(_relative_module_path(list(stack.values())[-1][0]).split("."))
    return paths


def _weight_modules(node, graph):
    """Module paths of the weights (graph inputs) the node reads."""
    stack = getattr(_compile_inputs, "stack", None)
    if not stack:
        return []
    placeholders = [n.name for n in graph.graph.find_nodes(op="placeholder")]
    if len(placeholders) != len(stack[-1]):
        return []
    dynamo_name = dict(zip(placeholders, stack[-1]))
    paths = []
    for dep in node.read_writes.reads:
        match = _PLACEHOLDER_MODULE.match(dynamo_name.get(dep.name, ""))
        if match:
            paths.append(match.group(1).split("_modules_"))
    return paths


def _common_module(paths):
    if not paths:
        return None
    common = paths[0]
    for parts in paths[1:]:
        k = 0
        while k < min(len(common), len(parts)) and common[k] == parts[k]:
            k += 1
        common = common[:k]
    return ".".join(common)


def patch_compile_fx(module):
    orig = getattr(module, "compile_fx", None)
    if orig is None or getattr(orig, "_infx_patched", False):
        return

    def compile_fx(model_, example_inputs_, *args, **kwargs):
        graph = getattr(model_, "graph", None)
        names = [n.name for n in graph.find_nodes(op="placeholder")] if graph is not None else []
        stack = getattr(_compile_inputs, "stack", None)
        if stack is None:
            stack = _compile_inputs.stack = []
        stack.append(names)
        try:
            return orig(model_, example_inputs_, *args, **kwargs)
        finally:
            stack.pop()

    compile_fx._infx_patched = True
    module.compile_fx = compile_fx


def patch_inductor_scheduler(module):
    cls = module.Scheduler
    if getattr(cls, "_infx_patched", False) or not hasattr(cls, "enter_context"):
        return
    orig = cls.enter_context

    def enter_context(self, node):
        orig(self, node)
        try:
            from torch._inductor.codegen.wrapper import PythonWrapperCodegen
            from torch._inductor.virtualized import V

            wrapper = V.graph.wrapper_code
            if getattr(V.graph, "cpp_wrapper", False) or not isinstance(wrapper, PythonWrapperCodegen):
                runtime.note_once("compile", "module_line_skipped", type(wrapper).__name__)
                return
            stacked = _stack_modules(node)
            source = "nn_module_stack" if stacked else "weights"
            path = _common_module(stacked or _weight_modules(node, V.graph))
            if path is None:
                path, source = "", "none"
            if getattr(wrapper, "_infx_module", None) != path:
                wrapper.writeline(f"__import__('infx_prof.modules').modules.compiled_module({path!r})")
                wrapper._infx_module = path
            runtime.note_once("compile", "module_source", f"{type(wrapper).__name__}:{source}")
        except Exception:
            runtime.write_error("inductor module line")

    cls.enter_context = enter_context
    cls._infx_patched = True


def set_compiled_root(root):
    """'model' or 'draft': whose modules the compiled code about to run belongs to."""
    _tls.compiled_root = root


def compiled_module(path):
    """Called from Inductor wrapper code: mark the following kernels as `path`'s."""
    if not runtime.profiling():
        return
    import torch

    current = getattr(_tls, "compiled", None)
    if current is not None and current[0] == path:
        return
    close_compiled_module()
    if not path:
        return
    root = getattr(_tls, "compiled_root", "model")
    name = _compiled_names.get((root, path))
    if name is None:
        suffix = f".{path}"
        matches = sorted((n for n in list(_names.values()) if n.startswith(root + ".") and n.endswith(suffix)),
                         key=len)
        name = _compiled_names[(root, path)] = matches[0] if matches else f"{root}:{path}"
    rf = torch.autograd.profiler.record_function(f"infx_mod#{name}#")
    rf.__enter__()
    _tls.compiled = (path, rf)


def close_compiled_module():
    current = getattr(_tls, "compiled", None)
    if current is not None:
        _tls.compiled = None
        try:
            current[1].__exit__(None, None, None)
        except Exception:
            pass


def install(registry):
    registry.on("torch._inductor.scheduler", patch_inductor_scheduler)
    registry.on("torch._inductor.compile_fx", patch_compile_fx)
