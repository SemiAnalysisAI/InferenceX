"""Python launcher markers."""

import functools
import os
import sys
import types

from . import runtime

CALLER_FRAMES = 16
_WRAP_SKIP_PREFIXES = ("_", "is_", "has_", "should_", "supports")
_OWN_DIR = os.path.dirname(os.path.abspath(__file__))
_caller_roots = ()


def callers(skip):
    """The engine's frames above the caller, outermost first (path:line:function)."""
    frame = sys._getframe(skip)
    chain = []
    while frame is not None and len(chain) < CALLER_FRAMES:
        path = frame.f_code.co_filename
        if not os.path.abspath(path).startswith(_OWN_DIR):
            for root in _caller_roots:
                cut = path.rfind(f"{os.sep}{root}{os.sep}")
                if cut >= 0:
                    chain.append(f"{path[cut + 1:]}:{frame.f_lineno}:{frame.f_code.co_name}")
                    break
        frame = frame.f_back
    return chain[::-1]


def marker(label, skip=3):
    import torch

    return torch.autograd.profiler.record_function(f"infx_py#{label}#{'|'.join(callers(skip))}")


def wrap(fn, label):
    @functools.wraps(fn)
    def launcher(*args, **kwargs):
        if runtime.compiling() or not runtime.profiling():
            return fn(*args, **kwargs)
        with marker(label):
            return fn(*args, **kwargs)

    launcher._infx_launcher = True
    return runtime.keep_signature(launcher, fn)


_COMPILED_FUNCTION_TYPES = {"builtin_function_or_method", "nb_func", "nb_method", "cython_function_or_method"}


def compiled_function(obj):
    return type(obj).__name__ in _COMPILED_FUNCTION_TYPES


def patch_module(module, package=None):
    """Mark every public Python function `module` defines, and its classes' public methods."""
    private = package is not None
    root = os.path.dirname(getattr(sys.modules.get(package), "__file__", None) or "") if private else ""

    def launchable(name, obj):
        skipped = name.startswith("__") or (
            name.startswith(_WRAP_SKIP_PREFIXES) and not (private and name.startswith("_")))
        return (isinstance(obj, types.FunctionType) and obj.__module__ == module.__name__
                and not skipped and not getattr(obj, "_infx_launcher", False))

    def package_code(name, obj):
        if not root or name.startswith("__") or getattr(obj, "_infx_launcher", False):
            return False
        if isinstance(obj, types.FunctionType):
            return obj.__module__ != module.__name__ and obj.__code__.co_filename.startswith(root + os.sep)
        owner_file = getattr(getattr(obj, "__self__", None), "__file__", None) or ""
        return compiled_function(obj) and (
            (getattr(obj, "__module__", None) or "").startswith(package + ".")
            or owner_file.startswith(root + os.sep))

    for name, obj in list(vars(module).items()):
        if launchable(name, obj) or package_code(name, obj):
            setattr(module, name, wrap(obj, f"{module.__name__}.{name}"))
        elif (private and callable(obj) and not isinstance(obj, (type, types.ModuleType))
              and not name.startswith("__") and not getattr(obj, "_infx_launcher", False)):
            kind = type(obj)
            runtime.note_once("launchers", "unwrapped",
                              f"{module.__name__}.{name}: {kind.__module__}.{kind.__qualname__}"
                              f" from {getattr(obj, '__module__', None)}")
        elif isinstance(obj, type) and obj.__module__ == module.__name__ and not name.startswith("_"):
            for attr, method in list(vars(obj).items()):
                label = f"{module.__name__}.{name}.{attr}"
                if launchable(attr, method):
                    setattr(obj, attr, wrap(method, label))
                elif isinstance(method, (staticmethod, classmethod)) and launchable(attr, method.__func__):
                    setattr(obj, attr, type(method)(wrap(method.__func__, label)))


def wrap_builtins(module, label_prefix, names=None):
    """Mark compiled-extension functions (nanobind / pybind builtins) bound as `module` attributes."""
    for name, obj in list(vars(module).items()):
        if names is not None and name not in names:
            continue
        if (compiled_function(obj) and not name.startswith("_")
                and not getattr(obj, "_infx_launcher", False)):
            setattr(module, name, wrap(obj, f"{label_prefix}.{name}"))


def patch_calls(module, attr, label_of):
    """Mark calls of every class in `module` that defines `attr` itself."""
    for cls in list(vars(module).values()):
        if not isinstance(cls, type) or cls.__module__ != module.__name__ or attr not in vars(cls):
            continue
        orig = vars(cls)[attr]
        if getattr(orig, "_infx_launcher", False):
            continue

        def call(self, *args, _orig=orig, **kwargs):
            if runtime.compiling() or not runtime.profiling():
                return _orig(self, *args, **kwargs)
            try:
                label = label_of(self)
            except Exception:
                label = type(self).__qualname__
            with marker(label):
                return _orig(self, *args, **kwargs)

        call._infx_launcher = True
        setattr(cls, attr, call)


def triton_label(jit_function):
    fn = getattr(jit_function, "fn", None)
    return f"triton:{getattr(fn, '__qualname__', None) or type(jit_function).__qualname__}"


def named_label(prefix):
    def label(obj):
        for attr in ("kernel_name", "name", "__name__", "func_name"):
            value = getattr(obj, attr, None)
            if isinstance(value, str) and value:
                return f"{prefix}:{value}"
        return f"{prefix}:{type(obj).__qualname__}"

    return label


def install(registry, caller_roots, modules=(), packages=()):
    """Hook the DSL launchers, the engine's launcher `modules` and every module under `packages`."""
    global _caller_roots
    _caller_roots = tuple(caller_roots)
    registry.on("triton.runtime.jit", lambda m: patch_calls(m, "run", triton_label))
    registry.on("tilelang.jit.kernel", lambda m: patch_calls(m, "__call__", named_label("tilelang")))
    registry.on("cutlass.cutlass_dsl.tvm_ffi_provider", lambda m: patch_calls(m, "__call__", named_label("cute")))
    for name in modules:
        registry.on(name, patch_module)
    for package in packages:
        registry.on_package(package, functools.partial(patch_module, package=package))
