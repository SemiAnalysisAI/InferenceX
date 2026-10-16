"""Patch modules as they are imported."""

import importlib.abc
import sys

from . import runtime


class HookRegistry:
    def __init__(self):
        self.exact = {}
        self.packages = []

    def on(self, name, fn):
        self.exact.setdefault(name, []).append(fn)

    def on_package(self, package, fn):
        self.packages.append((package, fn))

    def hooks_for(self, name):
        hooks = list(self.exact.get(name, ()))
        hooks += [fn for package, fn in self.packages if name == package or name.startswith(package + ".")]
        return hooks

    def run(self, name, module):
        for hook in self.hooks_for(name):
            try:
                hook(module)
            except Exception:
                runtime.write_error(f"patch {name}")

    def install(self):
        for name in list(sys.modules):
            if self.hooks_for(name):
                self.run(name, sys.modules[name])
        sys.meta_path.insert(0, _PostImportFinder(self))


class _PostImportFinder(importlib.abc.MetaPathFinder):
    def __init__(self, registry):
        self.registry = registry

    def find_spec(self, name, path, target=None):
        if not self.registry.hooks_for(name):
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            spec = find(name, path, target)
            if spec is not None and spec.loader is not None:
                break
        else:
            return None
        loader = spec.loader
        exec_module = loader.exec_module
        registry = self.registry

        def exec_and_patch(module, _exec=exec_module, _name=name):
            _exec(module)
            registry.run(_name, module)

        loader.exec_module = exec_and_patch
        return spec
