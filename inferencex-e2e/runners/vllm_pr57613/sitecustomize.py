"""vLLM PR #57613 at import time, for serving YAMLs that cannot patch the image.

Put this directory on PYTHONPATH (serving YAML env:). When vLLM imports
``vllm.v1.core.single_type_kv_cache_manager``, the module is compiled from its source
with the upstream diff applied: under sparse SWA retention, keep one extra alignment
block below each reachable boundary, so a prompt that diverges inside the last
aligned block still hits (vllm-project/vllm#57613, fixes #57610). Nothing on disk
changes. If the target text is missing (other vLLM version), the import fails loudly
instead of running unpatched. Other processes on this PYTHONPATH are untouched.
"""

import importlib.abc
import importlib.machinery
import sys

TARGET = "vllm.v1.core.single_type_kv_cache_manager"
OLD = """            for boundary_tokens in reachable_boundaries:
                aligned = boundary_tokens // alignment_tokens * alignment_tokens
                end = aligned // block_size + shift
                for j in range(max(start_block, end - need), min(end_block, end)):
                    mask[j - start_block] = True
"""
NEW = """            extra = alignment_tokens // block_size
            for boundary_tokens in reachable_boundaries:
                aligned = boundary_tokens // alignment_tokens * alignment_tokens
                end = aligned // block_size + shift
                for j in range(
                    max(start_block, end - need - extra), min(end_block, end)
                ):
                    mask[j - start_block] = True
"""


def patch_source(source: str) -> str:
    if source.count(OLD) != 1:
        raise ImportError(f"PR #57613 patch: expected 1 match in {TARGET}, found {source.count(OLD)}")
    return source.replace(OLD, NEW)


class _Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):  # compile patched source, never the cached .pyc
        source = patch_source(self.get_data(self.path).decode())
        print(f"[pr57613] patched {fullname} ({self.path})", file=sys.stderr, flush=True)
        return compile(source, self.path, "exec", dont_inherit=True)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            spec.loader = _Loader(fullname, spec.origin)
        return spec


sys.meta_path.insert(0, _Finder())
