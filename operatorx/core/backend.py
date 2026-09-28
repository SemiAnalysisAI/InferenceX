from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import Any, Callable

from operatorx.core.op import Op


@dataclass(frozen=True)
class BackendImpl:
    """prepare(op) -> ctx; kernel(ctx) runs the op once.

    launcher(ctx), if set, returns (callable, cuda_graph): what the runner times instead of
    kernel(ctx) (how the framework would execute the op) and whether it is a graph replay."""
    op_type: str
    prepare: Callable[[Op], Any]
    kernel: Callable[[Any], None]
    launcher: Callable[[Any], tuple[Callable[[], None], bool]] | None = None


def lookup_versions(*pkg_names: str) -> dict[str, str]:
    """Installed versions: package metadata, else the module's __version__ (PYTHONPATH
    checkouts); missing packages are skipped."""
    import importlib as _importlib

    out: dict[str, str] = {}
    for n in pkg_names:
        try:
            out[n] = importlib_metadata.version(n)
            continue
        except Exception:
            pass
        try:
            mod = _importlib.import_module(n.replace("-", "_"))
            v = getattr(mod, "__version__", None)
            if v:
                out[n] = str(v)
        except Exception:
            pass
    return out
