from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import Any, Callable

from operatorx.core.op import Op


@dataclass(frozen=True)
class BackendImpl:
    op_type: str
    prepare: Callable[[Op], Any]
    kernel: Callable[[Any], None]


def lookup_versions(*pkg_names: str) -> dict[str, str]:
    """Resolve installed versions for a set of package names.

    Tries ``importlib.metadata.version`` first (catches pip-installed
    packages), then falls back to importing the module and reading
    ``__version__`` (catches PYTHONPATH-installed checkouts like MaxText).
    Missing packages are silently skipped.

    Used by each backend's ``versions()`` to declare what it depends on.
    """
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
