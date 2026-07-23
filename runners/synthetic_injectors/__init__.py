"""Framework-specific synthetic-acceptance injectors.

The generic driver (``inject_synthetic_acceptance.py``) resolves *whether* to
inject and *what* acceptance length to use, then hands the recipe text to a
framework-specific injector registered here. This keeps the driver
framework-agnostic: adding a new backend (e.g. sglang, trtllm) means dropping a
module in this package and registering it, without touching the driver.

A backend is any object exposing:

    rewrite(content: str, al: float, log) -> tuple[str, int]
        Return the recipe text with the synthetic acceptance parameters applied
        and the number of entries modified. ``log`` is a ``callable(str)`` used
        for human-readable progress output.

    spec_tokens_from_recipe(content: str) -> int | None
        Best-effort extraction of the speculative-token count from the recipe,
        used only for reference-AL auto-lookup. May return ``None``.
"""

# framework value passed by the runner (e.g. "dynamo-vllm") -> backend module.
# Populated by backend modules registering themselves (see e.g. vllm.py, added
# in the framework-support PR). Empty here: the generic layer alone injects
# nothing, so merging it changes no behavior.
_INJECTORS = {}


def register(framework, backend):
    _INJECTORS[framework] = backend


def get_injector(framework):
    """Return the backend for ``framework`` (as passed by the runner), or None."""
    return _INJECTORS.get(framework)
