"""Committed golden acceptance-length (AL) curves for AgentX synthetic acceptance.

Each ``<curve>.yaml`` beside this module maps one measured model to
``{thinking_mode: {num_speculative_tokens: AL}}``. ``curve_name`` turns an
InferenceX model prefix and speculative config into a curve, and
``golden_length`` returns the committed value. From a shell, use
``python -m infx.golden_al_distribution {list,show,lookup}``.
"""

from .curves import (
    GOLDEN_DIR,
    THINKING_MODES,
    Curve,
    curve_name,
    golden_length,
    list_curves,
    load_curve,
)

__all__ = [
    "GOLDEN_DIR",
    "THINKING_MODES",
    "Curve",
    "curve_name",
    "golden_length",
    "list_curves",
    "load_curve",
]
