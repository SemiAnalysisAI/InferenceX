# SPDX-License-Identifier: GPL-3.0-only
"""Two-GPU Grace/Blackwell board assemblies."""

from .bianca import BiancaBoard
from .gb200 import GB200BiancaBoard
from .gb300 import GB300BiancaBoard

__all__ = ["BiancaBoard", "GB200BiancaBoard", "GB300BiancaBoard"]
