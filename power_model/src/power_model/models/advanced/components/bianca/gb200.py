# SPDX-License-Identifier: GPL-3.0-only
"""GB200 board variant."""

from typing import ClassVar, Literal

from .bianca import BiancaBoard


class GB200BiancaBoard(BiancaBoard):
    gpu_tdp_w: Literal[1200] = 1200
    gpu_family: ClassVar[str] = "Blackwell"
