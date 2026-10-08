# SPDX-License-Identifier: GPL-3.0-only
"""GB300 logical board variant, with a provisional shared LPDDR5X profile."""

from typing import ClassVar, Literal

from .bianca import BiancaBoard


class GB300BiancaBoard(BiancaBoard):
    gpu_tdp_w: Literal[1400] = 1400
    gpu_family: ClassVar[str] = "Blackwell Ultra"
