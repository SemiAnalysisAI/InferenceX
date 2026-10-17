# SPDX-License-Identifier: GPL-3.0-only
"""MI355 UBB with its fixed GPU and retimer inventory."""

from typing import Literal

from power_model.models.advanced.components.ubb.amd_8way_ubb import AMD8WayUBB


class MI355UBB(AMD8WayUBB):
    """Eight MI355 GPUs and eight PCIe 5.0 x16 retimers at 12 W each."""

    gpu_tdp_w: Literal[1400] = 1400
