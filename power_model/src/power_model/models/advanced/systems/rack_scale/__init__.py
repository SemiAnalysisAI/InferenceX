# SPDX-License-Identifier: GPL-3.0-only
"""NVL72 rack systems composed from trays and shared power shelves."""

from .gb200 import GB200NVL72RackScaleSystem
from .gb300 import GB300NVL72RackScaleSystem
from .rack_scale import RackScaleSystem

__all__ = ["GB200NVL72RackScaleSystem", "GB300NVL72RackScaleSystem", "RackScaleSystem"]
