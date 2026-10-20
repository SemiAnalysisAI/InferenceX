# SPDX-License-Identifier: GPL-3.0-only
"""Common board interface for NVIDIA eight-way HGX systems."""

from power_model.models.advanced.components.ubb.ubb import UniversalBaseBoard


class NVIDIA8WayHGXBoard(UniversalBaseBoard):
    """Eight GPUs with non-GPU inventory modeled by a concrete board."""
