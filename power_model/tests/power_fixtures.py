# SPDX-License-Identifier: GPL-3.0-only
"""Controlled inputs to the real fan and PSU models."""

from power_model.models.advanced.components import (
    HGXFanPolicy,
    PSUEfficiencyModel,
    PSUEfficiencyPoint,
)


def controlled_fan_policy() -> HGXFanPolicy:
    """Zero-floor cubic cooling with fan watts at 10% of design heat at full load."""
    return HGXFanPolicy(min_pwm_frac=0, normal_max_pwm_frac=1, fan_power_ratio_at_design=0.1)


def constant_efficiency_psu(efficiency: float = 1.0) -> PSUEfficiencyModel:
    return PSUEfficiencyModel(
        n_installed_psu=1,
        n_load_sharing_psu=1,
        n_redundant_capacity_psu=1,
        psu_capacity_w=100000,
        redundancy="none",
        efficiency_curve=(
            PSUEfficiencyPoint(load_fraction=0, efficiency=efficiency),
            PSUEfficiencyPoint(load_fraction=1, efficiency=efficiency),
        ),
    )
