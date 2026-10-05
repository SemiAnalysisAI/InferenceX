# SPDX-License-Identifier: GPL-3.0-only
"""Public construction of a power model from scenario inputs."""

from power_model.base import PowerModel, WorkloadState
from power_model.models.advanced.model import AdvancedAllInPowerModel
from power_model.models.advanced.systems.base import GPUSystem
from power_model.models.advanced.systems.catalog import get_system_class
from power_model.models.basic_example import BasicExamplePowerModel

MODELS = {"advanced": AdvancedAllInPowerModel, "basic-example": BasicExamplePowerModel}


def model_name(value: str) -> str:
    aliases = {model.__name__.lower(): name for name, model in MODELS.items()}
    return aliases.get(value.lower(), value.lower())


def create_power_model(
    *,
    system: str | type[GPUSystem],
    model: str = "advanced",
    workload_state: WorkloadState | str = WorkloadState.FIXED_SEQ_LEN,
    using_scale_out: bool = False,
    systems: int = 1,
) -> PowerModel:
    selected = model_name(model)
    if selected not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    system_class = get_system_class(system)
    if selected == "basic-example":
        return BasicExamplePowerModel(
            cooling=system_class.default_cooling,
            workload_state=workload_state,
            using_scale_out=using_scale_out,
        )
    return AdvancedAllInPowerModel.for_system(
        system_class,
        workload_state=workload_state,
        using_scale_out=using_scale_out,
        systems=systems,
    )
