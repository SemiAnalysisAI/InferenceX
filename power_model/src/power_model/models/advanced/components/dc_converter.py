# SPDX-License-Identifier: GPL-3.0-only
"""DC/DC loss curves with enabled idle draw and explicit capacity."""

from typing import Annotated, Self

from pydantic import Field, model_validator

from power_model.base import (
    FrozenModel,
    PowerComponentBreakdown,
    Provenance,
    Watts,
    validate_watts,
)


class DCLossPoint(FrozenModel):
    output_fraction: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    loss_w: Watts


class DCConverterAssembly(FrozenModel):
    module_capacity_w: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1600.0
    input_voltage_v: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 50.0
    output_voltage_v: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = 12.0
    loss_curve: Annotated[tuple[DCLossPoint, ...], Field(min_length=2)] = (
        DCLossPoint(output_fraction=0, loss_w=5),
        DCLossPoint(output_fraction=0.1, loss_w=160 / 0.94 - 160),
        DCLossPoint(output_fraction=0.2, loss_w=320 / 0.96 - 320),
        DCLossPoint(output_fraction=0.5, loss_w=800 / 0.967 - 800),
        DCLossPoint(output_fraction=1, loss_w=1600 / 0.96 - 1600),
    )
    provenance: Provenance = Provenance(
        profile_id="rack-48v-converter",
        version="1",
        kind="estimated",
        source="Engineering curve informed by Infineon REF-IBC-1600W-GAN peak efficiency",
        input_boundary="48 V-class tray bus to intermediate rails; loss only",
        assumptions=(
            "One 1600 W converter with 5 W enabled no-load loss.",
            "Assumed 94/96/96.7/96% efficiency at 10/20/50/100% output.",
            "Only the reference design peak informs this curve; these are not NVL72 measurements.",
            "All converter losses are assigned to air cooling for the initial comparison.",
        ),
    )

    @model_validator(mode="after")
    def validate_curve(self) -> Self:
        if self.loss_curve[0].output_fraction != 0 or self.loss_curve[-1].output_fraction != 1:
            raise ValueError("DC loss curve must span zero through full output")
        for left, right in zip(self.loss_curve, self.loss_curve[1:], strict=False):
            if left.output_fraction >= right.output_fraction or left.loss_w > right.loss_w:
                raise ValueError("DC loss curve requires increasing load and nondecreasing loss")
        return self

    def loss_w(self, output_w: float) -> float:
        output = validate_watts(output_w)
        fraction = output / self.module_capacity_w
        if fraction > 1:
            raise ValueError("DC converter output exceeds installed continuous capacity")
        for left, right in zip(self.loss_curve, self.loss_curve[1:], strict=False):
            if fraction <= right.output_fraction:
                weight = (fraction - left.output_fraction) / (
                    right.output_fraction - left.output_fraction
                )
                return validate_watts(left.loss_w + weight * (right.loss_w - left.loss_w))
        raise ValueError("DC converter curve does not cover output")

    def estimate_loss_breakdown(self, output_w: float) -> PowerComponentBreakdown:
        loss = self.loss_w(output_w)
        return PowerComponentBreakdown(
            name="DC/DC converter loss",
            power_w=loss,
            provenance=self.provenance,
            details=(
                ("output_power_w", output_w),
                ("input_power_w", output_w + loss),
                ("input_voltage_v", self.input_voltage_v),
                ("module_capacity_w", self.module_capacity_w),
            ),
        )


def compute_tray_converter() -> DCConverterAssembly:
    """One converter for the entire tray, retaining its aggregate 8 kW loss profile."""
    return DCConverterAssembly(
        module_capacity_w=8000,
        loss_curve=(
            DCLossPoint(output_fraction=0, loss_w=25),
            DCLossPoint(output_fraction=0.1, loss_w=800 / 0.94 - 800),
            DCLossPoint(output_fraction=0.2, loss_w=1600 / 0.96 - 1600),
            DCLossPoint(output_fraction=0.5, loss_w=4000 / 0.967 - 4000),
            DCLossPoint(output_fraction=1, loss_w=8000 / 0.96 - 8000),
        ),
        provenance=Provenance(
            profile_id="compute-tray-48v-converter",
            version="2",
            kind="estimated",
            source="User-specified single tray converter with shared engineering loss curve",
            input_boundary="48 V-class compute-tray bus to all modeled tray loads; loss only",
            assumptions=(
                "One converter per compute tray supplies both Bianca boards and tray auxiliaries.",
                "8 kW capacity and 25 W enabled no-load loss are provisional whole-unit values.",
                "Assumed 94/96/96.7/96% efficiency at 10/20/50/100% output.",
                "This is an estimated aggregate curve, not a converter internal-module BoM.",
                "Converter loss is assigned to air cooling for the initial comparison.",
            ),
        ),
    )
