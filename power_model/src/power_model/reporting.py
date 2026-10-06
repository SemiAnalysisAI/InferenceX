# SPDX-License-Identifier: GPL-3.0-only
"""Readable power bills of materials from the model's computed component tree."""

from power_model.base import PowerComponentBreakdown, PowerEstimate


def _component_rows(
    component: PowerComponentBreakdown,
    chassis_count: int | float,
    *,
    allow_fractional_quantities: bool = False,
    label_prefix: str = "",
    child_prefix: str = "",
) -> list[tuple[str, str, str, str]]:
    quantity, remainder = divmod(component.quantity, chassis_count)
    if remainder and not allow_fractional_quantities:
        raise ValueError("Component quantity cannot be normalized to one chassis")
    if allow_fractional_quantities:
        quantity = component.quantity / chassis_count
    rows = [
        (
            label_prefix + component.name,
            f"{component.power_w / component.quantity:,.2f}",
            f"{quantity:g}",
            f"{component.power_w / chassis_count:,.2f}",
        )
    ]
    for index, child in enumerate(component.children):
        last = index == len(component.children) - 1
        rows.extend(
            _component_rows(
                child,
                chassis_count,
                allow_fractional_quantities=allow_fractional_quantities,
                label_prefix=child_prefix + ("└─ " if last else "├─ "),
                child_prefix=child_prefix + ("   " if last else "│  "),
            )
        )
    return rows


def _format_table(rows: list[tuple[str, str, str, str]]) -> str:
    header = ("Component", "Power / unit (W)", "Quantity", "Extended power (W)")
    widths = tuple(max(len(row[index]) for row in [header, *rows]) for index in range(4))

    def line(row: tuple[str, str, str, str]) -> str:
        return "  ".join(
            value.ljust(width) if index == 0 else value.rjust(width)
            for index, (value, width) in enumerate(zip(row, widths, strict=True))
        )

    return "\n".join(
        (line(header), "  ".join("-" * width for width in widths), *(line(row) for row in rows))
    )


def format_power_breakdown_per_chassis(estimate: PowerEstimate) -> str:
    """Normalize each system group's existing tree to one chassis; keep cluster totals separate."""
    if estimate.scope != "cluster":
        raise ValueError("--power-breakdown-per-chassis requires the advanced model")
    systems, networking = estimate.components
    units = {dict(system.details).get("system_unit", "chassis") for system in systems.children}
    unit = next(iter(units)) if len(units) == 1 else "system"
    sections = [
        f"Power BoM per {unit}",
        f"GPU input: {estimate.gpu_level_power_per_gpu:,.2f} W/GPU | "
        f"Workload: {estimate.workload_state.value} | "
        f"Scale-out: {'on' if estimate.using_scale_out else 'off'}",
    ]
    for chassis in systems.children:
        system_unit = dict(chassis.details).get("system_unit", "chassis")
        sections.extend(
            (
                "",
                f"{chassis.name} — one {system_unit} (configured quantity: {chassis.quantity})",
                _format_table(_component_rows(chassis, chassis.quantity)),
            )
        )
    count = sum(chassis.quantity for chassis in systems.children)
    if networking.power_w:
        network_rows = [
            row
            for gear in networking.children
            for row in _component_rows(gear, count, allow_fractional_quantities=True)
        ]
        sections.extend(
            (
                "",
                f"Allocated external network per {unit} (cluster average)",
                _format_table(network_rows),
            )
        )
    sections.extend(
        (
            "",
            f"Parent rows are subtotals. Quantities are expanded to one {unit}.",
            f"{unit.capitalize()} power includes conversion loss, "
            "before shared networking and PUE.",
            "",
            f"Cluster totals ({count} {unit if unit == 'chassis' or count == 1 else unit + 's'}, "
            f"{estimate.gpu_count} GPUs)",
            f"Shared external network AC power: {networking.power_w:,.2f} W",
            f"IT AC power: {estimate.it_power_w:,.2f} W",
            f"PUE: {estimate.pue:g}",
            f"Facility overhead: {estimate.facility_overhead_w:,.2f} W",
            f"All in Utility Power: {estimate.facility_power_w:,.2f} W",
            f"AllInPower_per_gpu: {estimate.AllInPower_per_gpu:,.2f} W/GPU",
        )
    )
    return "\n".join(sections)
