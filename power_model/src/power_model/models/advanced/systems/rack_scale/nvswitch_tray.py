# SPDX-License-Identifier: GPL-3.0-only
"""Internal NVLink fabric tray with management CPU, fans, and DC/DC conversion."""

from power_model.base import FrozenModel, PowerComponentBreakdown, Provenance, sum_watts
from power_model.models.advanced.components.blackwell_nvswitch import BlackwellNVSwitch
from power_model.models.advanced.components.dc_converter import DCConverterAssembly
from power_model.models.advanced.components.epyc3151 import EPYCEmbedded3151CPU
from power_model.models.advanced.components.rack_fans import RackFanAssembly, solve_cooling


class NVSwitchTray(FrozenModel):
    switch: BlackwellNVSwitch = BlackwellNVSwitch()
    cpu: EPYCEmbedded3151CPU = EPYCEmbedded3151CPU()
    fans: RackFanAssembly = RackFanAssembly(fan_count=4, design_air_heat_w=250)
    converter: DCConverterAssembly = DCConverterAssembly()
    provenance: Provenance = Provenance(
        profile_id="nvl72-nvswitch-tray",
        version="1",
        kind="estimated",
        source="User inventory with provisional fan and converter assembly",
        input_boundary="One switch-tray rack-bus input, before rack AC/DC",
        assumptions=(
            "Two liquid-cooled switch electrical budgets plus one 45 W management CPU.",
            "Four representative fan modules and a 250 W air-heat anchor are provisional.",
            "Management CPU and tray converter heat are air cooled.",
            "Switch local regulation is included in its electrical budget; management CPU "
            "45 W is treated as an intermediate-rail budget for this tray profile.",
            "Unspecified management DRAM, storage and other auxiliaries are outside this BoM.",
        ),
    )

    def estimate_breakdown(self) -> PowerComponentBreakdown:
        switches = self.switch.estimate_breakdown().scaled(2)
        cpu = self.cpu.estimate_breakdown()
        load = sum_watts((switches.power_w, cpu.power_w))
        fans = solve_cooling(load, cpu.power_w, self.fans, self.converter.loss_w)
        loss = self.converter.estimate_loss_breakdown(load + fans.power_w)
        return PowerComponentBreakdown.group(
            "NVSwitchTray",
            (switches, cpu, fans, loss),
            provenance=self.provenance,
        )
