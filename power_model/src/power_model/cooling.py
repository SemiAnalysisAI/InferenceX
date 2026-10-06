# SPDX-License-Identifier: GPL-3.0-only
"""The requested facility PUE policy; independent of equipment fan power."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

CoolingMode = Literal["air", "liquid"]


class CoolingProfile(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    mode: CoolingMode

    @property
    def pue(self) -> float:
        return {"air": 1.3, "liquid": 1.1}[self.mode]
