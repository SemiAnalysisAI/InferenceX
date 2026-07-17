"""Minimal cluster routing table.

Maps cluster id -> platform (for runner dispatch) and cluster id -> chip
(for legacy/grouped layouts). No peak-throughput or bandwidth info — that
lives in dashboard_build/hardware.py.
"""
from __future__ import annotations


CLUSTER_PLATFORMS: dict[str, str] = {
    "b200_dgx_8x":   "nvidia",
    "b300_hgx_8x":   "nvidia",
    "b200_nvl72":    "nvidia",
    "mi355x_8x":     "amd",
    "v6e_1x":        "tpu",
    "v6e_4x":        "tpu",
    "v6e_pod":       "tpu",
    "v7x_4x":        "tpu",
    "trn3_1x":       "trainium",
    "trn3_8x":       "trainium",
    "trn3_16x": "trainium",
}

CLUSTER_CHIPS: dict[str, str] = {
    "b200_dgx_8x":   "b200",
    "b300_hgx_8x":   "b300",
    "b200_nvl72":    "b200",
    "mi355x_8x":     "mi355x",
    "v6e_1x":        "v6e",
    "v6e_4x":        "v6e",
    "v6e_pod":       "v6e",
    "v7x_4x":        "v7x",
    "trn3_1x":       "trn3",
    "trn3_8x":       "trn3",
    "trn3_16x": "trn3",
}
