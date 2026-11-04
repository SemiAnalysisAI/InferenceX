# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from typing import Any


def convert_to_pytorch_benchmark_format(
    args: argparse.Namespace, metrics: dict[str, list], extra_info: dict[str, Any]
) -> list:
    """Build one PyTorch OSS benchmark record per entry of ``metrics``, tagged with ``args`` and ``extra_info``.

    Returns an empty list unless ``SAVE_TO_PYTORCH_BENCHMARK_FORMAT`` is set. Format reference is at
    https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database
    """
    records = []
    if not os.environ.get("SAVE_TO_PYTORCH_BENCHMARK_FORMAT"):
        return records

    for name, benchmark_values in metrics.items():
        record = {
            "benchmark": {
                "name": "vLLM benchmark",
                "extra_info": {
                    "args": vars(args),
                },
            },
            "model": {
                "name": args.model,
            },
            "metric": {
                "name": name,
                "benchmark_values": benchmark_values,
                "extra_info": extra_info,
            },
        }
        records.append(record)

    return records
