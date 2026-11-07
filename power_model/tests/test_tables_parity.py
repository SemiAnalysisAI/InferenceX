# SPDX-License-Identifier: GPL-3.0-only
"""The tables evaluator reproduces the class graph on the app's parity grid."""

from math import isclose

import pytest
from power_parity_grid import SYSTEMS, cases
from pydantic import ValidationError

from power_model import create_power_model
from power_model.evaluate import estimate_breakdown
from power_model.tables import default_tables

# Not bytes: libm pow may differ by an ULP across platforms.
REL_TOL = 1e-12


def from_classes(case):
    return create_power_model(
        system=case.system,
        model=case.model,
        workload_state=case.workload_state,
        using_scale_out=case.using_scale_out,
        systems=case.systems,
    ).estimate_breakdown(
        case.gpu_level_power_per_gpu,
        cpu_and_dram_measured_power_per_socket=case.cpu_and_dram_measured_power_per_socket,
        cpu_measured_power_per_socket=case.cpu_measured_power_per_socket,
    )


def from_tables(case):
    return estimate_breakdown(**case._asdict())


def outcome(evaluate, case):
    """The estimate as JSON, or the rejection reason without pydantic's wrapper."""
    try:
        return evaluate(case).model_dump(mode="json"), None
    except ValidationError as error:
        first = error.errors()[0]
        return None, str(first["ctx"]["error"]) if first["type"] == "value_error" else first["type"]
    except ValueError as error:
        return None, str(error)


def differences(expected, actual, path="$"):
    """Where actual differs: structure, types, strings, or numbers beyond REL_TOL.

    The class graph's configuration also dumps its own constructor parameters; the
    tables evaluator records only the scenario, so only its keys are compared.
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        if path.endswith(".configuration"):
            keys = set(actual)
            if not keys <= set(expected):
                return [f"{path}: extra keys {sorted(keys - set(expected))}"]
        elif set(expected) != set(actual):
            return [f"{path}: keys {sorted(set(expected) ^ set(actual))}"]
        else:
            keys = set(expected)
        return [
            d
            for key in sorted(keys)
            for d in differences(expected[key], actual[key], f"{path}.{key}")
        ]
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: {len(expected)} != {len(actual)} items"]
        return [
            d
            for i, pair in enumerate(zip(expected, actual, strict=True))
            for d in differences(*pair, f"{path}[{i}]")
        ]
    if type(expected) is not type(actual):
        return [f"{path}: {expected!r} != {actual!r}"]
    if isinstance(expected, float) and isclose(expected, actual, rel_tol=REL_TOL, abs_tol=0):
        return []
    return [] if expected == actual else [f"{path}: {expected!r} != {actual!r}"]


def test_grid_covers_every_table_system():
    assert set(default_tables().systems) == set(SYSTEMS)


@pytest.mark.parametrize("system", SYSTEMS)
def test_tables_match_class_graph(system):
    compared = 0
    mismatches = []
    for case in cases():
        if case.system != system:
            continue
        compared += 1
        expected, expected_error = outcome(from_classes, case)
        actual, actual_error = outcome(from_tables, case)
        if expected_error or actual_error:
            if expected_error != actual_error:
                mismatches.append(f"{case}: {expected_error!r} != {actual_error!r}")
            continue
        mismatches += [f"{case}: {d}" for d in differences(expected, actual)]
    assert compared
    assert not mismatches, "\n".join(mismatches[:20])
