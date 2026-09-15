import copy
from functools import cache

import pytest

from infx.config import MASTER_CONFIGS, RUNNER_CONFIG
from infx.matrix.generate import generate_config_matrix
from infx.matrix.validation import load_config_files, load_runner_file
from infx.workflows.benchmark_schema import validate_matrix
from infx.workflows.ci_priority import PriorityContext, annotate_jobs, load_policy
from utils.matrix_logic.test_generate_sweep_configs import sample_runner_config

from cases import ROOT


@cache
def inventory():
    return (load_config_files([str(ROOT / path) for path in MASTER_CONFIGS]),
            load_runner_file(str(ROOT / RUNNER_CONFIG)))


@pytest.mark.parametrize("key", sorted(inventory()[0]))
@pytest.mark.parametrize("mode", ["none", "subset", "all"])
def test_recipes_generate_valid_schedulable_rows_with_runner_metadata(key, mode, sample_runner_config):
    master, runners = inventory()
    runners = {**runners, "hardware": {**sample_runner_config["hardware"], **runners["hardware"]}}
    before = copy.deepcopy(master[key])
    rows = generate_config_matrix([key], master, runners, eval_mode=mode)
    validate_matrix(rows)
    scheduled = annotate_jobs(rows, load_policy(ROOT / "configs/ci-priority.yaml"), PriorityContext(queue_namespace=key))
    assert [{k: v for k, v in row.items() if k not in {"priority", "queue-token"}} for row in scheduled] == rows
    assert master[key] == before
