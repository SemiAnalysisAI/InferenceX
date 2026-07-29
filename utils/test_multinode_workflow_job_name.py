from pathlib import Path

import yaml


WORKFLOW_PATH = (
    Path(__file__).parents[1]
    / ".github"
    / "workflows"
    / "benchmark-multinode-tmpl.yml"
)


def _benchmark_job_name() -> str:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())
    return workflow["jobs"]["benchmark"]["name"]


def test_non_disaggregated_multinode_job_name_omits_worker_role_label():
    job_name = _benchmark_job_name()

    assert (
        "inputs.disagg == 'true' && format('{0}P ', inputs.prefill-num-worker)"
        in job_name
    )
    assert "workers=" not in job_name


def test_decode_topology_is_only_shown_for_disaggregated_jobs():
    job_name = _benchmark_job_name()

    assert (
        "inputs.disagg == 'true' && format('x {0}D "
        "(TP{1}{2}{3}{4}{5}{6})'"
    ) in job_name
