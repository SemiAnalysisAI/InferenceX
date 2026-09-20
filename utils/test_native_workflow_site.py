"""Actionable deployment failures precede any native preparation or scheduler work."""

import json

import pytest
from test_native_pilot import inputs as pilot_inputs  # noqa: F401

from infx.srt_slurm import workflow


def test_missing_site_variables_fail_before_workflow_or_scheduler_inputs(monkeypatch):
    monkeypatch.setattr(workflow.os, "environ", {"NATIVE_SITE_JSON": "  "})
    with pytest.raises(ValueError) as error:
        workflow.main()
    assert str(error.value) == (
        "Native H100 pilot requires repository variables: INFX_H100_PHASE1_SITE_JSON, "
        "INFX_PHASE1_READER_REVISION, INFX_PHASE1_COLLECTOR_REVISION"
    )


@pytest.fixture
def environment(pilot_inputs):
    *_, site = pilot_inputs
    return {
        "NATIVE_SITE_JSON": site.model_dump_json(),
        "NATIVE_READER_REVISION": "a" * 40,
        "NATIVE_COLLECTOR_REVISION": "b" * 40,
    }


def test_explicit_matching_deployed_revisions_load_site(environment):
    site = workflow.load_site(environment)
    assert site.cluster == "h100-dgxc"
    assert site.reader_revision == "a" * 40
    assert site.collector_revision == "b" * 40


@pytest.mark.parametrize(
    "value,diagnostic",
    [("{", "JSON: Invalid JSON"), ('{"schema_version":2}', "schema_version:")],
)
def test_invalid_site_names_repository_variable(environment, value, diagnostic):
    environment["NATIVE_SITE_JSON"] = value
    with pytest.raises(ValueError) as error:
        workflow.load_site(environment)
    assert str(error.value).startswith(
        "Repository variable INFX_H100_PHASE1_SITE_JSON must contain valid PilotSite JSON: "
    )
    assert diagnostic in str(error.value)


@pytest.mark.parametrize("role", ["READER", "COLLECTOR"])
def test_mismatched_deployment_names_the_specific_variable(environment, role):
    environment[f"NATIVE_{role}_REVISION"] = "c" * 40
    with pytest.raises(ValueError) as error:
        workflow.load_site(environment)
    assert str(error.value) == (
        f"Repository variables INFX_PHASE1_{role}_REVISION must match the deployed revisions "
        "recorded in INFX_H100_PHASE1_SITE_JSON"
    )


def test_invalid_site_diagnostic_excludes_input_values(environment):
    site = json.loads(environment["NATIVE_SITE_JSON"])
    site["schema_version"] = "not-a-version-sensitive-value"
    environment["NATIVE_SITE_JSON"] = json.dumps(site)
    with pytest.raises(ValueError) as error:
        workflow.load_site(environment)
    assert "schema_version:" in str(error.value)
    assert "not-a-version-sensitive-value" not in str(error.value)
