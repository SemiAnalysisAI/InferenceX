"""The GitHub annotation of a failed job survives arbitrary failure text."""

from infx.launch.event import JobError, annotation


def test_annotation_escapes_workflow_command_data_and_title():
    error = JobError(
        stage="batch:run,2", type="JobFailed", message="100% done\r\nthen: died, twice",
        exit_code=1, retriable=False,
    )  # fmt: skip
    assert annotation(error) == (
        "::error title=batch%3Arun%2C2::JobFailed: 100%25 done%0D%0Athen: died, twice"
    )
