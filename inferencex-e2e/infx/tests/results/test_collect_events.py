"""A sweep's job records join into one JSON Lines file."""

from infx.results.collect_events import main


def test_every_readable_job_record_becomes_one_line_in_artifact_order(tmp_path, capsys):
    artifacts = tmp_path / "job_events"
    records = {
        "job_event_b": '{\n  "result_filename": "b",\n  "outcome": "failure"\n}\n',
        "job_event_a": '{"result_filename": "a", "outcome": "success"}',
        "job_event_c": '{"result_filename": "c", "outco',
        "job_event_d": '["not", "a", "record"]',
    }
    for artifact, text in records.items():
        (artifacts / artifact).mkdir(parents=True)
        (artifacts / artifact / "job_event.json").write_text(text)
    (artifacts / "job_event_a" / "point.json").write_text('{"not": "a job record"}')
    output = tmp_path / "events.jsonl"

    assert main([str(artifacts), str(output)]) == 0

    assert output.read_text() == (
        '{"result_filename":"a","outcome":"success"}\n'
        '{"result_filename":"b","outcome":"failure"}\n'
    )
    warnings = capsys.readouterr().err
    assert "job_event_c" in warnings and "job_event_d" in warnings
