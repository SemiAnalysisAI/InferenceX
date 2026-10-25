import pytest

from infx.serving.slo import SLO, SLOError, parse_slo


def test_two_tuples_mixed_stats():
    got = parse_slo("time_to_first_token:avg:lt:8000;inter_token_latency:p90:lt:50")
    assert got == [
        SLO("time_to_first_token", "avg", "lt", 8000.0),
        SLO("inter_token_latency", "p90", "lt", 50.0),
    ]


def test_tuple_string_round_trips_without_trailing_zeros():
    assert SLO("inter_token_latency", "p90", "lt", 50.0).tuple() == "inter_token_latency:p90:lt:50"
    assert SLO("time_to_first_token", "avg", "le", 0.5).tuple() == "time_to_first_token:avg:le:0.5"


def test_whitespace_and_trailing_separator_tolerated():
    assert len(parse_slo(" time_to_first_token:p99:lt:200 ; ")) == 1


@pytest.mark.parametrize(
    "text, fragment",
    [
        ("", "empty"),
        ("   ;  ", "empty"),
        ("time_to_first_token:avg:lt", "metric:stat:op:threshold"),
        ("time_to_first_token:median:lt:8000", "stat"),
        ("time_to_first_token:avg:<:8000", "op"),
        ("time_to_first_token:avg:lt:fast", "threshold"),
        ("time_to_first_token:avg:lt:-5", "threshold"),
        ("time_to_first_token:avg:lt:nan", "threshold"),
        ("Time To First:avg:lt:8000", "metric"),
    ],
)
def test_rejects_malformed(text, fragment):
    with pytest.raises(SLOError, match=fragment):
        parse_slo(text)
