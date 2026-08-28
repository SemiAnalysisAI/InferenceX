import pytest

from validate_vllm_cache_source_metrics import ValidationError, _parse_labels


def test_parse_labels() -> None:
    assert _parse_labels(None) == {}
    assert _parse_labels('engine="0", source="cpu"') == {
        "engine": "0",
        "source": "cpu",
    }


def test_parse_labels_unescapes_prometheus_values() -> None:
    assert _parse_labels(r'value="line\nquoted\"slash\\"') == {
        "value": 'line\nquoted"slash\\'
    }


@pytest.mark.parametrize(
    "labels",
    [
        '1bad="value"',
        'key="unterminated',
        'key="bad\\tvalue"',
        'key="value",',
        'key="value" unexpected',
    ],
)
def test_parse_labels_rejects_malformed_input(labels: str) -> None:
    with pytest.raises(ValidationError):
        _parse_labels(labels)


def test_parse_labels_handles_long_escape_sequences_linearly() -> None:
    escaped_backslashes = r"\\" * 100_000
    assert _parse_labels(f'value="{escaped_backslashes}"') == {
        "value": "\\" * 100_000
    }
