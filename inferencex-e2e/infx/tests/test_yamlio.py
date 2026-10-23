import pytest
import yaml

from infx import yamlio
from infx.workflows.validate_perf_changelog import UniqueKeyLoader


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("a:\n  b\tc: 1\n", "that cannot start any token"),
        ("a: {b: c?d}\n", "expected ',' or '}', but got '?'"),
        ("a: 1\n\ufeff\nb: 2\n", "could not find expected ':'"),
        ("a: |#\n  x\n", "expected chomping or indentation indicators"),
        ("[!!str, a]\n", "could not determine a constructor for the tag"),
    ],
    ids=[
        "tab-in-plain-scalar",
        "question-mark-in-flow-scalar",
        "inner-byte-order-mark",
        "comment-on-block-scalar-header",
        "tag-before-flow-comma",
    ],
)
def test_documents_only_libyaml_accepts_raise_pyyamls_error(text, message):
    with pytest.raises(yaml.YAMLError, match=message):
        yamlio.safe_load(text)


def test_a_tagged_empty_flow_key_is_null_as_in_pyyaml():
    assert yamlio.safe_load("{a: 1, ! }\n") == {"a": 1, None: None}


@pytest.mark.parametrize(
    "text",
    ["[" * 2000 + "]" * 2000, "#\r" + "- " * 600 + "x\r"],
    ids=["flow", "block-after-carriage-returns"],
)
def test_nesting_past_pyyamls_recursion_limit_raises_its_recursion_error(text):
    with pytest.raises(RecursionError):
        yamlio.safe_load(text)


def test_stream_errors_name_the_file_with_pyyamls_wording(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("a: [1\n")

    with path.open() as stream, pytest.raises(yaml.YAMLError) as error:
        yamlio.safe_load(stream)

    assert str(error.value) == (
        f'while parsing a flow sequence\n  in "{path}", line 1, column 4\n'
        f"expected ',' or ']', but got '<stream end>'\n  in \"{path}\", line 2, column 1"
    )


def test_custom_loader_constructors_apply():
    with pytest.raises(yaml.YAMLError, match="found duplicate key 'a'"):
        yamlio.load("x: 1\ny: {a: 1, a: 2}\n", UniqueKeyLoader)
