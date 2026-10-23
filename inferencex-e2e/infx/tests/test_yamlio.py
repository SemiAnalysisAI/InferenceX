import pytest
import yaml

from infx import yamlio


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
        yamlio.load("x: 1\ny: {a: 1, a: 2}\n", yamlio.UniqueKeyLoader)


def test_safe_unique_key_loader_merges_and_rejects_repeated_own_keys():
    merged = "base: &b {x: 1, y: 1}\nk: {<<: *b, y: 2}\n"
    repeated = "base: &b {x: 1}\nk: {<<: *b, y: 2, y: 3}\n"

    assert yamlio.load(merged, yamlio.UniqueKeySafeLoader) == {
        "base": {"x": 1, "y": 1},
        "k": {"x": 1, "y": 2},
    }
    with pytest.raises(yaml.YAMLError, match="found duplicate key 'y'"):
        yamlio.load(repeated, yamlio.UniqueKeySafeLoader)


@pytest.mark.parametrize("load", [yamlio.load, lambda text, loader: yaml.load(text, Loader=loader)])
@pytest.mark.parametrize(
    ("merging", "expected"), [("{<<: *b}", {"k": 2}), ("!!set {<<: *b}", {"k"})]
)
def test_safe_unique_key_loader_allows_overriding_a_mapping_flattened_elsewhere_first(
    load, merging, expected
):
    text = f"x:\n  b: &b {{<<: &a {{k: 1}}, k: 2}}\ny: {merging}\n"

    assert load(text, yamlio.UniqueKeySafeLoader) == {"x": {"b": {"k": 2}}, "y": expected}


def test_safe_unique_key_loader_reports_a_non_mapping_map_as_safeloader_does():
    with pytest.raises(yaml.constructor.ConstructorError, match="expected a mapping node"):
        yamlio.load("a: !!map x\n", yamlio.UniqueKeySafeLoader)
