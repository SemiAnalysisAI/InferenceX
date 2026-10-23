import io
import sys

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
        ("[?]]\n", "expected '<document start>', but found ']'"),
    ],
    ids=[
        "tab-in-plain-scalar",
        "question-mark-in-flow-scalar",
        "inner-byte-order-mark",
        "comment-on-block-scalar-header",
        "tag-before-flow-comma",
        "token-after-empty-flow-key",
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


def test_a_file_that_fails_to_decode_late_raises_pyyamls_earlier_syntax_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_bytes(b"a: b: c\n" + b"# padding\n" * 2000 + b"x: \xff\n")

    with path.open(encoding="utf-8") as stream, pytest.raises(yaml.YAMLError) as error:
        yamlio.safe_load(stream)

    assert "mapping values are not allowed here" in str(error.value)


def test_a_file_iterated_before_loading_loads_its_remaining_lines(tmp_path):
    path = tmp_path / "x.yaml"
    path.write_text("skip: 1\na: 1\n")

    with path.open() as stream:
        next(stream)
        assert yamlio.safe_load(stream) == {"a": 1}


def test_a_closed_file_raises_reads_closed_file_error(tmp_path):
    path = tmp_path / "x.yaml"
    path.write_text("a: 1\n")
    with path.open() as stream:
        pass

    with pytest.raises(ValueError, match=r"^I/O operation on closed file\.$"):
        yamlio.safe_load(stream)


def test_text_streams_that_need_a_read_size_load():
    class Chunks(io.TextIOBase):
        def __init__(self) -> None:
            self.left = "a: 1\n"

        def read(self, size: int) -> str:
            chunk, self.left = self.left[:size], self.left[size:]
            return chunk

    assert yamlio.safe_load(Chunks()) == {"a": 1}


def test_str_subclasses_load_like_str():
    class Text(str):
        __slots__ = ()

    assert yamlio.safe_load(Text("a: 1\n")) == {"a": 1}


def test_a_metaclass_table_property_is_ignored_as_pyyaml_ignores_it():
    class Meta(type):
        @property
        def yaml_constructors(cls):
            return {
                **yaml.SafeLoader.yaml_constructors,
                "tag:yaml.org,2002:int": lambda _, node: f"META:{node.value}",
            }

    class Loader(yaml.SafeLoader, metaclass=Meta):
        pass

    assert yamlio.load("n: 1\n", Loader) == {"n": 1}


def test_a_replaced_constructor_that_refuses_comparison_applies():
    class Fussy:
        __hash__ = object.__hash__

        def __init__(self, label):
            self.label = label

        def __eq__(self, other):
            raise TypeError("ambiguous comparison")

        def __call__(self, _, node):
            return f"{self.label}:{node.value}"

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor("tag:yaml.org,2002:int", Fussy("OLD"))
    first = yamlio.load("n: 1\n", Loader)
    Loader.add_constructor("tag:yaml.org,2002:int", Fussy("NEW"))

    assert (first, yamlio.load("n: 1\n", Loader)) == ({"n": "OLD:1"}, {"n": "NEW:1"})


def test_methods_patched_onto_a_loader_after_a_load_apply_to_the_same_text():
    class Loader(yaml.SafeLoader):
        pass

    def construct_mapping(self, node, deep=False):
        return {**yaml.SafeLoader.construct_mapping(self, node, deep), "patched": True}

    first = yamlio.load("a: 1\n", Loader)
    Loader.construct_mapping = construct_mapping

    assert (first, yamlio.load("a: 1\n", Loader)) == ({"a": 1}, {"a": 1, "patched": True})


def test_constructors_added_after_a_load_apply_to_the_same_text():
    class Loader(yaml.SafeLoader):
        pass

    first = yamlio.load("n: 12\n", Loader)
    Loader.add_constructor("tag:yaml.org,2002:int", lambda _, node: f"INT {node.value}")

    assert (first, yamlio.load("n: 12\n", Loader)) == ({"n": 12}, {"n": "INT 12"})


def test_the_int_digit_limit_applies_to_a_text_loaded_under_a_higher_limit():
    text = "n: " + "1" * 4400 + "\n"
    limit = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(5000)
    try:
        yamlio.safe_load(text)
    finally:
        sys.set_int_max_str_digits(4300)
    try:
        with pytest.raises(ValueError, match="Exceeds the limit"):
            yamlio.safe_load(text)
    finally:
        sys.set_int_max_str_digits(limit)


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


def test_repeated_loads_return_independent_values():
    text = "roles:\n  agg: {gpus: 8}\n"
    first = yamlio.safe_load(text)
    first["roles"]["agg"]["gpus"] = 4

    assert yamlio.safe_load(text) == {"roles": {"agg": {"gpus": 8}}}


def test_repeated_loads_keep_anchor_sharing():
    text = "base: &env {A: '1'}\nother: *env\n"
    yamlio.safe_load(text)

    loaded = yamlio.safe_load(text)

    assert loaded["base"] is loaded["other"]


def test_a_deep_alias_chain_loads_as_pyyaml_builds_it():
    chain = "".join(f"k{i}: &a{i} [{f'*a{i - 1}' if i else 'x'}]\n" for i in range(1200))
    text = "first: null\n" + chain + "first: *a1199\n"

    loaded = yamlio.safe_load(text)

    depth, node = 0, loaded["first"]
    while isinstance(node, list):
        depth, node = depth + 1, node[0]
    assert (depth, node, loaded["first"] is loaded["k1199"]) == (1200, "x", True)
