import pytest
import yaml

from infx import yamlio


def test_stream_errors_name_the_file_with_pyyamls_wording(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("a: [1\n")

    with path.open() as stream, pytest.raises(yaml.YAMLError) as error:
        yamlio.safe_load(stream)

    assert str(error.value) == (
        f'while parsing a flow sequence\n  in "{path}", line 1, column 4\n'
        f"expected ',' or ']', but got '<stream end>'\n  in \"{path}\", line 2, column 1"
    )


def test_a_text_that_fails_to_parse_raises_on_every_load():
    for _ in range(2):
        with pytest.raises(yaml.YAMLError, match="expected ',' or ']'"):
            yamlio.safe_load("a: [1\n")


def test_custom_loader_constructors_apply():
    with pytest.raises(yaml.YAMLError, match="found duplicate key 'a'"):
        yamlio.load("x: 1\ny: {a: 1, a: 2}\n", yamlio.UniqueKeyLoader)


def test_one_text_loads_through_each_loader_separately():
    text = "a: 1\na: 2\n"

    assert yamlio.safe_load(text) == {"a": 2}
    with pytest.raises(yaml.YAMLError, match="found duplicate key 'a'"):
        yamlio.load(text, yamlio.UniqueKeyLoader)


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
    yamlio.safe_load(text)

    loaded = yamlio.safe_load(text)

    depth, node = 0, loaded["first"]
    while isinstance(node, list):
        depth, node = depth + 1, node[0]
    assert (depth, node, loaded["first"] is loaded["k1199"]) == (1200, "x", True)
