"""PyYAML loading through libyaml, with PyYAML's own results and errors."""

from __future__ import annotations

import functools
import io
import os
import re
from collections.abc import Iterator
from contextlib import suppress
from typing import IO, Any

import yaml
from yaml.constructor import ConstructorError
from yaml.resolver import BaseResolver

from infx import fastcopy

_C_SAFE_LOADER = (
    None if os.environ.get("INFX_PURE_YAML") == "1" else getattr(yaml, "CSafeLoader", None)
)
_MERGE_TAG = "tag:yaml.org,2002:merge"
_VALUE_TAG = "tag:yaml.org,2002:value"
_TABLES = (
    "yaml_constructors",
    "yaml_multi_constructors",
    "yaml_implicit_resolvers",
    "yaml_path_resolvers",
)
_CLASS_ATTRIBUTES = {
    *_TABLES,
    "__module__",
    "__qualname__",
    "__doc__",
    "__firstlineno__",
    "__static_attributes__",
}
# libyaml accepts documents PyYAML rejects when they hold directives, tabs, byte order marks, a
# comment glued to a block scalar header, or characters PyYAML's reader refuses. Line breaks
# other than "\n" go to PyYAML too, because the line-start patterns here only see "\n".
_DIVERGENT = re.compile(r"(?m)^%|[\t\ufeff\r\x85\u2028\u2029]|[|>][-+0-9]*#")
# Deeper documents are left to PyYAML: its recursive composer raises RecursionError near 500
# levels, where libyaml's recursive C composer returns a value or overflows the C stack.
# Indentless sequences add at most one level per counted collection token.
_MAX_TOKEN_DEPTH = 50
_OPENING_TOKENS = (
    yaml.BlockMappingStartToken,
    yaml.BlockSequenceStartToken,
    yaml.FlowMappingStartToken,
    yaml.FlowSequenceStartToken,
)
_CLOSING_TOKENS = (yaml.BlockEndToken, yaml.FlowMappingEndToken, yaml.FlowSequenceEndToken)
_FLOW_TOKENS = {
    yaml.FlowMappingStartToken: 1,
    yaml.FlowSequenceStartToken: 1,
    yaml.FlowMappingEndToken: -1,
    yaml.FlowSequenceEndToken: -1,
}
# The largest srt recipe is 110 KiB; master configs and the changelog are parsed once a run.
_CACHED_TEXT_CHARACTERS = 128 * 1024
_PYYAML = object()


class _NamedText(io.StringIO):
    """Text that reports the original stream's name in PyYAML error marks."""

    def __init__(self, text: str, name: str) -> None:
        super().__init__(text)
        self.name = name


@functools.cache
def _fast_loader(loader: type) -> type | None:
    """libyaml twin of a SafeLoader subclass that only customizes its lookup tables."""
    if _C_SAFE_LOADER is None or not issubclass(loader, yaml.SafeLoader):
        return None
    mro = loader.__mro__
    if any(set(vars(cls)) - _CLASS_ATTRIBUTES for cls in mro[: mro.index(yaml.SafeLoader)]):
        return None
    return type(
        f"Fast{loader.__name__}", (_C_SAFE_LOADER,), {t: getattr(loader, t) for t in _TABLES}
    )


def _may_nest_deeply(text: str) -> bool:
    """Whether the token depth can exceed the limit: flow nesting is at most the number of
    openers, and block nesting at most one more than the longest run of spaces and block
    indicators that starts a line."""
    room = _MAX_TOKEN_DEPTH - 1 - text.count("[") - text.count("{")
    return room < 0 or re.search(rf"(?m)^[ ?:-]{{{room + 1},}}", text) is not None


def _needs_pyyaml(text: str, fast: type) -> bool:
    """Whether the document has a tag, which libyaml and PyYAML scan differently, nesting too
    deep for libyaml's composer, or a plain scalar inside a flow collection holding ``?``,
    which ends it in PyYAML."""
    if "?" not in text and "!" not in text and not _may_nest_deeply(text):
        return False
    depth = flow = 0
    for token in yaml.scan(text, Loader=fast):
        kind = type(token)
        if kind in _OPENING_TOKENS:
            depth += 1
            if depth > _MAX_TOKEN_DEPTH:
                return True
        elif kind in _CLOSING_TOKENS:
            depth -= 1
        elif kind is yaml.TagToken or (
            flow and kind is yaml.ScalarToken and token.plain and "?" in token.value
        ):
            return True
        flow += _FLOW_TOKENS.get(kind, 0)
    return False


def _libyaml_load(text: str, fast: type) -> Any:
    """libyaml's result for ``text``, or ``_PYYAML`` where PyYAML has to decide."""
    if _DIVERGENT.search(text) or yaml.reader.Reader.NON_PRINTABLE.search(text):
        return _PYYAML
    try:
        if _needs_pyyaml(text, fast):
            return _PYYAML
        return yaml.load(text, Loader=fast)  # noqa: S506
    except yaml.YAMLError:
        return _PYYAML


@functools.lru_cache(maxsize=512)
def _cached_libyaml_load(text: str, fast: type) -> Any:
    """Planner runs bind the same recipes and shared blocks once per matrix row."""
    return _libyaml_load(text, fast)


def load(stream: str | IO[str], loader: type = yaml.SafeLoader) -> Any:
    """Return exactly what ``yaml.load(stream, Loader=loader)`` returns or raises.

    Documents libyaml accepts are parsed by libyaml, which builds the same nodes as PyYAML;
    PyYAML reparses everything else, so errors and their marks are PyYAML's own. Texts up to
    128 KiB are parsed once per process and returned as fresh deep copies.
    ``INFX_PURE_YAML=1`` turns libyaml off.
    """
    fast = _fast_loader(loader)
    if fast is None or not isinstance(stream, str | io.TextIOBase):
        return yaml.load(stream, Loader=loader)  # noqa: S506
    if isinstance(stream, str):
        text, fallback = stream, stream
    else:
        text = stream.read()
        fallback = _NamedText(text, getattr(stream, "name", "<file>"))
    if len(text) <= _CACHED_TEXT_CHARACTERS:
        value = _cached_libyaml_load(text, fast)
        if value is not _PYYAML:
            # Chains of aliases can nest values far deeper than the document's own syntax.
            with suppress(RecursionError):
                return fastcopy.deepcopy(value)
    else:
        value = _libyaml_load(text, fast)
        if value is not _PYYAML:
            return value
    return yaml.load(fallback, Loader=loader)  # noqa: S506


def safe_load(stream: str | IO[str]) -> Any:
    """Drop-in for ``yaml.safe_load`` on text."""
    return load(stream, yaml.SafeLoader)


class UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def construct_unique_mapping(
    loader: UniqueKeyLoader,
    node: yaml.Node,
    deep: bool = False,
) -> dict[Any, Any]:
    """Construct a mapping while rejecting keys that would overwrite values."""
    if not isinstance(node, yaml.MappingNode):
        raise ConstructorError(
            None, None, f"expected a mapping node, but found {node.id}", node.start_mark
        )
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found unhashable key {key!r}",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


class UniqueKeySafeLoader(yaml.SafeLoader):
    """SafeLoader, merge keys included, that rejects a key repeated within one mapping."""


def _reject_repeated_keys(loader: UniqueKeySafeLoader, node: yaml.MappingNode) -> None:
    """Check ``node``'s own keys, and those of the mappings it merges, before PyYAML's
    ``flatten_mapping`` rewrites any of them in place."""
    # Holding the nodes keeps their ids from being reused by a later document's nodes.
    checked = loader.__dict__.setdefault("_infx_checked_mappings", {})
    if id(node) in checked:
        return
    checked[id(node)] = node
    seen = set()
    for key_node, value_node in node.value:
        if key_node.tag == _MERGE_TAG:
            sources = (
                value_node.value if isinstance(value_node, yaml.SequenceNode) else [value_node]
            )
            for source in sources:
                if isinstance(source, yaml.MappingNode):
                    _reject_repeated_keys(loader, source)
            continue
        if key_node.tag == _VALUE_TAG:
            key = loader.construct_scalar(key_node)
        else:
            key = loader.construct_object(key_node)
        try:
            duplicate = key in seen
        except TypeError:
            continue
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        seen.add(key)


def construct_unique_safe_mapping(
    loader: UniqueKeySafeLoader, node: yaml.Node
) -> Iterator[dict[Any, Any]]:
    """``SafeConstructor.construct_yaml_map`` after checking the mapping's own keys."""
    data: dict[Any, Any] = {}
    yield data
    if isinstance(node, yaml.MappingNode):
        _reject_repeated_keys(loader, node)
    data.update(loader.construct_mapping(node))


def construct_unique_safe_set(loader: UniqueKeySafeLoader, node: yaml.Node) -> Iterator[set[Any]]:
    """``SafeConstructor.construct_yaml_set`` after checking the mapping's own keys."""
    data: set[Any] = set()
    yield data
    if isinstance(node, yaml.MappingNode):
        _reject_repeated_keys(loader, node)
    data.update(loader.construct_mapping(node))


UniqueKeyLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, construct_unique_mapping)
UniqueKeySafeLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, construct_unique_safe_mapping)
UniqueKeySafeLoader.add_constructor("tag:yaml.org,2002:set", construct_unique_safe_set)
