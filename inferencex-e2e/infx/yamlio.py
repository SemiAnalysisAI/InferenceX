"""PyYAML loading through libyaml, with PyYAML's own results and errors."""

from __future__ import annotations

import functools
import io
import os
import re
import sys
from collections.abc import Iterator
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
_DIVERGENT = re.compile(r"(?m)^%|[\t\ufeff\r\x85\u2028\u2029]|[|>][-+0-9]*#")
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
_CACHED_TEXT_CHARACTERS = 128 * 1024
_PYYAML = object()


_TWINS: dict[type, tuple[type, tuple[type, ...], tuple[dict[str, Any], ...], tuple[Any, ...]]] = {}
_AFTER_EMPTY_FLOW_KEY = (yaml.ValueToken, yaml.FlowEntryToken, yaml.FlowSequenceEndToken)


def _class_state(classes: tuple[type, ...]) -> tuple[dict[str, Any], ...]:
    """Return a snapshot of each class's own attributes in ``classes``, lookup tables excluded."""
    return tuple(
        {key: value for key, value in dict(vars(cls)).items() if key not in _TABLES}
        for cls in classes
    )


_PYYAML_CLASSES = tuple(
    dict.fromkeys(
        cls
        for loader in (yaml.SafeLoader, _C_SAFE_LOADER)
        if loader is not None
        for cls in loader.__mro__
        if cls is not object
    )
)
_PYYAML_STATE = _class_state(_PYYAML_CLASSES)


def _table_state(loader: type) -> tuple[dict[Any, Any], ...]:
    """Return copies of the constructor and resolver tables of ``loader``."""
    return tuple(
        {key: list(value) if isinstance(value, list) else value for key, value in table.items()}
        for table in (dict(getattr(loader, name)) for name in _TABLES)
    )


def _fast_loader(loader: type) -> type | None:
    """libyaml twin of a SafeLoader subclass that only customizes its lookup tables.

    The twin parses with a copy of the tables and is rebuilt, with the parse cache cleared,
    once ``add_constructor`` and friends change them or an attribute of any class involved
    changes. Patched PyYAML classes, metaclasses and attributes that cannot be compared leave
    every load to PyYAML.
    """
    if _C_SAFE_LOADER is None or type(loader) is not type:
        return None
    if not issubclass(loader, yaml.SafeLoader):
        return None
    known = _TWINS.get(loader)
    mro = loader.__mro__
    try:
        if known is not None:
            twin, classes, attributes, tables = known
            if all(
                saved == vars(cls) for saved, cls in zip(attributes, classes, strict=True)
            ) and all(
                saved == getattr(loader, name) for saved, name in zip(tables, _TABLES, strict=True)
            ):
                return twin
        if (
            any(set(vars(cls)) - _CLASS_ATTRIBUTES for cls in mro[: mro.index(yaml.SafeLoader)])
            or _class_state(_PYYAML_CLASSES) != _PYYAML_STATE
        ):
            return None
        classes = tuple(dict.fromkeys((*mro[:-1], *_PYYAML_CLASSES)))
        attributes = tuple(dict(vars(cls)) for cls in classes)
        tables = _table_state(loader)
    except Exception:  # noqa: BLE001
        return None
    if known is not None:
        _cached_libyaml_load.cache_clear()
    twin = type(
        f"Fast{loader.__name__}", (_C_SAFE_LOADER,), dict(zip(_TABLES, tables, strict=True))
    )
    _TWINS[loader] = (twin, classes, attributes, tables)
    return twin


def _may_nest_deeply(text: str) -> bool:
    """Whether the token depth can exceed the limit: flow nesting is at most the number of
    openers, and block nesting at most one more than the longest run of spaces and block
    indicators that starts a line."""
    room = _MAX_TOKEN_DEPTH - 1 - text.count("[") - text.count("{")
    return room < 0 or re.search(rf"(?m)^[ ?:-]{{{room + 1},}}", text) is not None


def _needs_pyyaml(text: str, fast: type) -> bool:
    """Whether the document has a tag, which libyaml and PyYAML scan differently, nesting too
    deep for libyaml's composer, a plain scalar inside a flow collection holding ``?``, which
    ends it in PyYAML, or a token after an empty explicit flow key, which libyaml skips.

    Past 50 levels PyYAML nears its RecursionError while libyaml's C composer may overflow."""
    if "?" not in text and "!" not in text and not _may_nest_deeply(text):
        return False
    depth = flow = 0
    previous: type | None = None
    for token in yaml.scan(text, Loader=fast):
        kind = type(token)
        if kind in _OPENING_TOKENS:
            depth += 1
            if depth > _MAX_TOKEN_DEPTH:
                return True
        elif kind in _CLOSING_TOKENS:
            depth -= 1
        if (
            kind is yaml.TagToken
            or (flow and kind is yaml.ScalarToken and token.plain and "?" in token.value)
            or (flow and previous is yaml.KeyToken and kind in _AFTER_EMPTY_FLOW_KEY)
        ):
            return True
        flow += _FLOW_TOKENS.get(kind, 0)
        previous = kind
    return False


def _libyaml_load(text: str, fast: type) -> Any:
    """libyaml's result for ``text``, or ``_PYYAML`` where PyYAML has to decide.

    Directives, tabs, byte order marks, a comment glued to a block scalar header, characters
    PyYAML's reader refuses, and line breaks other than newline all go to PyYAML."""
    if _DIVERGENT.search(text) or yaml.reader.Reader.NON_PRINTABLE.search(text):
        return _PYYAML
    try:
        if _needs_pyyaml(text, fast):
            return _PYYAML
        return yaml.load(text, Loader=fast)  # noqa: S506
    except Exception:  # noqa: BLE001
        return _PYYAML


@functools.lru_cache(maxsize=512)
def _cached_libyaml_load(text: str, fast: type, int_digits: int) -> Any:  # noqa: ARG001
    """Return the libyaml result for ``text``, cached per loader and integer digit limit."""
    return _libyaml_load(text, fast)


def _fast_value(text: str, fast: type) -> Any:
    """Return a fresh copy of the libyaml result for ``text``, or ``_PYYAML`` for PyYAML to decide.

    Only texts up to 128 KiB are cached, which covers the largest srt recipe at 110 KiB. Alias
    chains too deep to copy go to PyYAML."""
    if len(text) > _CACHED_TEXT_CHARACTERS:
        return _libyaml_load(text, fast)
    value = _cached_libyaml_load(text, fast, sys.get_int_max_str_digits())
    if value is _PYYAML:
        return _PYYAML
    try:
        return fastcopy.deepcopy(value)
    except RecursionError:
        return _PYYAML


def load(stream: str | IO[str], loader: type = yaml.SafeLoader) -> Any:
    """Return exactly what ``yaml.load(stream, Loader=loader)`` returns or raises, except
    within a few frames of the interpreter's recursion limit, where either one can raise
    RecursionError while the other returns.

    A ``str`` or a seekable ``io.TextIOWrapper``/``io.StringIO`` whose text libyaml accepts is
    parsed by libyaml, which builds the same nodes as PyYAML; everything else, including every
    error, goes to PyYAML from the stream's original position. Texts up to 128 KiB are parsed
    once per process and returned as fresh deep copies. ``INFX_PURE_YAML=1`` turns libyaml off.
    PyYAML internals patched before this module is imported, or mutated in place (such as
    ``Scanner.ESCAPE_REPLACEMENTS``), are not detected. Each loader class is kept for the life
    of the process. A stream whose ``tell()`` fails, as after iteration or closing, goes to PyYAML.
    """
    fast = _fast_loader(loader)
    if fast is not None and type(stream) is str:
        value = _fast_value(stream, fast)
        if value is not _PYYAML:
            return value
    elif fast is not None and type(stream) in (io.TextIOWrapper, io.StringIO):
        try:
            start = stream.tell() if stream.seekable() else None
        except (OSError, ValueError):
            start = None
        if start is not None:
            try:
                value = _fast_value(stream.read(), fast)
            except (OSError, ValueError):
                value = _PYYAML
            if value is not _PYYAML:
                return value
            stream.seek(start)
    return yaml.load(stream, Loader=loader)  # noqa: S506


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
    ``flatten_mapping`` rewrites any of them in place. Checked nodes stay referenced on the
    loader so a later document cannot reuse their ids."""
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
