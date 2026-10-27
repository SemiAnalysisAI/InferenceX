"""PyYAML loading that parses each repeated text once per process."""

from __future__ import annotations

import functools
from collections.abc import Iterator
from typing import IO, Any

import yaml
from yaml.constructor import ConstructorError
from yaml.resolver import BaseResolver

from infx import fastcopy

_MERGE_TAG = "tag:yaml.org,2002:merge"
_VALUE_TAG = "tag:yaml.org,2002:value"
_CACHED_TEXT_CHARACTERS = 128 * 1024


@functools.lru_cache(maxsize=512)
def _parse(text: str, loader: type) -> Any:
    """Return ``yaml.load(text, Loader=loader)``, parsed once per text and loader."""
    return yaml.load(text, Loader=loader)  # noqa: S506


def load(stream: str | IO[str], loader: type = yaml.SafeLoader) -> Any:
    """Return what ``yaml.load(stream, Loader=loader)`` returns or raises.

    A ``str`` of at most 128 KiB is parsed once per process and loader, and every call gets a
    fresh deep copy. Streams and larger texts are parsed on each call. Constructors added to a
    loader after it parsed a text do not apply to that cached text.
    """
    if type(stream) is not str or len(stream) > _CACHED_TEXT_CHARACTERS:
        return yaml.load(stream, Loader=loader)  # noqa: S506
    value = _parse(stream, loader)
    try:
        return fastcopy.deepcopy(value)
    except RecursionError:
        return yaml.load(stream, Loader=loader)  # noqa: S506


def safe_load(stream: str | IO[str]) -> Any:
    """Return what ``yaml.safe_load(stream)`` returns, using the same cache as ``load``."""
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
