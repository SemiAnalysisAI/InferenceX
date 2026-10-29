"""Golden curve identities: which runs select a curve, how it was collected, and the audit.

``identities.yaml`` gives every curve a ``select`` block, which ``select_curve`` matches against
a run's speculative config, and a ``collection`` block, which ``compare`` checks a run against.
The curve file name is derived from the identity (``canonical_name``).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from infx import yamlio

IDENTITY_DIR = Path(__file__).resolve().parent
IDENTITIES_STEM = "identities"
SOURCES = ("explicit", "documented", "assumed", "default", "unknown")
FACTS = (
    "draft_sample", "verify", "adaptive", "topology",
    "draft_precision", "draft_kv_cache_dtype", "kv_cache_dtype",
)  # fmt: skip
ENGINES = ("vllm", "sglang", "trtllm", "atom", "tilert")
METHODS = ("mtp", "eagle3", "dspark", "dflash")
DRAFT_FAMILIES = ("default", "gqa")
SAMPLERS = ("greedy", "probabilistic")


@dataclass(frozen=True)
class Fact:
    """A drafting or verification setting and how it is known; null when default or unknown."""

    value: Any
    source: str

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"Fact source {self.source!r} is not one of {', '.join(SOURCES)}")
        if (self.value is None) != (self.source in ("default", "unknown")):
            raise ValueError(
                "A fact has a null value exactly when its source is default or unknown"
            )

    @property
    def known(self) -> bool:
        return self.value is not None

    def render(self) -> str:
        return self.source if not self.known else f"{self.value} ({self.source})"


@dataclass(frozen=True)
class Collection:
    """How drafting and verification ran, for a curve's collection or for a launch."""

    engine: str
    draft: str
    draft_sample: Fact
    verify: Fact
    adaptive: Fact
    topology: Fact
    draft_precision: Fact
    draft_kv_cache_dtype: Fact
    kv_cache_dtype: Fact


@dataclass(frozen=True)
class Selector:
    """The runs that select a curve."""

    model_prefixes: tuple[str, ...]
    method: str
    draft_family: str | None = None
    draft_sample: str | None = None


@dataclass(frozen=True)
class CurveIdentity:
    name: str
    select: Selector
    collection: Collection
    provisional: bool = False


@dataclass(frozen=True)
class Finding:
    """One way a run differs from its curve's collection.

    ``kind`` is ``mismatch`` (both values known and different), ``unverified`` (either side
    is an engine default or unknown, and they are not the same engine's default),
    ``adaptive`` (the run adapts verification, so no fixed-k curve applies), ``provisional``
    (the curve was not measured for this model) or ``engine``.
    """

    kind: str
    field: str
    run: str
    curve: str

    def as_list(self) -> list[str]:
        return [self.kind, self.field, self.run, self.curve]


def normalize_method(model_prefix: str, method: str) -> str:
    """The identity vocabulary's method for an engine's method name.

    SGLang names a model-native MTP head EAGLE or NEXTN; MiniMax-M3's EAGLE head is EAGLE3.
    """
    method = method.lower()
    if method in ("eagle", "nextn"):
        return "eagle3" if model_prefix == "minimaxm3" else "mtp"
    return method


def draft_family(draft_model: str) -> str:
    return "gqa" if "gqa" in draft_model.lower() else "default"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", text.rsplit("/", 1)[-1].lower()).strip("-")


def canonical_name(model: str, identity: CurveIdentity) -> str:
    """``<model>__<method>__<draft>__<draft_sample>-<verify>``, "default" for unset settings."""
    collection = identity.collection
    parts = [
        _slug(model),
        identity.select.method,
        _slug(collection.draft),
        "-".join(
            str(fact.value) if fact.source == "explicit" else "default"
            for fact in (collection.draft_sample, collection.verify)
        ),
    ]
    return "__".join(parts)


def _fact(field: str, data: Any) -> Fact:
    if not isinstance(data, Mapping) or set(data) != {"value", "source"}:
        raise ValueError(f"{field} must be {{value, source}}")
    return Fact(data["value"], data["source"])


def parse_identity(name: str, data: Any) -> CurveIdentity:
    """One curve's identity from ``identities.yaml``, or ValueError naming the curve."""
    try:
        if not isinstance(data, Mapping):
            raise ValueError("an identity must be a mapping")
        select, collection = data["select"], data["collection"]
        prefixes = tuple(select["model_prefixes"])
        selector = Selector(
            model_prefixes=prefixes,
            method=select["method"],
            draft_family=select.get("draft_family"),
            draft_sample=select.get("draft_sample"),
        )
        if not prefixes or selector.method not in METHODS:
            raise ValueError("select needs model_prefixes and a known method")
        if selector.draft_family not in (None, *DRAFT_FAMILIES):
            raise ValueError(f"draft_family must be one of {', '.join(DRAFT_FAMILIES)}")
        if selector.draft_sample not in (None, *SAMPLERS):
            raise ValueError(f"select draft_sample must be one of {', '.join(SAMPLERS)}")
        identity = CurveIdentity(
            name=name,
            select=selector,
            collection=Collection(
                engine=collection["engine"],
                draft=collection["draft"],
                **{field: _fact(field, collection[field]) for field in FACTS},
            ),
            provisional=data.get("provisional", False) is True,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid identity of golden curve {name}: {error}") from error
    if identity.collection.engine not in ENGINES:
        raise ValueError(f"Identity of {name} names unknown engine {identity.collection.engine!r}")
    return identity


def load_identities(golden_dir: Path = IDENTITY_DIR) -> dict[str, CurveIdentity]:
    """Every curve identity in ``golden_dir``'s ``identities.yaml``, by curve name."""
    path = golden_dir / f"{IDENTITIES_STEM}.yaml"
    if not path.is_file():
        raise ValueError(f"No committed golden curve identities {path.name} in {golden_dir}")
    data = yamlio.safe_load(path.read_text())
    if not isinstance(data, Mapping):
        raise ValueError(f"{IDENTITIES_STEM}.yaml must map curve names to identities")
    return {str(name): parse_identity(str(name), value) for name, value in data.items()}


def _selects(selector: Selector, model_prefix: str, spec: Mapping[str, Any]) -> bool:
    if model_prefix not in selector.model_prefixes:
        return False
    if normalize_method(model_prefix, str(spec.get("method", ""))) != selector.method:
        return False
    if selector.draft_family and draft_family(str(spec.get("model", ""))) != selector.draft_family:
        return False
    return not selector.draft_sample or spec.get("draft_sample_method") == selector.draft_sample


def select_curve(
    model_prefix: str, spec: Mapping[str, Any], golden_dir: Path = IDENTITY_DIR
) -> str:
    """The one curve whose ``select`` block matches, or ValueError naming the near misses."""
    identities = load_identities(golden_dir).values()
    matches = [i.name for i in identities if _selects(i.select, model_prefix, spec)]
    if len(matches) == 1:
        return matches[0]
    method = normalize_method(model_prefix, str(spec.get("method", "")))
    if matches:
        raise ValueError(f"{model_prefix} {method} matches several golden curves: {matches}")
    near = [
        i.select
        for i in identities
        if model_prefix in i.select.model_prefixes and i.select.method == method
    ]
    if near:
        options = "; ".join(
            ", ".join(
                f"{key}={value}"
                for key, value in (
                    ("draft_family", s.draft_family),
                    ("draft_sample", s.draft_sample),
                )
                if value
            )
            for s in near
        )
        raise ValueError(
            f"No {model_prefix} {method} golden curve for draft model "
            f"{spec.get('model')!r} and draft sampling {spec.get('draft_sample_method')!r}; "
            f"curves need {options}"
        )
    raise ValueError(f"No golden curve for model prefix {model_prefix!r} and method {method!r}")


def compare(run: Collection, curve: CurveIdentity) -> list[Finding]:
    """Where ``run`` differs from ``curve``'s collection, in field order."""
    theirs_all = curve.collection
    same_engine = run.engine == theirs_all.engine
    findings = []
    if run.adaptive.value is True:
        findings.append(
            Finding("adaptive", "adaptive", run.adaptive.render(), theirs_all.adaptive.render())
        )
    for field in FACTS:
        mine, theirs = getattr(run, field), getattr(theirs_all, field)
        if field == "adaptive" and mine.value is True:
            continue
        if mine.known and theirs.known:
            if mine.value != theirs.value:
                findings.append(Finding("mismatch", field, mine.render(), theirs.render()))
        elif not (same_engine and mine.source == theirs.source == "default"):
            findings.append(Finding("unverified", field, mine.render(), theirs.render()))
    if curve.provisional:
        findings.append(Finding("provisional", "values", "-", "not measured for this model"))
    if not same_engine:
        findings.append(Finding("engine", "engine", run.engine, theirs_all.engine))
    return findings
