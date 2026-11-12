"""Stable model product-line families declared in the prices document.

A model entry may name the product line it belongs to, such as Claude Sonnet
across its versions. A family is display and grouping metadata only: it never
says one model succeeds, upgrades or can replace another, and it plays no part
in alias resolution, routing or pricing. A model without one has no known
family; nothing is ever inferred from its name.
"""

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

_FAMILY_FIELDS = frozenset(("id", "display_name"))
_FAMILY_ID = re.compile(r"[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*")


@dataclass(frozen=True, slots=True)
class ModelFamily:
    """A publisher's stable product line. ``id`` is ``<publisher>/<line>``."""

    id: str
    display_name: str


def parse_model_families(document: Mapping[str, Any], error: type[Exception]) -> dict[str, ModelFamily]:
    """Each model's declared family by canonical id. A model without
    ``family`` is left out. Raises ``error`` for a malformed family, one
    outside its model's publisher namespace, or one id given two labels."""
    families: dict[str, ModelFamily] = {}
    labels: dict[str, str] = {}
    for entry in document.get("models") or []:
        if "family" not in entry:
            continue
        canonical = str(entry.get("canonical_id") or "")
        raw = entry["family"]
        if not isinstance(raw, Mapping):
            raise error(f"{canonical}: family must be a mapping")
        unknown = sorted(str(field) for field in raw if field not in _FAMILY_FIELDS)
        if unknown:
            raise error(f"{canonical}: family has unknown field {unknown[0]!r}")
        family_id = raw.get("id")
        if not isinstance(family_id, str) or not family_id:
            raise error(f"{canonical}: family needs id")
        display_name = raw.get("display_name")
        if not isinstance(display_name, str) or not display_name.strip():
            raise error(f"{canonical}: family needs display_name")
        # A label is shown as written, so one a consumer would trim or drop
        # is rejected rather than quietly cleaned.
        if display_name != display_name.strip() or not display_name.isprintable():
            raise error(
                f"{canonical}: family display_name {display_name!r} must be one "
                "line without surrounding whitespace"
            )
        if not _FAMILY_ID.fullmatch(family_id):
            raise error(f"{canonical}: family id {family_id!r} must be publisher/name")
        publisher = str(entry.get("publisher") or "").strip().lower()
        if not publisher:
            raise error(f"{canonical}: family needs the model's publisher")
        if family_id.split("/", 1)[0] != publisher:
            raise error(
                f"{canonical}: family id {family_id!r} must start with publisher {publisher!r}"
            )
        if canonical.split("/", 1)[0] != publisher:
            raise error(
                f"{canonical}: a model with a family needs a canonical id under "
                f"publisher {publisher!r}"
            )
        known = labels.setdefault(family_id, display_name)
        if known != display_name:
            raise error(
                f"family {family_id!r}: conflicting display_name {known!r} and {display_name!r}"
            )
        families[canonical] = ModelFamily(family_id, display_name)
    return families


def common_family(
    families: Mapping[str, ModelFamily], canonical_ids: Iterable[str]
) -> ModelFamily | None:
    """The family every one of ``canonical_ids`` declares, or None when there
    are none, any has no family, or they differ."""
    found = {families.get(canonical_id) for canonical_id in canonical_ids}
    if len(found) != 1:
        return None
    return found.pop()
