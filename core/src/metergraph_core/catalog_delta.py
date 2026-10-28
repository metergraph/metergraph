"""Pricing deltas: proposed changes to a catalog document.

A delta never serves prices. It names each target row by its identity
``(canonical_id, channel, region, effective_from)``, carries the row's
expected current state as a precondition, and carries the new
effective-dated value. The current schema supports price rows on models the
catalog already has:

- ``add_price`` appends a row; ``expect`` is null because no such row may
  exist.
- ``replace_price`` closes the open row at the new row's ``effective_from``
  and appends the new row. History is never rewritten.
- ``close_price`` closes the open row without a replacement.

It also supports reviewed catalog expansion:

- ``add_model`` appends one complete model entry.
- ``add_alias`` appends one alias to an existing model. A spelling another
  model holds conflicts only while both aliases are in effect, so a spelling
  can move to another model from a date.
- ``close_alias`` sets ``effective_to`` on an open alias. With an
  ``add_alias`` for the same spelling on another model in one delta, it
  moves that spelling: calls before the date keep the old model, calls from
  it resolve to the new one.

``provenance`` references the observations and evidence the delta came
from.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from .loader import CatalogError, _canonical_json, load_catalog_document

CATALOG_DELTA_SCHEMA_VERSION = 2
PRICING_DELTA_KIND = "pricing_delta"
PRICE_OPERATIONS = ("add_price", "replace_price", "close_price")
CATALOG_OPERATIONS = ("add_model", "add_alias", "close_alias")
OPERATIONS = PRICE_OPERATIONS + CATALOG_OPERATIONS

RATE_FIELDS = (
    "input_per_mtok",
    "output_per_mtok",
    "cache_read_per_mtok",
    "cache_write_5m_per_mtok",
    "cache_write_1h_per_mtok",
    "batch_input_per_mtok",
    "batch_output_per_mtok",
    "batch_cache_read_per_mtok",
    "batch_cache_write_5m_per_mtok",
    "request_fee_per_1k_usd",
)
_PRICE_FIELDS = frozenset(
    ("channel", "region", "effective_from", "source_url", "rules") + RATE_FIELDS
)
_KEY_FIELDS = ("canonical_id", "channel", "region", "effective_from")


class CatalogDeltaError(CatalogError):
    """A delta document that does not follow the schema."""


@dataclass(frozen=True, slots=True)
class PriceRowKey:
    canonical_id: str
    channel: str
    region: str
    effective_from: str


@dataclass(frozen=True, slots=True)
class ModelKey:
    canonical_id: str


@dataclass(frozen=True, slots=True)
class AliasKey:
    canonical_id: str
    provider: str
    alias: str
    channel: str


OperationKey = PriceRowKey | ModelKey | AliasKey


@dataclass(frozen=True, slots=True)
class DeltaConflict:
    """One operation whose precondition does not hold against the document."""

    index: int
    op: str
    key: OperationKey
    reason: str
    detail: str = ""


class CatalogDeltaConflict(CatalogDeltaError):
    """The delta does not apply; ``conflicts`` names every failing operation."""

    def __init__(self, conflicts: tuple[DeltaConflict, ...]) -> None:
        self.conflicts = conflicts
        summary = "; ".join(
            f"operation {c.index} {c.op} {_key_label(c.key)}: {c.reason}"
            + (f" ({c.detail})" if c.detail else "")
            for c in conflicts
        )
        super().__init__(f"delta conflicts: {summary}")


@dataclass(frozen=True, slots=True)
class DeltaOperation:
    index: int
    op: str
    key: OperationKey
    # The exact current row for replace/close; None for add.
    expect: Mapping[str, Any] | None
    # The new row for add/replace; None for close.
    price: Mapping[str, Any] | None
    # The closing date for close_price and close_alias.
    effective_to: str | None
    # The complete entry for add_model.
    model: Mapping[str, Any] | None = None
    # The alias row for add_alias.
    alias: Mapping[str, Any] | None = None
    # The source a close_alias cites, for an alias that cites none.
    source_url: str | None = None


@dataclass(frozen=True, slots=True)
class CatalogDelta:
    schema_version: int
    operations: tuple[DeltaOperation, ...]
    provenance: Mapping[str, Any]


def _instant(value: Any) -> datetime | None:
    """An ISO date or datetime as an aware UTC instant, or None when
    ``value`` is not one. A value without an offset is UTC, as the catalog
    loader reads it."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime(value.year, value.month, value.day)
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text[-1] in "Zz":
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso_date(value: Any) -> str | None:
    """The canonical text of an effective date: ``YYYY-MM-DD`` for UTC
    midnight, otherwise the UTC ISO datetime. Two spellings of one instant
    get one text, so they name one row. None when ``value`` is not a date."""
    instant = _instant(value)
    if instant is None:
        return None
    if instant.time() == time(0):
        return instant.date().isoformat()
    return instant.isoformat()


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _parse_key(raw: Any, label: str) -> PriceRowKey:
    if not isinstance(raw, Mapping):
        raise CatalogDeltaError(f"{label}: key must be a mapping")
    for field in ("canonical_id", "channel", "region"):
        if _text(raw.get(field)) is None:
            raise CatalogDeltaError(f"{label}: key needs {field}")
    if raw.get("effective_from") is None:
        raise CatalogDeltaError(f"{label}: key needs effective_from")
    effective_from = _iso_date(raw.get("effective_from"))
    if effective_from is None:
        raise CatalogDeltaError(f"{label}: invalid key effective_from")
    return PriceRowKey(
        canonical_id=_text(raw["canonical_id"]),
        channel=_text(raw["channel"]),
        region=_text(raw["region"]),
        effective_from=effective_from,
    )


def _parse_model_key(raw: Any, label: str) -> ModelKey:
    if not isinstance(raw, Mapping):
        raise CatalogDeltaError(f"{label}: key must be a mapping")
    if set(raw) != {"canonical_id"} or _text(raw.get("canonical_id")) is None:
        raise CatalogDeltaError(f"{label}: add_model key needs only canonical_id")
    return ModelKey(_text(raw["canonical_id"]))


def _parse_alias_key(raw: Any, label: str, op: str) -> AliasKey:
    if not isinstance(raw, Mapping):
        raise CatalogDeltaError(f"{label}: key must be a mapping")
    fields = ("canonical_id", "provider", "alias", "channel")
    for field in fields:
        if _text(raw.get(field)) is None:
            raise CatalogDeltaError(f"{label}: key needs {field}")
    if set(raw) != set(fields):
        raise CatalogDeltaError(f"{label}: {op} key has unknown fields")
    return AliasKey(
        _text(raw["canonical_id"]),
        _text(raw["provider"]).lower(),
        _text(raw["alias"]).lower(),
        _text(raw["channel"]),
    )


def _decimal_string(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        return False
    return parsed.is_finite() and parsed >= 0


def _parse_price(raw: Any, key: PriceRowKey, label: str, *, same_date: bool) -> dict:
    if not isinstance(raw, Mapping):
        raise CatalogDeltaError(f"{label}: price must be a mapping")
    if "effective_to" in raw:
        raise CatalogDeltaError(f"{label}: a new price row is open; drop effective_to")
    for field in raw:
        if field not in _PRICE_FIELDS:
            raise CatalogDeltaError(f"{label}: price has unknown field {field!r}")
    for field in ("channel", "region"):
        if _text(raw.get(field)) != getattr(key, field):
            raise CatalogDeltaError(f"{label}: price {field} must match key")
    effective_from = _iso_date(raw.get("effective_from"))
    if effective_from is None:
        raise CatalogDeltaError(f"{label}: invalid price effective_from")
    if same_date and effective_from != key.effective_from:
        raise CatalogDeltaError(f"{label}: price effective_from must match key")
    if _text(raw.get("source_url")) is None:
        raise CatalogDeltaError(f"{label}: price needs source_url")
    for field in RATE_FIELDS:
        if field in raw and not _decimal_string(raw[field]):
            raise CatalogDeltaError(
                f"{label}: price {field} must be a non-negative decimal string"
            )
    if "rules" in raw and not isinstance(raw["rules"], Mapping):
        raise CatalogDeltaError(f"{label}: price rules must be a mapping")
    price = copy.deepcopy(dict(raw))
    # Store the values that were validated, so the row lands exactly where
    # its key says and cites its source without stray whitespace.
    price["channel"] = key.channel
    price["region"] = key.region
    price["source_url"] = _text(raw["source_url"])
    price["effective_from"] = effective_from
    return price


def _probe_document(model: Mapping[str, Any]) -> None:
    load_catalog_document(
        {
            "version": "probe",
            "currency": "USD",
            "pricing_verified_at": "2026-10-20",
            "models": [copy.deepcopy(dict(model))],
        }
    )


def _parse_model(raw: Any, key: ModelKey, label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise CatalogDeltaError(f"{label}: invalid model: entry must be a mapping")
    model = copy.deepcopy(dict(raw))
    if _text(model.get("canonical_id")) != key.canonical_id:
        raise CatalogDeltaError(f"{label}: model canonical_id must match key")
    if _text(model.get("publisher")) is None:
        raise CatalogDeltaError(f"{label}: invalid model: entry needs publisher")
    if not isinstance(model.get("aliases"), list) or not model["aliases"]:
        raise CatalogDeltaError(f"{label}: invalid model: entry needs aliases")
    if not isinstance(model.get("prices"), list) or not model["prices"]:
        raise CatalogDeltaError(f"{label}: invalid model: entry needs prices")
    try:
        _probe_document(model)
    except CatalogError as exc:
        raise CatalogDeltaError(f"{label}: invalid model: {exc}") from exc
    return model


def _parse_alias(raw: Any, key: AliasKey, label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise CatalogDeltaError(f"{label}: alias must be a mapping")
    alias = copy.deepcopy(dict(raw))
    for field in ("provider", "alias", "channel"):
        value = _text(alias.get(field))
        if value is None:
            raise CatalogDeltaError(f"{label}: alias needs {field}")
        normalized = value.lower() if field in ("provider", "alias") else value
        if normalized != getattr(key, field):
            raise CatalogDeltaError(f"{label}: alias {field} must match key")
        alias[field] = normalized
    try:
        _probe_document(
            {
                "canonical_id": key.canonical_id,
                "publisher": "probe",
                "aliases": [alias],
                "prices": [],
            }
        )
    except CatalogError as exc:
        raise CatalogDeltaError(f"{label}: invalid alias: {exc}") from exc
    return alias


def _parse_operation(index: int, raw: Any) -> DeltaOperation:
    label = f"operation {index}"
    if not isinstance(raw, Mapping):
        raise CatalogDeltaError(f"{label}: must be a mapping")
    op = raw.get("op")
    if op not in OPERATIONS:
        raise CatalogDeltaError(f"{label}: unknown op {op!r}")
    if op == "add_model":
        key: OperationKey = _parse_model_key(raw.get("key"), label)
    elif op in ("add_alias", "close_alias"):
        key = _parse_alias_key(raw.get("key"), label, op)
    else:
        key = _parse_key(raw.get("key"), label)
    expect = raw.get("expect")
    if op in ("add_price", "add_model", "add_alias"):
        if "expect" not in raw or expect is not None:
            raise CatalogDeltaError(f"{label}: {op} expect must be null")
    elif not isinstance(expect, Mapping):
        raise CatalogDeltaError(f"{label}: {op} expect must be the full current row")
    price = None
    effective_to = None
    model = None
    alias = None
    source_url = None
    if op == "add_model":
        if "price" in raw or "alias" in raw or "effective_to" in raw:
            raise CatalogDeltaError(f"{label}: add_model takes only model")
        model = _parse_model(raw.get("model"), key, label)
    elif op == "add_alias":
        if "price" in raw or "model" in raw or "effective_to" in raw:
            raise CatalogDeltaError(f"{label}: add_alias takes only alias")
        alias = _parse_alias(raw.get("alias"), key, label)
    elif op == "close_alias":
        if set(raw) - {"op", "key", "expect", "effective_to", "source_url"}:
            raise CatalogDeltaError(
                f"{label}: close_alias takes only effective_to and source_url"
            )
        effective_to = _iso_date(raw.get("effective_to"))
        if effective_to is None:
            raise CatalogDeltaError(f"{label}: invalid effective_to")
        if "source_url" in raw:
            source_url = _text(raw.get("source_url"))
            if source_url is None:
                raise CatalogDeltaError(f"{label}: invalid source_url")
    elif op == "close_price":
        if "price" in raw:
            raise CatalogDeltaError(f"{label}: close_price takes no price")
        effective_to = _iso_date(raw.get("effective_to"))
        if effective_to is None:
            raise CatalogDeltaError(f"{label}: invalid effective_to")
        if _instant(effective_to) <= _instant(key.effective_from):
            raise CatalogDeltaError(
                f"{label}: effective_to must follow key effective_from"
            )
    else:
        price = _parse_price(
            raw.get("price"), key, label, same_date=op == "add_price"
        )
    return DeltaOperation(
        index=index,
        op=op,
        key=key,
        expect=copy.deepcopy(dict(expect)) if expect is not None else None,
        price=price,
        effective_to=effective_to,
        model=model,
        alias=alias,
        source_url=source_url,
    )


def parse_catalog_delta(delta_document: Mapping[str, Any]) -> CatalogDelta:
    """Validate a delta document without applying
    it. Raises :class:`CatalogDeltaError` naming the first malformed part."""
    if not isinstance(delta_document, Mapping):
        raise CatalogDeltaError("delta document must be a mapping")
    version = delta_document.get("schema_version")
    if version != CATALOG_DELTA_SCHEMA_VERSION:
        raise CatalogDeltaError(f"unsupported delta schema_version {version}")
    if delta_document.get("kind") != PRICING_DELTA_KIND:
        raise CatalogDeltaError(f"delta kind must be {PRICING_DELTA_KIND}")
    provenance = delta_document.get("provenance")
    if not isinstance(provenance, Mapping):
        raise CatalogDeltaError("delta needs provenance")
    if not isinstance(provenance.get("observations"), list):
        raise CatalogDeltaError("delta provenance needs an observations list")
    if "evidence" in provenance and not isinstance(provenance["evidence"], list):
        raise CatalogDeltaError("delta provenance evidence must be a list")
    raw_operations = delta_document.get("operations")
    if not isinstance(raw_operations, list):
        raise CatalogDeltaError("delta operations must be a list")
    operations: list[DeltaOperation] = []
    seen: set[tuple[str, OperationKey]] = set()
    for index, raw in enumerate(raw_operations):
        operation = _parse_operation(index, raw)
        family = "price" if operation.op in PRICE_OPERATIONS else operation.op
        target = (family, operation.key)
        if target in seen:
            raise CatalogDeltaError(f"operation {index}: duplicate target")
        seen.add(target)
        operations.append(operation)
    return CatalogDelta(
        schema_version=version,
        operations=tuple(operations),
        provenance=copy.deepcopy(dict(provenance)),
    )


def row_key(canonical_id: str, row: Mapping[str, Any]) -> PriceRowKey | None:
    """The identity of a catalog price row, or None when it has no readable
    channel or effective date."""
    channel = _text(row.get("channel"))
    effective_from = _iso_date(row.get("effective_from"))
    if channel is None or effective_from is None:
        return None
    return PriceRowKey(
        canonical_id=canonical_id,
        channel=channel,
        region=_text(row.get("region")) or "global",
        effective_from=effective_from,
    )


def _locate(
    document: Mapping[str, Any], operation: DeltaOperation
) -> tuple[int | None, list[int], str | None]:
    """The target model's index and the indexes of its rows with the
    operation's key, or the reason the target cannot be located."""
    key = operation.key
    models = [
        index
        for index, entry in enumerate(document.get("models") or [])
        if isinstance(entry, Mapping) and entry.get("canonical_id") == key.canonical_id
    ]
    if not models:
        return None, [], "unknown_model"
    if len(models) > 1:
        return None, [], "duplicate_model"
    entry = document["models"][models[0]]
    rows = [
        index
        for index, row in enumerate(entry.get("prices") or [])
        if isinstance(row, Mapping) and row_key(key.canonical_id, row) == key
    ]
    if len(rows) > 1:
        return models[0], rows, "duplicate_row"
    return models[0], rows, None


def _model_indexes(document: Mapping[str, Any], canonical_id: str) -> list[int]:
    return [
        index
        for index, entry in enumerate(document.get("models") or [])
        if isinstance(entry, Mapping) and entry.get("canonical_id") == canonical_id
    ]


def _aliases(document: Mapping[str, Any]):
    for entry in document.get("models") or []:
        if not isinstance(entry, Mapping):
            continue
        canonical_id = str(entry.get("canonical_id") or "")
        for alias in entry.get("aliases") or []:
            if not isinstance(alias, Mapping):
                continue
            provider = _text(alias.get("provider"))
            name = _text(alias.get("alias"))
            if provider and name:
                yield canonical_id, provider.lower(), name.lower()


def _key_label(key: OperationKey) -> str:
    if isinstance(key, PriceRowKey):
        return f"{key.canonical_id} {key.channel}/{key.region}@{key.effective_from}"
    if isinstance(key, AliasKey):
        return f"{key.canonical_id} {key.provider}:{key.alias}@{key.channel}"
    return key.canonical_id


def _alias_rows(
    entry: Mapping[str, Any], key: AliasKey
) -> list[Mapping[str, Any]]:
    return [
        alias
        for alias in entry.get("aliases") or []
        if isinstance(alias, Mapping)
        and (_text(alias.get("provider")) or "").lower() == key.provider
        and (_text(alias.get("alias")) or "").lower() == key.alias
        and _text(alias.get("channel")) == key.channel
    ]


def _windows_overlap(
    left_from: Any, left_to: Any, right_from: Any, right_to: Any
) -> bool:
    """Whether two alias windows share an instant; a missing bound is open,
    as the catalog loader reads it."""
    left_from, left_to = _instant(left_from), _instant(left_to)
    right_from, right_to = _instant(right_from), _instant(right_to)
    return (left_to is None or right_from is None or right_from < left_to) and (
        right_to is None or left_from is None or left_from < right_to
    )


def _precondition(
    document: Mapping[str, Any],
    operation: DeltaOperation,
    closing: Mapping[AliasKey, str] | None = None,
) -> str | None:
    """Why ``operation`` does not apply to ``document``, or None when it does.

    ``closing`` maps the key of each alias a close_alias in the same delta
    ends to its ``effective_to``, so an alias can move in one delta."""
    if operation.op == "add_model":
        key = operation.key
        if _model_indexes(document, key.canonical_id):
            return "model_exists"
        known = {(provider, name) for _, provider, name in _aliases(document)}
        for alias in operation.model.get("aliases") or []:
            pair = (
                str(alias.get("provider") or "").strip().lower(),
                str(alias.get("alias") or "").strip().lower(),
            )
            if pair in known:
                return "alias_taken"
        return None
    if operation.op == "add_alias":
        key = operation.key
        models = _model_indexes(document, key.canonical_id)
        if not models:
            return "unknown_model"
        if len(models) > 1:
            return "duplicate_model"
        closing = closing or {}
        new = operation.alias or {}
        for entry in document.get("models") or []:
            if not isinstance(entry, Mapping):
                continue
            canonical_id = str(entry.get("canonical_id") or "")
            for alias in entry.get("aliases") or []:
                if not isinstance(alias, Mapping):
                    continue
                provider = (_text(alias.get("provider")) or "").lower()
                name = (_text(alias.get("alias")) or "").lower()
                if (provider, name) != (key.provider, key.alias):
                    continue
                ends = alias.get("effective_to")
                if ends is None:
                    ends = closing.get(
                        AliasKey(
                            canonical_id,
                            provider,
                            name,
                            _text(alias.get("channel")) or "",
                        )
                    )
                if _windows_overlap(
                    alias.get("effective_from"),
                    ends,
                    new.get("effective_from"),
                    new.get("effective_to"),
                ):
                    return (
                        "alias_exists"
                        if canonical_id == key.canonical_id
                        else "alias_conflict"
                    )
        return None
    if operation.op == "close_alias":
        key = operation.key
        models = _model_indexes(document, key.canonical_id)
        if not models:
            return "unknown_model"
        if len(models) > 1:
            return "duplicate_model"
        rows = _alias_rows(document["models"][models[0]], key)
        if not rows:
            return "alias_missing"
        if len(rows) > 1:
            return "duplicate_alias"
        row = rows[0]
        if _canonical_json(row) != _canonical_json(operation.expect):
            return "expect_mismatch"
        if row.get("effective_to") is not None:
            return "alias_not_open"
        starts = _instant(row.get("effective_from"))
        if starts is not None and _instant(operation.effective_to) <= starts:
            return "not_after_start"
        # A dated alias must cite its source, so a close supplies one for an
        # alias that has none, and never replaces one it has.
        if _text(row.get("source_url")) is None:
            if operation.source_url is None:
                return "source_url_needed"
        elif operation.source_url is not None:
            return "source_url_exists"
        return None
    model_index, rows, reason = _locate(document, operation)
    if reason is not None:
        return reason
    if operation.op == "add_price":
        if rows:
            return "row_exists"
        # A price id names its row by UTC date, so a new row may not share a
        # UTC start date with any row on its model, channel and region, even
        # one whose window it does not overlap.
        key = operation.key
        day = _instant(key.effective_from).date()
        for other in document["models"][model_index].get("prices") or []:
            other_key = row_key(key.canonical_id, other) if isinstance(other, Mapping) else None
            if (
                other_key is not None
                and (other_key.channel, other_key.region) == (key.channel, key.region)
                and _instant(other_key.effective_from).date() == day
            ):
                return "price_id_collision"
        return None
    if not rows:
        return "row_missing"
    row = document["models"][model_index]["prices"][rows[0]]
    if row.get("effective_to") is not None:
        return "row_not_open"
    if _canonical_json(row) != _canonical_json(operation.expect):
        return "expect_mismatch"
    if operation.op == "replace_price":
        starts = _instant(operation.price["effective_from"])
        current = _instant(operation.key.effective_from)
        if starts <= current:
            return "not_after_current"
        # A price id names its row by UTC date, so a replacement must start
        # on a later UTC date than the row it closes to keep ids distinct.
        if starts.date() == current.date():
            return "same_day_replacement"
    return None


def apply_catalog_delta(
    document: Mapping[str, Any], delta_document: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a new catalog document with ``delta_document`` applied.

    Every operation's precondition is checked against ``document`` first; if
    any fails, :class:`CatalogDeltaConflict` lists all of them and nothing is
    returned. A replaced or closed row gets an ``effective_to``; new rows are
    appended; no other row changes. Neither input is mutated. The top-level
    ``version`` and ``pricing_verified_at`` are preserved: publishing a new
    catalog version is not this function's job. The complete result is
    validated with :func:`load_catalog_document`.
    """
    delta = parse_catalog_delta(delta_document)
    if not isinstance(document, Mapping):
        raise CatalogDeltaError("catalog document must be a mapping")
    closing = {
        operation.key: operation.effective_to
        for operation in delta.operations
        if operation.op == "close_alias"
    }
    conflicts = tuple(
        DeltaConflict(operation.index, operation.op, operation.key, reason)
        for operation in delta.operations
        if (reason := _precondition(document, operation, closing)) is not None
    )
    if conflicts:
        raise CatalogDeltaConflict(conflicts)

    result = copy.deepcopy(dict(document))
    for operation in delta.operations:
        if operation.op == "add_model":
            result.setdefault("models", []).append(
                copy.deepcopy(dict(operation.model))
            )
            continue
        if operation.op == "add_alias":
            [model_index] = _model_indexes(result, operation.key.canonical_id)
            result["models"][model_index].setdefault("aliases", []).append(
                copy.deepcopy(dict(operation.alias))
            )
            continue
        if operation.op == "close_alias":
            [model_index] = _model_indexes(result, operation.key.canonical_id)
            [alias] = _alias_rows(result["models"][model_index], operation.key)
            alias["effective_to"] = operation.effective_to
            if operation.source_url is not None:
                alias["source_url"] = operation.source_url
            continue
        model_index, rows, _ = _locate(result, operation)
        prices = result["models"][model_index].setdefault("prices", [])
        if operation.op == "add_price":
            prices.append(copy.deepcopy(dict(operation.price)))
        elif operation.op == "replace_price":
            prices[rows[0]]["effective_to"] = operation.price["effective_from"]
            prices.insert(rows[0] + 1, copy.deepcopy(dict(operation.price)))
        else:
            prices[rows[0]]["effective_to"] = operation.effective_to
    try:
        load_catalog_document(result)
    except CatalogError as exc:
        raise CatalogDeltaError(f"delta result is not a valid catalog: {exc}") from exc
    return result
