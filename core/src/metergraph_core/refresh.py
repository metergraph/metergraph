"""Pure low-risk refresh planning and discovery evidence.

:func:`plan_low_risk_refresh` compares supplied source content with a
supplied catalog document and returns a pricing delta for the changes a
low-risk source may make, one observation per source (including runs with
no change, so a missing run is detectable), and an escalation for every
change it leaves out. It never fetches, applies the delta, or writes.

A low-risk change updates an existing rate, in one of the source's allowed
dimensions, on an existing row of an existing model, starting on the UTC
observation date. New models, new regions, new dimensions, models a page no longer
lists, ambiguous parses, format changes, invalid rates and backdated or
same-day effective dates escalate instead.

:func:`discovery_evidence` builds the deterministic evidence envelope a
reviewer needs for a source that is not low-risk or for an escalation.
"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Iterable, Mapping

from .catalog_delta import (
    CATALOG_DELTA_SCHEMA_VERSION,
    PRICING_DELTA_KIND,
    RATE_FIELDS,
    PriceRowKey,
    _instant,
    _iso_date,
    parse_catalog_delta,
    row_key,
)
from .extractors import (
    ExtractedRate,
    Extraction,
    extract,
    page_match,
    redact_evidence,
    regional_key,
)
from .loader import _canonical_json, _canonical_value, load_catalog_document
from .sources import PriceSource, SourceRegistry

# Provider names that are not a page's own spelling of a model.
_ALIAS_ONLY_PROVIDERS = {"litellm", "unknown"}


@dataclass(frozen=True, slots=True)
class Observation:
    """One source's result for one run."""

    id: str
    source_id: str
    url: str
    # unchanged | changed | escalated | changed_and_escalated | missing
    result: str
    fingerprint: str | None
    expected_fingerprint: str | None
    rows_compared: int
    changes: int
    escalations: int

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "url": self.url,
            "result": self.result,
            "fingerprint": self.fingerprint,
            "expected_fingerprint": self.expected_fingerprint,
            "rows_compared": self.rows_compared,
            "changes": self.changes,
            "escalations": self.escalations,
        }


@dataclass(frozen=True, slots=True)
class Escalation:
    """A change left out of the delta for review."""

    source_id: str
    reason: str
    key: PriceRowKey | None = None
    model_key: str | None = None
    region: str | None = None
    dimension: str | None = None
    current: str | None = None
    observed: str | None = None
    detail: str = ""
    result: str = "escalated"

    def as_json(self) -> dict[str, Any]:
        return {
            "result": self.result,
            "source_id": self.source_id,
            "reason": self.reason,
            "key": _key_json(self.key) if self.key else None,
            "model_key": self.model_key,
            "region": self.region,
            "dimension": self.dimension,
            "current": self.current,
            "observed": self.observed,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class RefreshResult:
    delta: dict[str, Any]
    observations: tuple[Observation, ...]
    escalations: tuple[Escalation, ...]

    def as_json(self) -> dict[str, Any]:
        return {
            # Rows may carry date or datetime values; report them as ISO text.
            "delta": _canonical_value(self.delta),
            "observations": [o.as_json() for o in self.observations],
            "escalations": [e.as_json() for e in self.escalations],
        }


def _key_json(key: PriceRowKey) -> dict[str, str]:
    return {
        "canonical_id": key.canonical_id,
        "channel": key.channel,
        "region": key.region,
        "effective_from": key.effective_from,
    }


def _money(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _rate(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except ArithmeticError:
        return None


def _valid_rate(value: Any) -> Decimal | None:
    """``value`` as a positive finite rate, or None when it is absent,
    unparsable, non-finite, zero or negative. The catalog loader accepts all
    of those, so the planner and discovery must check."""
    rate = _rate(value)
    return rate if rate is not None and rate.is_finite() and rate > 0 else None


def _carryable(value: Any) -> bool:
    """Whether a rate can be copied into a delta: finite and not negative."""
    rate = _rate(value)
    return rate is not None and rate.is_finite() and rate >= 0


def _rate_text(value: Any) -> str:
    """A rate as evidence text: normalized when valid, else as written."""
    rate = _valid_rate(value)
    return _money(rate) if rate is not None else str(value)


def _day(value: Any) -> date:
    """The UTC date of an effective date given as a date, a naive or aware
    datetime, or an ISO string with or without an offset."""
    instant = _instant(value)
    if instant is None:
        raise ValueError(f"invalid effective date {value!r}")
    return instant.date()


def _observation_instant(day: date) -> datetime:
    """A run observed on ``day`` sees the catalog as of 00:00 UTC that day."""
    return datetime(day.year, day.month, day.day, tzinfo=timezone.utc)


def _boundary(value: Any) -> datetime:
    instant = _instant(value)
    if instant is None:
        raise ValueError(f"invalid effective date {value!r}")
    return instant


def _in_effect(item: Mapping[str, Any], day: date, *, dated: bool) -> bool:
    """Whether ``item``'s window contains the observation instant of
    ``day``. Date-only boundaries are UTC midnight, so they keep their
    meaning; a datetime boundary is compared as the instant it names."""
    starts = item.get("effective_from")
    ends = item.get("effective_to")
    if starts is None and dated:
        return False
    at = _observation_instant(day)
    return (starts is None or _boundary(starts) <= at) and (ends is None or at < _boundary(ends))


def current_rows(document: Mapping[str, Any], channel: str, observed_on: date) -> list[dict[str, Any]]:
    """The price rows on ``channel`` in effect at the observation instant of
    ``observed_on``, each a copy labelled with its ``canonical_id``."""
    rows = []
    for entry in document.get("models") or []:
        for row in entry.get("prices") or []:
            if row.get("channel") == channel and _in_effect(row, observed_on, dated=True):
                labelled = {"canonical_id": entry["canonical_id"]}
                labelled.update(copy.deepcopy(row))
                rows.append(labelled)
    return rows


# --- Matching catalog rows to extracted rates ------------------------------------


@dataclass(frozen=True, slots=True)
class _RowMatch:
    canonical_id: str
    key: PriceRowKey
    row: Mapping[str, Any]
    rate: ExtractedRate | None
    duplicate_model: bool
    # Page entries for the row's other aliases whose rates differ from
    # ``rate``; any makes the row ambiguous.
    disagreeing: tuple[ExtractedRate, ...] = ()


def _names(entry: Mapping[str, Any], channel: str, day: date) -> list[str]:
    """Model ids a page might use for ``entry`` on ``channel``, most
    specific first."""
    on_channel = [a for a in entry.get("aliases") or [] if a.get("channel") == channel]
    names = [
        str(alias["alias"])
        for alias in on_channel
        if str(alias.get("provider") or "").lower() not in _ALIAS_ONLY_PROVIDERS
        and _in_effect(alias, day, dated=False)
    ]
    tail = str(entry["canonical_id"]).split("/", 1)[-1]
    # The canonical tail is a fallback spelling, except where the channel
    # defines that same name as an alias that is not in effect on ``day``.
    defined = {str(a.get("alias") or "").lower() for a in on_channel}
    active = {str(a.get("alias") or "").lower() for a in on_channel if _in_effect(a, day, dated=False)}
    if tail not in names and (tail.lower() not in defined or tail.lower() in active):
        names.append(tail)
    return names


def _match(
    source: PriceSource, extraction: Extraction, document: Mapping[str, Any], day: date
) -> tuple[list[_RowMatch], list[tuple[ExtractedRate, str]]]:
    """Each current catalog row on the source's channel with its extracted
    rate (or None), and every extracted rate no row claimed, with whether it
    is an unknown model or a new region of a known one."""
    by_key = {regional_key(r.model_key, r.region): r for r in extraction.rates}
    tables = {key: rate.rates for key, rate in by_key.items()}
    entries = [e for e in document.get("models") or [] if isinstance(e, Mapping)]
    counts = Counter(str(e.get("canonical_id")) for e in entries)
    matches: list[_RowMatch] = []
    claimed: set[str] = set()
    matched_models: set[str] = set()
    for entry in entries:
        canonical_id = str(entry["canonical_id"])
        names = _names(entry, source.channel, day)
        for row in entry.get("prices") or []:
            if row.get("channel") != source.channel or not _in_effect(row, day, dated=True):
                continue
            region = str(row.get("region") or "global")
            # Every page entry an active alias names belongs to this row: the
            # first is its rate, and any that disagrees makes it ambiguous.
            found: list[ExtractedRate] = []
            for name in names:
                rates = page_match(source.parser, tables, name, region)
                if rates is not None:
                    entry_rate = next(r for k, r in by_key.items() if tables[k] is rates)
                    if entry_rate not in found:
                        found.append(entry_rate)
            for entry_rate in found:
                claimed.add(regional_key(entry_rate.model_key, entry_rate.region))
                matched_models.add(entry_rate.model_key)
            first = found[0] if found else None
            matches.append(
                _RowMatch(
                    canonical_id,
                    row_key(canonical_id, row),
                    row,
                    first,
                    counts[canonical_id] > 1,
                    tuple(r for r in found[1:] if dict(r.rates) != dict(first.rates)),
                )
            )
    unclaimed = [
        (rate, "new_region" if rate.model_key in matched_models else "unknown_model")
        for key, rate in sorted(by_key.items())
        if key not in claimed
    ]
    return matches, unclaimed


# --- Low-risk refresh ----------------------------------------------------------------


def _compare_row(source: PriceSource, match: _RowMatch, ambiguous: Iterable[str], observed_on: date) -> tuple[dict | None, list[Escalation]]:
    """The replace_price operation for one matched row, or its escalations."""
    rate = match.rate
    row = match.row

    def escalate(reason: str, **fields: Any) -> Escalation:
        return Escalation(
            source.id, reason, key=match.key, model_key=rate.model_key, region=match.key.region, **fields
        )

    # Ambiguities carry the extraction's own regional key, so a duplicate in
    # one region never blocks, or hides behind, another region's row.
    if regional_key(rate.model_key, rate.region) in ambiguous:
        return None, [escalate("ambiguous_parse", detail="the page gives two candidate values")]
    if match.disagreeing:
        names = ", ".join(r.model_key for r in (rate,) + match.disagreeing)
        return None, [escalate("ambiguous_parse", detail=f"the page prices this model's aliases differently: {names}")]
    escalations: list[Escalation] = []
    # Every represented rate the row carries must still be on the page;
    # otherwise copying the row forward would keep a stale value.
    for dimension in source.dimensions:
        if row.get(dimension) is not None and dimension not in rate.rates:
            escalations.append(escalate("missing_rate", dimension=dimension, current=_rate_text(row[dimension])))
    changed: dict[str, Decimal] = {}
    for dimension in source.dimensions:
        observed = rate.rates.get(dimension)
        if observed is None:
            continue
        raw = row.get(dimension)
        current = _valid_rate(raw)
        values = {"dimension": dimension, "current": _rate_text(raw) if raw is not None else None, "observed": _money(observed)}
        if not observed.is_finite() or observed <= 0:
            escalations.append(escalate("non_positive_rate", **values))
        elif raw is None:
            escalations.append(escalate("new_dimension", **values))
        elif current is None:
            # A catalog rate that is not a positive finite number is not a
            # trustworthy row to compare or continue.
            escalations.append(escalate("invalid_current_rate", detail="the catalog rate is not a positive finite number", **values))
        elif observed == current:
            continue
        elif dimension not in source.low_risk.dimensions:
            escalations.append(escalate("dimension_not_allowed", **values))
        else:
            changed[dimension] = observed
    if escalations or not changed:
        return None, escalations
    if row.get("effective_to") is not None:
        # A replacement would need to fit a window that already ends; that
        # schedule is a reviewer's decision, not an automated one.
        return None, [escalate("scheduled_window", detail=f"the current row already ends {_iso_date(row['effective_to'])}")]
    unrepresented = [f for f in RATE_FIELDS if row.get(f) is not None and f not in source.dimensions]
    if unrepresented:
        return None, [escalate("unrepresented_dimension", dimension=unrepresented[0], detail="the source cannot update this rate with the others")]
    if _day(row["effective_from"]) >= observed_on:
        return None, [escalate("not_after_current", detail=f"the current row starts {match.key.effective_from}")]
    return _replace_operation(source, match, changed, observed_on), []


def _replace_operation(
    source: PriceSource, match: _RowMatch, changed: Mapping[str, Decimal], observed_on: date
) -> dict[str, Any]:
    """Close ``match``'s row on ``observed_on`` and continue it with the
    ``changed`` rates, citing the source."""
    row = match.row
    price: dict[str, Any] = {
        "channel": match.key.channel,
        "region": match.key.region,
        "effective_from": observed_on.isoformat(),
    }
    for field in RATE_FIELDS:
        value = changed.get(field, _rate(row.get(field)))
        if value is not None:
            price[field] = _money(value)
    if row.get("rules"):
        price["rules"] = copy.deepcopy(row["rules"])
    price["source_url"] = source.url
    return {
        "op": "replace_price",
        "key": _key_json(match.key),
        "expect": copy.deepcopy(dict(row)),
        "price": price,
    }


def _refresh_source(
    source: PriceSource, content: str | None, document: Mapping[str, Any], observed_on: date
) -> tuple[Observation, list[dict], list[Escalation]]:
    observation_id = f"{source.id}@{observed_on.isoformat()}"

    def observe(result: str, fingerprint: str | None, rows: int, operations: list, escalations: list) -> tuple:
        observation = Observation(
            observation_id, source.id, source.url, result, fingerprint, source.fingerprint,
            rows, len(operations), len(escalations),
        )
        return observation, operations, escalations

    if content is None:
        return observe("missing", None, 0, [], [Escalation(source.id, "content_missing", detail="no content was supplied for this source")])
    extraction = extract(source.parser, content)
    fingerprint = extraction.fingerprint
    if not source.low_risk.eligible:
        return observe("escalated", fingerprint, 0, [], [Escalation(source.id, "not_low_risk")])
    if source.fingerprint is None:
        return observe("escalated", fingerprint, 0, [], [Escalation(source.id, "fingerprint_missing", detail="the registry records no expected fingerprint")])
    if fingerprint != source.fingerprint:
        return observe("escalated", fingerprint, 0, [], [Escalation(source.id, "fingerprint_mismatch", detail=f"expected {source.fingerprint}")])
    if not extraction.rates:
        return observe("escalated", fingerprint, 0, [], [Escalation(source.id, "extraction_empty")])

    matches, unclaimed = _match(source, extraction, document, observed_on)
    operations: list[dict] = []
    escalations: list[Escalation] = []
    reported_duplicates: set[str] = set()
    compared = 0
    for match in matches:
        if match.duplicate_model:
            if match.canonical_id not in reported_duplicates:
                reported_duplicates.add(match.canonical_id)
                escalations.append(Escalation(source.id, "duplicate_model_row", key=match.key, region=match.key.region, detail="the catalog has more than one entry for this model"))
            continue
        if match.rate is None:
            escalations.append(Escalation(source.id, "model_not_listed", key=match.key, region=match.key.region, detail="the page no longer lists this row"))
            continue
        compared += 1
        operation, row_escalations = _compare_row(source, match, extraction.ambiguities, observed_on)
        escalations.extend(row_escalations)
        if operation is not None:
            operations.append(operation)
    for rate, reason in unclaimed:
        escalations.append(
            Escalation(
                source.id, reason, model_key=rate.model_key, region=rate.region,
                detail="the catalog has no row for this model" if reason == "unknown_model" else "the catalog has no row for this region",
            )
        )
    return observe(_result(operations, escalations), fingerprint, compared, operations, escalations)


def _result(operations: list, escalations: list) -> str:
    if operations and escalations:
        return "changed_and_escalated"
    if operations:
        return "changed"
    if escalations:
        return "escalated"
    return "unchanged"


def _names_list(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _resolve_claims(
    planned: list[tuple[Observation, list[dict], list[Escalation]]],
) -> list[tuple[Observation, list[dict], list[Escalation]]]:
    """Withdraw every operation on a row that more than one source proposes,
    whether or not they agree, and escalate each claim naming the others, so
    the delta never targets a row twice and no source silently wins."""
    claims: dict[tuple, list[tuple[str, dict]]] = {}
    for observation, operations, _ in planned:
        for operation in operations:
            claims.setdefault(tuple(sorted(operation["key"].items())), []).append(
                (observation.source_id, operation)
            )
    contested = {key: claim for key, claim in claims.items() if len({s for s, _ in claim}) > 1}
    if not contested:
        return planned
    resolved = []
    for observation, operations, escalations in planned:
        kept = []
        added = []
        for operation in operations:
            claim = contested.get(tuple(sorted(operation["key"].items())))
            if claim is None:
                kept.append(operation)
                continue
            sources = list(dict.fromkeys(s for s, _ in claim))
            # Each source cites its own page, so agreement ignores source_url.
            same = len({
                _canonical_json({k: v for k, v in op["price"].items() if k != "source_url"})
                for _, op in claim
            }) == 1
            expect, price = operation["expect"], operation["price"]
            changed = [
                f for f in RATE_FIELDS
                if price.get(f) is not None and (expect.get(f) is None or _money(_rate(expect[f])) != price[f])
            ]
            values = {}
            if len(changed) == 1:
                current = _rate(expect.get(changed[0]))
                values = {
                    "dimension": changed[0],
                    "current": _money(current) if current is not None else None,
                    "observed": price[changed[0]],
                }
            added.append(
                Escalation(
                    observation.source_id,
                    "conflicting_sources",
                    key=PriceRowKey(**operation["key"]),
                    region=operation["key"]["region"],
                    detail=(
                        f"{_names_list(sources)} both propose this row, with "
                        + ("the same values" if same else "different values")
                    ),
                    **values,
                )
            )
        escalations = escalations + added
        resolved.append(
            (
                replace(
                    observation,
                    result=_result(kept, escalations),
                    changes=len(kept),
                    escalations=len(escalations),
                ),
                kept,
                escalations,
            )
        )
    return resolved


def plan_low_risk_refresh(
    document: Mapping[str, Any],
    registry: SourceRegistry,
    contents: Mapping[str, str],
    *,
    observed_on: date,
) -> RefreshResult:
    """Plan a low-risk refresh of ``document`` from ``contents`` (source id
    -> fetched text). ``registry`` is the selected scope (narrow it to the
    sources a run covers): every source in it gets an observation, a source
    without content escalates as content_missing, and a source that is not
    low-risk escalates as not_low_risk. Content for a source outside the
    registry is rejected. The document is validated first and never
    modified."""
    load_catalog_document(document)
    for source_id in contents:
        registry.get(source_id)
    # The registry passed in is the selected scope: every source in it is
    # observed, so a source whose content could not be had is reported as
    # missing rather than silently skipped.
    sources = list(registry.sources)
    observations: list[Observation] = []
    operations: list[dict] = []
    escalations: list[Escalation] = []
    evidence: list[dict[str, Any]] = []
    planned = _resolve_claims(
        [_refresh_source(source, contents.get(source.id), document, observed_on) for source in sources]
    )
    for source, (observation, source_operations, source_escalations) in zip(sources, planned):
        observations.append(observation)
        operations.extend(source_operations)
        escalations.extend(source_escalations)
        if observation.fingerprint is not None:
            evidence.append({"source_id": source.id, "url": source.url, "fingerprint": observation.fingerprint})
    delta = {
        "schema_version": CATALOG_DELTA_SCHEMA_VERSION,
        "kind": PRICING_DELTA_KIND,
        "provenance": {
            "generator": "metergraph-core low-risk refresh",
            "observed_on": observed_on.isoformat(),
            "observations": [o.id for o in observations],
            "evidence": evidence,
        },
        "operations": operations,
    }
    # The returned proposal must always be a valid delta.
    parse_catalog_delta(delta)
    return RefreshResult(delta, tuple(observations), tuple(escalations))


# --- Discovery evidence ------------------------------------------------------------


def _rates_json(rates: Mapping[str, Any], dimensions: Iterable[str]) -> dict[str, str]:
    return {
        dimension: _rate_text(rates[dimension])
        for dimension in dimensions
        if rates.get(dimension) is not None
    }


def _gateway_alias(source: PriceSource, rate: ExtractedRate) -> dict[str, Any]:
    alias = {
        "provider": source.provider,
        "alias": rate.model_key,
        "channel": source.channel,
    }
    rules = dict(rate.meta.get("alias_rules") or {})
    if rules:
        alias["rules"] = rules
    return alias


def _gateway_price(
    source: PriceSource, rate: ExtractedRate, observed_on: date
) -> dict[str, Any]:
    price: dict[str, Any] = {
        "channel": source.channel,
        "region": rate.region,
        "effective_from": observed_on.isoformat(),
    }
    for field in RATE_FIELDS:
        if field in rate.rates:
            price[field] = _money(rate.rates[field])
    rules = dict(rate.meta.get("price_rules") or {})
    if rules:
        price["rules"] = rules
    price["source_url"] = source.url
    return price


# Reviewed gateway spellings for publishers the catalog already names.
_GATEWAY_PUBLISHER_PREFIXES = {
    "bytedance-seed": "bytedance",
    "meta-llama": "meta",
    "mistralai": "mistral",
    "qwen": "alibaba",
    "z-ai": "zai",
}


def _catalog_gateway_identity(model_key: str) -> str:
    prefix, separator, suffix = model_key.strip().lower().partition("/")
    if not separator:
        return model_key.strip().lower()
    return f"{_GATEWAY_PUBLISHER_PREFIXES.get(prefix, prefix)}/{suffix}"


def _catalog_publisher(provider: Any) -> str:
    value = str(provider or "").strip().lower()
    return _GATEWAY_PUBLISHER_PREFIXES.get(value, value)


def _identity_shape(model_key: str) -> tuple[str, str]:
    prefix, _, suffix = _catalog_gateway_identity(model_key).partition("/")
    return prefix, "".join(character for character in suffix if character.isalnum())


def _current_identity(
    entry: Mapping[str, Any], day: date
) -> tuple[str | None, list[Mapping[str, Any]]]:
    """``entry``'s canonical id and its aliases in effect on ``day``.

    An alias that has ended no longer names the model, so a spelling moved to
    another model matches only there. The canonical id is left out when it is
    also one of the model's alias spellings and none of those is in effect,
    as ``_names`` treats the canonical tail."""
    aliases = [a for a in entry.get("aliases") or [] if isinstance(a, Mapping)]
    current = [a for a in aliases if _in_effect(a, day, dated=False)]
    canonical_id = str(entry.get("canonical_id") or "").strip().lower()
    spelled = {str(a.get("alias") or "").strip().lower() for a in aliases}
    active = {str(a.get("alias") or "").strip().lower() for a in current}
    if canonical_id in spelled and canonical_id not in active:
        return None, current
    return canonical_id, current


def _exact_models(
    document: Mapping[str, Any], model_key: str, day: date
) -> list[Mapping[str, Any]]:
    publisher, _, _ = _catalog_gateway_identity(model_key).partition("/")
    wanted = {
        model_key.strip().lower(),
        _catalog_gateway_identity(model_key),
    }
    found: list[Mapping[str, Any]] = []
    for entry in document.get("models") or []:
        if not isinstance(entry, Mapping):
            continue
        canonical_id, aliases = _current_identity(entry, day)
        names = {canonical_id} if canonical_id else set()
        names.update(
            str(alias.get("alias") or "").strip().lower() for alias in aliases
        )
        names.update(
            f"{publisher}/{str(alias.get('alias') or '').strip().lower()}"
            for alias in aliases
            if _catalog_publisher(alias.get("provider")) == publisher
        )
        if wanted & names:
            found.append(entry)
    return found


def _near_models(
    document: Mapping[str, Any], model_key: str, day: date
) -> list[Mapping[str, Any]]:
    wanted = _identity_shape(model_key)
    found = []
    for entry in document.get("models") or []:
        if not isinstance(entry, Mapping):
            continue
        canonical_id, aliases = _current_identity(entry, day)
        shapes = {_identity_shape(canonical_id)} if canonical_id else set()
        shapes.update(
            _identity_shape(
                f"{_catalog_publisher(alias.get('provider'))}/"
                f"{str(alias.get('alias') or '').strip().lower()}"
            )
            for alias in aliases
        )
        if wanted in shapes:
            found.append(entry)
    return found


def _gateway_additions(
    source: PriceSource,
    extraction: Extraction,
    document: Mapping[str, Any],
    observed_on: date,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Reviewed additive operations for exact gateway model identities."""
    operations: list[dict[str, Any]] = []
    identity_exclusions: list[dict[str, Any]] = []
    for rate in extraction.rates:
        matches = _exact_models(document, rate.model_key, observed_on)
        if not matches:
            near = _near_models(document, rate.model_key, observed_on)
            if near:
                identity_exclusions.append(
                    {
                        "model_key": rate.model_key,
                        "reason": "ambiguous_identity",
                        "candidates": sorted(
                            str(entry["canonical_id"]) for entry in near
                        ),
                    }
                )
                continue
            canonical_id = _catalog_gateway_identity(rate.model_key)
            publisher = str(
                rate.meta.get("publisher") or canonical_id.split("/", 1)[0]
            ).strip().lower()
            publisher = _GATEWAY_PUBLISHER_PREFIXES.get(publisher, publisher)
            operations.append(
                {
                    "op": "add_model",
                    "key": {"canonical_id": canonical_id},
                    "expect": None,
                    "model": {
                        "canonical_id": canonical_id,
                        "publisher": publisher,
                        "aliases": [_gateway_alias(source, rate)],
                        "prices": [_gateway_price(source, rate, observed_on)],
                    },
                }
            )
            continue
        if len(matches) != 1:
            identity_exclusions.append(
                {
                    "model_key": rate.model_key,
                    "reason": "ambiguous_identity",
                    "candidates": sorted(
                        str(entry["canonical_id"]) for entry in matches
                    ),
                }
            )
            continue
        entry = matches[0]
        canonical_id = str(entry["canonical_id"])
        raw_names = {canonical_id.strip().lower()}
        raw_names.update(
            str(current.get("alias") or "").strip().lower()
            for current in entry.get("aliases") or []
            if isinstance(current, Mapping)
        )
        derived_identity = not (
            {
                rate.model_key.strip().lower(),
                _catalog_gateway_identity(rate.model_key),
            }
            & raw_names
        )
        alias = _gateway_alias(source, rate)
        channel_aliases = [
            current
            for current in entry.get("aliases") or []
            if isinstance(current, Mapping)
            and current.get("channel") == source.channel
        ]
        if channel_aliases:
            alias.pop("rules", None)
            if channel_aliases[0].get("rules"):
                alias["rules"] = copy.deepcopy(channel_aliases[0]["rules"])
        has_alias = any(
            isinstance(current, Mapping)
            and str(current.get("provider") or "").strip().lower()
            == source.provider.lower()
            and str(current.get("alias") or "").strip().lower()
            == rate.model_key.lower()
            and current.get("channel") == source.channel
            for current in entry.get("aliases") or []
        )
        active_channel_prices = [
            row
            for row in entry.get("prices") or []
            if isinstance(row, Mapping)
            and row.get("channel") == source.channel
            and str(row.get("region") or "global") == rate.region
            and _in_effect(row, observed_on, dated=True)
        ]
        candidate_rates = _rates_json(rate.rates, source.dimensions)
        if (
            not has_alias
            and derived_identity
            and active_channel_prices
            and not any(
                _rates_json(row, candidate_rates) == candidate_rates
                for row in active_channel_prices
            )
        ):
            identity_exclusions.append(
                {
                    "model_key": rate.model_key,
                    "reason": "channel_price_conflict",
                    "candidates": [canonical_id],
                }
            )
            continue
        if not has_alias:
            operations.append(
                {
                    "op": "add_alias",
                    "key": {
                        "canonical_id": canonical_id,
                        "provider": source.provider,
                        "alias": rate.model_key,
                        "channel": source.channel,
                    },
                    "expect": None,
                    "alias": alias,
                }
            )
        has_current_price = bool(active_channel_prices)
        if not has_current_price:
            operations.append(
                {
                    "op": "add_price",
                    "key": {
                        "canonical_id": canonical_id,
                        "channel": source.channel,
                        "region": rate.region,
                        "effective_from": observed_on.isoformat(),
                    },
                    "expect": None,
                    "price": _gateway_price(source, rate, observed_on),
                }
            )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for operation in operations:
        target = operation["op"], _canonical_json(operation["key"])
        grouped.setdefault(target, []).append(operation)
    retained: list[dict[str, Any]] = []
    colliding_models: set[str] = set()
    operation_collisions: list[dict[str, Any]] = []
    for (operation_name, _), candidates in grouped.items():
        if len(candidates) == 1:
            retained.extend(candidates)
            continue
        canonical_id = str(candidates[0]["key"]["canonical_id"])
        if len({_canonical_json(candidate) for candidate in candidates}) == 1:
            retained.append(candidates[0])
            operation_collisions.append(
                {
                    "canonical_id": canonical_id,
                    "operation": operation_name,
                    "reason": "identical_operation_deduplicated",
                    "candidate_count": len(candidates),
                }
            )
            continue
        colliding_models.add(canonical_id)
        operation_collisions.append(
            {
                "canonical_id": canonical_id,
                "operation": operation_name,
                "reason": "conflicting_operation",
                "candidate_count": len(candidates),
            }
        )
    operations = [
        operation
        for operation in retained
        if operation["key"]["canonical_id"] not in colliding_models
    ]
    operation_collisions.sort(
        key=lambda item: (item["canonical_id"], item["operation"])
    )
    identity_exclusions.sort(key=lambda item: item["model_key"])
    return operations, identity_exclusions, operation_collisions


def discovery_evidence(
    source: PriceSource,
    content: str,
    document: Mapping[str, Any],
    *,
    observed_on: date,
) -> dict[str, Any]:
    """The evidence a reviewer needs for ``source``: its fingerprint against
    the registry's, the normalized extraction, and -- where the parser yields
    rates -- each candidate value with the prior catalog value, the reason it
    needs review, and a proposed operation for each changed catalog row. It
    is deterministic, never includes raw page content, and its operations are
    proposals that need review."""
    extraction = extract(source.parser, content)
    candidates: list[dict[str, Any]] = []
    operations: list[dict[str, Any]] = []
    identity_exclusions: list[dict[str, Any]] = []
    operation_collisions: list[dict[str, Any]] = []
    if extraction.applicable:
        matches, unclaimed = _match(source, extraction, document, observed_on)
        for match in matches:
            if match.rate is None:
                continue
            candidate = _rates_json(match.rate.rates, source.dimensions)
            prior = _rates_json(match.row, source.dimensions)
            # Every represented rate the row carries is compared, so one the
            # page no longer shows is evidence even when the rest agree.
            missing = [d for d in prior if d not in candidate]
            # A catalog rate that is not a positive finite number is shown as
            # written and blocks any proposed operation.
            invalid = [
                dimension
                for dimension in prior
                if _valid_rate(match.row[dimension]) is None
                and not (
                    _carryable(match.row[dimension])
                    and _rate(candidate.get(dimension)) == 0
                )
            ]
            # A replacement copies every other rate the row carries; one that
            # is not a valid delta value would break the delta or be dropped.
            invalid += [
                f for f in RATE_FIELDS
                if f not in source.dimensions and match.row.get(f) is not None and not _carryable(match.row[f])
            ]
            differs = any(prior.get(d) != v for d, v in candidate.items())
            if differs or missing or invalid:
                reason = "invalid_current_rate" if invalid else "changed" if differs else "missing_rate"
                candidates.append(
                    {
                        "model_key": match.rate.model_key,
                        "region": match.key.region,
                        "canonical_id": match.canonical_id,
                        "reason": reason,
                        "prior": prior,
                        "candidate": candidate,
                        "missing_dimensions": missing,
                        "invalid_dimensions": invalid,
                    }
                )
                complete = not missing and not invalid
                ambiguous = regional_key(match.rate.model_key, match.rate.region) in extraction.ambiguities
                if (
                    complete
                    and not match.disagreeing
                    and match.row.get("effective_to") is None
                    and not ambiguous
                    and not match.duplicate_model
                    and _day(match.row["effective_from"]) < observed_on
                ):
                    changed = {d: v for d, v in match.rate.rates.items() if d in source.dimensions}
                    operations.append(_replace_operation(source, match, changed, observed_on))
        for rate, reason in unclaimed:
            candidates.append(
                {"model_key": rate.model_key, "region": rate.region, "canonical_id": None, "reason": reason, "prior": None, "candidate": _rates_json(rate.rates, source.dimensions), "missing_dimensions": [], "invalid_dimensions": []}
            )
        candidates.sort(key=lambda c: (c["model_key"], c["region"]))
        if source.parser in {"vercel-gateway-json", "openrouter-json"}:
            additions, identity_exclusions, operation_collisions = _gateway_additions(
                source, extraction, document, observed_on
            )
            additions_by_target = {
                (operation["op"], _canonical_json(operation["key"])): operation
                for operation in additions
            }
            existing = {
                (operation["op"], _canonical_json(operation["key"]))
                for operation in operations
            }
            operations.extend(
                operation
                for target, operation in additions_by_target.items()
                if target not in existing
            )
    operations_blocked_reason = None
    if (
        source.fingerprint is not None
        and extraction.fingerprint != source.fingerprint
    ):
        operations = []
        operations_blocked_reason = "fingerprint_mismatch"
    exclusions = dict(
        sorted(Counter(item.reason for item in extraction.exclusions).items())
    )
    return {
        "source_id": source.id,
        "provider": source.provider,
        "channel": source.channel,
        "url": source.url,
        "parser": source.parser,
        "observed_on": observed_on.isoformat(),
        "fingerprint": extraction.fingerprint,
        "expected_fingerprint": source.fingerprint,
        "fingerprint_matches": extraction.fingerprint == source.fingerprint,
        "applicable": extraction.applicable,
        "ambiguities": list(extraction.ambiguities),
        "exclusions": exclusions,
        "identity_exclusions": identity_exclusions,
        "operation_collisions": operation_collisions,
        "candidates": candidates,
        # Reviewable changes. Exact structured gateway identities may add
        # models or aliases; other unknown models stay evidence only.
        "operations": _canonical_value(operations),
        "operations_blocked_reason": operations_blocked_reason,
        # Raw page content is never emitted: extractor evidence is always
        # reduced to normalized facts, with no page cell or sentence verbatim.
        "evidence": redact_evidence(extraction.evidence),
    }
