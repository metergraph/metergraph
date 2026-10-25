"""Parse and load a prices document, from a file or already parsed, into a
CatalogSnapshot."""

import copy
import hashlib
import json
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timezone
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from .catalog import (
    Alias,
    PRICE_ENDED,
    CatalogSnapshot,
    CostResult,
    ModelUsage,
    Price,
    PriceResult,
    _coerce_datetime,
    _cost_result,
    _decimal,
    _effective_windows_overlap,
    _iso_text,
    normalize_provider,
)
from .billing import Dispute
from .retrieval import RetrievalCatalog, RetrievalCostResult, RetrievalPrice

DEFAULT_CATALOG_PATH = Path(__file__).parent / "data" / "prices.yaml"
_PROVIDER_SYNONYMS = {"bedrock": ("aws-bedrock", "aws")}


class CatalogError(ValueError):
    pass


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _canonical_index(document: dict[str, Any]) -> dict[tuple[str, str], str]:
    """Map every ``(provider, name)`` -- for each declared alias and, under the
    providers that publish it, the canonical id itself -- to its canonical id,
    dropping any pair that would resolve ambiguously. Qualifying by provider
    lets a capture path canonicalize a name two providers share (each to its own
    canonical) while still never guessing when the provider is unknown."""
    index: dict[tuple[str, str], str] = {}
    ambiguous: set[tuple[str, str]] = set()
    for entry in document.get("models", []):
        canonical = str(entry["canonical_id"])
        providers: set[str] = set()
        pairs: list[tuple[str, str]] = []
        for alias in entry.get("aliases") or []:
            provider = str(alias.get("provider") or "").strip().lower()
            if not provider:
                continue
            providers.add(provider)
            pairs.append((provider, str(alias["alias"])))
        for provider in providers:
            pairs.append((provider, canonical))
        for key in pairs:
            if index.get(key, canonical) != canonical:
                ambiguous.add(key)
            index[key] = canonical
    for key in ambiguous:
        index.pop(key, None)
    return index


@dataclass(frozen=True, slots=True)
class LoadedCatalog:
    version: str
    currency: str
    pricing_verified_at: date
    content_hash: str
    document: dict[str, Any]
    snapshot: CatalogSnapshot
    canonical_ids: Mapping[tuple[str, str], str]
    retrieval: RetrievalCatalog
    # Gateways whose amount billing declines per publisher; see Dispute.
    disputes: tuple[Dispute, ...] = ()

    def canonical_model_id(self, provider: str, model_id: str) -> str:
        """Resolve a captured provider-native model id to its canonical id,
        using the known source ``provider`` to disambiguate names shared across
        providers. An unknown provider, unknown id, or non-string input is left
        unchanged so none is ever misattributed."""
        if not isinstance(model_id, str) or not isinstance(provider, str):
            return model_id
        return self.canonical_ids.get(
            (normalize_provider(provider), model_id), model_id
        )

    def infer_direct_channel(self, model: Any) -> str | None:
        """Infer a direct pricing channel only when the catalog is unambiguous."""
        return self.snapshot.infer_direct_channel(model)

    def price(
        self,
        *,
        model: Any,
        channel: Any,
        at: Any,
        input_tokens: Any,
        output_tokens: Any,
        cache_read_tokens: Any = None,
        cache_write_tokens: Any = None,
        cache_write_5m_tokens: Any = None,
        cache_write_1h_tokens: Any = None,
        batch: bool = False,
    ) -> CostResult:
        """Price an observed deployment channel-exactly. See
        :meth:`CatalogSnapshot.price_deployment`."""
        return self.snapshot.price_deployment(
            model=model,
            channel=channel,
            at=at,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            cache_write_5m_tokens=cache_write_5m_tokens,
            cache_write_1h_tokens=cache_write_1h_tokens,
            batch=batch,
        )

    def lookup_price(
        self,
        *,
        model: Any,
        at: Any,
        channel: Any = None,
        provider: Any = None,
        region: Any = None,
    ) -> PriceResult:
        """The price a call to ``model`` at ``at`` is billed at.

        ``channel`` is authoritative; without one, ``provider`` supplies its
        direct channel. ``region`` overrides the region the catalog was
        loaded with. A result that depends on a dimension the caller left out
        lists it in ``missing_dimensions``."""
        when = _coerce_datetime(at)
        resolved, missing = self.snapshot._lookup(
            model=model, channel=channel, provider=provider, at=when, region=region
        )
        if resolved is None:
            return PriceResult("unpriced", None, None, when, missing)
        return PriceResult(
            "partial" if missing else "priced",
            resolved.canonical_model,
            resolved.price,
            when,
            missing,
            (PRICE_ENDED,) if resolved.ended else (),
        )

    def calculate_cost(
        self,
        *,
        model: Any,
        usage: ModelUsage,
        at: Any,
        channel: Any = None,
        provider: Any = None,
        region: Any = None,
    ) -> CostResult:
        """Cost ``usage`` at the price :meth:`lookup_price` resolves. A
        missing dimension, from the lookup or the usage, turns the result
        ``partial`` and keeps the subtotal that is known."""
        when = _coerce_datetime(at)
        resolved, missing = self.snapshot._lookup(
            model=model, channel=channel, provider=provider, at=when, region=region
        )
        if resolved is None:
            reasons = tuple(f"missing_{dimension}" for dimension in missing)
            return CostResult(
                None,
                None,
                None,
                "unpriced",
                reasons or ("unknown_deployment",),
                effective_at=when,
                missing_dimensions=missing,
            )
        return _cost_result(resolved, at=when, usage=usage, missing=missing)

    def price_retrieval(
        self,
        *,
        channel: Any,
        operation: Any,
        units: Any,
        at: Any,
        region: Any = "global",
    ) -> RetrievalCostResult:
        """Price a retrieval operation (searches, tool calls, grounded queries)
        by counted unit. See :meth:`RetrievalCatalog.price`. This is kept
        distinct from :meth:`price`, which costs model-token usage."""
        return self.retrieval.price(
            channel=channel,
            operation=operation,
            units=units,
            at=at,
            region=region,
        )


def _date(value: Any, *, field: str, model: str) -> datetime:
    if value is None:
        raise CatalogError(f"{model}: missing {field}")
    try:
        parsed = datetime.fromisoformat(_iso_text(value))
    except ValueError as exc:
        raise CatalogError(f"{model}: invalid {field} {value!r}") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _id_date(effective_from: datetime) -> date:
    """The date a price id names: the UTC date of its start. A date-only or
    UTC start keeps its written date; an offset start uses its UTC date."""
    return effective_from.astimezone(timezone.utc).date()


def _unique_ids(items: list) -> list:
    """``items`` with ids made unique. An id names its row by the UTC date of
    its start, which stays the id whenever it is unique; rows that would
    share one also name their UTC start time, except a row starting at UTC
    midnight, whose date alone already says when it starts."""
    counts: dict[str, int] = {}
    for item in items:
        counts[item.id] = counts.get(item.id, 0) + 1
    unique = []
    for item in items:
        start = item.effective_from.astimezone(timezone.utc)
        if counts[item.id] > 1 and start.timetz().replace(tzinfo=None) != time(0):
            moment = f"{start:%H:%M:%S}" + (f".{start:%f}" if start.microsecond else "")
            item = replace(item, id=f"{item.id}T{moment}Z")
        unique.append(item)
    return unique


def _price_region(value: Any, *, model: str) -> tuple[str, ...]:
    """An alias's optional ``price_region``: one region or a list of them."""
    if value is None:
        return ()
    regions = [value] if isinstance(value, str) else value
    if (
        not isinstance(regions, (list, tuple))
        or not regions
        or not all(isinstance(region, str) and region.strip() for region in regions)
    ):
        raise CatalogError(f"{model}: invalid price_region {value!r}")
    return tuple(dict.fromkeys(region.strip().lower() for region in regions))


def _catalog_metadata(document: dict[str, Any]) -> tuple[str, date]:
    currency = str(document.get("currency") or "").strip().upper()
    if currency != "USD":
        raise CatalogError("prices document currency must be USD")
    try:
        pricing_verified_at = date.fromisoformat(
            str(document.get("pricing_verified_at") or "")
        )
    except ValueError as exc:
        raise CatalogError(
            "prices document must have an ISO pricing_verified_at date"
        ) from exc
    return currency, pricing_verified_at


def parse_catalog(
    document: Any,
) -> tuple[str, dict[tuple[str, str], tuple[Alias, ...]], list[Price]]:
    if not isinstance(document, dict) or not isinstance(document.get("models"), list):
        raise CatalogError("prices document must have a models list")
    version = str(document.get("version") or "")
    if not version:
        raise CatalogError("prices document must have a version")
    currency, _ = _catalog_metadata(document)
    aliases: dict[tuple[str, str], list[Alias]] = {}
    prices: list[Price] = []
    seen_models: set[str] = set()
    for entry in document["models"]:
        canonical = str(entry.get("canonical_id") or "")
        if not canonical:
            raise CatalogError("model entry missing canonical_id")
        # One entry per model: its aliases and price windows are checked
        # within that entry, so a second entry would escape those checks.
        if canonical in seen_models:
            raise CatalogError(f"duplicate model entry {canonical!r}")
        seen_models.add(canonical)
        publisher = str(entry.get("publisher") or "").strip().lower() or None
        for alias in entry.get("aliases") or []:
            provider = str(alias.get("provider") or "").lower()
            name = str(alias.get("alias") or "").lower()
            channel = str(alias.get("channel") or "")
            if not provider or not name or not channel:
                raise CatalogError(f"{canonical}: alias needs provider/alias/channel")
            effective_from = (
                _date(
                    alias.get("effective_from"),
                    field="effective_from",
                    model=f"{canonical} alias {name}",
                )
                if alias.get("effective_from") is not None
                else None
            )
            effective_to = (
                _date(
                    alias.get("effective_to"),
                    field="effective_to",
                    model=f"{canonical} alias {name}",
                )
                if alias.get("effective_to") is not None
                else None
            )
            if (
                effective_from is not None
                and effective_to is not None
                and effective_to <= effective_from
            ):
                raise CatalogError(
                    f"{canonical} alias {name}: effective_to before effective_from"
                )
            source_url = str(alias.get("source_url") or "").strip() or None
            if (effective_from is not None or effective_to is not None) and not source_url:
                raise CatalogError(
                    f"{canonical} alias {name}: effective alias needs source_url"
                )
            parsed_alias = Alias(
                model_id=canonical,
                canonical_id=canonical,
                pricing_channel=channel,
                rules=_freeze(alias.get("rules") or {}),
                publisher=publisher,
                effective_from=effective_from,
                effective_to=effective_to,
                source_url=source_url,
                price_region=_price_region(
                    alias.get("price_region"), model=f"{canonical} alias {name}"
                ),
            )
            keys = [(provider, name)]
            for synonym in _PROVIDER_SYNONYMS.get(provider, ()):
                keys.append((synonym, name))
            for key in keys:
                candidates = aliases.setdefault(key, [])
                for other in candidates:
                    if _effective_windows_overlap(parsed_alias, other):
                        raise CatalogError(
                            f"{canonical}: overlapping alias windows for {key}"
                        )
                candidates.append(parsed_alias)
        seen_windows: list[tuple[str, str, datetime, datetime | None]] = []
        for price in entry.get("prices") or []:
            channel = str(price.get("channel") or "")
            region = str(price.get("region") or "global")
            if not channel:
                raise CatalogError(f"{canonical}: price entry needs channel")
            if not str(price.get("source_url") or "").strip():
                raise CatalogError(f"{canonical}: price entry needs source_url")
            effective_from = _date(
                price.get("effective_from"), field="effective_from", model=canonical
            )
            effective_to = (
                _date(price.get("effective_to"), field="effective_to", model=canonical)
                if price.get("effective_to") is not None
                else None
            )
            if effective_to is not None and effective_to <= effective_from:
                raise CatalogError(f"{canonical}: effective_to before effective_from")
            for other_channel, other_region, other_from, other_to in seen_windows:
                if (other_channel, other_region) != (channel, region):
                    continue
                if (effective_to is None or other_from < effective_to) and (
                    other_to is None or effective_from < other_to
                ):
                    raise CatalogError(
                        f"{canonical}: overlapping {channel}/{region} price windows"
                    )
            seen_windows.append((channel, region, effective_from, effective_to))
            batch_cache_read = price.get("batch_cache_read_per_mtok")
            batch_cache_read_per_mtok = _decimal(batch_cache_read)
            # A stated batch cache rate replaces the standard one on batch
            # calls, so one that cannot be read must not quietly fall back.
            if batch_cache_read is not None and (
                batch_cache_read_per_mtok is None or batch_cache_read_per_mtok < 0
            ):
                raise CatalogError(
                    f"{canonical}: invalid batch_cache_read_per_mtok "
                    f"{batch_cache_read!r}"
                )
            prices.append(
                Price(
                    id=f"{canonical}:{channel}:{region}:{_id_date(effective_from)}",
                    model_id=canonical,
                    pricing_channel=channel,
                    region=region,
                    input_per_mtok=_decimal(price.get("input_per_mtok")),
                    output_per_mtok=_decimal(price.get("output_per_mtok")),
                    cache_read_per_mtok=_decimal(price.get("cache_read_per_mtok")),
                    cache_write_5m_per_mtok=_decimal(
                        price.get("cache_write_5m_per_mtok")
                    ),
                    cache_write_1h_per_mtok=_decimal(
                        price.get("cache_write_1h_per_mtok")
                    ),
                    batch_input_per_mtok=_decimal(price.get("batch_input_per_mtok")),
                    batch_output_per_mtok=_decimal(price.get("batch_output_per_mtok")),
                    request_fee_per_1k_usd=_decimal(price.get("request_fee_per_1k_usd")),
                    rules=_freeze(price.get("rules") or {}),
                    effective_from=effective_from,
                    effective_to=effective_to,
                    source_url=str(price["source_url"]).strip(),
                    publisher=publisher,
                    currency=currency,
                    batch_cache_read_per_mtok=batch_cache_read_per_mtok,
                )
            )
    return version, {key: tuple(value) for key, value in aliases.items()}, _unique_ids(prices)


def parse_retrieval(document: Any) -> list[RetrievalPrice]:
    """Parse the optional top-level ``retrieval`` list into effective-dated
    per-1,000-unit prices, applying the same effective-date validation as the
    token catalog (ISO dates, ``effective_to`` after ``effective_from``, no
    overlapping windows per channel/operation/region). A per-unit rate must be
    present and non-negative. An absent ``retrieval`` key yields no prices."""
    if not isinstance(document, dict):
        raise CatalogError("prices document must be a mapping")
    entries = document.get("retrieval")
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise CatalogError("prices document retrieval must be a list")
    prices: list[RetrievalPrice] = []
    seen_windows: list[tuple[str, str, str, datetime, datetime | None]] = []
    for entry in entries:
        channel = str(entry.get("channel") or "").strip().lower()
        operation = str(entry.get("operation") or "").strip().lower()
        region = str(entry.get("region") or "global")
        unit = str(entry.get("unit") or "").strip()
        if not channel or not operation:
            raise CatalogError("retrieval entry needs channel/operation")
        label = f"{channel}/{operation}"
        if not unit:
            raise CatalogError(f"{label}: retrieval entry needs a unit")
        if not str(entry.get("source_url") or "").strip():
            raise CatalogError(f"{label}: retrieval entry needs source_url")
        per_1k_usd = _decimal(entry.get("per_1k_usd"))
        if per_1k_usd is None or per_1k_usd < 0:
            raise CatalogError(
                f"{label}: retrieval entry needs a non-negative per_1k_usd"
            )
        effective_from = _date(
            entry.get("effective_from"), field="effective_from", model=label
        )
        effective_to = (
            _date(entry.get("effective_to"), field="effective_to", model=label)
            if entry.get("effective_to") is not None
            else None
        )
        if effective_to is not None and effective_to <= effective_from:
            raise CatalogError(f"{label}: effective_to before effective_from")
        for other in seen_windows:
            other_channel, other_operation, other_region, other_from, other_to = other
            if (other_channel, other_operation, other_region) != (
                channel,
                operation,
                region,
            ):
                continue
            if (effective_to is None or other_from < effective_to) and (
                other_to is None or effective_from < other_to
            ):
                raise CatalogError(
                    f"{label}: overlapping {region} retrieval price windows"
                )
        seen_windows.append((channel, operation, region, effective_from, effective_to))
        prices.append(
            RetrievalPrice(
                id=f"{channel}:{operation}:{region}:{_id_date(effective_from)}",
                channel=channel,
                operation=operation,
                region=region,
                unit=unit,
                per_1k_usd=per_1k_usd,
                effective_from=effective_from,
                effective_to=effective_to,
                source_url=str(entry["source_url"]).strip(),
            )
        )
    return _unique_ids(prices)


def parse_disputes(document: Any) -> list[Dispute]:
    """Parse the optional top-level ``disputes`` list: a gateway whose
    reported amount is not believed for one publisher's models from a date,
    with the note a product can show and the page that shows the figure is
    wrong. An absent key yields no disputes."""
    if not isinstance(document, dict):
        raise CatalogError("prices document must be a mapping")
    entries = document.get("disputes")
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise CatalogError("prices document disputes must be a list")
    disputes: list[Dispute] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise CatalogError("dispute entry must be a mapping")
        gateway = str(entry.get("gateway") or "").strip().lower()
        publisher = str(entry.get("publisher") or "").strip().lower()
        if not gateway or not publisher:
            raise CatalogError("dispute entry needs gateway/publisher")
        label = f"{gateway}/{publisher}"
        note = str(entry.get("note") or "").strip()
        if not note:
            raise CatalogError(f"{label}: dispute entry needs a note")
        if not str(entry.get("source_url") or "").strip():
            raise CatalogError(f"{label}: dispute entry needs source_url")
        disputes.append(
            Dispute(
                gateway=gateway,
                publisher=publisher,
                since=_date(entry.get("since"), field="since", model=label),
                note=note,
                source_url=str(entry["source_url"]).strip(),
            )
        )
    return disputes


def _canonical_value(value: Any) -> Any:
    """``value`` in the form canonical JSON compares and hashes: mappings and
    sequences recursively, dates as ISO strings and decimals as strings, so a
    quoted and an unquoted YAML date are the same document."""
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    return value


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            _canonical_value(value), sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError) as exc:
        raise CatalogError(f"prices document is not JSON-representable: {exc}") from exc


def _build_loaded_catalog(
    document: Any, *, content_hash: str, region: str
) -> LoadedCatalog:
    """The one construction and validation path behind both loaders."""
    version, aliases, prices = parse_catalog(document)
    currency, pricing_verified_at = _catalog_metadata(document)
    return LoadedCatalog(
        version=version,
        currency=currency,
        pricing_verified_at=pricing_verified_at,
        content_hash=content_hash,
        document=document,
        snapshot=CatalogSnapshot(aliases, prices, region=region),
        canonical_ids=_canonical_index(document),
        retrieval=RetrievalCatalog(parse_retrieval(document)),
        disputes=tuple(parse_disputes(document)),
    )


def load_catalog(
    path: str | Path | None = None, *, region: str = "global"
) -> LoadedCatalog:
    """Load a catalog file. ``content_hash`` is the SHA-256 of its bytes."""
    resolved = Path(path) if path is not None else DEFAULT_CATALOG_PATH
    content = resolved.read_bytes()
    return _build_loaded_catalog(
        yaml.safe_load(content),
        content_hash=hashlib.sha256(content).hexdigest(),
        region=region,
    )


def load_catalog_document(
    document: Mapping[str, Any], *, region: str = "global"
) -> LoadedCatalog:
    """Build a catalog from an already parsed document, validated exactly as
    :func:`load_catalog` validates a file. The document is deep-copied, so later
    changes to the caller's mapping never reach the loaded catalog.
    ``content_hash`` is the SHA-256 of the document's canonical JSON (sorted
    keys, compact separators, ISO dates), so equal documents hash equally
    whatever their key order; it differs from a file's raw-byte hash."""
    if not isinstance(document, Mapping):
        raise CatalogError("prices document must have a models list")
    copied = copy.deepcopy(dict(document))
    content_hash = hashlib.sha256(_canonical_json(copied).encode("utf-8")).hexdigest()
    return _build_loaded_catalog(copied, content_hash=content_hash, region=region)
