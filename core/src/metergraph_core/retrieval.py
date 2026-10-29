"""Effective-dated retrieval pricing: searches, tool calls, grounded queries.

Retrieval pricing sits alongside -- and deliberately apart from -- the
model-token catalog. A retrieval operation is billed per counted unit (a
completed web search, a tool call, an executed grounding query), never per
token, so it carries its own price shape and its own catalog object.

The contract mirrors the token path's explicit priced/partial/unpriced
semantics: a resolved operation with a valid unit count is ``priced`` (a $0
fee is a real priced zero), while a missing, non-integer, or negative unit
count and an unknown operation or channel come back ``unpriced`` with a reason
-- never silently priced at zero.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from .catalog import _COST_QUANTUM, _coerce_datetime

_THOUSAND = Decimal("1000")


@dataclass(frozen=True, slots=True)
class RetrievalPrice:
    """An effective-dated per-1,000-unit price for one channel/operation."""

    id: str
    channel: str
    operation: str
    region: str
    unit: str
    per_1k_usd: Decimal
    effective_from: datetime
    effective_to: datetime | None
    source_url: str


@dataclass(frozen=True, slots=True)
class RetrievalCostResult:
    """The outcome of pricing a retrieval operation.

    ``status`` reuses the token path's vocabulary (``priced``/``partial``/
    ``unpriced``); retrieval never produces ``partial`` because a unit count is
    all-or-nothing -- a bad count leaves nothing knowable to price.
    """

    price_id: str | None
    cost_usd: Decimal | None
    status: str
    reasons: tuple[str, ...] = ()
    source_url: str | None = None


# The retrieval operation a counted search or fetch is billed as on each
# channel. A channel absent here charges no per-use fee for that operation.
SEARCH_OPERATION_BY_CHANNEL: Mapping[str, str] = MappingProxyType({
    "openai-api": "web_search",
    "anthropic-api": "web_search",
    "google-api": "google_search_grounding",
    "google-vertex-ai": "google_search_grounding",
    # Historical Sonar request fees use this operation; no current fee is
    # assigned until its API and billing unit can be verified.
    "perplexity-api": "search_request_low",
})
FETCH_OPERATION_BY_CHANNEL: Mapping[str, str] = MappingProxyType({
    "anthropic-api": "web_fetch",
})

# The tool-call item types a provider emits for a search or a fetch it ran
# itself, as the Responses API and the relay record them.
_SEARCH_TOOL_TYPES = frozenset({"web_search_call"})
_FETCH_TOOL_TYPES = frozenset({"web_fetch_call"})
_PERPLEXITY = frozenset({"perplexity", "perplexity-ai"})


def search_operation_for_channel(channel: Any) -> str | None:
    """The retrieval operation a counted search is billed as on ``channel``,
    or ``None`` where the channel charges no per-search fee."""
    return SEARCH_OPERATION_BY_CHANNEL.get(str(channel or "").strip().lower())


def fetch_operation_for_channel(channel: Any) -> str | None:
    """The retrieval operation a counted fetch is billed as on ``channel``,
    or ``None`` where the channel charges no per-fetch fee."""
    return FETCH_OPERATION_BY_CHANNEL.get(str(channel or "").strip().lower())


def _count(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return count if count >= 0 else None


def _tool_items(row: Mapping[str, Any], types: frozenset[str]) -> int | None:
    """Items of the given types among ``tool_calls``: a measured zero when the
    list is there and holds none, ``None`` when there is no list to read."""
    calls = row.get("tool_calls")
    if not isinstance(calls, list):
        return None
    return sum(
        1
        for call in calls
        if isinstance(call, Mapping) and str(call.get("type") or "").strip().lower() in types
    )


def count_search_units(row: Mapping[str, Any]) -> int | None:
    """The searches one call row made, read from whichever field its capture
    path carries, in the order they are believed: ``web_search_calls`` and
    ``grounding_queries`` (the relay's per-provider counts), Anthropic's
    ``server_tool_use.web_search_requests``, then the ``web_search_call``
    items among ``tool_calls``. A Perplexity call is one search, since
    Perplexity bills every request as one. A field that is present and says
    nothing was searched is a measured zero; ``None`` means no field says."""
    for key in ("web_search_calls", "grounding_queries"):
        count = _count(row.get(key))
        if count is not None:
            return count
    server = row.get("server_tool_use")
    if isinstance(server, Mapping):
        count = _count(server.get("web_search_requests"))
        if count is not None:
            return count
    if str(row.get("provider") or "").strip().lower() in _PERPLEXITY:
        return 1
    return _tool_items(row, _SEARCH_TOOL_TYPES)


def count_fetch_units(row: Mapping[str, Any]) -> int | None:
    """The page fetches one call row made: Anthropic's
    ``server_tool_use.web_fetch_requests``, then the ``web_fetch_call``
    items among ``tool_calls``, with the same measured-zero rule."""
    server = row.get("server_tool_use")
    if isinstance(server, Mapping):
        count = _count(server.get("web_fetch_requests"))
        if count is not None:
            return count
    return _tool_items(row, _FETCH_TOOL_TYPES)


def _units(value: Any) -> tuple[int | None, str | None]:
    """Coerce a retrieval unit count under a strict-integer contract.

    Only a real, non-negative ``int`` is a valid count. ``bool`` (``type`` is
    ``bool``, not ``int``), ``float``, strings, and ``Decimal`` are rejected as
    ``invalid_units`` rather than truncated or parsed; ``None`` is
    ``missing_units`` and a negative int is ``negative_units``. Each rejection
    keeps the operation explicitly unpriced instead of turning it into zero.
    """
    if value is None:
        return None, "missing_units"
    if type(value) is not int:
        return None, "invalid_units"
    if value < 0:
        return None, "negative_units"
    return value, None


class RetrievalCatalog:
    """Resolve and cost retrieval operations by ``(channel, operation)``.

    Channel selection is exact and region resolution falls back
    ``region -> "*" -> "global"``, mirroring the token snapshot: an operation a
    channel does not carry stays unpriced rather than being repriced off another
    channel or region.
    """

    def __init__(self, prices: list[RetrievalPrice]) -> None:
        self._by_key: dict[tuple[str, str], list[RetrievalPrice]] = {}
        self._channels: set[str] = set()
        for price in prices:
            self._channels.add(price.channel)
            self._by_key.setdefault((price.channel, price.operation), []).append(price)
        for candidates in self._by_key.values():
            candidates.sort(key=lambda price: price.effective_from, reverse=True)

    def _select(
        self, channel: str, operation: str, region: str, at: datetime
    ) -> RetrievalPrice | None:
        at = at if at.tzinfo else at.replace(tzinfo=timezone.utc)
        candidates = self._by_key.get((channel, operation), [])
        for candidate_region in dict.fromkeys((region, "*", "global")):
            for price in candidates:
                if price.region.lower() != candidate_region:
                    continue
                if price.effective_from <= at and (
                    price.effective_to is None or at < price.effective_to
                ):
                    return price
        return None

    def price(
        self,
        *,
        channel: Any,
        operation: Any,
        units: Any,
        at: Any,
        region: Any = "global",
    ) -> RetrievalCostResult:
        """Price ``units`` of ``operation`` on ``channel`` at time ``at``.

        Resolution happens before unit validation: an unknown channel or
        operation is reported as such regardless of the unit count, because an
        operation that cannot be identified cannot be priced at all.
        """
        channel_key = str(channel or "").strip().lower()
        operation_key = str(operation or "").strip().lower()
        region_key = str(region or "global").strip().lower() or "global"

        if channel_key not in self._channels:
            return RetrievalCostResult(None, None, "unpriced", ("unknown_channel",))
        if (channel_key, operation_key) not in self._by_key:
            return RetrievalCostResult(None, None, "unpriced", ("unknown_operation",))

        price = self._select(channel_key, operation_key, region_key, _coerce_datetime(at))
        if price is None:
            return RetrievalCostResult(None, None, "unpriced", ("no_effective_price",))

        count, reason = _units(units)
        if reason is not None:
            return RetrievalCostResult(None, None, "unpriced", (reason,))

        cost = (Decimal(count) * price.per_1k_usd / _THOUSAND).quantize(
            _COST_QUANTUM, rounding=ROUND_HALF_UP
        )
        return RetrievalCostResult(price.id, cost, "priced", (), price.source_url)
