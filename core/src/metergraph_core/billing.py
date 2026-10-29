"""Pure selection of effective LLM cost from catalog and gateway evidence."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping

from .catalog import CostResult, _coerce_datetime

@dataclass(frozen=True, slots=True)
class _QualifiedSource:
    """A gateway whose reported amount this module will bill from.

    Three things have to line up before an amount is believed, because the field
    carrying one is open: a customer's own client can write to it.

    * The **gateway**, so an amount is attributed to a system we have checked.
    * The **endpoint**, because a gateway prices its endpoints differently and a
      figure is only interpretable alongside the one that produced it.
    * The **source**, naming the field the amount was read from, so a row cannot
      qualify by claiming a gateway name alone.

    A gateway absent from this table is not distrusted so much as unverified:
    its amount is still recorded, and the catalog still prices the call, so the
    two can be compared before it is added here.
    """

    gateway: str
    endpoints: frozenset[str]
    cost_source: str
    upstream_cost_source: str | None = None


# Each entry reconciles against the providers' own published rates, including
# the per-request and per-query charges a token rate cannot express. Portkey was
# checked across a 46,241-call export: its figure matches published rates for
# Anthropic, xAI and Perplexity to the cent, and for OpenAI once the per-search
# charge is counted -- 20,590 of 20,590 rows, exactly. Anthropic's calls come
# back in the Messages API shape, which the relay reports as `messages`; before
# that endpoint was listed here every Anthropic call fell back to the catalog,
# and a model the catalog did not know yet (Opus 5.5, 2,154 calls in one
# customer's week) was stored at $0 with the gateway's figure sitting unused.
_QUALIFIED_SOURCES = (
    _QualifiedSource(
        gateway="openrouter",
        endpoints=frozenset({"chat.completions"}),
        cost_source="openrouter.usage.cost",
        upstream_cost_source=(
            "openrouter.usage.cost_details.upstream_inference_cost"
        ),
    ),
    _QualifiedSource(
        gateway="portkey",
        endpoints=frozenset({"chat.completions", "responses", "messages"}),
        cost_source="portkey.cost",
    ),
)


@dataclass(frozen=True, slots=True)
class Dispute:
    """A gateway whose reported amount is not believed for one publisher's
    models from a date: the catalog has shown its figure wrong there, and the
    note says how. Billing takes the catalog price instead and keeps the
    gateway's figure as evidence."""

    gateway: str
    publisher: str
    since: datetime
    note: str
    source_url: str


@dataclass(frozen=True, slots=True)
class GatewayBillingEvidence:
    """Validated gateway billing fields from one content-blind call row."""

    gateway: str | None
    endpoint: str | None
    reported_cost_usd: Decimal | None
    reported_cost_source: str | None
    reported_upstream_cost_usd: Decimal | None
    reported_upstream_cost_source: str | None


@dataclass(frozen=True, slots=True)
class BillingDecision:
    """Effective cost plus the independent evidence used to select it."""

    cost_usd: Decimal | None
    cost_status: str
    cost_provenance: str
    reported_cost_usd: Decimal | None
    reported_upstream_cost_usd: Decimal | None
    catalog_cost_usd: Decimal | None
    catalog_price_id: str | None
    catalog_reasons: tuple[str, ...]
    cost_discrepancy_status: str | None = None


def _text(value: Any, *, limit: int = 128) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped[:limit] if stripped else None


def _non_negative_decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    if not result.is_finite() or result < 0:
        return None
    return result


def normalize_gateway_evidence(row: Mapping[str, Any]) -> GatewayBillingEvidence:
    """Allowlist and validate gateway billing evidence from a call mapping."""

    gateway = _text(row.get("gateway"), limit=64)
    if gateway is not None:
        gateway = gateway.lower()
    endpoint = _text(row.get("endpoint"), limit=64)
    reported_source = _text(row.get("reported_cost_source"))
    upstream_source = _text(row.get("reported_upstream_cost_source"))

    return GatewayBillingEvidence(
        gateway=gateway,
        endpoint=endpoint,
        reported_cost_usd=_non_negative_decimal(row.get("reported_cost_usd")),
        reported_cost_source=reported_source,
        reported_upstream_cost_usd=_non_negative_decimal(
            row.get("reported_upstream_cost_usd")
        ),
        reported_upstream_cost_source=upstream_source,
    )


def _qualified_source(evidence: GatewayBillingEvidence) -> _QualifiedSource | None:
    """The entry this row's gateway and endpoint match, if any."""
    for source in _QUALIFIED_SOURCES:
        if evidence.gateway == source.gateway and evidence.endpoint in source.endpoints:
            return source
    return None


def _disputed(
    qualified: _QualifiedSource | None,
    publisher: str | None,
    disputes: Iterable[Dispute],
    at: Any,
) -> bool:
    # Without the call time nothing is disputed: a dispute runs from a date,
    # and declining a figure for a call that may predate it is the wrong
    # side to err on.
    if qualified is None or publisher is None or at is None:
        return False
    when = _coerce_datetime(at)
    return any(
        dispute.gateway == qualified.gateway
        and dispute.publisher == publisher
        and dispute.since <= when
        for dispute in disputes
    )


def resolve_billing(
    catalog_result: "CostResult | CallCostResult",
    evidence: GatewayBillingEvidence,
    *,
    disputes: Iterable[Dispute] = (),
    at: Any = None,
) -> BillingDecision:
    """Select effective cost without combining independent reported amounts.

    A qualified gateway's amount is declined when the catalog disputes that
    gateway for the priced model's publisher at ``at`` (with no ``at`` nothing
    is disputed): the catalog price is
    billed, the gateway's figure stays in ``reported_cost_usd`` as evidence,
    and ``gateway_disputed`` is recorded among the reasons. When the catalog
    cannot price the call, the gateway's figure still stands, with the same
    reason recorded, because a disputed figure is better evidence than none.
    """

    qualified = _qualified_source(evidence)
    reported_cost = (
        evidence.reported_cost_usd
        if qualified is not None
        and evidence.reported_cost_source == qualified.cost_source
        else None
    )
    disputed = reported_cost is not None and _disputed(
        qualified, catalog_result.publisher, disputes, at
    )
    reasons = catalog_result.reasons
    if disputed:
        reasons = tuple(dict.fromkeys((*reasons, "gateway_disputed")))
    upstream_cost = (
        evidence.reported_upstream_cost_usd
        if qualified is not None
        and qualified.upstream_cost_source is not None
        and evidence.reported_upstream_cost_source == qualified.upstream_cost_source
        else None
    )

    if reported_cost is not None and not (
        disputed and catalog_result.cost_usd is not None
    ):
        cost_usd = reported_cost
        cost_status = "priced"
        provenance = "gateway_reported"
    elif catalog_result.cost_usd is not None:
        cost_usd = catalog_result.cost_usd
        cost_status = catalog_result.status
        provenance = "catalog"
    else:
        cost_usd = None
        cost_status = catalog_result.status
        provenance = "none"

    return BillingDecision(
        cost_usd=cost_usd,
        cost_status=cost_status,
        cost_provenance=provenance,
        reported_cost_usd=reported_cost,
        reported_upstream_cost_usd=upstream_cost,
        catalog_cost_usd=catalog_result.cost_usd,
        catalog_price_id=catalog_result.price_id,
        catalog_reasons=reasons,
    )

# Imported only for type checking to keep the response adapter independent.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .calls import CallCostResult
