"""Compose token and counted retrieval costs from one catalog snapshot."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Sequence

from .catalog import CostResult, ModelUsage, direct_channel_for_provider, normalize_provider
from .loader import LoadedCatalog
from .responses import _as_mapping, _bundled_catalog, _first, cost_from_response, response_format

_ZERO = Decimal('0')


@dataclass(frozen=True, slots=True)
class RetrievalUsage:
    operation: str
    units: Any


@dataclass(frozen=True, slots=True)
class RetrievalComponent:
    operation: str
    units: Any
    cost_usd: Decimal | None
    price_id: str | None
    basis: str
    reasons: tuple[str, ...] = ()
    source_url: str | None = None


@dataclass(frozen=True, slots=True)
class CallCostResult:
    token_cost: CostResult
    retrieval: tuple[RetrievalComponent, ...]
    cost_usd: Decimal | None
    known_subtotal_usd: Decimal
    estimated_total_usd: Decimal | None
    status: str
    token_basis: str
    missing_dimensions: tuple[str, ...]
    catalog_hash: str


    @property
    def publisher(self) -> str | None:
        return self.token_cost.publisher

    @property
    def price_id(self) -> str | None:
        return self.token_cost.price_id

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.token_cost.reasons,
            *(f"missing_{d}" for d in self.missing_dimensions))))


def compose_call_cost(
    token_cost: CostResult, *, catalog: LoadedCatalog,
    retrieval: Sequence[RetrievalUsage] = (), provider: str | None = None,
    channel: str | None = None, region: str = 'global',
) -> CallCostResult:
    """Combine retrieval with a token result from the same catalog and time.

    Partial token results remain estimates, excluded from the known subtotal.
    """
    if token_cost.effective_at is None:
        raise ValueError('token_cost must carry its effective_at')
    if token_cost.price is not None:
        selected = catalog.lookup_price(model=token_cost.canonical_model,
            channel=token_cost.price.pricing_channel, region=token_cost.price.region,
            at=token_cost.effective_at)
        if selected.price != token_cost.price:
            raise ValueError('token price does not match the catalog snapshot')
    if len({item.operation for item in retrieval}) != len(retrieval):
        raise ValueError('retrieval operations must be aggregated once per call')
    resolved_channel = token_cost.price.pricing_channel if token_cost.price else (channel or direct_channel_for_provider(provider))
    publisher = token_cost.publisher or normalize_provider(provider or '')
    missing = list(token_cost.missing_dimensions)
    token_basis = ('exact' if token_cost.status == 'priced' else
                   'fallback_estimate' if token_cost.cost_usd is not None else 'unknown')
    # Retrieval rates are USD. Do not add them to a different currency.
    currency_ok = token_cost.currency in (None, 'USD')
    known = token_cost.cost_usd if token_basis == 'exact' and currency_ok else _ZERO
    if not currency_ok:
        missing.append('currency')
        token_basis = 'unknown'
    parts = []
    for usage in retrieval:
        if resolved_channel == 'anthropic-api' and publisher == 'anthropic' and usage.operation in {'web_search', 'web_fetch'}:
            fee = catalog.price_retrieval(channel=resolved_channel, operation=usage.operation,
                units=usage.units, at=token_cost.effective_at, region=region)
            part = RetrievalComponent(usage.operation, usage.units, fee.cost_usd,
                fee.price_id, 'exact' if fee.status == 'priced' else 'unknown', fee.reasons, fee.source_url)
        else:
            part = RetrievalComponent(usage.operation, usage.units, None, None,
                'unknown', ('unsupported_retrieval_scope',))
        part = RetrievalComponent(part.operation,
            part.units if type(part.units) is int and part.units >= 0 else None,
            part.cost_usd, part.price_id, part.basis, part.reasons, part.source_url)
        parts.append(part)
        if part.cost_usd is None:
            missing.append(f'retrieval.{usage.operation}')
        else:
            known += part.cost_usd
    unresolved = any(p.basis == 'unknown' for p in parts)
    status = ('unpriced' if token_cost.price is None or token_cost.status == 'unpriced' else
              'partial' if token_basis != 'exact' or unresolved or missing else 'priced')
    estimated = None
    if token_basis == 'fallback_estimate' and not unresolved:
        estimated = token_cost.cost_usd + sum((p.cost_usd or _ZERO for p in parts), _ZERO)
    return CallCostResult(token_cost, tuple(parts), known if status == 'priced' else None,
        known, estimated, status, token_basis, tuple(dict.fromkeys(missing)), catalog.content_hash)


def price_call(
    *, model: Any, usage: ModelUsage, at: Any, catalog: LoadedCatalog | None = None,
    provider: str | None = None, channel: str | None = None, region: str | None = None,
    retrieval: Sequence[RetrievalUsage] = (),
) -> CallCostResult:
    """Price normalized token and retrieval usage against one snapshot."""
    loaded = catalog if catalog is not None else _bundled_catalog()
    tokens = loaded.calculate_cost(model=model, usage=usage, at=at, provider=provider,
        channel=channel, region=region)
    return compose_call_cost(tokens, catalog=loaded, retrieval=retrieval, provider=provider,
        channel=channel or direct_channel_for_provider(provider), region=region or 'global')


def response_retrieval_usage(response: Any, *, format: str | None = None) -> tuple[RetrievalUsage, ...]:
    """Read operation metadata without reading prompt or response text.

    Tool evidence without a valid usage counter remains unresolved.
    """
    data = _as_mapping(response)
    detected = format or response_format(data)
    if detected == 'anthropic.messages':
        usage = data.get('usage')
        usage = usage if isinstance(usage, Mapping) else {}
        server = usage.get('server_tool_use')
        if server is not None and not isinstance(server, Mapping):
            return (RetrievalUsage('web_search', None),)
        server = server or {}
        found = [RetrievalUsage(op, server[key]) for key, op in (
            ('web_search_requests', 'web_search'), ('web_fetch_requests', 'web_fetch')) if key in server]
        observed = set()
        content = data.get('content')
        if isinstance(content, list):
            for block in content:
                if isinstance(block, Mapping) and block.get('type') == 'server_tool_use':
                    name = block.get('name')
                    if name in ('web_search', 'web_fetch'):
                        observed.add(name)
        found.extend(RetrievalUsage(op, None) for op in sorted(observed) if op not in {f.operation for f in found})
        return tuple(found)
    if detected == 'gemini.generate_content':
        candidates = data.get('candidates')
        if isinstance(candidates, list):
            for item in candidates:
                metadata = _first(item, 'groundingMetadata', 'grounding_metadata') if isinstance(item, Mapping) else None
                if isinstance(metadata, Mapping) and any(
                    _first(metadata, camel, snake) is not None
                    for camel, snake in (("webSearchQueries", "web_search_queries"),
                                         ("groundingChunks", "grounding_chunks"))
                ):
                    # Grounding evidence does not establish billable query count.
                    return (RetrievalUsage('google_search_grounding', None),)
    if detected == 'openai.responses':
        output = data.get('output')
        if isinstance(output, list):
            count = sum(1 for item in output if isinstance(item, Mapping) and item.get('type') == 'web_search_call')
            if count:
                return (RetrievalUsage('web_search', count),)
    return ()


def price_call_from_response(
    provider: str, response: Any, *, catalog: LoadedCatalog | None = None,
    model: str | None = None, at: Any = None, channel: str | None = None,
    region: str | None = None, service_tier: str | None = None, format: str | None = None,
) -> CallCostResult:
    """Price a provider response, including supported counted retrieval fees."""
    loaded = catalog if catalog is not None else _bundled_catalog()
    tokens = cost_from_response(provider, response, catalog=loaded, model=model, at=at,
        channel=channel, region=region, service_tier=service_tier, format=format)
    return compose_call_cost(tokens, catalog=loaded,
        retrieval=response_retrieval_usage(response, format=format), provider=provider,
        channel=channel or direct_channel_for_provider(provider), region=region or 'global')
